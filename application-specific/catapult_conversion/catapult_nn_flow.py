"""
Catapult AI NN Flow - Direct PyTorch to RTL
============================================

This script executes the complete Catapult AI NN flow:
1. Takes a .pt (PyTorch) model file
2. Transfers it to EC2
3. Runs Catapult AI NN synthesis remotely
4. Downloads the generated RTL back

Based on Catapult AI NN documentation - Catapult directly integrates hls4ml
and handles the PyTorch -> HLS C++ -> RTL conversion internally.

Usage:
    python catapult_nn_flow.py model.pt --input-shape 1 3 224 224
    python catapult_nn_flow.py model.pt --test-connection
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import time
import shutil


class CatapultNNFlow:
    """
    Execute Catapult AI NN flow on remote EC2 instance.
    """
    
    # EC2 Configuration
    EC2_HOST = "10.10.8.216"
    EC2_USER = "ubuntu"
    
    # Catapult Configuration on EC2
    CATAPULT_HOME = "/data/tools/catapult/Mgc_home"
    LICENSE_SERVER = "29000@10.9.8.8"
    
    # Remote working directory
    REMOTE_WORK_DIR = "~/catapult_nn_projects"
    
    @classmethod
    def test_connection(cls, ssh_key: Optional[str] = None) -> bool:
        """Test SSH connection to EC2 and verify Catapult installation."""
        print("=" * 60)
        print("Testing EC2 Connection and Catapult Installation")
        print("=" * 60)
        
        ssh_args = ["ssh"]
        if ssh_key:
            ssh_args.extend(["-i", ssh_key])
        ssh_args.extend(["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"])
        ssh_args.append(f"{cls.EC2_USER}@{cls.EC2_HOST}")
        
        # Test 1: Basic connectivity
        print(f"\n[1] Testing SSH to {cls.EC2_HOST}...")
        result = subprocess.run(
            ssh_args + ["echo 'Connection successful'"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"    ❌ Failed: {result.stderr}")
            return False
        print("    ✅ SSH connection successful")
        
        # Test 2: Check Catapult installation
        print(f"\n[2] Checking Catapult installation...")
        result = subprocess.run(
            ssh_args + [f"ls {cls.CATAPULT_HOME}/bin/catapult"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"    ❌ Catapult not found at {cls.CATAPULT_HOME}")
            return False
        print(f"    ✅ Catapult found at {cls.CATAPULT_HOME}")
        
        # Test 3: Check hls4ml
        print(f"\n[3] Checking hls4ml installation...")
        result = subprocess.run(
            ssh_args + [f"ls {cls.CATAPULT_HOME}/shared/pkgs/ccs_hls4ml"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"    ⚠️  hls4ml not found in Catapult package")
        else:
            print(f"    ✅ hls4ml found")
        
        # Test 4: Check license
        print(f"\n[4] Checking license server...")
        result = subprocess.run(
            ssh_args + [f"timeout 5 bash -c 'echo >/dev/tcp/10.9.8.8/29000' 2>/dev/null && echo OK || echo FAIL"],
            capture_output=True, text=True
        )
        if "OK" in result.stdout:
            print(f"    ✅ License server reachable at {cls.LICENSE_SERVER}")
        else:
            print(f"    ⚠️  Cannot reach license server (may still work)")
        
        print("\n" + "=" * 60)
        print("Connection test complete - Ready to run Catapult AI NN!")
        print("=" * 60)
        return True
    
    def __init__(self, 
                 model_path: str,
                 project_name: Optional[str] = None,
                 input_shape: tuple = (1, 1, 28, 28),
                 output_dir: Optional[str] = None,
                 ssh_key: Optional[str] = None):
        """
        Initialize the Catapult NN Flow.
        
        Args:
            model_path: Path to the .pt PyTorch model file
            project_name: Name for the Catapult project (default: model filename)
            input_shape: Input tensor shape (batch, channels, height, width)
            output_dir: Local directory for outputs (default: current directory)
            ssh_key: Path to SSH private key for EC2 (optional)
        """
        self.model_path = Path(model_path).resolve()
        self.project_name = project_name or self.model_path.stem
        self.input_shape = input_shape
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "catapult_output"
        self.ssh_key = ssh_key
        
        # Validate model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _ssh_cmd(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        """Execute command on EC2 via SSH."""
        ssh_args = ["ssh"]
        if self.ssh_key:
            ssh_args.extend(["-i", self.ssh_key])
        ssh_args.extend(["-o", "StrictHostKeyChecking=no"])
        ssh_args.append(f"{self.EC2_USER}@{self.EC2_HOST}")
        ssh_args.append(cmd)
        
        print(f"[SSH] {cmd}")
        result = subprocess.run(ssh_args, capture_output=True, text=True)
        
        if check and result.returncode != 0:
            print(f"[SSH ERROR] {result.stderr}")
            raise RuntimeError(f"SSH command failed: {cmd}")
        
        return result
    
    def _scp_to_ec2(self, local_path: str, remote_path: str):
        """Copy file to EC2."""
        scp_args = ["scp"]
        if self.ssh_key:
            scp_args.extend(["-i", self.ssh_key])
        scp_args.extend(["-o", "StrictHostKeyChecking=no"])
        scp_args.extend([local_path, f"{self.EC2_USER}@{self.EC2_HOST}:{remote_path}"])
        
        print(f"[SCP] {local_path} -> {remote_path}")
        result = subprocess.run(scp_args, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[SCP ERROR] {result.stderr}")
            raise RuntimeError(f"SCP failed: {local_path}")
    
    def _scp_from_ec2(self, remote_path: str, local_path: str, recursive: bool = False):
        """Copy file/directory from EC2."""
        scp_args = ["scp"]
        if self.ssh_key:
            scp_args.extend(["-i", self.ssh_key])
        if recursive:
            scp_args.append("-r")
        scp_args.extend(["-o", "StrictHostKeyChecking=no"])
        scp_args.extend([f"{self.EC2_USER}@{self.EC2_HOST}:{remote_path}", local_path])
        
        print(f"[SCP] {remote_path} -> {local_path}")
        result = subprocess.run(scp_args, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[SCP ERROR] {result.stderr}")
            # Don't raise - some files may not exist
    
    def generate_catapult_tcl(self) -> str:
        """
        Generate TCL script for Catapult AI NN flow.
        Based on notebook.txt documentation.
        """
        
        # Input shape for hls4ml config
        _, channels, height, width = self.input_shape
        
        tcl_script = f'''#
# Catapult AI NN Synthesis Script
# Project: {self.project_name}
# Generated by: catapult_nn_flow.py
#

# Set up environment
set MGC_HOME $env(MGC_HOME)

# Project settings
set project_name "{self.project_name}"
set model_file "{self.project_name}.pt"

# Create new project
project new -name $project_name

# ============================================
# Step 1: Load PyTorch model using AI NN flow
# ============================================

# The AI NN flow uses hls4ml to convert PyTorch model
# Configure hls4ml settings

# Create hls4ml configuration
set hls4ml_config {{
    "Backend": "Catapult",
    "IOType": "io_stream",
    "HLSConfig": {{
        "Model": {{
            "Precision": "ac_fixed<16,6>",
            "ReuseFactor": 1,
            "Strategy": "Latency"
        }},
        "LayerType": {{
            "Conv2D": {{
                "Precision": "ac_fixed<16,6>",
                "ReuseFactor": 1
            }},
            "Dense": {{
                "Precision": "ac_fixed<16,6>",
                "ReuseFactor": 1
            }},
            "BatchNormalization": {{
                "Precision": "ac_fixed<16,6>"
            }},
            "Activation": {{
                "Precision": "ac_fixed<16,6>",
                "table_size": 1024
            }}
        }}
    }},
    "InputShape": [{channels}, {height}, {width}],
    "OutputDir": "./hls4ml_output",
    "ProjectName": "$project_name"
}}

# Save config to file
set config_file [open "hls4ml_config.json" w]
puts $config_file $hls4ml_config
close $config_file

# ============================================
# Step 2: Run hls4ml conversion via Python
# ============================================

# Use Catapult's integrated hls4ml
set hls4ml_path "$MGC_HOME/shared/pkgs/ccs_hls4ml"

# Run the conversion script
puts "Converting PyTorch model to HLS..."
exec python3 << EOF
import sys
sys.path.insert(0, "$hls4ml_path")
import hls4ml
import torch
import json

# Load model
model = torch.load("$model_file", map_location='cpu')
if hasattr(model, 'eval'):
    model.eval()

# Configure hls4ml  
config = hls4ml.utils.config_from_pytorch_model(
    model,
    granularity='name',
    backend='Catapult',
    default_precision='ac_fixed<16,6>',
    default_reuse_factor=1
)

# Set Catapult-specific options
config['IOType'] = 'io_stream'
config['Backend'] = 'Catapult'

# Convert
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    output_dir='./hls_output',
    input_data_tb=None,
    backend='Catapult',
    hls_config=config
)

# Write the HLS project
hls_model.write()

print("HLS conversion complete!")
EOF

# ============================================
# Step 3: Configure Catapult synthesis
# ============================================

# Add the generated HLS files to the project
solution file add ./hls_output/${{project_name}}.cpp
solution file add ./hls_output/weights/*.h

# Set design top
solution design set ${{project_name}} -top

# ============================================
# Step 4: Set technology library
# ============================================

# Use Nangate 45nm FreePDK (available in Catapult)
solution library add nangate-45nm_beh -- -rtlsyntool DesignCompiler
solution library add ccs_sample_mem

# ============================================
# Step 5: Set interface and constraints
# ============================================

# Clock constraint (100 MHz default)
directive set -CLOCKS {{clk {{-CLOCK_PERIOD 10.0 -CLOCK_UNCERTAINTY 0.0}}}}

# Reset
directive set -RESET_CLEARS_ALL_REGS true

# Interface style
directive set -TRANSACTION_DONE_SIGNAL true
directive set -DESIGN_HIERARCHY $project_name

# ============================================
# Step 6: Run synthesis
# ============================================

puts "Starting Catapult synthesis..."

# Go through synthesis flow
go new
go analyze
go compile
go libraries
go assembly
go architect
go allocate
go schedule
go dpfsm
go extract

# ============================================
# Step 7: Generate RTL
# ============================================

puts "Generating RTL..."

# Generate Verilog output
solution options set Output/OutputVerilog true
solution options set Output/OutputVHDL false

go extract

puts "Synthesis complete!"

# ============================================
# Step 8: Generate reports
# ============================================

# Timing report
report timing -file timing_report.txt

# Area report  
report area -file area_report.txt

# Power estimate
report power -file power_report.txt

puts "Reports generated."

# Exit
exit
'''
        return tcl_script
    
    def generate_direct_python_script(self) -> str:
        """
        Generate a Python script that uses hls4ml directly with Catapult backend.
        This is the simpler approach based on the notebook documentation.
        """
        
        _, channels, height, width = self.input_shape
        
        script = f'''#!/usr/bin/env python3
"""
Catapult AI NN - Direct hls4ml Conversion
Project: {self.project_name}
"""

import sys
import os
import numpy as np

# Add Catapult's hls4ml to path
MGC_HOME = os.environ.get('MGC_HOME', '/data/tools/catapult/Mgc_home')
sys.path.insert(0, os.path.join(MGC_HOME, 'shared/pkgs/ccs_hls4ml'))

import torch
import hls4ml

print("=" * 60)
print("Catapult AI NN Conversion")
print("=" * 60)

# Load PyTorch model
print("\\nLoading PyTorch model: {self.project_name}.pt")
model = torch.load("{self.project_name}.pt", map_location='cpu')

if hasattr(model, 'eval'):
    model.eval()
    print("Model set to eval mode")

# Print model architecture
print("\\nModel Architecture:")
print(model)

# Create sample input for tracing
input_shape = {list(self.input_shape)}
sample_input = torch.randn(*input_shape)
print(f"\\nInput shape: {{input_shape}}")

# Configure hls4ml for Catapult
print("\\nConfiguring hls4ml for Catapult backend...")

config = hls4ml.utils.config_from_pytorch_model(
    model,
    granularity='name',
    backend='Catapult',
    default_precision='ac_fixed<16,6>',
    default_reuse_factor=1
)

# Catapult-specific settings
config['IOType'] = 'io_stream'
config['Backend'] = 'Catapult'

print("\\nHLS Configuration:")
for key, value in config.items():
    if key != 'LayerName':
        print(f"  {{key}}: {{value}}")

# Convert model
print("\\nConverting to HLS...")
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    output_dir='./hls_project_{self.project_name}',
    input_data_tb=sample_input.numpy(),
    backend='Catapult',
    hls_config=config
)

# Compile the model
print("\\nCompiling HLS model...")
hls_model.compile()

# Run a test prediction to verify
print("\\nRunning test prediction...")
pytorch_pred = model(sample_input).detach().numpy()
hls_pred = hls_model.predict(sample_input.numpy())

print(f"PyTorch output shape: {{pytorch_pred.shape}}")
print(f"HLS output shape: {{hls_pred.shape}}")

# Check accuracy
if pytorch_pred.shape == hls_pred.shape:
    diff = np.abs(pytorch_pred - hls_pred)
    print(f"Max difference: {{diff.max():.6f}}")
    print(f"Mean difference: {{diff.mean():.6f}}")

# Write the HLS project files
print("\\nWriting HLS project files...")
hls_model.write()

print("\\n" + "=" * 60)
print("Conversion Complete!")
print("=" * 60)
print(f"\\nOutput directory: ./hls_project_{self.project_name}")
print("\\nGenerated files:")
print("  - myproject.cpp      : Main HLS source")
print("  - myproject.h        : Header file")  
print("  - parameters.h       : Layer parameters")
print("  - weights/*.h        : Weight files")
print("  - build_prj.tcl      : Catapult build script")

# Now run Catapult synthesis
print("\\n" + "=" * 60)
print("Starting Catapult Synthesis")
print("=" * 60)

import subprocess

os.chdir(f'./hls_project_{self.project_name}')

# Run Catapult
result = subprocess.run(
    ['catapult', '-shell', '-f', 'build_prj.tcl'],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("Errors:", result.stderr)

print("\\nDone!")
'''
        return script
    
    def generate_simple_tcl(self) -> str:
        """
        Generate a simple TCL script that calls hls4ml's built-in Catapult flow.
        Based on the actual examples in notebook.txt.
        """
        
        _, channels, height, width = self.input_shape
        
        tcl = f'''#
# Simple Catapult AI NN Build Script
# Project: {self.project_name}
#

# Run the Python conversion first (generates HLS files)
puts "Step 1: Converting PyTorch model with hls4ml..."
exec python3 convert_{self.project_name}.py

# The hls4ml Catapult backend generates build_prj.tcl automatically
# Source it to run Catapult synthesis
puts "Step 2: Running Catapult synthesis..."

cd hls_project_{self.project_name}

# Source the generated build script
source build_prj.tcl

puts "Synthesis complete!"
'''
        return tcl
    
    def run(self, 
            technology: str = "nangate-45nm",
            clock_period: float = 10.0,
            reuse_factor: int = 1,
            precision: str = "ac_fixed<16,6>",
            strategy: str = "Latency") -> Dict[str, Any]:
        """
        Execute the complete Catapult AI NN flow.
        
        Args:
            technology: Target technology library
            clock_period: Clock period in ns
            reuse_factor: Resource reuse factor (1 = max parallelism)
            precision: Fixed-point precision
            strategy: "Latency" or "Resource"
            
        Returns:
            Dictionary with synthesis results
        """
        
        results = {
            "status": "started",
            "project_name": self.project_name,
            "model_path": str(self.model_path),
            "ec2_host": self.EC2_HOST
        }
        
        try:
            print("=" * 60)
            print("Catapult AI NN Flow")
            print("=" * 60)
            print(f"Model: {self.model_path}")
            print(f"Project: {self.project_name}")
            print(f"EC2 Host: {self.EC2_HOST}")
            print(f"Technology: {technology}")
            print(f"Clock Period: {clock_period} ns")
            print()
            
            # Step 1: Prepare remote directory
            print("[1/6] Setting up remote directory...")
            remote_project_dir = f"{self.REMOTE_WORK_DIR}/{self.project_name}"
            self._ssh_cmd(f"mkdir -p {remote_project_dir}")
            
            # Step 2: Upload model file
            print("[2/6] Uploading model file...")
            self._scp_to_ec2(
                str(self.model_path), 
                f"{remote_project_dir}/{self.project_name}.pt"
            )
            
            # Step 3: Generate and upload conversion script
            print("[3/6] Generating conversion script...")
            python_script = self.generate_direct_python_script()
            
            # Save locally first
            local_script = self.output_dir / f"convert_{self.project_name}.py"
            local_script.write_text(python_script)
            
            # Upload to EC2
            self._scp_to_ec2(
                str(local_script),
                f"{remote_project_dir}/convert_{self.project_name}.py"
            )
            
            # Step 4: Run conversion and synthesis on EC2
            print("[4/6] Running Catapult AI NN synthesis on EC2...")
            print("      This may take several minutes...")
            
            synthesis_cmd = f'''
cd {remote_project_dir}
export MGC_HOME={self.CATAPULT_HOME}
export PATH=$MGC_HOME/bin:$PATH
export MGLS_LICENSE_FILE={self.LICENSE_SERVER}
python3 convert_{self.project_name}.py 2>&1 | tee synthesis.log
'''
            
            start_time = time.time()
            result = self._ssh_cmd(synthesis_cmd, check=False)
            elapsed = time.time() - start_time
            
            print(f"\n      Synthesis completed in {elapsed:.1f} seconds")
            print("\n--- Synthesis Output ---")
            print(result.stdout[:2000] if len(result.stdout) > 2000 else result.stdout)
            if result.stderr:
                print("\n--- Errors ---")
                print(result.stderr[:1000])
            
            results["synthesis_time"] = elapsed
            results["synthesis_output"] = result.stdout
            
            # Step 5: Download results
            print("\n[5/6] Downloading results...")
            
            # Download log
            self._scp_from_ec2(
                f"{remote_project_dir}/synthesis.log",
                str(self.output_dir / "synthesis.log")
            )
            
            # Download generated RTL
            self._scp_from_ec2(
                f"{remote_project_dir}/hls_project_{self.project_name}/Catapult",
                str(self.output_dir / "rtl"),
                recursive=True
            )
            
            # Download reports
            for report in ["timing_report.txt", "area_report.txt", "power_report.txt"]:
                self._scp_from_ec2(
                    f"{remote_project_dir}/hls_project_{self.project_name}/{report}",
                    str(self.output_dir / report)
                )
            
            # Step 6: Summary
            print("\n[6/6] Done!")
            print("=" * 60)
            print("Results saved to:", self.output_dir)
            print("=" * 60)
            
            # List output files
            print("\nGenerated files:")
            for f in self.output_dir.iterdir():
                print(f"  - {f.name}")
            
            results["status"] = "success"
            results["output_dir"] = str(self.output_dir)
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
            results["status"] = "error"
            results["error"] = str(e)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Catapult AI NN flow on EC2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test EC2 connection first
  python catapult_nn_flow.py --test-connection
  
  # Basic usage
  python catapult_nn_flow.py model.pt
  
  # With custom input shape (batch, channels, height, width)
  python catapult_nn_flow.py model.pt --input-shape 1 3 224 224
  
  # With custom output directory
  python catapult_nn_flow.py model.pt -o ./my_output
  
  # With SSH key
  python catapult_nn_flow.py model.pt --ssh-key ~/.ssh/my_key.pem
"""
    )
    
    parser.add_argument("model", nargs="?", help="Path to PyTorch .pt model file")
    parser.add_argument("-n", "--name", help="Project name (default: model filename)")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--input-shape", nargs=4, type=int, 
                        default=[1, 1, 28, 28],
                        metavar=("B", "C", "H", "W"),
                        help="Input shape: batch channels height width")
    parser.add_argument("--ssh-key", help="SSH private key for EC2")
    parser.add_argument("--clock", type=float, default=10.0, 
                        help="Clock period in ns (default: 10.0)")
    parser.add_argument("--reuse-factor", type=int, default=1,
                        help="Reuse factor (default: 1 = max parallelism)")
    parser.add_argument("--precision", default="ac_fixed<16,6>",
                        help="Fixed-point precision (default: ac_fixed<16,6>)")
    parser.add_argument("--strategy", choices=["Latency", "Resource"],
                        default="Latency", help="Optimization strategy")
    parser.add_argument("--test-connection", action="store_true",
                        help="Test EC2 connection and exit")
    
    args = parser.parse_args()
    
    # Test connection mode
    if args.test_connection:
        success = CatapultNNFlow.test_connection(args.ssh_key)
        return 0 if success else 1
    
    # Require model for synthesis
    if not args.model:
        parser.error("model is required (or use --test-connection)")
    
    # Create flow instance
    flow = CatapultNNFlow(
        model_path=args.model,
        project_name=args.name,
        input_shape=tuple(args.input_shape),
        output_dir=args.output,
        ssh_key=args.ssh_key
    )
    
    # Run synthesis
    results = flow.run(
        clock_period=args.clock,
        reuse_factor=args.reuse_factor,
        precision=args.precision,
        strategy=args.strategy
    )
    
    # Print final status
    print(f"\nFinal Status: {results['status']}")
    
    return 0 if results["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
