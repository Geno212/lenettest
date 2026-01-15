#!/usr/bin/env python3
"""
Catapult AI NN Flow - Local EC2 Execution
==========================================

Run this script ON the EC2 instance (via DCV session).
It takes a PyTorch .pt file and generates RTL using Catapult AI NN.

Usage:
    python3 catapult_local.py model.pt --input-shape 1 3 224 224
    python3 catapult_local.py /path/to/model.pt -o ./output
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Catapult Configuration
CATAPULT_HOME = "/data/tools/catapult/Mgc_home"
LICENSE_SERVER = "29000@10.9.8.8"
HLS4ML_PATH = f"{CATAPULT_HOME}/shared/pkgs/ccs_hls4ml"


def setup_environment():
    """Set up Catapult environment variables."""
    os.environ["MGC_HOME"] = CATAPULT_HOME
    os.environ["MGLS_LICENSE_FILE"] = LICENSE_SERVER
    os.environ["PATH"] = f"{CATAPULT_HOME}/bin:" + os.environ.get("PATH", "")
    
    # Add hls4ml to Python path
    if HLS4ML_PATH not in sys.path:
        sys.path.insert(0, HLS4ML_PATH)


def check_installation():
    """Verify Catapult and hls4ml are installed."""
    print("Checking installation...")
    
    # Check Catapult
    catapult_bin = Path(CATAPULT_HOME) / "bin" / "catapult"
    if not catapult_bin.exists():
        print(f"❌ Catapult not found at {catapult_bin}")
        return False
    print(f"✅ Catapult found: {catapult_bin}")
    
    # Check hls4ml
    hls4ml_dir = Path(HLS4ML_PATH)
    if not hls4ml_dir.exists():
        print(f"⚠️  hls4ml not found at {hls4ml_dir}")
        print("   Will try system hls4ml...")
    else:
        print(f"✅ hls4ml found: {hls4ml_dir}")
    
    return True


def convert_pytorch_to_hls(model_path: str, 
                           output_dir: str,
                           input_shape: tuple,
                           precision: str = "ac_fixed<16,6>",
                           reuse_factor: int = 1,
                           io_type: str = "io_stream",
                           strategy: str = "Latency"):
    """
    Convert PyTorch model to HLS using hls4ml with Catapult backend.
    """
    import torch
    import numpy as np
    
    # Try to import hls4ml
    try:
        import hls4ml
        print(f"✅ hls4ml version: {hls4ml.__version__}")
    except ImportError:
        print("❌ hls4ml not found. Install with: pip install hls4ml")
        return None
    
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    project_name = model_path.stem
    
    print(f"\n{'='*60}")
    print(f"Converting: {model_path.name}")
    print(f"Output: {output_dir}")
    print(f"Input shape: {input_shape}")
    print(f"{'='*60}\n")
    
    # Load PyTorch model
    print("[1/5] Loading PyTorch model...")
    try:
        model = torch.load(str(model_path), map_location='cpu')
        if hasattr(model, 'eval'):
            model.eval()
        print(f"      Model loaded successfully")
        print(f"      Model type: {type(model).__name__}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Print model architecture
    print("\n[2/5] Model Architecture:")
    print("-" * 40)
    if hasattr(model, 'modules'):
        for name, module in model.named_modules():
            if name:  # Skip root module
                print(f"      {name}: {type(module).__name__}")
    else:
        print(f"      {model}")
    print("-" * 40)
    
    # Create sample input for tracing
    print("\n[3/5] Configuring hls4ml...")
    sample_input = torch.randn(*input_shape)
    
    # Configure hls4ml
    try:
        config = hls4ml.utils.config_from_pytorch_model(
            model,
            granularity='name',
            backend='Catapult',
            default_precision=precision,
            default_reuse_factor=reuse_factor
        )
        
        # Set Catapult-specific options
        config['IOType'] = io_type
        config['Backend'] = 'Catapult'
        config['ProjectName'] = project_name
        config['OutputDir'] = str(output_dir / f"hls_{project_name}")
        
        print(f"      Backend: {config.get('Backend')}")
        print(f"      IOType: {config.get('IOType')}")
        print(f"      Precision: {precision}")
        print(f"      ReuseFactor: {reuse_factor}")
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        print("   Trying alternative approach...")
        
        # Alternative: manual config
        config = {
            'Backend': 'Catapult',
            'IOType': io_type,
            'ProjectName': project_name,
            'OutputDir': str(output_dir / f"hls_{project_name}"),
            'HLSConfig': {
                'Model': {
                    'Precision': precision,
                    'ReuseFactor': reuse_factor,
                    'Strategy': strategy
                }
            }
        }
    
    # Convert model
    print("\n[4/5] Converting to HLS...")
    try:
        hls_model = hls4ml.converters.convert_from_pytorch_model(
            model,
            output_dir=str(output_dir / f"hls_{project_name}"),
            input_data_tb=sample_input.numpy(),
            backend='Catapult',
            hls_config=config
        )
        print("      Conversion successful!")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print("\n   This may be because:")
        print("   - The model uses unsupported layers")
        print("   - hls4ml Catapult backend is not available")
        print("\n   Trying Vivado backend as fallback...")
        
        try:
            hls_model = hls4ml.converters.convert_from_pytorch_model(
                model,
                output_dir=str(output_dir / f"hls_{project_name}"),
                input_data_tb=sample_input.numpy(),
                backend='Vivado',
                hls_config=config
            )
            print("      Conversion successful (Vivado backend)")
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return None
    
    # Compile and write project
    print("\n[5/5] Writing HLS project files...")
    try:
        hls_model.compile()
        hls_model.write()
        print("      Project files written!")
    except Exception as e:
        print(f"⚠️  Compile/write warning: {e}")
        # Try just write
        try:
            hls_model.write()
            print("      Project files written (without compile)")
        except:
            pass
    
    # Test prediction comparison
    print("\n" + "="*60)
    print("Validation")
    print("="*60)
    try:
        pytorch_pred = model(sample_input).detach().numpy()
        hls_pred = hls_model.predict(sample_input.numpy())
        
        diff = np.abs(pytorch_pred.flatten() - hls_pred.flatten())
        print(f"PyTorch output shape: {pytorch_pred.shape}")
        print(f"HLS output shape: {hls_pred.shape}")
        print(f"Max difference: {diff.max():.6f}")
        print(f"Mean difference: {diff.mean():.6f}")
        
        if diff.max() < 0.1:
            print("✅ Outputs match within tolerance!")
        else:
            print("⚠️  Large difference - check quantization settings")
    except Exception as e:
        print(f"⚠️  Could not validate: {e}")
    
    return output_dir / f"hls_{project_name}"


def run_catapult_synthesis(hls_project_dir: str, clock_period: float = 10.0):
    """
    Run Catapult HLS synthesis on the generated project.
    """
    hls_project_dir = Path(hls_project_dir)
    
    # Look for build script
    build_tcl = hls_project_dir / "build_prj.tcl"
    
    if not build_tcl.exists():
        print(f"⚠️  No build_prj.tcl found, generating one...")
        
        # Generate a basic Catapult TCL script
        tcl_content = f'''#
# Catapult HLS Synthesis Script
# Auto-generated by catapult_local.py
#

# Project settings
project new -name {hls_project_dir.name}

# Add source files
solution file add {hls_project_dir / "myproject.cpp"}

# Set top module
solution design set myproject -top

# Technology library
solution library add nangate-45nm_beh -- -rtlsyntool DesignCompiler
solution library add ccs_sample_mem

# Clock constraint
directive set -CLOCKS {{clk {{-CLOCK_PERIOD {clock_period} -CLOCK_UNCERTAINTY 0.0}}}}

# Run synthesis flow
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

# Generate RTL
solution options set Output/OutputVerilog true
go extract

# Reports
report timing -file timing_report.txt
report area -file area_report.txt

puts "Synthesis complete!"
exit
'''
        build_tcl.write_text(tcl_content)
        print(f"      Generated: {build_tcl}")
    
    print(f"\n{'='*60}")
    print("Running Catapult Synthesis")
    print(f"{'='*60}")
    print(f"Project: {hls_project_dir}")
    print(f"Clock: {clock_period} ns")
    print()
    
    # Run Catapult
    catapult_cmd = f"catapult -shell -f {build_tcl}"
    print(f"Command: {catapult_cmd}")
    print("-" * 60)
    
    result = subprocess.run(
        catapult_cmd,
        shell=True,
        cwd=str(hls_project_dir),
        capture_output=False  # Show output live
    )
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Catapult AI NN - Local EC2 Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python3 catapult_local.py model.pt
  
  # With input shape
  python3 catapult_local.py model.pt --input-shape 1 3 224 224
  
  # Full options
  python3 catapult_local.py model.pt -o ./output --precision "ac_fixed<8,4>" --clock 5.0
  
  # Just convert, no Catapult synthesis
  python3 catapult_local.py model.pt --convert-only
"""
    )
    
    parser.add_argument("model", help="Path to PyTorch .pt model file")
    parser.add_argument("-o", "--output", default="./catapult_output",
                        help="Output directory (default: ./catapult_output)")
    parser.add_argument("--input-shape", nargs=4, type=int,
                        default=[1, 1, 28, 28],
                        metavar=("B", "C", "H", "W"),
                        help="Input shape: batch channels height width")
    parser.add_argument("--precision", default="ac_fixed<16,6>",
                        help="Fixed-point precision")
    parser.add_argument("--reuse-factor", type=int, default=1,
                        help="Reuse factor (1=max parallelism)")
    parser.add_argument("--io-type", choices=["io_stream", "io_parallel"],
                        default="io_stream", help="I/O type")
    parser.add_argument("--strategy", choices=["Latency", "Resource"],
                        default="Latency", help="Optimization strategy")
    parser.add_argument("--clock", type=float, default=10.0,
                        help="Clock period in ns")
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert, don't run Catapult synthesis")
    parser.add_argument("--check", action="store_true",
                        help="Check installation and exit")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Catapult AI NN Flow - Local Execution")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup environment
    setup_environment()
    
    # Check installation
    if not check_installation():
        return 1
    
    if args.check:
        print("\n✅ Installation check passed!")
        return 0
    
    # Validate model file
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return 1
    
    # Convert
    hls_project = convert_pytorch_to_hls(
        model_path=str(model_path),
        output_dir=args.output,
        input_shape=tuple(args.input_shape),
        precision=args.precision,
        reuse_factor=args.reuse_factor,
        io_type=args.io_type,
        strategy=args.strategy
    )
    
    if hls_project is None:
        print("\n❌ Conversion failed!")
        return 1
    
    print(f"\n✅ HLS project created: {hls_project}")
    
    # Run synthesis
    if not args.convert_only:
        print()
        success = run_catapult_synthesis(hls_project, args.clock)
        if success:
            print("\n✅ Synthesis complete!")
        else:
            print("\n⚠️  Synthesis had issues - check logs")
    else:
        print("\n📝 Skipping synthesis (--convert-only)")
        print(f"   Run manually: catapult -shell -f {hls_project}/build_prj.tcl")
    
    # Summary
    print("\n" + "="*60)
    print("Output Files")
    print("="*60)
    for f in Path(args.output).rglob("*"):
        if f.is_file():
            print(f"  {f.relative_to(args.output)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
