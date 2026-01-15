#!/usr/bin/env python3
"""
PyTorch to Catapult AI NN HLS Conversion Script

This script converts trained PyTorch models (.pt files) to a format compatible
with Siemens Catapult AI NN for RTL synthesis.

The workflow:
1. Load PyTorch model (.pt file)
2. Convert to ONNX format (intermediate)
3. Generate Keras/TensorFlow model (Catapult AI NN compatible)
4. Create configuration files for Catapult AI NN
5. Package for transfer to EC2 for Catapult synthesis

Usage:
    python convert_pytorch_to_hls.py --model path/to/model.pt --output output_dir
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Check for required packages
def check_dependencies():
    """Check and report missing dependencies."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
    
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    
    try:
        import onnx2keras
    except ImportError:
        missing.append("onnx2keras")
    
    try:
        import tensorflow as tf
    except ImportError:
        missing.append("tensorflow")
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


def load_pytorch_model(model_path: str, model_class=None, input_shape=None):
    """
    Load a PyTorch model from a .pt file.
    
    Args:
        model_path: Path to the .pt file
        model_class: Optional model class for state_dict loading
        input_shape: Expected input shape (for tracing)
    
    Returns:
        Loaded PyTorch model
    """
    import torch
    
    print(f"Loading PyTorch model from: {model_path}")
    
    # Try loading as complete model first
    try:
        model = torch.load(model_path, map_location='cpu')
        if isinstance(model, dict):
            # It's a state dict, need model class
            if model_class is None:
                raise ValueError(
                    "Model file contains state_dict. Please provide model_class."
                )
            instance = model_class()
            instance.load_state_dict(model)
            model = instance
        print(f"Model loaded successfully: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def convert_to_onnx(model, output_path: str, input_shape: tuple, 
                    input_names=None, output_names=None):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path for ONNX output file
        input_shape: Input tensor shape (batch, channels, height, width)
        input_names: Names for input tensors
        output_names: Names for output tensors
    
    Returns:
        Path to ONNX file
    """
    import torch
    
    print(f"Converting to ONNX format...")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Default names
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True
    )
    
    print(f"ONNX model saved to: {output_path}")
    return output_path


def convert_onnx_to_keras(onnx_path: str, output_path: str):
    """
    Convert ONNX model to Keras format for Catapult AI NN.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path for Keras model output
    
    Returns:
        Keras model
    """
    import onnx
    from onnx2keras import onnx_to_keras
    
    print(f"Converting ONNX to Keras...")
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Get input names
    input_names = [inp.name for inp in onnx_model.graph.input]
    
    # Convert to Keras
    keras_model = onnx_to_keras(onnx_model, input_names)
    
    # Save Keras model
    keras_model.save(output_path)
    
    print(f"Keras model saved to: {output_path}")
    return keras_model


def generate_catapult_config(model_name: str, output_dir: str, 
                             precision: str = "ac_fixed<16,6>",
                             reuse_factor: int = 1,
                             io_type: str = "io_stream",
                             strategy: str = "Latency",
                             target: str = "asic"):
    """
    Generate Catapult AI NN configuration files.
    
    Args:
        model_name: Name of the model
        output_dir: Output directory
        precision: Fixed-point precision (e.g., "ac_fixed<16,6>")
        reuse_factor: Parallelism control (1=fully parallel, higher=less area)
        io_type: "io_stream" or "io_parallel"
        strategy: "Latency" or "Resource"
        target: "asic" or "fpga"
    """
    
    config = {
        "OutputDir": output_dir,
        "ProjectName": model_name,
        "Backend": "Catapult",
        "Version": "1.0.0",
        "Technology": target,
        "ClockPeriod": 10,
        "IOType": io_type,
        "ProjectDir": "Catapult",
        "HLSConfig": {
            "Model": {
                "Precision": {
                    "default": precision,
                    "maximum": precision
                },
                "ReuseFactor": reuse_factor,
                "Strategy": strategy,
                "BramFactor": 1000000000,
                "TraceOutput": False
            }
        },
        "WriterConfig": {
            "Namespace": None,
            "WriteWeightsTxt": 1,
            "WriteTar": 0
        },
        "ROMLocation": "Local",
        "BuildOptions": {
            "csim": 0,
            "SCVerify": 0,
            "Synth": 1,
            "vhdl": 1,
            "verilog": 1,
            "RTLSynth": 0,
            "RandomTBFrames": 2,
            "PowerEst": 0,
            "PowerOpt": 0,
            "BuildBUP": 0,
            "BUPWorkers": 0,
            "LaunchDA": 0
        }
    }
    
    config_path = os.path.join(output_dir, "hls4ml_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Catapult config saved to: {config_path}")
    return config_path


def generate_catapult_script(model_name: str, output_dir: str, keras_model_path: str):
    """
    Generate the Python script to run on EC2 with Catapult AI NN.
    
    This script will be executed on the EC2 instance where Catapult is installed.
    """
    
    script_content = f'''#!/usr/bin/env python3
"""
Catapult AI NN Synthesis Script
Generated: {datetime.now().isoformat()}

Run this script on the EC2 instance with Catapult installed:
    source $HOME/ccs_venv/bin/activate
    python run_catapult_synthesis.py
"""

import os
import sys
import json

# Add Catapult AI NN to path
MGC_HOME = os.environ.get('MGC_HOME', '/data/tools/catapult/Mgc_home')
sys.path.insert(0, os.path.join(MGC_HOME, 'shared', 'pkgs', 'ccs_hls4ml'))

# Import Catapult AI NN (hls4ml with Catapult backend)
try:
    import hls4ml
    print(f"hls4ml version: {{hls4ml.__version__}}")
except ImportError:
    print("Error: Could not import hls4ml. Make sure you're in the Catapult environment.")
    print("Run: source $HOME/ccs_venv/bin/activate")
    sys.exit(1)

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Configuration
MODEL_NAME = "{model_name}"
KERAS_MODEL_PATH = "keras_model.h5"
CONFIG_PATH = "hls4ml_config.json"

def main():
    print("=" * 60)
    print("Catapult AI NN Synthesis")
    print("=" * 60)
    
    # Load configuration
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    print(f"Project: {{config['ProjectName']}}")
    print(f"Precision: {{config['HLSConfig']['Model']['Precision']['default']}}")
    print(f"ReuseFactor: {{config['HLSConfig']['Model']['ReuseFactor']}}")
    print(f"IOType: {{config['IOType']}}")
    print(f"Strategy: {{config['HLSConfig']['Model']['Strategy']}}")
    
    # Load Keras model
    print(f"\\nLoading Keras model from: {{KERAS_MODEL_PATH}}")
    model = keras.models.load_model(KERAS_MODEL_PATH)
    model.summary()
    
    # Configure hls4ml
    print("\\nConfiguring hls4ml...")
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    
    # Apply custom configuration
    hls_config['Model']['Precision'] = config['HLSConfig']['Model']['Precision']['default']
    hls_config['Model']['ReuseFactor'] = config['HLSConfig']['Model']['ReuseFactor']
    
    # Print layer configuration
    print("\\nLayer Configuration:")
    for layer_name in hls_config['LayerName']:
        print(f"  {{layer_name}}: {{hls_config['LayerName'][layer_name]}}")
    
    # Convert to HLS model
    print("\\nConverting to HLS model...")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=hls_config,
        output_dir=config['OutputDir'],
        project_name=config['ProjectName'],
        part='xcu250-figd2104-2L-e',  # Default Xilinx part, can be changed
        clock_period=config['ClockPeriod'],
        io_type=config['IOType'],
        backend='Catapult'
    )
    
    # Compile the model (generates C++ code)
    print("\\nCompiling HLS model (generating C++)...")
    hls_model.compile()
    
    # Build (runs Catapult synthesis)
    print("\\nRunning Catapult synthesis...")
    print("This may take several minutes...")
    
    build_options = config['BuildOptions']
    hls_model.build(
        csim=build_options.get('csim', False),
        synth=build_options.get('Synth', True),
        vsynth=build_options.get('RTLSynth', False)
    )
    
    # Print reports
    print("\\n" + "=" * 60)
    print("Synthesis Complete!")
    print("=" * 60)
    
    # Read and display results
    results_file = os.path.join(
        config['OutputDir'], 'Catapult', 
        f"{{config['ProjectName']}}.v1", 'nnet_layer_results.txt'
    )
    
    if os.path.exists(results_file):
        print("\\nSynthesis Results:")
        with open(results_file, 'r') as f:
            print(f.read())
    
    print(f"\\nOutput directory: {{config['OutputDir']}}")
    print(f"RTL files in: {{config['OutputDir']}}/Catapult/{{config['ProjectName']}}.v1/")

if __name__ == "__main__":
    main()
'''
    
    script_path = os.path.join(output_dir, "run_catapult_synthesis.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Catapult synthesis script saved to: {script_path}")
    return script_path


def generate_transfer_script(output_dir: str, ec2_user: str = "ubuntu", 
                              ec2_host: str = "10.10.8.216"):
    """Generate script to transfer files to EC2."""
    
    script_content = f'''#!/bin/bash
# Transfer Catapult AI NN project to EC2
# Generated: {datetime.now().isoformat()}

EC2_USER="{ec2_user}"
EC2_HOST="{ec2_host}"
REMOTE_DIR="~/catapult_projects/$(basename $(pwd))"

echo "Transferring to $EC2_USER@$EC2_HOST:$REMOTE_DIR"

# Create remote directory
ssh $EC2_USER@$EC2_HOST "mkdir -p $REMOTE_DIR"

# Transfer files
scp -r ./* $EC2_USER@$EC2_HOST:$REMOTE_DIR/

echo ""
echo "Files transferred successfully!"
echo ""
echo "To run synthesis on EC2:"
echo "  ssh $EC2_USER@$EC2_HOST"
echo "  cd $REMOTE_DIR"
echo "  source \\$HOME/ccs_venv/bin/activate"
echo "  export MGLS_LICENSE_FILE=29000@10.9.8.8"
echo "  python run_catapult_synthesis.py"
'''
    
    script_path = os.path.join(output_dir, "transfer_to_ec2.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Also create a PowerShell version for Windows
    ps_script_content = f'''# Transfer Catapult AI NN project to EC2
# Generated: {datetime.now().isoformat()}

$EC2_USER = "{ec2_user}"
$EC2_HOST = "{ec2_host}"
$LOCAL_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_NAME = Split-Path -Leaf $LOCAL_DIR
$REMOTE_DIR = "~/catapult_projects/$PROJECT_NAME"

Write-Host "Transferring to $EC2_USER@${{EC2_HOST}}:$REMOTE_DIR"

# Create remote directory
ssh "$EC2_USER@$EC2_HOST" "mkdir -p $REMOTE_DIR"

# Transfer files
scp -r "$LOCAL_DIR/*" "$EC2_USER@${{EC2_HOST}}:$REMOTE_DIR/"

Write-Host ""
Write-Host "Files transferred successfully!"
Write-Host ""
Write-Host "To run synthesis on EC2:"
Write-Host "  ssh $EC2_USER@$EC2_HOST"
Write-Host "  cd $REMOTE_DIR"
Write-Host '  source $HOME/ccs_venv/bin/activate'
Write-Host "  export MGLS_LICENSE_FILE=29000@10.9.8.8"
Write-Host "  python run_catapult_synthesis.py"
'''
    
    ps_script_path = os.path.join(output_dir, "transfer_to_ec2.ps1")
    with open(ps_script_path, 'w') as f:
        f.write(ps_script_content)
    
    print(f"Transfer scripts saved to: {script_path}, {ps_script_path}")
    return script_path


def create_readme(output_dir: str, model_name: str):
    """Create README with instructions."""
    
    readme_content = f'''# Catapult AI NN Project: {model_name}

Generated: {datetime.now().isoformat()}

## Contents

- `keras_model.h5` - Keras model converted from PyTorch
- `model.onnx` - ONNX intermediate format
- `hls4ml_config.json` - Catapult AI NN configuration
- `run_catapult_synthesis.py` - Script to run on EC2
- `transfer_to_ec2.ps1` - PowerShell script to transfer to EC2
- `transfer_to_ec2.sh` - Bash script to transfer to EC2

## Quick Start

### 1. Transfer to EC2

**PowerShell (Windows):**
```powershell
cd {output_dir}
.\\transfer_to_ec2.ps1
```

**Or manually:**
```powershell
scp -r {output_dir} ubuntu@10.10.8.216:~/catapult_projects/
```

### 2. Run Synthesis on EC2

```bash
ssh ubuntu@10.10.8.216
cd ~/catapult_projects/{model_name}

# Set up environment
source $HOME/ccs_venv/bin/activate
export MGLS_LICENSE_FILE=29000@10.9.8.8

# Run synthesis
python run_catapult_synthesis.py
```

### 3. Retrieve Results

After synthesis completes, the RTL files will be in:
- `{model_name}/Catapult/{model_name}.v1/` - Verilog/VHDL files
- `{model_name}/Catapult/{model_name}.v1/nnet_layer_results.txt` - Synthesis report

## Configuration Options

Edit `hls4ml_config.json` to modify:

- **Precision**: Fixed-point bit width (e.g., "ac_fixed<16,6>")
- **ReuseFactor**: Parallelism (1=fully parallel, higher=smaller area)
- **IOType**: "io_stream" (pipelined) or "io_parallel"
- **Strategy**: "Latency" or "Resource"

## Troubleshooting

1. **License error**: Ensure `MGLS_LICENSE_FILE=29000@10.9.8.8` is set
2. **Import error**: Run `source $HOME/ccs_venv/bin/activate` first
3. **Missing hls4ml**: Create venv with `$MGC_HOME/bin/python3 $HOME/ccs_venv`
'''
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"README saved to: {readme_path}")
    return readme_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch model to Catapult AI NN format'
    )
    parser.add_argument(
        '--model', '-m', required=True,
        help='Path to PyTorch model (.pt file)'
    )
    parser.add_argument(
        '--output', '-o', default='catapult_output',
        help='Output directory'
    )
    parser.add_argument(
        '--input-shape', '-s', default='1,1,28,28',
        help='Input shape as comma-separated values (batch,channels,height,width)'
    )
    parser.add_argument(
        '--precision', '-p', default='ac_fixed<16,6>',
        help='Fixed-point precision (default: ac_fixed<16,6>)'
    )
    parser.add_argument(
        '--reuse-factor', '-r', type=int, default=1,
        help='Reuse factor for parallelism control (default: 1)'
    )
    parser.add_argument(
        '--io-type', choices=['io_stream', 'io_parallel'], default='io_stream',
        help='I/O type (default: io_stream)'
    )
    parser.add_argument(
        '--strategy', choices=['Latency', 'Resource'], default='Latency',
        help='Optimization strategy (default: Latency)'
    )
    parser.add_argument(
        '--target', choices=['asic', 'fpga'], default='asic',
        help='Target technology (default: asic)'
    )
    parser.add_argument(
        '--ec2-host', default='10.10.8.216',
        help='EC2 host IP address'
    )
    parser.add_argument(
        '--skip-conversion', action='store_true',
        help='Skip model conversion, only generate config files'
    )
    
    args = parser.parse_args()
    
    # Parse input shape
    input_shape = tuple(int(x) for x in args.input_shape.split(','))
    
    # Create output directory
    model_name = Path(args.model).stem
    output_dir = os.path.join(args.output, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PyTorch to Catapult AI NN Converter")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print(f"Input shape: {input_shape}")
    print(f"Precision: {args.precision}")
    print(f"Reuse factor: {args.reuse_factor}")
    print(f"IO type: {args.io_type}")
    print(f"Strategy: {args.strategy}")
    print(f"Target: {args.target}")
    print("=" * 60)
    
    if not args.skip_conversion:
        # Check dependencies
        if not check_dependencies():
            print("\nInstall missing dependencies and retry.")
            sys.exit(1)
        
        # Load PyTorch model
        model = load_pytorch_model(args.model)
        
        # Convert to ONNX
        onnx_path = os.path.join(output_dir, "model.onnx")
        convert_to_onnx(model, onnx_path, input_shape)
        
        # Convert ONNX to Keras
        keras_path = os.path.join(output_dir, "keras_model.h5")
        try:
            convert_onnx_to_keras(onnx_path, keras_path)
        except Exception as e:
            print(f"Warning: ONNX to Keras conversion failed: {e}")
            print("You may need to manually convert the model or use a different approach.")
    
    # Generate Catapult configuration
    generate_catapult_config(
        model_name=model_name,
        output_dir=output_dir,
        precision=args.precision,
        reuse_factor=args.reuse_factor,
        io_type=args.io_type,
        strategy=args.strategy,
        target=args.target
    )
    
    # Generate synthesis script for EC2
    keras_path = os.path.join(output_dir, "keras_model.h5")
    generate_catapult_script(model_name, output_dir, keras_path)
    
    # Generate transfer scripts
    generate_transfer_script(output_dir, ec2_host=args.ec2_host)
    
    # Create README
    create_readme(output_dir, model_name)
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print(f"  1. cd {output_dir}")
    print(f"  2. .\\transfer_to_ec2.ps1")
    print(f"  3. SSH to EC2 and run: python run_catapult_synthesis.py")


if __name__ == "__main__":
    main()
