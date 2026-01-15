"""
Direct PyTorch to Catapult AI NN Converter

This is a simpler approach that works directly with PyTorch models
using Catapult AI NN's native PyTorch support (via hls4ml).

This script is designed to run on the EC2 instance where Catapult is installed.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

def create_pytorch_catapult_project(
    model_path: str,
    output_dir: str,
    model_name: str = None,
    input_shape: tuple = (1, 1, 28, 28),
    precision: str = "ac_fixed<16,6>",
    reuse_factor: int = 1,
    io_type: str = "io_stream",
    strategy: str = "Latency"
):
    """
    Create a Catapult AI NN project from a PyTorch model.
    
    This function creates all necessary files to run Catapult synthesis.
    """
    
    if model_name is None:
        model_name = Path(model_path).stem
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the model file
    import shutil
    dest_model = os.path.join(output_dir, "model.pt")
    shutil.copy2(model_path, dest_model)
    
    # Create configuration
    config = {
        "model_path": "model.pt",
        "model_name": model_name,
        "input_shape": list(input_shape),
        "precision": precision,
        "reuse_factor": reuse_factor,
        "io_type": io_type,
        "strategy": strategy,
        "clock_period": 10,
        "output_dir": f"{model_name}_hls",
        "backend": "Catapult",
        "build_options": {
            "csim": False,
            "synth": True,
            "cosim": False,
            "vsynth": False
        }
    }
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create the synthesis script
    synthesis_script = f'''#!/usr/bin/env python3
"""
Catapult AI NN PyTorch Synthesis Script
Model: {model_name}
Generated: {datetime.now().isoformat()}

Prerequisites:
    export MGLS_LICENSE_FILE=29000@10.9.8.8
    source $HOME/ccs_venv/bin/activate
"""

import os
import sys
import json
import torch
import numpy as np

# Setup paths
MGC_HOME = os.environ.get('MGC_HOME', '/data/tools/catapult/Mgc_home')
HLS4ML_PATH = os.path.join(MGC_HOME, 'shared', 'pkgs', 'ccs_hls4ml')
sys.path.insert(0, HLS4ML_PATH)

print(f"MGC_HOME: {{MGC_HOME}}")
print(f"HLS4ML_PATH: {{HLS4ML_PATH}}")

# Import hls4ml
try:
    import hls4ml
    print(f"hls4ml loaded successfully")
except ImportError as e:
    print(f"Error importing hls4ml: {{e}}")
    print("Make sure you've activated the Catapult Python environment:")
    print("  source $HOME/ccs_venv/bin/activate")
    sys.exit(1)

def load_model():
    """Load the PyTorch model."""
    print("\\nLoading PyTorch model...")
    model = torch.load("model.pt", map_location='cpu')
    
    # Handle state dict vs full model
    if isinstance(model, dict) and 'state_dict' in model:
        print("Model contains state_dict, need model architecture")
        # Try to find model class
        raise ValueError("Model is a state_dict. Need full model or model class.")
    
    model.eval()
    print(f"Model type: {{type(model).__name__}}")
    return model

def convert_and_synthesize():
    """Main conversion and synthesis flow."""
    
    # Load configuration
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    print("=" * 60)
    print("Catapult AI NN PyTorch Synthesis")
    print("=" * 60)
    print(f"Model: {{config['model_name']}}")
    print(f"Input shape: {{config['input_shape']}}")
    print(f"Precision: {{config['precision']}}")
    print(f"Reuse factor: {{config['reuse_factor']}}")
    print(f"IO type: {{config['io_type']}}")
    print(f"Strategy: {{config['strategy']}}")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    # Create dummy input for tracing
    input_shape = tuple(config['input_shape'])
    dummy_input = torch.randn(*input_shape)
    
    # Trace the model (creates a TorchScript model)
    print("\\nTracing model...")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save("traced_model.pt")
    print("Traced model saved to: traced_model.pt")
    
    # Configure hls4ml for PyTorch
    print("\\nConfiguring hls4ml...")
    
    # Create hls4ml configuration
    hls_config = {{
        'Model': {{
            'Precision': config['precision'],
            'ReuseFactor': config['reuse_factor'],
        }}
    }}
    
    # Convert PyTorch model to HLS
    print("\\nConverting to HLS model...")
    
    try:
        # Try using hls4ml's PyTorch converter
        hls_model = hls4ml.converters.convert_from_pytorch_model(
            model,
            input_shape=[input_shape],
            hls_config=hls_config,
            output_dir=config['output_dir'],
            project_name=config['model_name'],
            backend='Catapult',
            clock_period=config['clock_period'],
            io_type=config['io_type']
        )
    except AttributeError:
        # Fallback: Convert via ONNX
        print("Direct PyTorch conversion not available, using ONNX intermediate...")
        
        # Export to ONNX
        onnx_path = "model.onnx"
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=['input'], output_names=['output'],
            opset_version=11
        )
        print(f"ONNX model saved to: {{onnx_path}}")
        
        # Convert from ONNX
        hls_model = hls4ml.converters.convert_from_onnx_model(
            onnx_path,
            hls_config=hls_config,
            output_dir=config['output_dir'],
            project_name=config['model_name'],
            backend='Catapult',
            clock_period=config['clock_period'],
            io_type=config['io_type']
        )
    
    # Compile (generates C++ code)
    print("\\nCompiling HLS model...")
    hls_model.compile()
    print("C++ code generated successfully!")
    
    # Test the model (optional)
    print("\\nTesting HLS model prediction...")
    test_input = np.random.randn(*input_shape).astype(np.float32)
    
    # PyTorch prediction
    with torch.no_grad():
        pytorch_output = model(torch.from_numpy(test_input)).numpy()
    
    # HLS prediction
    hls_output = hls_model.predict(test_input)
    
    print(f"PyTorch output: {{pytorch_output[0][:5]}}...")
    print(f"HLS output: {{hls_output[0][:5]}}...")
    
    # Build (runs Catapult synthesis)
    print("\\nRunning Catapult HLS synthesis...")
    print("This may take several minutes...")
    
    build_opts = config['build_options']
    hls_model.build(
        csim=build_opts.get('csim', False),
        synth=build_opts.get('synth', True),
        cosim=build_opts.get('cosim', False),
        vsynth=build_opts.get('vsynth', False)
    )
    
    # Print results
    print("\\n" + "=" * 60)
    print("Synthesis Complete!")
    print("=" * 60)
    
    # Find and display results
    results_path = os.path.join(
        config['output_dir'], 'Catapult',
        f"{{config['model_name']}}.v1", 'nnet_layer_results.txt'
    )
    
    if os.path.exists(results_path):
        print("\\nSynthesis Results:")
        print("-" * 60)
        with open(results_path, 'r') as f:
            print(f.read())
    
    print(f"\\nOutput files in: {{config['output_dir']}}/")
    print(f"RTL files in: {{config['output_dir']}}/Catapult/{{config['model_name']}}.v1/")
    
    return hls_model

if __name__ == "__main__":
    convert_and_synthesize()
'''
    
    script_path = os.path.join(output_dir, "synthesize.py")
    with open(script_path, 'w') as f:
        f.write(synthesis_script)
    
    # Create setup script
    setup_script = f'''#!/bin/bash
# Setup script for Catapult AI NN synthesis
# Run this on the EC2 instance

echo "Setting up Catapult AI NN environment..."

# Set license
export MGLS_LICENSE_FILE=29000@10.9.8.8

# Set MGC_HOME if not already set
export MGC_HOME=${{MGC_HOME:-/data/tools/catapult/Mgc_home}}

# Create virtual environment if it doesn't exist
if [ ! -d "$HOME/ccs_venv" ]; then
    echo "Creating Python virtual environment..."
    $MGC_HOME/bin/python3 -m venv $HOME/ccs_venv
    source $HOME/ccs_venv/bin/activate
    pip install numpy torch
else
    source $HOME/ccs_venv/bin/activate
fi

echo ""
echo "Environment ready!"
echo "Run: python synthesize.py"
'''
    
    setup_path = os.path.join(output_dir, "setup.sh")
    with open(setup_path, 'w') as f:
        f.write(setup_script)
    
    # Create README
    readme = f'''# Catapult AI NN Project: {model_name}

Generated: {datetime.now().isoformat()}

## Files

- `model.pt` - PyTorch model
- `config.json` - Synthesis configuration
- `synthesize.py` - Main synthesis script (run on EC2)
- `setup.sh` - Environment setup script

## Usage

### 1. Transfer to EC2

```powershell
scp -r {output_dir} ubuntu@10.10.8.216:~/catapult_projects/
```

### 2. Setup Environment on EC2

```bash
ssh ubuntu@10.10.8.216
cd ~/catapult_projects/{model_name}
chmod +x setup.sh
./setup.sh
```

### 3. Run Synthesis

```bash
python synthesize.py
```

## Configuration

Edit `config.json` to change:

- `precision`: Fixed-point format (e.g., "ac_fixed<16,6>" = 16 bits, 6 integer)
- `reuse_factor`: 1=fully parallel (fast), higher=less resources
- `io_type`: "io_stream" (pipelined) or "io_parallel"
- `strategy`: "Latency" (fastest) or "Resource" (smallest)

## Output

After synthesis:
- RTL: `{model_name}_hls/Catapult/{model_name}.v1/concat_rtl.v`
- Report: `{model_name}_hls/Catapult/{model_name}.v1/nnet_layer_results.txt`
'''
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    print(f"Project created in: {output_dir}")
    print(f"Files: model.pt, config.json, synthesize.py, setup.sh, README.md")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Create Catapult AI NN project from PyTorch model'
    )
    parser.add_argument('model', help='Path to PyTorch model (.pt file)')
    parser.add_argument('-o', '--output', default='./catapult_project',
                        help='Output directory')
    parser.add_argument('-n', '--name', help='Project name (default: model filename)')
    parser.add_argument('-s', '--shape', default='1,1,28,28',
                        help='Input shape: batch,channels,height,width')
    parser.add_argument('-p', '--precision', default='ac_fixed<16,6>',
                        help='Fixed-point precision')
    parser.add_argument('-r', '--reuse', type=int, default=1,
                        help='Reuse factor (1=parallel)')
    parser.add_argument('--io-type', choices=['io_stream', 'io_parallel'],
                        default='io_stream', help='IO type')
    parser.add_argument('--strategy', choices=['Latency', 'Resource'],
                        default='Latency', help='Optimization strategy')
    
    args = parser.parse_args()
    
    input_shape = tuple(int(x) for x in args.shape.split(','))
    
    create_pytorch_catapult_project(
        model_path=args.model,
        output_dir=args.output,
        model_name=args.name,
        input_shape=input_shape,
        precision=args.precision,
        reuse_factor=args.reuse,
        io_type=args.io_type,
        strategy=args.strategy
    )


if __name__ == "__main__":
    main()
