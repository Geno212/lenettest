#!/usr/bin/env python3
"""
Catapult AI NN - PyTorch Model Flow
====================================

This script follows the EXACT Catapult AI NN flow from notebook.txt,
but adapted for PyTorch models instead of Keras.

The flow is:
1. Load PyTorch model (.pt file)
2. Configure hls4ml with Catapult backend
3. Convert to HLS C++ 
4. Run Catapult synthesis -> RTL

Run this ON EC2 where Catapult is installed:
    source $HOME/ccs_venv/bin/activate
    python3 catapult_pytorch_flow.py model.pt --input-shape 1 1 28 28

Or directly with Catapult's Python:
    $MGC_HOME/bin/python3 catapult_pytorch_flow.py model.pt --input-shape 1 1 28 28
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path

# Setup Catapult environment
MGC_HOME = os.environ.get('MGC_HOME', '/data/tools/catapult/Mgc_home')
os.environ['MGC_HOME'] = MGC_HOME

# Add Catapult's hls4ml to Python path
HLS4ML_PATH = f"{MGC_HOME}/shared/pkgs/ccs_hls4ml"
if HLS4ML_PATH not in sys.path:
    sys.path.insert(0, HLS4ML_PATH)

import torch


def load_pytorch_model(model_path: str):
    """Load a PyTorch model from .pt file."""
    print(f"Loading PyTorch model: {model_path}")
    
    model = torch.load(model_path, map_location='cpu')
    
    # Handle different save formats
    if isinstance(model, dict):
        # State dict - need the model architecture
        print("  Model is a state dict - need model architecture")
        raise ValueError("Please save the complete model with torch.save(model, path)")
    
    if hasattr(model, 'eval'):
        model.eval()
    
    print(f"  Model type: {type(model).__name__}")
    return model


def configure_hls4ml_for_catapult(
    model,
    input_shape: tuple,
    output_dir: str,
    project_name: str = "myproject",
    precision: str = "ac_fixed<16,6>",
    reuse_factor: int = 1,
    io_type: str = "io_stream",
    strategy: str = "Latency"
):
    """
    Configure hls4ml for Catapult backend.
    This mirrors the notebook.txt In[13] and In[14] steps.
    """
    import hls4ml
    
    print("\n" + "="*60)
    print("Configuring hls4ml for Catapult")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample input for tracing
    sample_input = torch.randn(*input_shape)
    
    # Get predictions for testbench
    with torch.no_grad():
        sample_output = model(sample_input).numpy()
    
    # Save testbench data (like notebook.txt In[13])
    tb_input_path = output_dir / "tb_input_features.dat"
    tb_output_path = output_dir / "tb_output_predictions.dat"
    
    np.savetxt(tb_input_path, sample_input.numpy().flatten(), fmt='%.6f')
    np.savetxt(tb_output_path, sample_output.flatten(), fmt='%.6f')
    
    print(f"Saved testbench data:")
    print(f"  Input: {tb_input_path}")
    print(f"  Output: {tb_output_path}")
    
    # Configure hls4ml (Catapult-bundled version uses dict directly)
    print("\nCreating hls4ml configuration...")
    
    # Build config dict manually (Catapult hls4ml doesn't have utils module)
    output_dir_str = str(output_dir / f"{project_name}-Catapult-test")
    
    config = {
        'Backend': 'Catapult',
        'IOType': io_type,
        'ProjectName': project_name,
        'OutputDir': output_dir_str,
        'ClockPeriod': 10,
        'Model': {
            'Strategy': strategy,
            'ReuseFactor': reuse_factor,
            'Precision': precision
        },
        'LayerName': {}  # Per-layer config can be added here
    }
    
    # Print configuration summary
    print(f"\nConfiguration:")
    print(f"  Backend: {config['Backend']}")
    print(f"  IOType: {config['IOType']}")
    print(f"  Strategy: {config['Model']['Strategy']}")
    print(f"  Precision: {config['Model']['Precision']}")
    print(f"  ReuseFactor: {config['Model']['ReuseFactor']}")
    print(f"  OutputDir: {config['OutputDir']}")
    
    # Save config to JSON (like notebook.txt)
    config_path = output_dir / "hls4ml_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"\nSaved config: {config_path}")
    
    return config, sample_input.numpy()


def convert_and_synthesize(
    model,
    config: dict,
    test_input: np.ndarray,
    run_synthesis: bool = True
):
    """
    Convert model to HLS and run Catapult synthesis.
    This mirrors notebook.txt In[16], In[17], In[18], In[19].
    """
    import hls4ml
    
    print("\n" + "="*60)
    print("Converting to HLS")
    print("="*60)
    
    # Step 1: Convert model (In[16])
    print("\n[Step 1] Creating HLS model...")
    
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        output_dir=config['OutputDir'],
        project_name=config['ProjectName'],
        input_data_tb=test_input,
        backend='Catapult',
        hls_config=config
    )
    
    # Step 2: Write HLS project (In[17])
    print("\n[Step 2] Writing HLS project...")
    hls_model.write()
    print("  Done - HLS C++ generated")
    
    # Step 3: Compile C++ model (In[17])
    print("\n[Step 3] Compiling C++ model...")
    try:
        hls_model.compile()
        print("  C++ model compiled successfully")
        
        # Step 4: Test accuracy (In[18])
        print("\n[Step 4] Testing C++ model accuracy...")
        with torch.no_grad():
            pytorch_pred = model(torch.from_numpy(test_input)).numpy()
        cpp_pred = hls_model.predict(test_input)
        
        print(f"  PyTorch output: {pytorch_pred.flatten()[:5]}...")
        print(f"  C++ output: {cpp_pred.flatten()[:5]}...")
        
        diff = np.abs(pytorch_pred.flatten() - cpp_pred.flatten())
        print(f"  Max difference: {diff.max():.6f}")
        print(f"  Mean difference: {diff.mean():.6f}")
        
        if diff.max() < 0.1:
            print("  ✅ Outputs match within tolerance!")
        else:
            print("  ⚠️  Large difference - check quantization")
            
    except Exception as e:
        print(f"  ⚠️  Compile/test skipped: {e}")
    
    # Step 5: Run Catapult synthesis (In[19])
    if run_synthesis:
        print("\n[Step 5] Running Catapult synthesis...")
        print("  This may take several minutes...")
        
        try:
            # The build() method runs Catapult synthesis
            hls_model.build(
                csim=False,
                synth=True,
                verilog=True,
                vhdl=True
            )
            print("\n  ✅ Catapult synthesis complete!")
            
        except Exception as e:
            print(f"\n  ⚠️  Build error: {e}")
            print("\n  You can run synthesis manually:")
            print(f"    cd {config['OutputDir']}")
            print(f"    catapult -shell -f build_prj.tcl")
    else:
        print("\n[Step 5] Skipping synthesis (--no-synth)")
        print(f"  Run manually: cd {config['OutputDir']} && catapult -shell -f build_prj.tcl")
    
    return hls_model


def print_results(output_dir: str, project_name: str):
    """Print synthesis results like notebook.txt In[20], In[21]."""
    
    project_dir = Path(output_dir) / f"{project_name}-Catapult-test"
    
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    
    # Layer summary
    layer_summary = project_dir / "firmware" / "layer_summary.txt"
    if layer_summary.exists():
        print(f"\nLayer Summary: {layer_summary}")
        print(layer_summary.read_text()[:1000])
    
    # Catapult reports
    catapult_dir = project_dir / "Catapult"
    if catapult_dir.exists():
        # Find RTL files
        verilog_files = list(catapult_dir.rglob("*.v"))
        vhdl_files = list(catapult_dir.rglob("*.vhd"))
        
        print(f"\nGenerated RTL files:")
        print(f"  Verilog: {len(verilog_files)} files")
        print(f"  VHDL: {len(vhdl_files)} files")
        
        # Show first few
        for f in verilog_files[:3]:
            print(f"    - {f.name}")
        
        # Check for reports
        for report in ["rtl.rpt", "cycle.rpt", "nnet_layer_results.txt"]:
            report_files = list(catapult_dir.rglob(report))
            if report_files:
                print(f"\n  Report: {report_files[0]}")
    
    print("\n" + "="*60)
    print(f"Output directory: {project_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Catapult AI NN Flow for PyTorch Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script follows the exact Catapult AI NN flow from the tutorial,
adapted for PyTorch models.

Examples:
    # Basic usage
    python3 catapult_pytorch_flow.py model.pt --input-shape 1 1 28 28
    
    # With custom settings
    python3 catapult_pytorch_flow.py model.pt \\
        --input-shape 1 3 224 224 \\
        --precision "ac_fixed<16,6>" \\
        --reuse-factor 4 \\
        --io-type io_stream \\
        --strategy Latency
    
    # Generate HLS only, no Catapult synthesis
    python3 catapult_pytorch_flow.py model.pt --input-shape 1 1 28 28 --no-synth
"""
    )
    
    parser.add_argument("model", help="Path to PyTorch .pt model file")
    parser.add_argument("-o", "--output", default=".",
                        help="Output directory (default: current)")
    parser.add_argument("-n", "--name", default="myproject",
                        help="Project name (default: myproject)")
    parser.add_argument("--input-shape", nargs='+', type=int, default=None,
                        help="Input shape: batch channels height width (auto-detect if not provided)")
    parser.add_argument("--precision", default="ac_fixed<16,6>",
                        help="Fixed-point precision")
    parser.add_argument("--reuse-factor", type=int, default=1,
                        help="Reuse factor (1=max parallelism)")
    parser.add_argument("--io-type", choices=["io_stream", "io_parallel"],
                        default="io_stream")
    parser.add_argument("--strategy", choices=["Latency", "Resource"],
                        default="Latency")
    parser.add_argument("--no-synth", action="store_true",
                        help="Skip Catapult synthesis")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Catapult AI NN - PyTorch Model Flow")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"MGC_HOME: {MGC_HOME}")
    print()
    
    # Verify hls4ml is available
    try:
        import hls4ml
        version = getattr(hls4ml, '__version__', 'Catapult-bundled')
        print(f"hls4ml version: {version}")
    except ImportError:
        print("ERROR: hls4ml not found!")
        print("Make sure you're using Catapult's Python:")
        print("  $MGC_HOME/bin/python3 catapult_pytorch_flow.py ...")
        return 1
    
    # Load model
    model = load_pytorch_model(args.model)
    
    # Auto-detect input shape if not provided
    input_shape = args.input_shape
    if input_shape is None:
        print("\nAuto-detecting input shape...")
        # Try to infer from first layer
        try:
            first_layer = list(model.children())[0]
            if hasattr(first_layer, 'in_features'):
                # Linear layer - assume flattened input
                input_shape = [1, first_layer.in_features]
                print(f"  Detected Linear input: {input_shape}")
            elif hasattr(first_layer, 'in_channels'):
                # Conv layer - need spatial dims, use common defaults
                in_ch = first_layer.in_channels
                # Common sizes: MNIST=28x28, CIFAR=32x32, ImageNet=224x224
                print(f"  Detected Conv2d with {in_ch} input channels")
                print("  ERROR: Cannot auto-detect spatial dimensions for Conv layers.")
                print("  Please provide --input-shape, e.g.:")
                print(f"    --input-shape 1 {in_ch} 28 28   (for MNIST-like)")
                print(f"    --input-shape 1 {in_ch} 32 32   (for CIFAR-like)")
                print(f"    --input-shape 1 {in_ch} 224 224 (for ImageNet-like)")
                return 1
            else:
                print("  ERROR: Could not auto-detect input shape.")
                print("  Please provide --input-shape")
                return 1
        except Exception as e:
            print(f"  ERROR: Could not auto-detect input shape: {e}")
            print("  Please provide --input-shape")
            return 1
    
    print(f"Input shape: {input_shape}")
    
    # Configure hls4ml
    config, test_input = configure_hls4ml_for_catapult(
        model=model,
        input_shape=tuple(input_shape),
        output_dir=args.output,
        project_name=args.name,
        precision=args.precision,
        reuse_factor=args.reuse_factor,
        io_type=args.io_type,
        strategy=args.strategy
    )
    
    # Convert and synthesize
    hls_model = convert_and_synthesize(
        model=model,
        config=config,
        test_input=test_input,
        run_synthesis=not args.no_synth
    )
    
    # Print results
    print_results(args.output, args.name)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
