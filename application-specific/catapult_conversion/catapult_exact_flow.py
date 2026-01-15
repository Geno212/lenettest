#!/usr/bin/env python3
"""
Catapult AI NN - EXACT Tutorial Flow
=====================================

This script replicates the EXACT flow from:
$MGC_HOME/shared/examples/cat_ai_nn/Tutorial/notebook.ipynb

The official Catapult AI NN flow works with Keras models.
For PyTorch models, we first convert to Keras format.

SETUP (run once on EC2):
    sh $MGC_HOME/shared/pkgs/ccs_hls4ml/create_env.sh $MGC_HOME/bin/python3 $HOME/ccs_venv

RUN:
    source $HOME/ccs_venv/bin/activate
    python3 catapult_exact_flow.py model.pt --input-shape 1 1 28 28

Or with Catapult's Python directly:
    $MGC_HOME/bin/python3 catapult_exact_flow.py model.pt --input-shape 1 1 28 28
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Catapult environment
MGC_HOME = os.environ.get('MGC_HOME', '/data/tools/catapult/Mgc_home')
os.environ['MGC_HOME'] = MGC_HOME
os.environ['PATH'] = f"{MGC_HOME}/bin:" + os.environ.get('PATH', '')

# Add Catapult's packages to path
sys.path.insert(0, f"{MGC_HOME}/shared/pkgs/ccs_hls4ml")


def pytorch_to_keras(model_path: str, input_shape: tuple, output_dir: str):
    """
    Convert PyTorch model to Keras format.
    Catapult AI NN tutorial uses Keras models.
    """
    import torch
    
    print("="*60)
    print("Step 1: Convert PyTorch to Keras")
    print("="*60)
    
    # Load PyTorch model
    print(f"\nLoading PyTorch model: {model_path}")
    pt_model = torch.load(model_path, map_location='cpu')
    if hasattr(pt_model, 'eval'):
        pt_model.eval()
    
    print(f"Model type: {type(pt_model).__name__}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(model_path).stem
    
    # Export to ONNX first
    print("\nExporting to ONNX...")
    import torch.onnx
    
    onnx_path = output_dir / f"{model_name}.onnx"
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        pt_model,
        dummy_input,
        str(onnx_path),
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
    print(f"  Saved: {onnx_path}")
    
    # Convert ONNX to Keras
    print("\nConverting ONNX to Keras...")
    try:
        import onnx
        from onnx2keras import onnx_to_keras
        
        onnx_model = onnx.load(str(onnx_path))
        keras_model = onnx_to_keras(onnx_model, ['input'])
        
        # Save in Keras format (JSON + H5) - as used by Catapult tutorial
        json_path = output_dir / f"{model_name}.json"
        h5_path = output_dir / f"{model_name}_weights.h5"
        
        with open(json_path, 'w') as f:
            f.write(keras_model.to_json())
        keras_model.save_weights(str(h5_path))
        
        print(f"  Saved: {json_path}")
        print(f"  Saved: {h5_path}")
        
        return keras_model, str(json_path), str(h5_path)
        
    except ImportError as e:
        print(f"\n  ⚠️ Missing packages: {e}")
        print("  Install: pip install onnx onnx2keras")
        return None, None, None


def run_catapult_ai_nn_flow(
    keras_model,
    model_name: str,
    output_dir: str,
    input_shape: tuple,
    test_data=None,
    precision: str = "ac_fixed<16,6>",
    reuse_factor: int = 1,
    io_type: str = "io_stream",
    strategy: str = "Latency",
    run_synthesis: bool = True
):
    """
    Run the EXACT Catapult AI NN flow from the tutorial.
    This uses hls4ml with Catapult backend.
    """
    import hls4ml
    
    output_dir = Path(output_dir)
    project_dir = output_dir / f"my-{model_name}-Catapult-test"
    
    print("\n" + "="*60)
    print("Step 2: Configure Catapult AI NN (hls4ml)")
    print("="*60)
    
    # Generate test data if not provided
    if test_data is None:
        # Create random test data
        # Note: input_shape is NCHW for PyTorch, need NHWC for Keras
        batch, channels, height, width = input_shape
        test_data = np.random.randn(batch, height, width, channels).astype(np.float32)
    
    # Get predictions for testbench
    predictions = keras_model.predict(test_data)
    
    # Save testbench data (exactly as notebook.txt In[13])
    project_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(project_dir / "tb_input_features.dat", test_data.flatten(), fmt='%.6f')
    np.savetxt(project_dir / "tb_output_predictions.dat", predictions.flatten(), fmt='%.6f')
    
    print(f"\nProject directory: {project_dir}")
    print(f"Saved testbench data")
    
    # Configure hls4ml (exactly as notebook.txt In[13], In[14], In[15])
    print("\nCreating hls4ml configuration...")
    
    config = hls4ml.utils.config_from_keras_model(
        keras_model,
        granularity='name',
        default_precision=precision,
        default_reuse_factor=reuse_factor
    )
    
    # Catapult-specific settings (from notebook.txt config)
    config['Backend'] = 'Catapult'
    config['IOType'] = io_type
    config['ProjectName'] = 'myproject'
    config['OutputDir'] = str(project_dir)
    config['ClockPeriod'] = 10
    config['Technology'] = 'asic'
    
    if 'Model' not in config:
        config['Model'] = {}
    config['Model']['Strategy'] = strategy
    config['Model']['ReuseFactor'] = reuse_factor
    config['Model']['Precision'] = {'default': precision}
    
    print(f"\nConfiguration:")
    print(f"  Backend: Catapult")
    print(f"  IOType: {io_type}")
    print(f"  Strategy: {strategy}")
    print(f"  Precision: {precision}")
    print(f"  ReuseFactor: {reuse_factor}")
    
    # Step 3: Create HLS model (notebook.txt In[16])
    print("\n" + "="*60)
    print("Step 3: Generate HLS Model")
    print("="*60)
    
    print("\nCreating HLS model...")
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        hls_config=config,
        output_dir=str(project_dir),
        project_name='myproject',
        backend='Catapult'
    )
    
    # Step 4: Write and compile C++ (notebook.txt In[17])
    print("\n" + "="*60)
    print("Step 4: Write and Compile C++")
    print("="*60)
    
    print("\nWriting HLS project...")
    hls_model.write()
    print("  Done")
    
    print("\nCompiling C++ model...")
    try:
        hls_model.compile()
        print("  C++ compiled successfully")
        
        # Test accuracy (notebook.txt In[18])
        print("\nTesting C++ model accuracy...")
        keras_pred = keras_model.predict(test_data)
        cpp_pred = hls_model.predict(test_data)
        
        print(f"  Keras Accuracy: {np.argmax(keras_pred, axis=1)}")
        print(f"  C++ Accuracy: {np.argmax(cpp_pred, axis=1)}")
        
        diff = np.abs(keras_pred - cpp_pred)
        print(f"  Max difference: {diff.max():.6f}")
        
    except Exception as e:
        print(f"  ⚠️ Compile error: {e}")
    
    # Step 5: Run Catapult synthesis (notebook.txt In[19])
    if run_synthesis:
        print("\n" + "="*60)
        print("Step 5: Catapult Synthesis")
        print("="*60)
        
        print("\nRunning Catapult synthesis...")
        print("This may take several minutes...")
        
        try:
            hls_model.build(
                csim=False,
                synth=True,
                verilog=True,
                vhdl=True
            )
            print("\n✅ Synthesis complete!")
            
        except Exception as e:
            print(f"\n⚠️ Build error: {e}")
            print("\nRun manually:")
            print(f"  cd {project_dir}")
            print(f"  catapult -shell -f build_prj.tcl")
    
    # Print results (notebook.txt In[20], In[21])
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    
    # Check for generated files
    catapult_dir = project_dir / "Catapult"
    if catapult_dir.exists():
        v_files = list(catapult_dir.rglob("*.v"))
        vhd_files = list(catapult_dir.rglob("*.vhd"))
        print(f"\nGenerated RTL:")
        print(f"  Verilog files: {len(v_files)}")
        print(f"  VHDL files: {len(vhd_files)}")
    
    print(f"\nOutput: {project_dir}")
    
    return hls_model


def main():
    parser = argparse.ArgumentParser(
        description="Catapult AI NN - Exact Tutorial Flow",
        epilog="""
This follows the EXACT flow from:
  $MGC_HOME/shared/examples/cat_ai_nn/Tutorial/notebook.ipynb

For PyTorch models, converts to Keras first, then runs Catapult AI NN.

Examples:
    python3 catapult_exact_flow.py model.pt --input-shape 1 1 28 28
    python3 catapult_exact_flow.py model.pt --input-shape 1 3 224 224 --precision "ac_fixed<8,4>"
"""
    )
    
    parser.add_argument("model", help="PyTorch .pt or Keras .h5 model")
    parser.add_argument("-o", "--output", default="./catapult_output")
    parser.add_argument("--input-shape", nargs=4, type=int, required=True,
                        metavar=("N", "C", "H", "W"),
                        help="Input shape: batch channels height width")
    parser.add_argument("--precision", default="ac_fixed<16,6>")
    parser.add_argument("--reuse-factor", type=int, default=1)
    parser.add_argument("--io-type", default="io_stream",
                        choices=["io_stream", "io_parallel"])
    parser.add_argument("--strategy", default="Latency",
                        choices=["Latency", "Resource"])
    parser.add_argument("--no-synth", action="store_true",
                        help="Skip Catapult synthesis")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Catapult AI NN - Exact Tutorial Flow")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Input shape: {tuple(args.input_shape)}")
    print(f"MGC_HOME: {MGC_HOME}")
    print()
    
    model_path = Path(args.model)
    model_name = model_path.stem
    
    # Check if PyTorch or Keras model
    if model_path.suffix == '.pt' or model_path.suffix == '.pth':
        # Convert PyTorch to Keras
        keras_model, json_path, h5_path = pytorch_to_keras(
            str(model_path),
            tuple(args.input_shape),
            args.output
        )
        if keras_model is None:
            print("\nFailed to convert PyTorch model to Keras")
            return 1
    else:
        # Load Keras model directly
        import tensorflow as tf
        
        json_path = str(model_path).replace('.h5', '.json')
        if Path(json_path).exists():
            with open(json_path, 'r') as f:
                keras_model = tf.keras.models.model_from_json(f.read())
            keras_model.load_weights(str(model_path))
        else:
            keras_model = tf.keras.models.load_model(str(model_path))
    
    # Run Catapult AI NN flow
    run_catapult_ai_nn_flow(
        keras_model=keras_model,
        model_name=model_name,
        output_dir=args.output,
        input_shape=tuple(args.input_shape),
        precision=args.precision,
        reuse_factor=args.reuse_factor,
        io_type=args.io_type,
        strategy=args.strategy,
        run_synthesis=not args.no_synth
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
