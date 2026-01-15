#!/usr/bin/env python3
"""
Catapult AI NN - Direct PyTorch Flow
====================================

This script implements the direct PyTorch to HLS flow using catapult_ai_nn,
exactly as described in the official documentation (pytorchdocumentation.txt).

The flow is:
1. config_ccs = catapult_ai_nn.config_for_dataflow(model=model, x_test=..., ...)
2. hls_model_ccs = catapult_ai_nn.generate_dataflow(model=model, config_ccs=config_ccs)
3. hls_model_ccs.compile()
4. hls_model_ccs.predict(x_test)  # Verify accuracy
5. hls_model_ccs.build()  # Synthesize RTL

Usage:
    # Option 1: Model directory with model.pt + model.py (RECOMMENDED)
    PYTHONPATH=$MGC_HOME/shared/pkgs/ccs_hls4ml/hls4ml $MGC_HOME/bin/python3 \\
        catapult_keras_flow.py /path/to/model_dir --input-shape 1 1 32 32
    
    # Option 2: Project with NN Generator structure
    PYTHONPATH=$MGC_HOME/shared/pkgs/ccs_hls4ml/hls4ml $MGC_HOME/bin/python3 \\
        catapult_keras_flow.py /path/to/project --model-type manual --input-shape 1 1 32 32
    
    # Option 3: Explicit separate paths
    PYTHONPATH=$MGC_HOME/shared/pkgs/ccs_hls4ml/hls4ml $MGC_HOME/bin/python3 \\
        catapult_keras_flow.py /project \\
        --model-pt /project/Manual_Output/SystemC/Pt/model.pt \\
        --model-py-dir /project/Manual_Output/python \\
        --input-shape 1 1 32 32
    
    # Option 4: Direct .pt file (requires model class to be loadable)
    PYTHONPATH=$MGC_HOME/shared/pkgs/ccs_hls4ml/hls4ml $MGC_HOME/bin/python3 \\
        catapult_keras_flow.py model.pt --input-shape 1 1 32 32

Model Directory Structure (Simple):
    model_dir/
    ├── model.pt    # Trained weights (saved with torch.save(model.state_dict(), 'model.pt'))
    └── model.py    # Model class definition (must have CNN or Model class)

Project Structure (NN Generator Output):
    project/
    └── Manual_Output/     # or Pretrained_Output/ or YOLOX/
        ├── SystemC/
        │   └── Pt/
        │       └── model.pt
        └── python/
            ├── model.py
            └── residual.py  # optional
"""

import argparse
import os
import sys
import importlib.util

# Setup Catapult environment - must be done BEFORE any other imports
MGC_HOME = os.environ.get('MGC_HOME', '/data/tools/catapult/Mgc_home')
os.environ['MGC_HOME'] = MGC_HOME

# CRITICAL: PYTHONPATH must point to the INNER hls4ml directory
# This is $MGC_HOME/shared/pkgs/ccs_hls4ml/hls4ml (NOT ccs_hls4ml itself)
# This allows both:
#   - import catapult_ai_nn  (from ccs_hls4ml/hls4ml/catapult_ai_nn.py)
#   - import hls4ml.utils    (from ccs_hls4ml/hls4ml/hls4ml/utils/)
HLS4ML_PATH = f"{MGC_HOME}/shared/pkgs/ccs_hls4ml/hls4ml"
if HLS4ML_PATH not in sys.path:
    sys.path.insert(0, HLS4ML_PATH)

print(f"DEBUG: MGC_HOME = {MGC_HOME}")
print(f"DEBUG: HLS4ML_PATH = {HLS4ML_PATH}")
print(f"DEBUG: sys.path[0:3] = {sys.path[0:3]}")

# Import catapult_ai_nn DIRECTLY as shown in pytorchdocumentation.txt line 20:
#   "import catapult_ai_nn"
# catapult_ai_nn.py is at $MGC_HOME/shared/pkgs/ccs_hls4ml/hls4ml/catapult_ai_nn.py
try:
    import catapult_ai_nn
    print(f"DEBUG: Successfully imported catapult_ai_nn")
    print(f"DEBUG: catapult_ai_nn location: {catapult_ai_nn.__file__}")
    print(f"DEBUG: config_for_dataflow available: {hasattr(catapult_ai_nn, 'config_for_dataflow')}")
except ImportError as e:
    print(f"ERROR: Could not import catapult_ai_nn: {e}")
    print(f"Ensure PYTHONPATH includes: {HLS4ML_PATH}")
    print(f"Run with: PYTHONPATH={HLS4ML_PATH} $MGC_HOME/bin/python3 {sys.argv[0]}")
    sys.exit(1)

# Verify hls4ml.utils is accessible (required by catapult_ai_nn internally)
try:
    import hls4ml.utils
    print(f"DEBUG: hls4ml.utils available: {hls4ml.utils.__file__}")
except ImportError as e:
    print(f"ERROR: hls4ml.utils not found: {e}")
    print(f"This means PYTHONPATH is not set correctly.")
    sys.exit(1)

# Now import other dependencies
import torch
import torch.nn as nn
import numpy as np


def load_model_from_directory(model_dir: str, model_pt_override: str = None, model_py_dir_override: str = None):
    """
    Load a PyTorch model from a directory containing model.pt and model.py.
    
    Args:
        model_dir: Path to directory containing model.pt + model.py (simple format)
                   OR the project root when using overrides
        model_pt_override: Optional explicit path to model.pt (for project structure)
        model_py_dir_override: Optional explicit path to directory containing model.py (for project structure)
    
    Supports two formats:
    1. Simple: model_dir contains both model.pt and model.py
    2. Project: Files are in separate subdirectories (use overrides)
       - model.pt at {project}/Manual_Output/SystemC/Pt/model.pt
       - model.py at {project}/Manual_Output/python/model.py
    
    Returns:
        Loaded PyTorch model with weights
    """
    # Determine actual file paths
    if model_pt_override and model_py_dir_override:
        # Project structure: separate paths
        model_pt_path = model_pt_override
        model_py_dir = model_py_dir_override
        model_py_path = os.path.join(model_py_dir, 'model.py')
        print(f"    Using project structure:")
        print(f"      model.pt: {model_pt_path}")
        print(f"      model.py dir: {model_py_dir}")
    else:
        # Simple structure: both files in same directory
        model_pt_path = os.path.join(model_dir, 'model.pt')
        model_py_path = os.path.join(model_dir, 'model.py')
        model_py_dir = model_dir
    
    # Check files exist
    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"model.pt not found at {model_pt_path}")
    if not os.path.exists(model_py_path):
        raise FileNotFoundError(f"model.py not found at {model_py_path}")
    
    print(f"    Found model.pt: {model_pt_path}")
    print(f"    Found model.py: {model_py_path}")
    
    # Add the model.py directory to sys.path so imports can find sibling files (residual.py, etc.)
    if model_py_dir not in sys.path:
        sys.path.insert(0, model_py_dir)
        print(f"    Added to sys.path: {model_py_dir}")
    
    # Handle "from python.xxx import ..." patterns by creating a 'python' package
    # that points to the model.py directory (common in NN Generator outputs)
    python_pkg_path = os.path.join(model_py_dir, 'python')
    if not os.path.exists(python_pkg_path):
        # Create a virtual 'python' package pointing to model_py_dir
        import types
        python_module = types.ModuleType('python')
        python_module.__path__ = [model_py_dir]
        sys.modules['python'] = python_module
        print(f"    Created virtual 'python' package")
    
    # Pre-load any .py files in the model.py directory as potential dependencies
    for py_file in os.listdir(model_py_dir):
        if py_file.endswith('.py') and py_file != 'model.py':
            dep_name = py_file[:-3]  # Remove .py extension
            dep_path = os.path.join(model_py_dir, py_file)
            try:
                dep_spec = importlib.util.spec_from_file_location(dep_name, dep_path)
                dep_module = importlib.util.module_from_spec(dep_spec)
                sys.modules[dep_name] = dep_module
                sys.modules[f'python.{dep_name}'] = dep_module  # Also register as python.xxx
                dep_spec.loader.exec_module(dep_module)
                print(f"    Pre-loaded dependency: {dep_name}")
    except Exception as e:
                print(f"    Warning: Could not pre-load {py_file}: {e}")
    
    # Load the model class from model.py
    spec = importlib.util.spec_from_file_location("model_module", model_py_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = model_module
    sys.modules["python.model"] = model_module  # Also register as python.model
    spec.loader.exec_module(model_module)
    
    # Find the model class (look for CNN, Model, or Net)
    model_class = None
    for class_name in ['CNN', 'Model', 'Net', 'Network', 'LeNet', 'LeNet5']:
        if hasattr(model_module, class_name):
            model_class = getattr(model_module, class_name)
            print(f"    Found model class: {class_name}")
            break
    
    if model_class is None:
        # Try to find any class that inherits from nn.Module
        for name in dir(model_module):
            obj = getattr(model_module, name)
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj != nn.Module:
                model_class = obj
                print(f"    Found model class: {name}")
                break
    
    if model_class is None:
        raise ValueError(f"No model class found in {model_py_path}. "
                        f"Expected a class named CNN, Model, Net, or inheriting from nn.Module")
    
    # Instantiate the model
    model = model_class()
    print(f"    Instantiated model: {model_class.__name__}")
    
    # Load the weights - try state_dict first, then full model
    try:
        # Try loading as state_dict (recommended format)
        state_dict = torch.load(model_pt_path, map_location='cpu', weights_only=True)
        if isinstance(state_dict, dict) and all(isinstance(k, str) for k in state_dict.keys()):
            print(f"    Loaded state_dict from model.pt")
        else:
            raise ValueError("Not a state_dict")
    except Exception:
        # Fall back to loading full model and extracting state_dict
        print(f"    model.pt is full model, extracting state_dict...")
        loaded = torch.load(model_pt_path, map_location='cpu', weights_only=False)
        if hasattr(loaded, 'state_dict'):
            state_dict = loaded.state_dict()
        elif isinstance(loaded, dict):
            state_dict = loaded
        else:
            raise ValueError(f"Cannot extract state_dict from model.pt: {type(loaded)}")
    
    model.load_state_dict(state_dict)
    print(f"    ✅ Loaded weights from model.pt")
    
    model.eval()
    return model


def load_model_direct(model_path: str):
    """
    Load a PyTorch model directly from a .pt file.
    This works if the model was saved with torch.save(model, path) 
    and the class is loadable.
    
    Args:
        model_path: Path to .pt file
    
    Returns:
        Loaded PyTorch model
    """
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Check if it's a TorchScript model (not supported)
    if type(model).__name__ in ['RecursiveScriptModule', 'ScriptModule']:
        raise ValueError(
            f"TorchScript models are NOT supported!\n"
            f"Model type: {type(model).__name__}\n"
            f"Please re-save your model with: torch.save(model, 'model.pt')\n"
            f"NOT with torch.jit.save()"
        )
    
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Catapult AI NN - Direct PyTorch Flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From model directory (RECOMMENDED):
  python catapult_keras_flow.py /path/to/model_dir --input-shape 1 1 32 32
  
  # From NN Generator project (auto-detect model type):
  python catapult_keras_flow.py /path/to/project --model-type manual --input-shape 1 1 32 32
  
  # Explicit paths (for complex setups):
  python catapult_keras_flow.py /project --model-pt /project/Manual_Output/SystemC/Pt/model.pt \\
      --model-py-dir /project/Manual_Output/python --input-shape 1 1 32 32
  
  # Direct .pt file (legacy):
  python catapult_keras_flow.py model.pt --input-shape 1 1 32 32

Model Directory Structure:
  model_dir/
  ├── model.pt    # torch.save(model.state_dict(), 'model.pt')
  └── model.py    # class CNN(nn.Module): ...

Project Structure (NN Generator):
  project/
  └── Manual_Output/     # or Pretrained_Output/ or YOLOX/
      ├── SystemC/Pt/model.pt
      └── python/model.py
        """
    )
    parser.add_argument("model", help="Path to model directory (with model.pt + model.py) OR direct .pt file OR project path with --model-type")
    parser.add_argument("-o", "--output", default=".", help="Output directory")
    parser.add_argument("-n", "--name", default="myproject", help="Project name")
    parser.add_argument("--input-shape", nargs='+', type=int, required=True,
                        help="Input shape: batch channels height width")
    parser.add_argument("--precision", default="ac_fixed<16,8>", help="Default precision")
    parser.add_argument("--reuse-factor", type=int, default=100, help="Reuse factor")
    parser.add_argument("--io-type", default="io_stream", help="IO Type")
    parser.add_argument("--clock-period", type=float, default=10.0, help="Clock period in ns")
    parser.add_argument("--strategy", default="Latency", help="Strategy (Latency/Resource)")
    parser.add_argument("--no-synth", action="store_true", help="Skip synthesis")
    parser.add_argument("--run-questasim", action="store_true", help="Run QuestaSim verification")
    parser.add_argument("--step", choices=['all', 'config', 'verify', 'build'], default='all',
                        help="Execution step: config (generate C++), verify (check accuracy), build (synthesize RTL)")
    
    # Project structure support (files in separate directories)
    parser.add_argument("--model-type", choices=['manual', 'pretrained', 'yolox'],
                        help="Model type for project structure. Uses project paths: "
                             "{project}/{Type}_Output/SystemC/Pt/model.pt and {project}/{Type}_Output/python/model.py")
    parser.add_argument("--model-pt", help="Explicit path to model.pt (overrides auto-detection)")
    parser.add_argument("--model-py-dir", help="Explicit path to directory containing model.py (overrides auto-detection)")
    
    # Legacy/Coordinator compatibility arguments
    parser.add_argument("--questa-timeout", type=int, default=3600)
    parser.add_argument("--questasim-only", action="store_true")

    args = parser.parse_args()

    print("="*60)
    print("Catapult AI NN - Direct PyTorch Flow")
    print("="*60)
    print(f"Model: {args.model}")
    if args.model_type:
        print(f"Model Type: {args.model_type}")
    print(f"Input Shape: {args.input_shape}")
    print(f"Step: {args.step}")
    print(f"MGC_HOME: {MGC_HOME}")
    
    # 1. Load PyTorch Model
    print("\n[1] Loading PyTorch Model...")
    model = None
    
    # Determine model paths based on input format
    model_pt_path = None
    model_py_dir = None
    
    # Option 1: Explicit paths provided
    if args.model_pt and args.model_py_dir:
        model_pt_path = args.model_pt
        model_py_dir = args.model_py_dir
        print(f"    Using explicit paths:")
        print(f"      model.pt: {model_pt_path}")
        print(f"      model.py dir: {model_py_dir}")
    
    # Option 2: Project structure with --model-type
    elif args.model_type:
        project_path = args.model
        # Map model type to output directory name
        type_to_dir = {
            'manual': 'Manual_Output',
            'pretrained': 'Pretrained_Output',
            'yolox': 'YOLOX'
        }
        output_dir_name = type_to_dir.get(args.model_type)
        model_pt_path = os.path.join(project_path, output_dir_name, 'SystemC', 'Pt', 'model.pt')
        model_py_dir = os.path.join(project_path, output_dir_name, 'python')
        print(f"    Using project structure for {args.model_type}:")
        print(f"      Project: {project_path}")
        print(f"      model.pt: {model_pt_path}")
        print(f"      model.py dir: {model_py_dir}")
    
    # Option 3: AUTO-DETECT project structure (no --model-type provided)
    elif os.path.isdir(args.model):
        # Check if this is a project directory with Manual_Output, Pretrained_Output, or YOLOX
        detected_type = None
        for output_type, output_dir_name in [('manual', 'Manual_Output'), ('pretrained', 'Pretrained_Output'), ('yolox', 'YOLOX')]:
            check_pt = os.path.join(args.model, output_dir_name, 'SystemC', 'Pt', 'model.pt')
            check_py = os.path.join(args.model, output_dir_name, 'python', 'model.py')
            if os.path.exists(check_pt) and os.path.exists(check_py):
                detected_type = output_type
                model_pt_path = check_pt
                model_py_dir = os.path.join(args.model, output_dir_name, 'python')
                print(f"    ✨ Auto-detected project structure: {output_type}")
                print(f"      Project: {args.model}")
                print(f"      model.pt: {model_pt_path}")
                print(f"      model.py dir: {model_py_dir}")
                break
        
        # If not a project structure, check for simple model directory
        if detected_type is None:
            simple_pt = os.path.join(args.model, 'model.pt')
            simple_py = os.path.join(args.model, 'model.py')
            if os.path.exists(simple_pt) and os.path.exists(simple_py):
                print(f"    Found simple model directory: {args.model}")
                # model_pt_path and model_py_dir stay None, will use load_model_from_directory with defaults
    
    # Load the model
    try:
        if model_pt_path and model_py_dir:
            # Load from separate paths (project structure or explicit)
            model = load_model_from_directory(args.model, model_pt_path, model_py_dir)
            print(f"    ✅ Model loaded from project structure!")
        elif os.path.isdir(args.model):
            # Simple format: directory with model.pt + model.py together
            print(f"    Loading from model directory: {args.model}")
            model = load_model_from_directory(args.model)
            print(f"    ✅ Model loaded from directory!")
        else:
            # Legacy format: direct .pt file
            print(f"    Loading from .pt file: {args.model}")
            model = load_model_direct(args.model)
            print(f"    ✅ Model loaded directly!")
    except FileNotFoundError as e:
        print(f"    ❌ ERROR: {e}")
        print(f"")
        if args.model_type or model_pt_path:
            print(f"    Expected project structure:")
            print(f"      {model_pt_path or 'model.pt path'}")
            print(f"      {model_py_dir or 'model.py directory'}/model.py")
        else:
            print(f"    Could not find model files. Expected one of:")
            print(f"")
            print(f"    1. Project structure (NN Generator output):")
            print(f"       {args.model}/Manual_Output/SystemC/Pt/model.pt")
            print(f"       {args.model}/Manual_Output/python/model.py")
            print(f"")
            print(f"    2. Simple model directory:")
            print(f"       {args.model}/model.pt")
            print(f"       {args.model}/model.py")
        sys.exit(1)
    except ModuleNotFoundError as e:
        print(f"    ❌ ERROR: {e}")
        print(f"")
        print(f"    The model was saved with a custom module path that doesn't exist.")
        print(f"    Please use --model-type or --model-pt/--model-py-dir to specify paths.")
        sys.exit(1)
    except ValueError as e:
        print(f"    ❌ ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"    ❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if model is None:
        print(f"    ❌ ERROR: Could not load model")
        sys.exit(1)
        
    print(f"    Model type: {type(model).__name__}")

    # 2. Create dummy input and output for testing
    # args.input_shape is typically NCHW for PyTorch
    input_shape = tuple(args.input_shape)
    x_test = torch.randn(*input_shape)
    print(f"    Created dummy input with shape: {x_test.shape}")
    
    # Try to run inference to get output shape, otherwise create dummy y_test
    try:
        with torch.no_grad():
            y_test = model(x_test)
        print(f"    Created dummy output with shape: {y_test.shape}")
    except Exception as e:
        print(f"    WARNING: Could not run inference ({e})")
        print(f"    Creating dummy output (10 classes for MNIST)")
        # Create dummy output - assume 10 classes for MNIST
        y_test = torch.randn(input_shape[0], 10)
        print(f"    Created dummy output with shape: {y_test.shape}")
    
    # 3. Configure Catapult AI NN
    output_dir = os.path.join(args.output, f"{args.name}-Catapult-test")
    
    # Ensure parent directory exists (catapult_ai_nn only does mkdir, not makedirs)
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n[2] Configuring Catapult AI NN...")
    print(f"    Output Dir: {output_dir}")
    print(f"    Precision: {args.precision}")
    print(f"    Reuse Factor: {args.reuse_factor}")
    print(f"    IO Type: {args.io_type}")
    
    try:
        # Configuration based on pytorchdocumentation.txt
        config_ccs = catapult_ai_nn.config_for_dataflow(
            model=model,
            x_test=x_test.numpy(),
            y_test=y_test.numpy(),
            granularity='name',
            default_precision=args.precision,
            max_precision=args.precision,
            default_reuse_factor=args.reuse_factor,
            output_dir=output_dir,
            tech='asic', # Default to ASIC as per doc
            io_type=args.io_type,
            csim=1,
            SCVerify=1 if args.run_questasim else 0,
            Synth=1 if not args.no_synth else 0,
            verilog=1,
            vhdl=0
        )
    except AttributeError as e:
        print(f"    ERROR: catapult_ai_nn module missing expected function: {e}")
        print(f"    Module location: {catapult_ai_nn.__file__}")
        print(f"    Available attributes: {[x for x in dir(catapult_ai_nn) if 'config' in x.lower()]}")
        sys.exit(1)
    except Exception as e:
        print(f"    ERROR during configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Generate Dataflow (C++ Model)
    print("\n[3] Generating Dataflow (C++ Model)...")
    try:
        hls_model_ccs = catapult_ai_nn.generate_dataflow(model=model, config_ccs=config_ccs)
    except Exception as e:
        print(f"    ERROR during generation: {e}")
        sys.exit(1)

    # 5. Compile C++ Model
    print("\n[4] Compiling C++ Model...")
    try:
        hls_model_ccs.compile()
    except Exception as e:
        print(f"    ERROR during compilation: {e}")
        sys.exit(1)
    
    if args.step == 'config':
        print("\n[Config] Configuration and C++ generation complete.")
        return

    # 6. Verify C++ Model (Prediction)
    if args.step in ['all', 'verify']:
        print("\n[5] Verifying C++ Model Accuracy...")
        try:
            with torch.no_grad():
                torch_pred = model(x_test).numpy()
            
            hls_pred = hls_model_ccs.predict(x_test.numpy())
            
            # Calculate error
            diff = np.abs(torch_pred - hls_pred)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"    Max Diff: {max_diff:.6f}")
            print(f"    Mean Diff: {mean_diff:.6f}")
            
            if max_diff > 1e-3:
                print("    WARNING: Large difference detected between PyTorch and HLS model.")
            else:
                print("    SUCCESS: HLS model matches PyTorch model within tolerance.")
                
        except Exception as e:
            print(f"    Warning during verification: {e}")
            
        if args.step == 'verify':
            return
    
    # 7. Synthesize (Build)
    if args.step in ['all', 'build'] and not args.no_synth:
        print("\n[6] Synthesizing to RTL (Catapult Build)...")
        print("    This may take several minutes...")
        try:
            hls_model_ccs.build()
            print("    Synthesis Complete!")
        except Exception as e:
            print(f"    ERROR during synthesis: {e}")
            sys.exit(1)
    elif args.no_synth:
        print("\n[6] Synthesis skipped (--no-synth)")

    print("\n" + "="*60)
    print("Flow Complete")
    print("="*60)

if __name__ == "__main__":
    main()
