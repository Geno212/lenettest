"""
RTL Synthesis Coordinator - Converts trained PyTorch models to RTL hardware.

This assistant manages the conversion pipeline:
1. PyTorch (.pt) → Catapult AI NN (Direct Flow)
2. Catapult AI NN → HLS C++ (using Catapult's catapult_ai_nn library)
3. HLS → RTL (Verilog/VHDL using Catapult synthesis)
4. Optional: QuestaSim RTL verification

The coordinator uses the catapult_keras_flow.py script to perform the conversion.
It iteratively collects all required parameters before starting synthesis.
"""

import os
import sys
import json
import subprocess
import shlex
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field


class SynthesisParametersIntent(BaseModel):
    """Extracted synthesis parameters from user message."""
    reuse_factor: Optional[int] = Field(None, description="Reuse factor: any positive integer (common: 1=full parallel, 16=high, 32=good balance, 64=balanced, 128=efficient, 784=sequential)")
    clock_period: Optional[int] = Field(None, description="Clock period in nanoseconds (e.g., 10 for 100MHz)")
    strategy: Optional[str] = Field(None, description="Synthesis strategy: 'Latency' or 'Resource'")
    precision: Optional[str] = Field(None, description="Precision: e.g., 'ac_fixed<16,6>' or 'ac_fixed<16,8>'")
    io_type: Optional[str] = Field(None, description="IO Type: 'io_parallel' (recommended) or 'io_stream'")
    wants_questasim: Optional[bool] = Field(None, description="Whether user wants QuestaSim verification")
    ready_to_synthesize: bool = Field(False, description="User confirmed all parameters and ready to start")


class RTLSynthesisCoordinator:
    """Assistant for converting trained models to RTL hardware."""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "RTL Synthesis Coordinator"
        # LLM for extracting synthesis parameters
        self.param_extractor_llm = llm.with_structured_output(SynthesisParametersIntent)
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute RTL synthesis on the trained PyTorch model.
        Iteratively collects parameters before starting synthesis.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with RTL synthesis results
        """
        from src.agentic.src.core.state import NNGeneratorState
        
        # ------------------------------------------------------------------
        # Step 0: Get or ask for trained model path
        # ------------------------------------------------------------------
        trained_model_path = state.get("trained_model_path") or state.get("pretrained_model_path")
        
        # If no path in state, try to extract from user message
        if not trained_model_path:
            last_message = self._get_last_user_message(state)
            if last_message:
                extracted_path = await self._extract_model_path(last_message)
                if extracted_path:
                    # Validate path exists OR looks like a Unix path on Windows (assuming WSL/Remote)
                    is_unix_path = extracted_path.startswith('/')
                    is_windows = os.name == 'nt'
                    
                    if os.path.exists(extracted_path) or (is_windows and is_unix_path):
                        trained_model_path = extracted_path
                        # Store in state for next iteration - CLEAR post-training flags
                        return {
                            "trained_model_path": trained_model_path,
                            "pretrained_model_path": trained_model_path,
                            "model_params": state.get("model_params"),
                            "rtl_synthesis_config": state.get("rtl_synthesis_config", {}),
                            "awaiting_rtl_synthesis": True,
                            "awaiting_new_design_choice": False,  # Clear post-training state
                            "awaiting_test_image": False,  # Clear test image state
                            "current_node": "rtl_synthesis_node",
                            "dialog_state": ["rtl_synthesis_node"],
                            "needs_user_input": False,
                            "messages": [AIMessage(
                                content=f"✅ Found trained model: `{os.path.basename(trained_model_path)}`\n\nNow let's configure the synthesis parameters...",
                                name=self.name
                            )]
                        }
                    else:
                        # Path provided but doesn't exist
                        error_msg = (
                            f"❌ **Model Not Found**\n\n"
                            f"The model does not exist at: `{extracted_path}`\n\n"
                            f"Please provide a valid path to your model directory containing:\n"
                            f"  - `model.pt` (weights)\n"
                            f"  - `model.py` (class definition)\n\n"
                            f"Or a direct `.pt` file if the model class is loadable."
                        )
                        return {
                            "messages": [AIMessage(content=error_msg, name=self.name)],
                            "trained_model_path": state.get("trained_model_path"),
                            "pretrained_model_path": state.get("pretrained_model_path"),
                            "model_params": state.get("model_params"),
                            "rtl_synthesis_config": state.get("rtl_synthesis_config", {}),
                            "awaiting_rtl_synthesis": True,
                            "awaiting_new_design_choice": False,  # Clear post-training state
                            "awaiting_test_image": False,  # Clear test image state
                            "current_node": "rtl_synthesis_node",
                            "dialog_state": ["rtl_synthesis_node"],
                            "needs_user_input": True,
                        }
            
            # No path found - ask user
            ask_path_msg = (
                "🔧 **RTL Synthesis - Trained Model Required**\n\n"
                "To synthesize your neural network to RTL hardware, I need your trained model.\n\n"
                "**Option 1: UI-Generated Project** (If you trained through this UI)\n"
                "Just provide the project path - I'll find the model automatically:\n"
                "- `/home/user/my_lenet5_project/`\n"
                "- `D:\\projects\\my_cnn\\`\n\n"
                "**Option 2: Model Directory** (Direct upload)\n"
                "Provide a directory containing:\n"
                "- `model.pt` (weights saved with `torch.save(model.state_dict(), ...)`)\n"
                "- `model.py` (the CNN class definition)\n\n"
                "**Example paths:**\n"
                "- `/home/user/models/lenet5_output/`\n"
                "- `D:\\projects\\models\\my_cnn\\`\n\n"
                "Type the path to your project or model directory:"
            )
            return {
                "messages": [AIMessage(content=ask_path_msg, name=self.name)],
                "trained_model_path": state.get("trained_model_path"),
                "pretrained_model_path": state.get("pretrained_model_path"),
                "model_params": state.get("model_params"),
                "rtl_synthesis_config": state.get("rtl_synthesis_config", {}),
                "awaiting_rtl_synthesis": True,
                "awaiting_new_design_choice": False,  # Clear post-training state
                "awaiting_test_image": False,  # Clear test image state
                "current_node": "rtl_synthesis_node",
                "dialog_state": ["rtl_synthesis_node"],
                "needs_user_input": True,
            }
        
        # ------------------------------------------------------------------
        # Step 1: Validate trained model exists
        # ------------------------------------------------------------------
        is_unix_path = trained_model_path.startswith('/') if trained_model_path else False
        is_windows = os.name == 'nt'
        
        if not trained_model_path or (not os.path.exists(trained_model_path) and not (is_windows and is_unix_path)):
            error_msg = (
                "❌ **RTL Synthesis Error**\n\n"
                f"Trained model not found at: `{trained_model_path}`\n\n"
                "Please provide a valid path to your trained model file."
            )
            return {
                "messages": [AIMessage(content=error_msg, name=self.name)],
                "trained_model_path": None,  # Clear invalid path
                "pretrained_model_path": None,
                "model_params": state.get("model_params"),
                "rtl_synthesis_config": state.get("rtl_synthesis_config", {}),
                "awaiting_rtl_synthesis": True,
                "current_node": "rtl_synthesis_node",
                "dialog_state": ["rtl_synthesis_node"],
                "needs_user_input": True,
                "errors": state.get("errors", []) + [{
                    "stage": "rtl_synthesis",
                    "error": "Trained model file not found",
                    "path": trained_model_path
                }]
            }
        
        # ------------------------------------------------------------------
        # Step 2: Get or infer model input dimensions
        # ------------------------------------------------------------------
        model_params = state.get("model_params", {})
        if isinstance(model_params, dict):
            height = model_params.get("height")
            width = model_params.get("width")
            channels = model_params.get("channels")
        else:
            height = getattr(model_params, "height", None)
            width = getattr(model_params, "width", None)
            channels = getattr(model_params, "channels", None)
        
        # If dimensions not in state, try to extract from model file
        if not all([height, width, channels]) and trained_model_path:
            try:
                inferred_dims = await self._infer_model_dimensions(trained_model_path)
                if inferred_dims:
                    height, width, channels = inferred_dims
                    # Store in state for future use - PRESERVE trained_model_path!
                    return {
                        "model_params": {
                            "height": height,
                            "width": width,
                            "channels": channels
                        },
                        "trained_model_path": trained_model_path,  # Preserve path
                        "pretrained_model_path": trained_model_path,  # Preserve path
                        "rtl_synthesis_config": state.get("rtl_synthesis_config", {}),  # Preserve config
                        "awaiting_rtl_synthesis": True,
                        "current_node": "rtl_synthesis_node",  # Update graph
                        "dialog_state": ["rtl_synthesis_node"],  # Update UI state
                        "needs_user_input": False,
                        "messages": [AIMessage(
                            content=f"✅ Detected model input dimensions: {channels}x{height}x{width}\n\nNow configuring synthesis parameters...",
                            name=self.name
                        )]
                    }
            except Exception as e:
                pass
        
        # Try to extract dimensions from user message BEFORE asking
        if not all([height, width, channels]):
            last_message = self._get_last_user_message(state)
            if last_message:
                dims_from_msg = await self._extract_dimensions_from_message(last_message)
                if dims_from_msg:
                    height, width, channels = dims_from_msg
                    # Store and continue - PRESERVE trained_model_path!
                    return {
                        "model_params": {
                            "height": height,
                            "width": width,
                            "channels": channels
                        },
                        "trained_model_path": trained_model_path,  # Preserve path
                        "pretrained_model_path": trained_model_path,  # Preserve path
                        "rtl_synthesis_config": state.get("rtl_synthesis_config", {}),  # Preserve config
                        "awaiting_rtl_synthesis": True,
                        "current_node": "rtl_synthesis_node",  # Update graph
                        "dialog_state": ["rtl_synthesis_node"],  # Update UI state
                        "needs_user_input": False,
                        "messages": [AIMessage(
                            content=f"✅ Using input dimensions: {channels}x{height}x{width}\n\nNow configuring synthesis parameters...",
                            name=self.name
                        )]
                    }
        
        # If STILL missing dimensions after trying message extraction, ask user
        if not all([height, width, channels]):
            ask_dims_msg = (
                "🔧 **Model Input Dimensions Required**\n\n"
                "I need to know the input shape for your model to proceed with synthesis.\n\n"
                "**For LeNet-5:** Typically 1x28x28 (grayscale MNIST)\n"
                "**For ResNet:** Typically 3x224x224 (color ImageNet)\n"
                "**For custom models:** Check your training configuration\n\n"
                "**Please provide the input dimensions in this format:**\n"
                "`channels x height x width`\n\n"
                "**Examples:**\n"
                "- `1x28x28` (grayscale 28x28)\n"
                "- `3x32x32` (color 32x32)\n"
                "- `3x224x224` (color 224x224)"
            )
            return {
                "messages": [AIMessage(content=ask_dims_msg, name=self.name)],
                "trained_model_path": trained_model_path,  # Preserve path
                "pretrained_model_path": trained_model_path,  # Preserve path
                "model_params": state.get("model_params"),
                "rtl_synthesis_config": state.get("rtl_synthesis_config", {}),
                "awaiting_rtl_synthesis": True,
                "current_node": "rtl_synthesis_node",
                "dialog_state": ["rtl_synthesis_node"],
                "needs_user_input": True,
            }
        
        # ------------------------------------------------------------------
        # Step 3: Collect synthesis parameters iteratively
        # ------------------------------------------------------------------
        
        rtl_config = state.get("rtl_synthesis_config", {})
        reuse_factor = rtl_config.get("reuse_factor")
        clock_period = rtl_config.get("clock_period")
        strategy = rtl_config.get("strategy")
        precision = rtl_config.get("precision")
        io_type = rtl_config.get("io_type")
        run_questasim = rtl_config.get("run_questasim")
        ready_to_run = rtl_config.get("ready_to_run", False)
        
        # If user just provided a message, try to extract parameters
        last_message = self._get_last_user_message(state)
        if last_message:
            extracted = await self._extract_synthesis_params(last_message)
            
            # Update config with extracted values
            if extracted.reuse_factor is not None:
                reuse_factor = extracted.reuse_factor
                rtl_config["reuse_factor"] = reuse_factor
            if extracted.clock_period is not None:
                clock_period = extracted.clock_period
                rtl_config["clock_period"] = clock_period
            if extracted.strategy is not None:
                strategy = extracted.strategy
                rtl_config["strategy"] = strategy
            if extracted.precision is not None:
                precision = extracted.precision
                rtl_config["precision"] = precision
            if extracted.io_type is not None:
                io_type = extracted.io_type
                rtl_config["io_type"] = io_type
            if extracted.wants_questasim is not None:
                run_questasim = extracted.wants_questasim
                rtl_config["run_questasim"] = run_questasim
            if extracted.ready_to_synthesize:
                ready_to_run = True
                rtl_config["ready_to_run"] = True
            
            # Update state with new config - IMPORTANT: Preserve trained_model_path!
            state = {
                **state,
                "rtl_synthesis_config": rtl_config,
            }
        
        # Check what parameters are still missing
        missing_params = []
        if reuse_factor is None:
            missing_params.append("reuse_factor")
        if clock_period is None:
            missing_params.append("clock_period")
        if strategy is None:
            missing_params.append("strategy")
        if precision is None:
            # Default to ac_fixed<16,6> if not specified, but allow user to override
            # We won't block on this, just set a default if missing
            precision = "ac_fixed<16,6>"
            rtl_config["precision"] = precision
        if io_type is None:
            # Default to io_parallel as it avoids HIER-11 errors
            io_type = "io_parallel"
            rtl_config["io_type"] = io_type
        if run_questasim is None:
            missing_params.append("questasim_confirmation")
        
        # If any parameters are missing or user hasn't confirmed, ask for them
        if missing_params or not ready_to_run:
            return await self._ask_for_parameters(state, rtl_config, missing_params)
        
        # ------------------------------------------------------------------
        # Step 4: All parameters collected - Start Pipeline
        # Use trained_model_path from Step 0 (already fetched)
        # ------------------------------------------------------------------
        if not trained_model_path:
            return {
                "messages": [AIMessage(
                    content="❌ **Error**: Model path was lost. Please restart and provide the model path again.",
                    name=self.name
                )],
                "awaiting_rtl_synthesis": False,
                "current_node": "rtl_synthesis_node",
                "dialog_state": ["rtl_synthesis_node"],
                "needs_user_input": True,
            }
        
        # Get dimensions from state
        model_params = state.get("model_params", {})
        if isinstance(model_params, dict):
            height = model_params.get("height")
            width = model_params.get("width")
            channels = model_params.get("channels")
        else:
            height = getattr(model_params, "height", None)
            width = getattr(model_params, "width", None)
            channels = getattr(model_params, "channels", None)
        
        if not all([height, width, channels]):
            return {
                "messages": [AIMessage(
                    content="❌ **Error**: Model dimensions were lost. Please provide dimensions again.",
                    name=self.name
                )],
                "trained_model_path": trained_model_path,
                "pretrained_model_path": state.get("pretrained_model_path"),
                "model_params": state.get("model_params"),
                "rtl_synthesis_config": state.get("rtl_synthesis_config", {}),
                "awaiting_rtl_synthesis": True,
                "current_node": "rtl_synthesis_node",
                "dialog_state": ["rtl_synthesis_node"],
                "needs_user_input": True,
            }
            
        # Transition to first stage: Configure HLS
        return {
            "messages": [AIMessage(
                content=f"✅ **Configuration Confirmed**\n\nStarting synthesis pipeline...\n\n**Phase 1:** Configuring Catapult AI NN and generating C++ model...",
                name=self.name
            )],
            "trained_model_path": trained_model_path,
            "pretrained_model_path": state.get("pretrained_model_path"),
            "model_params": state.get("model_params"),
            "rtl_synthesis_config": rtl_config,
            "awaiting_rtl_synthesis": False, # Done with parameter collection
            "awaiting_hls_config": True,     # Trigger next node
            "current_node": "rtl_synthesis_node",
            "dialog_state": ["rtl_synthesis_node"],
            "needs_user_input": False,
        }

    async def configure_hls(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Step 1: Configure HLS and Generate C++."""
        self._stream_log_to_ui("🔧 [DEBUG] configure_hls() called", state)
        return await self._run_step(state, "config", "HLS Configuration")

    async def verify_hls(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Step 2: Verify C++ Model Accuracy."""
        return await self._run_step(state, "verify", "C++ Verification")

    async def synthesize_rtl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Step 3: Synthesize RTL."""
        return await self._run_step(state, "build", "RTL Synthesis")
    
    async def _run_step(self, state: Dict[str, Any], step: str, step_name: str) -> Dict[str, Any]:
        """Helper to run a specific synthesis step with live log streaming."""
        
        self._stream_log_to_ui(f"🔧 [DEBUG] _run_step() called: step={step}, step_name={step_name}", state)
        
        trained_model_path = state.get("trained_model_path")
        model_params = state.get("model_params", {})
        if isinstance(model_params, dict):
            height = model_params.get("height")
            width = model_params.get("width")
            channels = model_params.get("channels")
        else:
            height = getattr(model_params, "height", None)
            width = getattr(model_params, "width", None)
            channels = getattr(model_params, "channels", None)
            
        rtl_config = state.get("rtl_synthesis_config", {})
        reuse_factor = rtl_config.get("reuse_factor")
        clock_period = rtl_config.get("clock_period")
        strategy = rtl_config.get("strategy")
        precision = rtl_config.get("precision")
        io_type = rtl_config.get("io_type", "io_parallel")
        run_questasim = rtl_config.get("run_questasim")
        
        # Find catapult_keras_flow.py script
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            "catapult_conversion",
            "catapult_keras_flow.py"
        )
        
        if not os.path.exists(script_path):
            return {
                "messages": [AIMessage(content=f"❌ Script not found: {script_path}", name=self.name)],
                "needs_user_input": True
            }

        # Get Catapult Python path - MUST use Catapult's Python for catapult_ai_nn
        # The catapult_ai_nn module requires Catapult's bundled Python environment
        mgc_home = os.environ.get("MGC_HOME", "/data/tools/catapult/Mgc_home")
        
        # Always prefer Catapult Python - it has the correct hls4ml/catapult_ai_nn setup
        catapult_python_paths = [
            os.path.join(mgc_home, "bin", "python3"),
            "/data/tools/catapult/Mgc_home/bin/python3",  # Default EC2 path
        ]
        
        python_exe = None
        for catapult_python in catapult_python_paths:
            # On Windows, we'll run via WSL so check won't work - trust the path
            if os.name == 'nt' or os.path.exists(catapult_python):
                python_exe = catapult_python
                break
        
        if not python_exe:
            # Last resort fallback - but this likely won't work with catapult_ai_nn
            python_exe = sys.executable
            self._stream_log_to_ui(
                f"⚠️ WARNING: Catapult Python not found. Using {python_exe} - synthesis may fail!",
                state
            )
        
        self._stream_log_to_ui(f"🐍 Using Python: {python_exe}", state)
        self._stream_log_to_ui(f"📁 MGC_HOME: {mgc_home}", state)
        
        # Build command
        project_path = state.get("project_path") or "."
        project_name = state.get("project_name") or "myproject"
        rtl_output_dir = os.path.join(project_path, "outputs", "rtl_synthesis")
        os.makedirs(rtl_output_dir, exist_ok=True)

        # Detect if model_path has project structure (files in separate directories)
        model_type, model_pt_path, model_py_dir = self._detect_project_structure(trained_model_path)
        
        cmd = [
            python_exe,
            "-u", # Force unbuffered output
            script_path,
            trained_model_path,
            "--input-shape", "1", str(channels), str(height), str(width),
            "--output", rtl_output_dir,
            "--name", project_name,
            "--reuse-factor", str(reuse_factor),
            "--clock-period", str(clock_period),
            "--strategy", strategy,
            "--precision", precision,
            "--io-type", io_type,
            "--step", step
        ]
        
        # Add project structure arguments if detected
        if model_type and model_pt_path and model_py_dir:
            cmd.extend(["--model-type", model_type])
            self._stream_log_to_ui(
                f"📁 Using project structure: {model_type}",
                state
            )
            self._stream_log_to_ui(f"   model.pt: {model_pt_path}", state)
            self._stream_log_to_ui(f"   model.py dir: {model_py_dir}", state)
        
        if run_questasim:
            cmd.append("--run-questasim")
            
        # Setup environment for Catapult AI NN
        env = os.environ.copy()
        env['MGC_HOME'] = mgc_home
        
        # Add Catapult bin to PATH
        catapult_bin = os.path.join(mgc_home, "bin")
        env['PATH'] = f"{catapult_bin}:{env.get('PATH', '')}"
        
        # CRITICAL: PYTHONPATH must point to the INNER hls4ml directory
        # This is $MGC_HOME/shared/pkgs/ccs_hls4ml/hls4ml (NOT ccs_hls4ml itself!)
        # This allows both:
        #   - import catapult_ai_nn  (from ccs_hls4ml/hls4ml/catapult_ai_nn.py)
        #   - import hls4ml.utils    (from ccs_hls4ml/hls4ml/hls4ml/utils/)
        hls4ml_path = os.path.join(mgc_home, "shared", "pkgs", "ccs_hls4ml", "hls4ml")
        existing_pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{hls4ml_path}:{existing_pythonpath}" if existing_pythonpath else hls4ml_path
        
        # Prepare log
        log_path = os.path.join(rtl_output_dir, f"synthesis_{step}.log")
        
        # Notify start
        self._stream_log_to_ui(f"--- Starting {step_name} ---", state)
        
        cmd_str = " ".join(shlex.quote(str(arg)) for arg in cmd)
        
        # System libs hack
        system_lib_paths = ["/usr/lib/x86_64-linux-gnu", "/usr/lib", "/lib/x86_64-linux-gnu", "/lib"]
        lib_path_str = ":".join(system_lib_paths)
        system_include_paths = ["/usr/include/x86_64-linux-gnu", "/usr/include"]
        include_path_str = ":".join(system_include_paths)
        
        log_lines = []
        
        # Create a temporary script for the command to ensure environment is set correctly
        run_script_path = os.path.join(rtl_output_dir, f"run_{step}.sh")
        status_file = os.path.join(rtl_output_dir, f"status_{step}.txt")
        
        # QuestaSim flag for the command
        questa_flag = " --run-questasim" if run_questasim else ""
        
        # Model type flag for project structure
        model_type_flag = f" --model-type {model_type}" if model_type else ""
        
        # Script content: Setup env, run command, capture exit code, keep window open on error
        # CRITICAL: PYTHONPATH must be the INNER hls4ml directory for both:
        #   - import catapult_ai_nn
        #   - import hls4ml.utils
        script_content = f"""#!/bin/bash
# Catapult AI NN Environment Setup (matching pytorchdocumentation.txt)
export MGC_HOME="{mgc_home}"
export PATH="{mgc_home}/bin:$PATH"
export LIBRARY_PATH={lib_path_str}:$LIBRARY_PATH
export CPATH={include_path_str}:$CPATH

# CRITICAL: PYTHONPATH must point to the INNER hls4ml directory
# This is $MGC_HOME/shared/pkgs/ccs_hls4ml/hls4ml (NOT ccs_hls4ml itself!)
export PYTHONPATH="{hls4ml_path}:$PYTHONPATH"

echo "=================================================="
echo "   Running {step_name}"
echo "=================================================="
echo "   MGC_HOME: $MGC_HOME"
echo "   PYTHONPATH: $PYTHONPATH"
echo "   Python: {python_exe}"
echo "=================================================="
echo ""

# Run with Catapult's Python and correct PYTHONPATH
{python_exe} -u "{script_path}" "{trained_model_path}" --input-shape 1 {channels} {height} {width} --output "{rtl_output_dir}" --name "{project_name}" --reuse-factor {reuse_factor} --clock-period {clock_period} --strategy "{strategy}" --precision "{precision}" --io-type "{io_type}" --step {step}{model_type_flag}{questa_flag} | tee "{log_path}"
RET=${{PIPESTATUS[0]}}
echo "$RET" > "{status_file}"
echo ""
echo "=================================================="
if [ $RET -eq 0 ]; then
    echo "   {step_name} SUCCESS"
else
    echo "   {step_name} FAILED (Exit Code: $RET)"
    echo "   Check log: {log_path}"
fi
echo ""
echo "Press ENTER to close this window..."
read
"""
        with open(run_script_path, "w") as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(run_script_path, 0o755)
        
        # Remove previous status file
        if os.path.exists(status_file):
            os.remove(status_file)
            
        log_lines = []
        return_code = 1
        
        try:
            if os.name == 'nt':
                # Windows (WSL) - Set up full Catapult environment
                # CRITICAL: PYTHONPATH must be the INNER hls4ml directory
                wsl_cmd = (
                    f"export MGC_HOME='{mgc_home}'; "
                    f"export PATH='{mgc_home}/bin:$PATH'; "
                    f"export PYTHONPATH='{hls4ml_path}:$PYTHONPATH'; "
                    f"export LIBRARY_PATH={lib_path_str}:$LIBRARY_PATH; "
                    f"export CPATH={include_path_str}:$CPATH; "
                    f"{cmd_str}"
                )
                process = await asyncio.create_subprocess_exec(
                    "wsl", "bash", "-l", "-c", wsl_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT
                )
                
                # Stream output
                with open(log_path, 'w') as log_file:
                    buffer = []
                    last_flush_time = asyncio.get_event_loop().time()
                    
                    while True:
                        if process.stdout is None:
                            break
                        line = await process.stdout.readline()
                        if not line:
                            break
                        decoded_line = line.decode(errors='replace').rstrip()
                        
                        # Write to log file
                        log_file.write(decoded_line + '\n')
                        log_file.flush()
                        log_lines.append(decoded_line)
                        
                        # Buffer for UI
                        if decoded_line.strip():
                            buffer.append(decoded_line)
                        
                        current_time = asyncio.get_event_loop().time()
                        if len(buffer) >= 10 or (current_time - last_flush_time) > 0.2:
                            if buffer:
                                chunk = "\n".join(buffer)
                                self._stream_log_to_ui(chunk, state)
                                buffer = []
                            last_flush_time = current_time
                    
                    if buffer:
                        chunk = "\n".join(buffer)
                        self._stream_log_to_ui(chunk, state)
                
                await process.wait()
                return_code = process.returncode
            else:
                # Linux (EC2) - Launch gnome-terminal
                self._stream_log_to_ui(f"🚀 Launching terminal for {step_name}...", state)
                
                terminal_process = None
                terminal_launched = False
                
                try:
                    # Check if DISPLAY is set
                    display_env = os.environ.get("DISPLAY")
                    if not display_env:
                        self._stream_log_to_ui("⚠️ DISPLAY env var not set. Terminal might not appear.", state)
                    else:
                        self._stream_log_to_ui(f"✅ DISPLAY={display_env}", state)
                    
                    # Debug: Log script path and environment
                    self._stream_log_to_ui(f"📝 Script: {run_script_path}", state)
                    self._stream_log_to_ui(f"📝 Script exists: {os.path.exists(run_script_path)}", state)
                    
                    # Prepare environment with DISPLAY
                    term_env = os.environ.copy()
                    if display_env:
                        term_env['DISPLAY'] = display_env
                    
                    # Launch gnome-terminal executing our script
                    # We use --wait to ensure the process stays alive until the window closes.
                    # This allows us to correctly detect if the window failed to open (process exits immediately).
                    
                    # Try gnome-terminal
                    try:
                        # Note: --wait blocks until the terminal window closes
                        self._stream_log_to_ui("🔧 Attempting gnome-terminal launch...", state)
                        terminal_process = subprocess.Popen(
                            ["gnome-terminal", "--wait", "--", "bash", run_script_path],
                            env=term_env
                        )
                        terminal_launched = True
                        self._stream_log_to_ui(f"✅ Terminal launched! PID={terminal_process.pid}", state)
                    except FileNotFoundError:
                        # Try xterm (xterm doesn't need --wait, it blocks by default)
                        self._stream_log_to_ui("⚠️ gnome-terminal not found, trying xterm...", state)
                        try:
                            terminal_process = subprocess.Popen(
                                ["xterm", "-e", "bash", run_script_path],
                                env=term_env
                            )
                            terminal_launched = True
                            self._stream_log_to_ui(f"✅ xterm launched! PID={terminal_process.pid}", state)
                        except FileNotFoundError:
                            self._stream_log_to_ui("❌ Could not launch terminal (gnome-terminal/xterm not found). Falling back to background execution.", state)
                except Exception as e:
                    self._stream_log_to_ui(f"❌ Error launching terminal: {e}", state)
                
                if terminal_launched and terminal_process:
                    # Poll for status file OR process exit
                    max_wait = 7200 # 2 hours timeout
                    start_wait = asyncio.get_event_loop().time()
                    last_log_pos = 0
                    
                    # Give the terminal a moment to start and possibly fail
                    await asyncio.sleep(2)
                    
                    # Check if process died immediately (indicating launch failure)
                    if terminal_process.poll() is not None:
                        # Process exited. Check if status file exists.
                        if not os.path.exists(status_file):
                            self._stream_log_to_ui("⚠️ Terminal process exited immediately without running script. Falling back to background execution.", state)
                            terminal_launched = False # Trigger fallback
                    
                    if terminal_launched:
                        # We have a running terminal. Monitor it.
                        while True:
                            # Check if terminal window was closed by user or script finished
                            if terminal_process.poll() is not None:
                                # Terminal closed. Check status file.
                                if os.path.exists(status_file):
                                    break # Done!
                                else:
                                    # Window closed but no status file? Maybe user closed it early.
                                    self._stream_log_to_ui("⚠️ Terminal window closed before completion.", state)
                                    # We can't really recover here easily, but let's check log one last time
                                    break
                            
                            # Check for completion via status file (in case window stays open)
                            if os.path.exists(status_file):
                                # Give it a moment to ensure write is complete
                                await asyncio.sleep(0.5)
                                with open(status_file, "r") as f:
                                    content = f.read().strip()
                                    if content:
                                        try:
                                            return_code = int(content)
                                            # If success, the script sleeps 3s then closes.
                                            # If fail, it waits for user input.
                                            # We can break here if we want to proceed without waiting for window close
                                            # But user might want to see the window.
                                            # Let's just continue monitoring logs until window closes or we decide to move on.
                                            pass 
                                        except ValueError:
                                            pass
                            
                            # Check timeout
                            if asyncio.get_event_loop().time() - start_wait > max_wait:
                                self._stream_log_to_ui("❌ Timeout waiting for terminal process.", state)
                                break
                                
                            # Stream log updates to UI (tail)
                            if os.path.exists(log_path):
                                try:
                                    with open(log_path, "r") as f:
                                        f.seek(last_log_pos)
                                        new_lines = f.read()
                                        if new_lines:
                                            last_log_pos = f.tell()
                                            lines = new_lines.splitlines()
                                            if lines:
                                                # Just show the last line as progress
                                                self._stream_log_to_ui(lines[-1], state)
                                                log_lines.extend(lines)
                                except:
                                    pass
                                    
                            await asyncio.sleep(1)
                        
                        # Read full log at the end to ensure we have everything
                        if os.path.exists(log_path):
                            with open(log_path, "r") as f:
                                log_lines = f.readlines()
                        
                        # If we broke out of loop, ensure we have a return code
                        if os.path.exists(status_file):
                            with open(status_file, "r") as f:
                                try:
                                    return_code = int(f.read().strip())
                                except:
                                    return_code = 1
                        else:
                            # No status file, assume failure
                            return_code = 1

                if not terminal_launched:
                    # Fallback to direct execution if terminal failed
                    self._stream_log_to_ui(f"🔄 Running {step_name} in background (headless)...", state)
                    
                    # Use the same script but run it directly with bash
                    full_cmd = f"bash '{run_script_path}'"
                    process = await asyncio.create_subprocess_shell(
                        full_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT
                    )
                    
                    # Stream output (reuse logic)
                    while True:
                        line = await process.stdout.readline()
                        if not line:
                            break
                        decoded_line = line.decode(errors='replace').rstrip()
                        log_lines.append(decoded_line)
                        self._stream_log_to_ui(decoded_line, state)
                    
                    await process.wait()
                    
                    # Check status file first (script writes it)
                    if os.path.exists(status_file):
                        with open(status_file, "r") as f:
                            try:
                                return_code = int(f.read().strip())
                            except:
                                return_code = process.returncode
                    else:
                        return_code = process.returncode

        except Exception as e:
            return_code = 1
            log_lines.append(f"Execution error: {str(e)}")
            self._stream_log_to_ui(f"Execution error: {str(e)}", state)

        if return_code == 0:
            # Success
            next_state = {
                "messages": [AIMessage(content=f"✅ **{step_name} Successful**", name=self.name)],
                "trained_model_path": trained_model_path,
                "pretrained_model_path": state.get("pretrained_model_path"),
                "model_params": state.get("model_params"),
                "rtl_synthesis_config": rtl_config,
                "needs_user_input": False,
            }
            
            if step == "config":
                next_state["awaiting_hls_config"] = False
                next_state["awaiting_hls_verify"] = True
                next_state["current_node"] = "configure_hls_node"
                next_state["dialog_state"] = ["configure_hls_node"]  # Update UI graph
                next_state["needs_user_input"] = True
                next_state["messages"][0].content += "\n\nConfiguration complete. Type **'verify'** to check accuracy, or **'synthesize'** to build RTL."
            elif step == "verify":
                # Extract accuracy from logs
                acc_msg = ""
                for line in log_lines:
                    if "Max Diff:" in line or "Mean Diff:" in line or "WARNING:" in line or "SUCCESS:" in line:
                        acc_msg += f"\n- {line.strip()}"
                
                next_state["awaiting_hls_verify"] = False
                next_state["awaiting_rtl_build"] = True
                next_state["current_node"] = "verify_hls_node"
                next_state["dialog_state"] = ["verify_hls_node"]  # Update UI graph
                next_state["needs_user_input"] = True
                next_state["messages"][0].content += f"\n\n**Accuracy Report:**{acc_msg}\n\nType **'synthesize'** to generate RTL."
            elif step == "build":
                next_state["awaiting_rtl_build"] = False
                next_state["awaiting_new_design_choice"] = True
                next_state["current_node"] = "synthesize_rtl_node"
                next_state["dialog_state"] = ["synthesize_rtl_node"]  # Update UI graph
                next_state["needs_user_input"] = True
                next_state["messages"][0].content += f"\n\nOutput: `{rtl_output_dir}`\n\nFlow Complete!"
                
            return next_state
        else:
            # Failure - show the failed node
            failed_node = "configure_hls_node" if step == "config" else ("verify_hls_node" if step == "verify" else "synthesize_rtl_node")
            return {
                "messages": [AIMessage(content=f"❌ **{step_name} Failed**\n\nCheck log: `{log_path}`", name=self.name)],
                "needs_user_input": True,
                "awaiting_hls_config": False,
                "awaiting_hls_verify": False,
                "awaiting_rtl_build": False,
                "current_node": failed_node,
                "dialog_state": [failed_node]  # Show which step failed in UI
            }

    def _get_last_user_message(self, state: Dict[str, Any]) -> str:
        """Extract the last user message from state."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle list of content items
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            parts.append(str(item.get("text", "")))
                        else:
                            parts.append(str(item))
                    return " ".join(parts)
        return ""
    
    async def _extract_model_path(self, message: str) -> Optional[str]:
        """
        Extract model path from user message.
        Supports:
        - Project directories (with Manual_Output/, Pretrained_Output/, or YOLOX/) - NEW
        - Model directories (containing model.pt + model.py) - RECOMMENDED
        - Direct .pt/.pth files (legacy)
        """
        import re
        
        # First, look for directory paths
        # Windows: C:\path\to\model_dir or D:/path/to/model_dir
        # Linux/Mac: /path/to/model_dir or ./relative/model_dir
        
        # Pattern for directory paths (no file extension)
        dir_patterns = [
            r'([A-Za-z]:[\\\/][\w\s\-\\\/\.]+?)(?:\s|$|,|\.(?!\w))',  # Windows absolute (ends at space/comma/period)
            r'(\/[\w\s\-\/\.]+?)(?:\s|$|,|\.(?!\w))',  # Unix absolute
            r'(\.{1,2}\/[\w\s\-\/\.]+?)(?:\s|$|,)',  # Relative paths
        ]
        
        for pattern in dir_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                path = match.group(1).strip().strip('`').strip('"').strip("'")
                
                # Check if it's a project directory with NN Generator structure
                model_type, _, _ = self._detect_project_structure(path)
                if model_type:
                    self._stream_log_to_ui(f"Found project directory ({model_type}): {path}", {})
                    return path  # Return project path
                
                # Check if it's a simple directory with model.pt + model.py inside
                model_pt = os.path.join(path, 'model.pt')
                model_py = os.path.join(path, 'model.py')
                if os.path.isdir(path) and os.path.exists(model_pt) and os.path.exists(model_py):
                    self._stream_log_to_ui(f"Found model directory: {path}", {})
                    return path  # Return directory path
        
        # Fallback: Look for .pt/.pth file paths (legacy format)
        file_patterns = [
            r'([A-Za-z]:[\\\/][\w\s\-\\\/\.]+\.(?:pt|pth))',  # Windows absolute
            r'(\/[\w\s\-\/\.]+\.(?:pt|pth))',  # Unix absolute
            r'(\.{1,2}\/[\w\s\-\/\.]+\.(?:pt|pth))',  # Relative paths
            r'([\w\s\-\/\\\.]+\.(?:pt|pth))',  # Any path with .pt/.pth
        ]
        
        for pattern in file_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                path = match.group(1)
                # Clean up the path
                path = path.strip().strip('`').strip('"').strip("'")
                return path
        
        return None
    
    def _detect_project_structure(self, model_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Detect if the model_path is a project with the NN Generator output structure.
        
        Supported structures:
        - {project}/Manual_Output/SystemC/Pt/model.pt + {project}/Manual_Output/python/model.py
        - {project}/Pretrained_Output/SystemC/Pt/model.pt + {project}/Pretrained_Output/python/model.py
        - {project}/YOLOX/SystemC/Pt/model.pt + {project}/YOLOX/python/model.py
        
        Args:
            model_path: Path that could be a project directory or a model directory
            
        Returns:
            Tuple of (model_type, model_pt_path, model_py_dir) if project structure detected,
            otherwise (None, None, None) for simple directory or file.
        """
        if not model_path:
            return None, None, None
        
        # Check for each model type
        model_types = [
            ('manual', 'Manual_Output'),
            ('pretrained', 'Pretrained_Output'),
            ('yolox', 'YOLOX'),
        ]
        
        for model_type, output_dir_name in model_types:
            # Check if this looks like a project path with the expected structure
            model_pt = os.path.join(model_path, output_dir_name, 'SystemC', 'Pt', 'model.pt')
            model_py_dir = os.path.join(model_path, output_dir_name, 'python')
            model_py = os.path.join(model_py_dir, 'model.py')
            
            if os.path.exists(model_pt) and os.path.exists(model_py):
                self._stream_log_to_ui(
                    f"🔍 Detected project structure: {model_type} model in {output_dir_name}/",
                    {}
                )
                return model_type, model_pt, model_py_dir
        
        # Also check if model_path is already inside an output directory
        # e.g., /project/Manual_Output/python might be passed
        path_parts = model_path.replace('\\', '/').split('/')
        for i, part in enumerate(path_parts):
            if part in ['Manual_Output', 'Pretrained_Output', 'YOLOX']:
                # Found output directory in path, construct project root
                project_root = '/'.join(path_parts[:i])
                if os.name == 'nt':
                    project_root = project_root.replace('/', '\\')
                
                for model_type, output_dir_name in model_types:
                    if part == output_dir_name:
                        model_pt = os.path.join(project_root, output_dir_name, 'SystemC', 'Pt', 'model.pt')
                        model_py_dir = os.path.join(project_root, output_dir_name, 'python')
                        model_py = os.path.join(model_py_dir, 'model.py')
                        
                        if os.path.exists(model_pt) and os.path.exists(model_py):
                            self._stream_log_to_ui(
                                f"🔍 Detected project structure from path: {model_type} model",
                                {}
                            )
                            return model_type, model_pt, model_py_dir
                break
        
        return None, None, None
    
    async def _infer_model_dimensions(self, model_path: str) -> Optional[Tuple[int, int, int]]:
        """
        Try to infer model input dimensions by loading the model.
        Returns (height, width, channels) or None if cannot determine.
        """
        if not model_path:
            return None
            
        try:
            import torch
            
            # Try loading as TorchScript first (common for deployed models)
            try:
                model = torch.jit.load(model_path, map_location='cpu')
                # For TorchScript, we can't easily inspect layers
                # Default to 32x32 for LeNet-5 (more common than 28x28 for non-MNIST)
                # User can override if wrong
                return (32, 32, 1)  # Default LeNet-5 dimensions (compatible with both MNIST and CIFAR)
            except Exception:
                # Not TorchScript, try regular torch.load
                pass
            
            # Load as regular PyTorch model
            model = torch.load(model_path, map_location='cpu')
            
            # Handle different model formats
            if isinstance(model, dict):
                # Check for common keys
                if isinstance(model, dict):
                    if 'model' in model:
                        model = model['model']
                    elif 'state_dict' in model:
                        # Can't easily infer from state_dict alone
                        pass
            
            # Try to get first layer input shape
            if hasattr(model, 'modules') and callable(getattr(model, 'modules', None)):
                for module in model.modules():  # type: ignore[union-attr]
                    if isinstance(module, torch.nn.Conv2d):
                        # Conv2d in_channels is the number of input channels
                        channels = module.in_channels
                        # For LeNet-5: typically 1x28x28
                        # For common CNNs: 1x28x28, 3x32x32, 3x224x224
                        
                        # Common dimension inference based on channels
                        if channels == 1:
                            # Grayscale - likely MNIST (28x28)
                            return (28, 28, 1)
                        elif channels == 3:
                            # Color - check kernel size to guess
                            kernel_size = module.kernel_size
                            if kernel_size == (5, 5) or kernel_size == 5:
                                # Likely CIFAR-10 or similar
                                return (32, 32, 3)
                            else:
                                # Default to ImageNet size
                                return (224, 224, 3)
                        break
            
            return None
        except Exception as e:
            return None
    
    async def _extract_dimensions_from_message(self, message: str) -> Optional[Tuple[int, int, int]]:
        """
        Extract input dimensions from user message.
        Patterns: "1x28x28", "3x224x224", "28x28x1", etc.
        """
        import re
        
        # Pattern: CxHxW or HxWxC format
        patterns = [
            r'(\d+)x(\d+)x(\d+)',  # Matches 1x28x28 or 28x28x1
            r'(\d+)\s*x\s*(\d+)\s*x\s*(\d+)',  # With spaces
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                a, b, c = int(match.group(1)), int(match.group(2)), int(match.group(3))
                
                # Determine if it's CxHxW or HxWxC
                # Channels are typically 1, 3, or 4
                # Heights/widths are typically larger (28, 32, 64, 224, etc.)
                if a <= 4 and b > a and c > a:
                    # Likely CxHxW
                    return (b, c, a)  # Return as (height, width, channels)
                elif c <= 4 and a > c and b > c:
                    # Likely HxWxC
                    return (a, b, c)
                else:
                    # Ambiguous, assume CxHxW (most common in PyTorch)
                    return (b, c, a)
        
        return None
    
    async def _extract_synthesis_params(self, message: str) -> SynthesisParametersIntent:
        """Extract synthesis parameters from user message using LLM and Regex fallback."""
        import re
        
        # 1. Try LLM extraction first
        prompt = f"""Extract RTL synthesis parameters from the user's message.

User message: "{message}"

Look for:
1. reuse_factor: Any positive integer (common: 1, 16, 32, 64, 128, 256, 784)
   - 1 = full parallel (fastest/largest)
   - 16-32 = high parallelism
   - 64 = balanced (most common)
   - 128+ = resource-efficient
   - 784 = fully sequential (smallest)
   Keywords: reuse factor, parallelism, resource usage
   
2. clock_period: Integer in nanoseconds (5-20 typical, smaller=faster but harder timing)
   Keywords: clock, frequency, MHz (convert MHz to ns: 100MHz = 10ns)
   
3. strategy: "Latency" or "Resource"
   Keywords: optimization strategy, latency-optimized, resource-optimized
   
4. wants_questasim: Boolean - does user want RTL verification?
   Keywords: questasim, verify, verification, rtl verification, cosimulation
   
5. ready_to_synthesize: Boolean - is user ready to start synthesis?
   Keywords: start, run, proceed, go, synthesize now, yes, confirm

Respond with structured output."""
        
        extracted = SynthesisParametersIntent(
            reuse_factor=None,
            clock_period=None,
            strategy=None,
            precision=None,
            wants_questasim=None,
            ready_to_synthesize=False
        )
        
        try:
            result = await self.param_extractor_llm.ainvoke(prompt)
            if isinstance(result, dict):
                extracted = SynthesisParametersIntent.model_validate(result)
            elif isinstance(result, SynthesisParametersIntent):
                extracted = result
        except Exception as e:
            print(f"Parameter extraction error: {e}")
        
        # 2. Regex Fallback - Override/Fill missing values if regex finds them
        # This ensures robustness if LLM fails
        
        # Reuse Factor
        if extracted.reuse_factor is None:
            reuse_match = re.search(r'reuse\s*(?:factor)?\s*[:=]?\s*(\d+)', message, re.IGNORECASE)
            if reuse_match:
                extracted.reuse_factor = int(reuse_match.group(1))
        
        # Clock Period
        if extracted.clock_period is None:
            clock_match = re.search(r'clock\s*(?:period)?\s*[:=]?\s*(\d+)', message, re.IGNORECASE)
            if clock_match:
                extracted.clock_period = int(clock_match.group(1))
            # Handle MHz
            mhz_match = re.search(r'(\d+)\s*MHz', message, re.IGNORECASE)
            if mhz_match:
                mhz = int(mhz_match.group(1))
                if mhz > 0:
                    extracted.clock_period = int(1000 / mhz)
        
        # Strategy
        if extracted.strategy is None:
            if re.search(r'resource', message, re.IGNORECASE):
                extracted.strategy = "Resource"
            elif re.search(r'latency', message, re.IGNORECASE):
                extracted.strategy = "Latency"
        
        # Precision
        if extracted.precision is None:
            precision_match = re.search(r'precision\s*[:=]?\s*(ac_fixed<[\d,]+>)', message, re.IGNORECASE)
            if precision_match:
                extracted.precision = precision_match.group(1)
        
        # IO Type
        if extracted.io_type is None:
            if re.search(r'io[_\s]?parallel', message, re.IGNORECASE):
                extracted.io_type = "io_parallel"
            elif re.search(r'io[_\s]?stream', message, re.IGNORECASE):
                extracted.io_type = "io_stream"
        
        # QuestaSim
        if extracted.wants_questasim is None:
            if re.search(r'no\s*questasim', message, re.IGNORECASE) or re.search(r'skip\s*verification', message, re.IGNORECASE):
                extracted.wants_questasim = False
            elif re.search(r'run\s*questasim', message, re.IGNORECASE) or re.search(r'with\s*verification', message, re.IGNORECASE):
                extracted.wants_questasim = True
        
        # Ready to synthesize
        if not extracted.ready_to_synthesize:
            if re.search(r'start|proceed|go|run|synthesize|begin', message, re.IGNORECASE):
                extracted.ready_to_synthesize = True
                
        return extracted
    
    async def _ask_for_parameters(
        self, state: Dict[str, Any], rtl_config: Dict[str, Any], missing_params: list
    ) -> Dict[str, Any]:
        """Ask user for missing synthesis parameters."""
        
        # Build current configuration display
        current_config = "**Current Configuration:**\n"
        reuse = rtl_config.get("reuse_factor")
        clock = rtl_config.get("clock_period")
        strat = rtl_config.get("strategy")
        questa = rtl_config.get("run_questasim")
        
        if reuse is not None:
            if reuse == 1:
                parallelism = "Full Parallel"
            elif reuse <= 16:
                parallelism = "High Parallelism"
            elif reuse <= 64:
                parallelism = "Balanced"
            elif reuse <= 128:
                parallelism = "Resource-Efficient"
            else:
                parallelism = "Sequential"
            current_config += f"- ✅ **Reuse Factor**: {reuse} ({parallelism})\n"
        else:
            current_config += f"- ⚠️ **Reuse Factor**: Not set\n"
        
        if clock is not None:
            freq_mhz = 1000 / clock
            current_config += f"- ✅ **Clock Period**: {clock}ns ({freq_mhz:.0f} MHz)\n"
        else:
            current_config += f"- ⚠️ **Clock Period**: Not set\n"
        
        if strat is not None:
            current_config += f"- ✅ **Strategy**: {strat}\n"
        else:
            current_config += f"- ⚠️ **Strategy**: Not set\n"
        
        if questa is not None:
            current_config += f"- ✅ **QuestaSim Verification**: {'Enabled' if questa else 'Disabled'}\n"
        else:
            current_config += f"- ⚠️ **QuestaSim Verification**: Not set\n"
        
        # Precision
        precision = rtl_config.get("precision")
        if precision:
            current_config += f"- ✅ **Precision**: {precision}\n"
        else:
            current_config += f"- ⚠️ **Precision**: Not set (default: ac_fixed<16,6>)\n"
        
        # IO Type
        io_type = rtl_config.get("io_type")
        if io_type:
            current_config += f"- ✅ **IO Type**: {io_type}\n"
        else:
            current_config += f"- ⚠️ **IO Type**: Not set (default: io_parallel)\n"
        
        # Build prompt for missing parameters
        prompt_msg = "🔧 **RTL Synthesis Configuration**\n\n"
        prompt_msg += current_config + "\n"
        
        if missing_params:
            prompt_msg += "**Please provide the following parameters:**\n\n"
            
            if "reuse_factor" in missing_params:
                prompt_msg += (
                    "**1. Reuse Factor** (Hardware resource usage):\n"
                    "   Common values:\n"
                    "   - `1`: Full parallel (fastest, largest, ~15GB+ memory during synthesis)\n"
                    "   - `16`: High parallelism (fast, large)\n"
                    "   - `32`: Good balance (fast synthesis, moderate size)\n"
                    "   - `64`: Balanced (recommended for most models, ~2-3GB memory)\n"
                    "   - `128`: Resource-efficient\n"
                    "   - `784`: Sequential (slowest, smallest, minimal memory)\n"
                    "   \n"
                    "   💡 Tip: Higher reuse = less hardware, slower inference, faster synthesis\n"
                    "   Any positive integer works - choose based on your target FPGA size\n"
                    "   Example: \"Use reuse factor 32\" or \"reuse 64\"\n\n"
                )
            
            if "clock_period" in missing_params:
                prompt_msg += (
                    "**2. Clock Period** (Target frequency):\n"
                    "   - `5`ns (200 MHz) - Very fast, may have timing issues\n"
                    "   - `10`ns (100 MHz) - Good balance (recommended)\n"
                    "   - `20`ns (50 MHz) - Conservative, easier timing\n"
                    "   Example: \"Clock period 10ns\" or \"100 MHz\"\n\n"
                )
            
            if "strategy" in missing_params:
                prompt_msg += (
                    "**3. Synthesis Strategy**:\n"
                    "   - `Latency`: Optimize for speed\n"
                    "   - `Resource`: Optimize for area (recommended for high reuse factors)\n"
                    "   Example: \"Use Resource strategy\"\n\n"
                )
            
            if "questasim_confirmation" in missing_params:
                prompt_msg += (
                    "**4. QuestaSim RTL Verification**:\n"
                    "   Do you want to run QuestaSim co-simulation to verify RTL correctness?\n"
                    "   This adds extra time but ensures RTL matches C++ behavior.\n"
                    "   Example: \"Yes, run QuestaSim\" or \"No QuestaSim\"\n\n"
                )
            
            if "precision" in missing_params:
                prompt_msg += (
                    "**5. Precision** (Fixed-point bit width):\n"
                    "   Format: `ac_fixed<total_bits, integer_bits>`\n"
                    "   - `ac_fixed<16,6>`: 16 bits total, 6 integer bits (default, good balance)\n"
                    "   - `ac_fixed<16,8>`: More integer range, less fractional precision\n"
                    "   - `ac_fixed<32,16>`: Higher precision (larger hardware)\n"
                    "   - `ac_fixed<8,4>`: Minimal (smaller, faster, less accurate)\n"
                    "   Example: \"Use precision ac_fixed<16,8>\"\n\n"
                )
            
            if "io_type" in missing_params:
                prompt_msg += (
                    "**6. IO Type** (Data interface style):\n"
                    "   - `io_parallel`: Array-based interfaces (⭐ **recommended** - avoids HIER-11 errors)\n"
                    "   - `io_stream`: Streaming FIFO interfaces (may cause issues with some models)\n"
                    "   Example: \"Use io_parallel\"\n\n"
                )
        else:
            # All parameters collected, ask for confirmation
            prompt_msg += (
                "\n✅ **All parameters configured!**\n\n"
                "Type **'start synthesis'** or **'proceed'** to begin RTL synthesis.\n"
                "Or modify any parameter by providing new values."
            )
        
        return {
            "messages": [AIMessage(content=prompt_msg, name=self.name)],
            "rtl_synthesis_config": rtl_config,
            "trained_model_path": state.get("trained_model_path"),  # PRESERVE!
            "pretrained_model_path": state.get("pretrained_model_path"),  # PRESERVE!
            "model_params": state.get("model_params"),  # PRESERVE dimensions!
            "awaiting_rtl_synthesis": True,
            "current_node": "rtl_synthesis_node",  # Update graph
            "dialog_state": ["rtl_synthesis_node"],  # Update UI state
            "needs_user_input": True,
        }
    
    def _stream_log_to_ui(self, content: str, state: Dict[str, Any]):
        """
        Directly stream a message to the UI via stdout.
        This bypasses LangGraph's state update mechanism to provide real-time feedback.
        """
        # Construct a minimal state object that satisfies the frontend
        ui_state = {
            "project_name": state.get("project_name"),
            "project_path": state.get("project_path"),
            "current_stage": state.get("current_stage"),
            "dialog_state": ["rtl_synthesis_node"],
            "completed_stages": state.get("completed_stages", []),
            "awaiting_test_image": False,
            "awaiting_new_design_choice": False,
            "needs_user_input": False
        }
        
        # Use 'log' type to hint frontend to render as a console/terminal block
        # instead of separate chat bubbles
        message = {
            "type": "log", 
            "content": content,
            "state": ui_state
        }
        
        try:
            # Write to stdout with flush to ensure immediate delivery
            print(json.dumps(message), flush=True)
        except Exception:
            pass
