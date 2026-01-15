"""Training Coordinator - Validates and launches training via MCP tools.

This assistant ensures a valid project_output is available and then calls the
appropriate training tool on the MCP server based on the requested training type
("pretrained", "manual", or "yolox").
"""

from typing import Dict, Any, Optional
from pathlib import Path
from langchain_core.runnables import RunnableConfig
from regex import T

from .base_assistant import BaseAssistant
from src.agentic.src.core.state import NNGeneratorState
from src.agentic.src.mcp import get_mcp_helper


class TrainingCoordinator(BaseAssistant):
	"""Coordinates training runs using MCP server tools."""

	def __init__(self, llm=None):
		super().__init__(name="training_coordinator", llm=llm)

	def _determine_training_type(self, state: NNGeneratorState) -> str:
		"""Determine training type by inspecting state.
		
		Logic:
		- If manual_layers is not empty -> "manual"
		- Else if pretrained_model starts with "yolox" -> "yolox"
		- Else if pretrained_model exists -> "pretrained"
		- Default -> "pretrained"
		"""
		manual_layers = state.get("manual_layers", [])
		pretrained_model = state.get("pretrained_model")
		
		# Check manual layers first
		if manual_layers and len(manual_layers) > 0:
			return "manual"
		
		# Check pretrained model
		if pretrained_model:
			if str(pretrained_model).lower().startswith("yolox"):
				return "YOLOX"
			return "pretrained"
		
		# Default to pretrained
		return "pretrained"

	async def __call__(self, state: NNGeneratorState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
		"""Validate inputs and launch training based on training_type.

		Reads state directly (following project_manager pattern):
		- project_output: project root from state
		- project_path: fallback if project_output not set
		- manual_layers: to detect manual training
		- pretrained_model: to detect pretrained/yolox training
		
		Optional state fields:
		- output_dir: optional override for training outputs
		- log_dir: optional override for tensorboard logs
		- verbose: bool (default: True)
		"""

		# Read state directly
		project_output = state.get("project_output") or state.get("project_path")
		
		# Determine training type by inspecting state
		training_type = self._determine_training_type(state)

		# Check project_output exists
		if not project_output:
			return {
				"messages": [self.format_message(
					"❌ **Cannot Start Training**\n\n"
					"**Missing:** project_output was not found.\n\n"
					"Please run code generation first so I can capture the project output directory."
				)],
				"needs_user_input": False
			}

		project_output_path = Path(str(project_output))
		if not project_output_path.exists():
			return {
				"messages": [self.format_message(
					"❌ **Cannot Start Training**\n\n"
					f"The project output directory does not exist:\n`{project_output_path}`\n\n"
					"Please verify the path or run code generation again."
				)],
				"needs_user_input": False
			}

		# Read optional overrides from state
		output_dir = state.get("output_dir")
		if output_dir is not None:
			output_dir = Path(str(output_dir))
		else:
			# Default: project_output/outputs/training_type (determined from state)
			output_dir = project_output_path / "outputs" / training_type

		log_dir = state.get("log_dir")
		if log_dir is not None:
			log_dir = Path(str(log_dir))
		else:
			# Default consistent with MCP server tools
			log_dir = Path("data/tensorboardlogs")


		# Call the appropriate MCP training tool
		mcp = await get_mcp_helper()
		tool_name = {
			"pretrained": "train_pretrained_model",
			"manual": "train_manual_model",
			"YOLOX": "train_yolox_model",
		}.get(training_type)

		if tool_name is None:
			return self.create_error_response(
				ValueError(f"Unknown training_type: {training_type}"),
				stage="training",
				severity="high",
			)

		try:
			# Notify user that training is starting
			start_message = self.format_message(
				f"🚀 **Training Started**\n\n"
				f"- Training type: `{training_type}`\n"
				f"- Project: `{project_output_path}`\n"
				f"- Log directory: `{log_dir}`\n\n"
				f"Training is now running... This may take several minutes."
			)
			
			# Call MCP training tool with project_output
			# The MCP server will handle both project root and nested output directory paths
			result = await mcp.call(tool_name, {
				"project_output": str(project_output_path),
				"output_dir": str(output_dir),
				"log_dir": str(log_dir),
				"verbose": True,
			})

			status = result.get("status") if isinstance(result, dict) else None
			details = result.get("details", {}) if isinstance(result, dict) else {}

			if status != "success":
				msg = result.get("message", "Training failed") if isinstance(result, dict) else "Training failed"
				error_details = []
				error_details.append(f"**Error:** {msg}")
				
				if details.get("exit_code"):
					error_details.append(f"**Exit code:** {details['exit_code']}")
				if details.get("saved_model_path"):
					model_exists = details.get("saved_model_exists", False)
					error_details.append(
						f"**Model saved:** {'✅ Yes' if model_exists else '❌ No'} at `{details['saved_model_path']}`"
					)
				
				return {
					"messages": [
						start_message,
						self.format_message(f"❌ **Training Failed**\n\n" + "\n".join(error_details))
					],
					"needs_user_input": False
				}

			# Success message - extract saved model info from details
			success_parts = [
				"✅ **Training Completed Successfully**\n",
				f"- Training type: `{training_type}`",
				f"- Log directory: `{details.get('log_dir', str(log_dir))}`",
			]
			
			# Extract model path and prepare state update
			state_updates = {}
			if details.get("saved_model_path"):
				model_exists = details.get("saved_model_exists", False)
				if model_exists:
					success_parts.append(f"- **Model saved at:** `{details['saved_model_path']}` ✅")
					state_updates["trained_model_path"] = str(details["saved_model_path"])
				else:
					success_parts.append(f"- **Model path (not found):** `{details['saved_model_path']}` ⚠️")
			
			# For YOLOX models, try to locate class names file from dataset path
			if training_type == "YOLOX":
				dataset_path = state.get("model_params", {}).get("dataset_path") if isinstance(state.get("model_params"), dict) else None
				if dataset_path:
					dataset_path_obj = Path(str(dataset_path))
					# Common YOLOX class names file locations
					possible_class_files = [
						dataset_path_obj / "classes.txt",
						dataset_path_obj / "class_names.txt",
						dataset_path_obj / "annotations" / "classes.txt",
					]
					for class_file in possible_class_files:
						if class_file.exists():
							state_updates["class_names_path"] = str(class_file)
							success_parts.append(f"- **Class names file:** `{class_file}` ✅")
							break
			
			if details.get("training_script"):
				success_parts.append(f"- Training script: `{details['training_script']}`")

			return {
				"messages": [
					start_message,
					self.format_message("\n".join(success_parts))
				],
				**state_updates,
				**self.update_stage(state, "training_complete"),
				"needs_user_input": False
			}

		except Exception as e:
			return self.create_error_response(e, stage="training", severity="high")

