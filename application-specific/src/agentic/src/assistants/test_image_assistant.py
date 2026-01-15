from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, List

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agentic.src.core import llm_context
from src.agentic.src.core.state import NNGeneratorState


_IMAGE_EXT_PATTERN = r"(?:png|jpg|jpeg|bmp|gif)"


class ImagePathExtraction(BaseModel):
    image_path: Optional[str] = Field(
        default=None,
        description="Absolute or relative image file path extracted from the user message, or null if none.",
    )


class YoloxInferenceParams(BaseModel):
    """Parameters needed for YOLOX model inference."""
    image_path: Optional[str] = Field(
        default=None,
        description="Path to the image file to test (.png, .jpg, .jpeg, etc.)",
    )
    weights_path: Optional[str] = Field(
        default=None,
        description="Path to the trained YOLOX model weights file (e.g., best_ckpt.pth)",
    )
    class_names_path: Optional[str] = Field(
        default=None,
        description="Path to the class names file (e.g., classes.txt, class_names.txt)",
    )
    hgd_checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to the HGD denoising checkpoint file (e.g., best_ckpt.pt)",
    )


def _get_last_user_message(state: NNGeneratorState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) and message.content:
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text_value = item.get("text") or item.get("content")
                        if text_value:
                            parts.append(str(text_value))
                    else:
                        parts.append(str(item))
                return " ".join(parts)
            return str(content)
    return ""


def _strip_wrapping_quotes(text: str) -> str:
    return text.strip().strip('"').strip("'").strip()


def _get_last_ai_message_content(state: NNGeneratorState) -> Optional[str]:
    """Retrieve the content of the last AI message from the history."""
    from langchain_core.messages import AIMessage
    messages = state.get("messages", [])
    # Iterate backwards to find the last AI Message, skipping the immediate user message at the end
    for message in reversed(messages[:-1]):
        if isinstance(message, AIMessage) and message.content:
            if isinstance(message.content, str):
                return message.content
            elif isinstance(message.content, list):
                # Handle multimodal/complex content by joining text parts
                parts = []
                for item in message.content:
                    if isinstance(item, dict):
                        text_value = item.get("text") or item.get("content")
                        if text_value:
                            parts.append(str(text_value))
                    else:
                        parts.append(str(item))
                return " ".join(parts)
    return None





async def extract_image_path_from_message(message: str) -> Optional[str]:
    message = message or ""
    message = message.strip()
    if not message:
        return None

    llm = llm_context.design_intent_llm
    if llm is None:
        raise RuntimeError("LLM not configured for image path extraction")

    prompt = (
        "Extract ONE image file path from the text. "
        "If there is no image path, return null. "
        "Only extract paths that end with an image extension (png/jpg/jpeg/gif).\n"
        f"Text: {message}"
    )

    try:
        structured = llm.with_structured_output(ImagePathExtraction)
        parsed = await structured.ainvoke(prompt)
        if isinstance(parsed, dict):
            parsed = ImagePathExtraction.model_validate(parsed)
        if isinstance(parsed, ImagePathExtraction) and parsed.image_path:
            return _strip_wrapping_quotes(str(parsed.image_path))
    except Exception as e:
        raise RuntimeError(f"Failed to extract image path from message: {str(e)}") from e

    return None


async def extract_yolox_params_from_message(message: str, state: NNGeneratorState) -> YoloxInferenceParams:
    """Extract YOLOX inference parameters from user message using LLM structured output.
    
    Follows configuration_specialist.py pattern with conditional context injection.
    """
    message = message or ""
    message = message.strip()
    
    if not message:
        return YoloxInferenceParams()

    llm = llm_context.design_intent_llm
    if llm is None:
        raise RuntimeError("LLM not configured for YOLOX parameter extraction")

    # ------------------------------------------------------------------
    # CONDITIONAL CONTEXT INJECTION
    # ------------------------------------------------------------------
    # Only inject context if the system is awaiting test image input
    context_prompt_section = ""
    
    if state.get("awaiting_test_image"):
        last_ai_msg = _get_last_ai_message_content(state)
        if last_ai_msg:
            context_prompt_section = (
                f"\n\nCONTEXT (You previously said this to the user):\n"
                f"\"\"\"{last_ai_msg}\"\"\"\n"
                f"The user is likely answering the question about missing file paths from the context.\n"
                f"If the AI asked for 'image path' and user says 'test.jpg', map it to image_path.\n"
            )

    prompt = (
        "You are extracting YOLOX inference configuration from a user message.\n\n"
        "From the user's message, extract ANY file paths for:\n"
        "- image file (.png/.jpg/.jpeg) -> image_path\n"
        "- model weights (.pth) -> weights_path\n"
        "- class names file (.txt) -> class_names_path\n"
        "- HGD checkpoint (.pt) -> hgd_checkpoint_path\n\n"
        "Extract ONLY the fields mentioned - all fields are optional and can be provided incrementally.\n"
        f"{context_prompt_section}\n"
        f"User message: {message}"
    )

    try:
        structured = llm.with_structured_output(YoloxInferenceParams)
        parsed = await structured.ainvoke(prompt)
        if isinstance(parsed, dict):
            parsed = YoloxInferenceParams.model_validate(parsed)
        
        # Clean up extracted paths
        if isinstance(parsed, YoloxInferenceParams):
            if parsed.image_path:
                parsed.image_path = _strip_wrapping_quotes(parsed.image_path)
            if parsed.weights_path:
                parsed.weights_path = _strip_wrapping_quotes(parsed.weights_path)
            if parsed.class_names_path:
                parsed.class_names_path = _strip_wrapping_quotes(parsed.class_names_path)
            if parsed.hgd_checkpoint_path:
                parsed.hgd_checkpoint_path = _strip_wrapping_quotes(parsed.hgd_checkpoint_path)
            return parsed
    except Exception as e:
        raise RuntimeError(f"Failed to extract YOLOX parameters from message: {str(e)}") from e

    return YoloxInferenceParams()


def _get_model_params_hw(state: NNGeneratorState) -> tuple[int, int]:
    model_params = state.get("model_params")
    if hasattr(model_params, "height") and hasattr(model_params, "width"):
        try:
            return int(model_params.height), int(model_params.width)
        except Exception:
            raise RuntimeError("Invalid model_params height/width values")

    if isinstance(model_params, dict):
        try:
            return int(model_params.get("height", 224)), int(model_params.get("width", 224))
        except Exception:
            raise RuntimeError("Invalid model_params height/width values")

    return 224, 224


async def _handle_yolox_inference(state: NNGeneratorState) -> Dict[str, Any]:
    """Handle YOLOX inference by collecting required parameters incrementally."""
    from src.agentic.src.mcp import get_mcp_helper
    
    project_path = state.get("project_path") or state.get("project_output")
    
    # Get previously stored params from state
    stored_params = state.get("yolox_inference_params") or {}
    if isinstance(stored_params, dict):
        yolox_params = YoloxInferenceParams.model_validate(stored_params)
    else:
        yolox_params = YoloxInferenceParams()
    
    # Extract params from current message
    latest_msg = _get_last_user_message(state)
    new_params = await extract_yolox_params_from_message(latest_msg, state)
    
    # Merge new params with stored (new values override)
    if new_params.image_path:
        yolox_params.image_path = new_params.image_path
    if new_params.weights_path:
        yolox_params.weights_path = new_params.weights_path
    if new_params.class_names_path:
        yolox_params.class_names_path = new_params.class_names_path
    if new_params.hgd_checkpoint_path:
        yolox_params.hgd_checkpoint_path = new_params.hgd_checkpoint_path
    
    # Helper to resolve paths
    def resolve_path(path_str: Optional[str]) -> Optional[Path]:
        if not path_str:
            return None
        p = Path(path_str)
        if not p.is_absolute() and project_path:
            p = Path(project_path) / p
        return p
    
    # Check what's missing
    missing = []
    if not yolox_params.image_path:
        missing.append("**image path** (e.g., test_image.jpg)")
    if not yolox_params.weights_path:
        missing.append("**model weights path** (e.g., best_ckpt.pth or trained model path)")
    if not yolox_params.class_names_path:
        missing.append("**class names file path** (e.g., classes.txt)")
    if not yolox_params.hgd_checkpoint_path:
        missing.append("**HGD checkpoint path** (e.g., best_ckpt.pt)")
    
    # If any missing, prompt for them
    if missing:
        prompt_parts = [
            "📋 **YOLOX Model Testing**\n",
            "Please provide the following file paths (you can provide multiple at once):\n"
        ]
        for item in missing:
            prompt_parts.append(f"- {item}")
        
        # Show what we already have
        if yolox_params.image_path or yolox_params.weights_path or yolox_params.class_names_path or yolox_params.hgd_checkpoint_path:
            prompt_parts.append("\n**Already provided:**")
            if yolox_params.image_path:
                prompt_parts.append(f"- Image: `{yolox_params.image_path}` ✅")
            if yolox_params.weights_path:
                prompt_parts.append(f"- Weights: `{yolox_params.weights_path}` ✅")
            if yolox_params.class_names_path:
                prompt_parts.append(f"- Class names: `{yolox_params.class_names_path}` ✅")
            if yolox_params.hgd_checkpoint_path:
                prompt_parts.append(f"- HGD checkpoint: `{yolox_params.hgd_checkpoint_path}` ✅")
        
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(
                content="\n".join(prompt_parts),
                name="test_image"
            )],
            "awaiting_test_image": True,
            "awaiting_new_design_choice": False,
            "needs_user_input": True,
            "yolox_inference_params": yolox_params.model_dump(),
        }
    
    # All params collected - validate paths exist
    image_path = resolve_path(yolox_params.image_path)
    weights_path = resolve_path(yolox_params.weights_path)
    class_names_path = resolve_path(yolox_params.class_names_path)
    hgd_checkpoint_path = resolve_path(yolox_params.hgd_checkpoint_path)
    
    # Validate existence
    validation_errors = []
    if not image_path or not image_path.exists():
        validation_errors.append(f"Image file not found: `{yolox_params.image_path}`")
    if not weights_path or not weights_path.exists():
        validation_errors.append(f"Model weights file not found: `{yolox_params.weights_path}`")
    if not class_names_path or not class_names_path.exists():
        validation_errors.append(f"Class names file not found: `{yolox_params.class_names_path}`")
    if not hgd_checkpoint_path or not hgd_checkpoint_path.exists():
        validation_errors.append(f"HGD checkpoint file not found: `{yolox_params.hgd_checkpoint_path}`")
    
    if validation_errors:
        error_msg = "❌ **File Validation Errors:**\n\n" + "\n".join(f"- {err}" for err in validation_errors)
        error_msg += "\n\nPlease provide valid file paths."
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(
                content=error_msg,
                name="test_image"
            )],
            "awaiting_test_image": True,
            "awaiting_new_design_choice": False,
            "needs_user_input": True,
            "yolox_inference_params": yolox_params.model_dump(),
        }
    
    # All params valid - call test_image MCP endpoint
    try:
        mcp = await get_mcp_helper()
        result = await mcp.call(
            "test_image",
            {
                "image_path": str(image_path),
                "model_path": str(weights_path),
                "hgd_ckpt_path": str(hgd_checkpoint_path),
                "class_names_path": str(class_names_path),
            },
        )
        
        status = result.get("status") if isinstance(result, dict) else None
        details = result.get("details", {}) if isinstance(result, dict) else {}
        
        if status == "success":
            # For YOLOX, details might have detected objects
            detected_objects = details.get("detected_objects", [])
            if detected_objects:
                response_parts = [
                    "✅ **YOLOX Inference Complete!**\n",
                    f"**Detected {len(detected_objects)} object(s):**\n"
                ]
                for obj in detected_objects[:10]:  # Limit to 10
                    cls = obj.get("class", "unknown")
                    conf = obj.get("confidence", 0.0)
                    response_parts.append(f"- {cls} ({conf:.2%})")
                if len(detected_objects) > 10:
                    response_parts.append(f"- ... and {len(detected_objects) - 10} more")
            else:
                response_parts = [
                    "✅ **YOLOX Inference Complete!**\n",
                    "No objects detected in the image."
                ]
            
            response_parts.append("\nWould you like to:")
            response_parts.append("- **Test another image** (provide new paths)")
            response_parts.append("- **Create a new neural network** (say 'new design')")
            response_parts.append("- **Finish** (say 'done')")
            
            response = "\n".join(response_parts)
        else:
            msg = result.get("message", "Inference failed") if isinstance(result, dict) else "Inference failed"
            response = f"❌ **YOLOX Inference Failed**\n\n{msg}\n\nProvide new paths or say 'done'."
        
        # Persist weights/class_names/HGD paths for next test, only clear image_path
        persisted_params = {
            "weights_path": yolox_params.weights_path,
            "class_names_path": yolox_params.class_names_path,
            "hgd_checkpoint_path": yolox_params.hgd_checkpoint_path,
            "image_path": None,  # Clear image path for next test
        }
        
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(
                content=response,
                name="test_image"
            )],
            "awaiting_test_image": False,
            "awaiting_new_design_choice": True,
            "needs_user_input": True,
            "yolox_inference_params": persisted_params,  # Persist non-image params
        }
    
    except Exception as e:
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(
                content=(
                    f"❌ Error during YOLOX inference: {str(e)}\n\n"
                    "Would you like to:\n"
                    "- **Try again** (provide paths)\n"
                    "- **Finish** (say 'done')"
                ),
                name="test_image"
            )],
            "awaiting_test_image": False,
            "awaiting_new_design_choice": True,
            "needs_user_input": True,
            "yolox_inference_params": yolox_params.model_dump(),  # Keep all params for retry
        }





async def test_image_node(state: NNGeneratorState) -> Dict[str, Any]:
    """Run inference on an uploaded image for the trained model."""

    from src.agentic.src.mcp import get_mcp_helper

    manual_layers = state.get("manual_layers") or []
    pretrained_model = state.get("pretrained_model")
    
    # Determine model type with explicit precedence
    is_manual = len(manual_layers) > 0
    is_yolox = pretrained_model and str(pretrained_model).lower().startswith("yolox")
    is_pretrained = pretrained_model and not is_yolox
    
    # Handle ambiguous case
    if is_manual and (is_yolox or is_pretrained):
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(
                content="❌ Model type is ambiguous: both manual layers and pretrained model are set.\n\nPlease contact support.",
                name="test_image"
            )],
            "awaiting_test_image": False,
            "awaiting_new_design_choice": True,
            "needs_user_input": True,
        }
    
    # Special handling for YOLOX: collect all 4 required paths incrementally
    if is_yolox:
        return await _handle_yolox_inference(state)
    
    # For manual/pretrained models, check if router already extracted image path
    image_path_str = state.get("extracted_image_path")
    
    # If not pre-extracted, try to extract from current message
    if not image_path_str:
        latest_msg = _get_last_user_message(state)
        image_path_str = await extract_image_path_from_message(latest_msg)

    if not image_path_str:
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(
                content="Please upload an image (Upload button) or paste an image file path to test the trained model.",
                name="test_image"
            )],
            "awaiting_test_image": True,
            "awaiting_new_design_choice": False,
            "needs_user_input": True,
        }

    image_path = Path(_strip_wrapping_quotes(image_path_str))
    
    # Resolve relative paths relative to project directory if available
    if not image_path.is_absolute():
        project_path = state.get("project_path") or state.get("project_output")
        if project_path:
            image_path = Path(project_path) / image_path
    
    if not image_path.exists():
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(
                content=f"❌ Image file not found: `{image_path}`\n\nPlease upload a valid image or paste a valid image path.",
                name="test_image"
            )],
            "awaiting_test_image": True,
            "awaiting_new_design_choice": False,
            "needs_user_input": True,
        }

    trained_model_path = state.get("trained_model_path")
    if not trained_model_path:
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(
                content="❌ No trained model found in state. Please train a model first.",
                name="test_image"
            )],
            "awaiting_test_image": False,
            "awaiting_new_design_choice": True,
            "needs_user_input": True,
        }
    
    # Validate model path exists
    model_path_obj = Path(trained_model_path)
    if not model_path_obj.exists():
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(
                content=f"❌ Trained model file not found: `{trained_model_path}`\n\nPlease retrain the model.",
                name="test_image"
            )],
            "awaiting_test_image": False,
            "awaiting_new_design_choice": True,
            "needs_user_input": True,
        }

    try:
        mcp = await get_mcp_helper()

        if is_manual:
            height, width = _get_model_params_hw(state)
            result = await mcp.call(
                "test_manual_image",
                {
                    "image_path": str(image_path),
                    "model_path": str(trained_model_path),
                    "input_height": height,
                    "input_width": width,
                },
            )

        elif is_pretrained:
            height, width = _get_model_params_hw(state)
            result = await mcp.call(
                "test_pretrained_image",
                {
                    "image_path": str(image_path),
                    "model_path": str(trained_model_path),
                    "input_height": height,
                    "input_width": width,
                },
            )
        
        else:
            # No valid model type detected
            from langchain_core.messages import AIMessage
            return {
                "messages": [AIMessage(
                    content="❌ Cannot determine model type. No manual layers or pretrained model configuration found.\n\nPlease retrain with a valid architecture.",
                    name="test_image"
                )],
                "awaiting_test_image": False,
                "awaiting_new_design_choice": True,
                "needs_user_input": True,
            }

        status = result.get("status") if isinstance(result, dict) else None
        details = result.get("details", {}) if isinstance(result, dict) else {}

        if status == "success":
            predicted_class = details.get("predicted_class", "unknown")
            confidence = details.get("confidence", 0.0)
            response = (
                "✅ **Inference Complete!**\n\n"
                f"- **Predicted Class:** {predicted_class}\n"
                f"- **Confidence:** {confidence:.4f}\n"
                f"- **Image:** `{image_path}`\n\n"
                "Would you like to:\n"
                "- **Test another image** (upload or paste a path)\n"
                "- **Create a new neural network** (say 'new design')\n"
                "- **Finish** (say 'done')"
            )
        else:
            msg = result.get("message", "Inference failed") if isinstance(result, dict) else "Inference failed"
            response = f"❌ **Inference Failed**\n\n{msg}\n\nUpload another image or say 'done'."

        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(
                content=response,
                name="test_image"
            )],
            "awaiting_test_image": False,
            "awaiting_new_design_choice": True,
            "needs_user_input": True,
            "extracted_image_path": None,  # Clear extracted path after use
        }

    except Exception as e:
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(
                content=(
                    f"❌ Error during inference: {str(e)}\n\n"
                    "Would you like to:\n"
                    "- **Try another image** (upload or paste a path)\n"
                    "- **Finish** (say 'done' or 'finish')"
                ),
                name="test_image"
            )],
            "awaiting_test_image": False,
            "awaiting_new_design_choice": True,
            "needs_user_input": True,
            "extracted_image_path": None,  # Clear extracted path after use
        }
