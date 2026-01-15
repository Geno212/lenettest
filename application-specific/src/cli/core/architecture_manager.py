import json
import os
import sys
from pathlib import Path

from src.cli.utils.console import console

current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent.parent

# src directory to Python path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from Classes.ModelLoader import ModelLoaderFactory
    from Tests.Validation import Validation
    from utils.AutoExtraction import AutoExtraction
    from utils.Singleton import Singleton
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Trying to import from: {src_dir}")
    raise


class ArchitectureManager:
    """Manages neural network architectures."""

    def __init__(self):
        self.auto_extractor = AutoExtraction(debug=False)
        (
            self.layers,
            self.loss_funcs,
            self.optimizers,
            self.schedulers,
            self.pretrained_models,
            self.layers_without_res,
            self.datasets,
            self.complex_arch_models,
        ) = self.auto_extractor.extracted_data()
        self.complex_schedulers = [
            "cos",
            "warmcos",
            "yoloxwarmcos",
            "yoloxsemiwarmcos",
            "multistep",
        ]
        self.basedir = os.path.dirname(__file__)
        self.datadir = os.path.normpath(os.path.join(self.basedir, "./../../../data"))
        self.log_path = os.path.normpath(
            os.path.join(self.datadir, "./tensorboardlogs"),
        )
        self.current_architecture = {
            "layers": [],
            "misc_params": {
                "device": {
                    "value": "cpu",
                    "index": 0,
                },
                "width": 224,
                "height": 224,
                "channels": 3,
                "num_epochs": 10,
                "batch_size": 32,
                "dataset": {
                    "value": "CustomDataset",
                    "index": 0,
                },
                "dataset_path": "",
            },
            "optimizer": {},
            "loss_func": {},
            "scheduler": {},
            "pretrained": {"value": None, "index": -1},
            "log_dir": self.log_path
        }

    def add_layer(self, layer_type: str, params: dict, position: int | None = None):
        """Add a layer to the current architecture.

        Args:
            layer_type (str): Type of layer to add
            params (dict): Layer parameters
            position (Optional[int]): Position to insert layer (0-based). If None, appends to end.

        """
        layer = {
            "type": layer_type,
            "params": params,
            "name": f"{layer_type.lower()}_{len(self.current_architecture['layers']) + 1}",
        }

        if position is None:
            self.current_architecture["layers"].append(layer)
        else:
            # Insert at specified position (clamped to valid range)
            layers = self.current_architecture["layers"]
            if position < 0:
                position = 0
            elif position > len(layers):
                position = len(layers)

            layers.insert(position, layer)

        return layer

    def remove_layer(self, index: int):
        """Remove a layer from the current architecture."""
        if 0 <= index < len(self.current_architecture["layers"]):
            return self.current_architecture["layers"].pop(index)
        return None

    def move_layer(self, from_index: int, to_index: int):
        """Move a layer from one position to another."""
        layers = self.current_architecture["layers"]
        if 0 <= from_index < len(layers) and 0 <= to_index < len(layers):
            layer = layers.pop(from_index)
            layers.insert(to_index, layer)
            return True
        return False

    def load_architecture(self, file_path: Path):
        """Load an architecture from a file."""
        with open(file_path) as f:
            self.current_architecture = json.load(f)

    def save_architecture(self, file_path: Path):
        """Save the current architecture to a file."""
        with open(file_path, "w") as f:
            json.dump(self.current_architecture, f, indent=4)

    def list_available_pretrained_models(self):
        return self.pretrained_models

    def list_available_layers(self):
        """List all available layer types."""
        return list(self.layers.keys())

    def get_layer_info(self, layer_type: str):
        """Get information about a specific layer type."""
        if layer_type in self.layers:
            return self.layers[layer_type]
        return None

    def _ensure_complex_fields(self):
        """Ensure complex architecture fields exist in current architecture."""
        if "complex_misc_params" not in self.current_architecture:
            self.current_architecture["complex_misc_params"] = {
                "data_num_workers": 0,
                "eval_interval": 1,
                "warmup_epochs": 0,
                "scheduler": "cos",
                "num_classes": 0,
            }

        if "pretrained_weights" not in self.current_architecture:
            self.current_architecture["pretrained_weights"] = ""

        if "log_dir" not in self.current_architecture:
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "data" / "tensorboardlogs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.current_architecture["log_dir"] = str(log_dir)

    def get_model_dimensions(self, model_name):
        yolox_variants = {
            "yolox-nano": {"depth": 0.33, "width": 0.25},
            "yolox-tiny": {"depth": 0.33, "width": 0.375},
            "yolox-small": {"depth": 0.33, "width": 0.50},
            "yolox-medium": {"depth": 0.67, "width": 0.75},
            "yolox-large": {"depth": 1.00, "width": 1.00},
            "yolox-x-large": {"depth": 1.33, "width": 1.25},
        }
        if model_name.lower() in yolox_variants:
            return yolox_variants[model_name.lower()]["depth"], yolox_variants[
                model_name.lower()
            ]["width"]
        return 0, 0

    def create_complex_architecture(self, model_name: str):
        """Create a complex architecture with pretrained model."""
        self._ensure_complex_fields()
        depth, width = self.get_model_dimensions(model_name)

        # Find model index
        model_index = -1
        if model_name.lower() in [m.lower() for m in self.complex_arch_models]:
            model_index = next(
                i
                for i, m in enumerate(self.complex_arch_models)
                if m.lower() == model_name.lower()
            )

        self.current_architecture["pretrained"] = {
            "value": model_name,
            "index": model_index,
            "depth": depth,
            "width": width,
        }

        return self.current_architecture

    def save_complex_architecture(self, file_path: Path):
        """Save complex architecture with validation and model loading."""
        self._ensure_complex_fields()

        # Validate and prepare architecture
        try:
            current_arch = self.current_architecture.copy()
            validated_arch = self.validate_architecture(current_arch)
            prepared_arch = self.prepare_architecture(validated_arch)
            self.current_architecture = prepared_arch
        except Exception as e:
            console.print(
                f"[yellow]⚠️  Validation failed: {e}. Saving without validation.[/yellow]",
            )

        # Save to file
        with open(file_path, "w") as f:
            json.dump(self.current_architecture, f, indent=2)

    def validate_architecture(self, architecture):
        """Validate architecture layers and apply corrections.

        Args:
            architecture (dict): The architecture dictionary to validate

        Returns:
            dict: The validated architecture

        """
        try:
            layer_validator = Validation()
            # Get current layers for validation
            layers = architecture.get("layers", [])

            # Perform validation and correction
            temp_dict = layer_validator.validate_and_correct_layers(
                layers,
                architecture["misc_params"]["width"],
                architecture["misc_params"]["height"],
                architecture["misc_params"]["channels"],
            )
            if len(layers) != len(temp_dict):
                architecture["layers"] = {
                    "list": layers,
                }
                layer_validator.layer_naming(architecture["layers"]["list"])
            else:
                layer_validator.layer_naming(architecture["layers"])

            architecture["layers"] = {"list": layers}
            architecture["log_dir"] = self.log_path

        except Exception as e:
            console.print(
                f"[yellow]⚠️  Layer validation failed: {e}. Continuing without validation.[/yellow]",
            )

        return architecture

    def prepare_architecture(self, architecture):
        """Prepare architecture for saving with pretrained model adjustments.

        Args:
            architecture (dict): The validated architecture dictionary

        Returns:
            dict: The prepared architecture

        """
        # Check if pretrained model is set and adjust dimensions if needed
        pretrained_value = architecture.get("pretrained", {}).get("value")
        if pretrained_value:
            try:
                # Get minimum size for the pretrained model
                height, width = self._get_min_size(pretrained_value)
                architecture["misc_params"]["height"] = max(
                    architecture["misc_params"]["height"],
                    height,
                )
                architecture["misc_params"]["width"] = max(
                    architecture["misc_params"]["width"],
                    width,
                )

                # Load model to extract channels
                try:
                    model_loader = ModelLoaderFactory.get_model_loader(architecture)
                    model_loader.load_model()
                    console.print(
                        f"[green]✅ Model loaded successfully for {pretrained_value}[/green]",
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]⚠️  Failed to load model for channel extraction: {e}[/yellow]",
                    )

            except Exception as e:
                console.print(
                    f"[yellow]⚠️  Failed to adjust dimensions for pretrained model {pretrained_value}: {e}[/yellow]",
                )

        return architecture

    def _get_min_size(self, model_name):
        """Get minimum input size for pretrained model.

        Args:
            model_name (str): Name of the pretrained model

        Returns:
            tuple: (height, width) minimum input size

        """
        # Define YOLOX input sizes
        yolox_input_sizes = {
            "yolox-tiny": (416, 416),
            "yolox-small": (640, 640),
            "yolox-medium": (640, 640),
            "yolox-large": (640, 640),
            "yolox-x-large": (640, 640),
            "yolox-nano": (416, 416),
        }

        # Check if the model is a YOLOX model
        if model_name.lower() in yolox_input_sizes:
            return yolox_input_sizes[model_name.lower()]

        # Handle torchvision models
        try:
            import torchvision.models

            weights = torchvision.models.get_model_weights(
                torchvision.models.__dict__[model_name],
            )
            default_weights = getattr(weights, 'DEFAULT', None)
            if default_weights and hasattr(default_weights, 'meta'):
                min_size = default_weights.meta.get("min_size", (224, 224))
            else:
                min_size = (224, 224)
            return min_size
        except (KeyError, AttributeError):
            # Default fallback
            return (224, 224)


# Global architecture manager
arch_manager = ArchitectureManager()
