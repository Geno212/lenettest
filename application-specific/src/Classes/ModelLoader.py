import os
from abc import ABC, abstractmethod

import torch
from torchvision import models
from YOLOX.yolox.exp import get_exp

"""Abstract class for model loaders."""


class ModelLoaderTemplate(ABC):
    def __init__(self, architecture):
        self.architecture = architecture
        self.model = None
        self.weights_path = None

    def load_model(self):
        self.prepare_weights_path()
        self.load_weights()
        self.extract_channels()

    @abstractmethod
    def prepare_weights_path(self):
        """Prepare the path to the model weights."""

    @abstractmethod
    def load_weights(self):
        """Load the weights into the model."""

    @abstractmethod
    def extract_channels(self):
        """Abstract method to extract channels from the model."""


"""YoloX model loader."""


class YOLOXModelLoader(ModelLoaderTemplate):
    def prepare_weights_path(self):
        """Prepare the weights path for YOLOX model."""
        current_file_path = os.path.abspath(__file__)
        weights_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
            "weights",
        )
        file_name = f"{self.architecture['pretrained']['value'].lower()}.pth"
        self.weights_path = os.path.join(weights_dir, file_name)
        self.weights_path = os.path.normpath(self.weights_path)
        print(self.weights_path)

    def load_weights(self):
        """Load YOLOX model weights."""
        model_name = self.architecture["pretrained"]["value"].lower()  # e.g. "yolox_s" or "yolox_x"
        exp = get_exp(None, model_name)
        self.model = exp.get_model()
        self.model.eval()
        checkpoint = torch.load(self.weights_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])

    def extract_channels(self):
        """Extract channels dynamically for YOLOX."""
        first_layer = list(self.model.state_dict().keys())[0]
        self.architecture["misc_params"]["channels"] = self.model.state_dict()[
            first_layer
        ].shape[1]


"""Torchvision model loader."""


class TorchvisionModelLoader(ModelLoaderTemplate):
    def prepare_weights_path(self):
        """Prepare the weights path for Torchvision model."""
        self.weights_path = (
            None  # For torchvision, the weights path is handled by torchvision itself.
        )

    def load_weights(self):
        """Load a Torchvision model with default weights."""
        model_name = self.architecture["pretrained"]["value"]

        # Torchvision lazily exposes many models via __getattr__, so we cannot rely on __dict__.
        try:
            model_builder = getattr(models, model_name)
        except AttributeError:
            # Try common sub-modules where the model might live (e.g. video, detection, segmentation)
            model_builder = None
            for submod_name in [
                "video",
                "detection",
                "segmentation",
                "quantization",
                "optical_flow",
            ]:
                submod = getattr(models, submod_name, None)
                if submod is not None and hasattr(submod, model_name):
                    model_builder = getattr(submod, model_name)
                    break
            if model_builder is None:
                raise ValueError(
                    f"Model '{model_name}' is not available in torchvision",
                )

        # Call the builder with default weights if supported, otherwise fall back to pretrained flag
        try:
            self.model = model_builder(weights="DEFAULT")
        except TypeError:
            # Older torchvision versions use the 'pretrained' boolean flag instead of 'weights'
            self.model = model_builder(pretrained=True)

    def extract_channels(self):
        """Extract channels dynamically for Torchvision models."""
        (name, param) = list(self.model.named_parameters())[0]
        self.architecture["misc_params"]["channels"] = param.shape[1]


"""Model loader factory."""


class ModelLoaderFactory:
    @staticmethod
    def get_model_loader(architecture):
        if architecture["pretrained"]["value"].lower().startswith("yolox"):
            return YOLOXModelLoader(architecture)
        return TorchvisionModelLoader(architecture)
