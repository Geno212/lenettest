import copy
import inspect
import types

import torch.nn.modules as nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, models

from .Singleton import Singleton


class AutoExtraction(metaclass=Singleton):
    def __init__(self, debug) -> None:
        self.debug = debug
        if self.debug:
            print("Auto Extraction")
        self.unnecessary_datasets = [
            "FakeData",
            "wrap_dataset_for_transforms_v2",
        ]
        self.unnecessary_params = [
            "dilation",
            "divisor_override",
            "approximate",
        ]
        self.unnecessary_optimizer_params = ["params"]
        self.unnecessary_loss_params = ["reduce", "size_average", "weight"]
        self.unnecessary_schedulers_params = [
            "optimizer",
            "lr_lambda",
            "milestones",
            "schedulers",
            "verbose",
            "scale_fn",
        ]

        self.extract_torch_layers()
        self.extract_res_block()
        self.extract_torch_lossfunctions()
        self.extract_torch_optimizers()
        self.extract_scheduler_learning()
        self.extract_pretrained_models()
        self.extract_Datasets()
        self.extract_complex_arch_models()

    def extracted_data(self):
        return (
            self.LAYERS,
            self.LOSSFUNC,
            self.OPTIMIZERS,
            self.SCHEDULERS,
            self.PRETRAINED_MODELS,
            self.LAYERS_WITHOUT_RES,
            self.DATASETS,
            self.COMPLEX_ARCH_MODELS,
        )

    def extract_res_block(self):
        res_params = [
            {"name": "in_channels", "defaultvalue": 0, "type": int},
            {"name": "out_channels", "defaultvalue": 0, "type": int},
        ]
        res_block_dict = {"Residual_Block": res_params}
        self.LAYERS.update(res_block_dict)
        self.LAYERS_WITHOUT_RES = copy.deepcopy(self.LAYERS)
        self.LAYERS_WITHOUT_RES.pop("Residual_Block")

    def extract_torch_layers(self) -> dict:
        torch_layers_names = dir(nn)
        torch_layers_dict = dict()

        for layer_name in torch_layers_names:
            obj = getattr(nn, layer_name)
            # print(f"Layer Name: {layer_name}, Object: {obj}")
            if (
                isinstance(obj, type)
                and obj is not nn.Module
                and issubclass(obj, nn.Module)
                and "Loss" not in layer_name
                and "3d" not in layer_name
            ):
                # print(f"Layer Name: {layer_name}, Object: {obj}")
                inspector = inspect.signature(obj).parameters
                params_list = list()

                for i in inspector:
                    if (
                        inspector[i].kind
                        == inspect._ParameterKind.POSITIONAL_OR_KEYWORD
                        and inspector[i].name not in self.unnecessary_params
                    ):
                        params_list.append(
                            {
                                "name": inspector[i].name,
                                "defaultvalue": inspector[i].default,
                                "type": inspector[i].annotation,
                            },
                        )

                # Include layer if it has parameters or is a whitelisted activation
                activation_whitelist = {
                    "Tanh",
                    "Sigmoid",
                    "Softmax",
                    "ReLU",
                    "ReLU6",
                    "LeakyReLU",
                    "GELU",
                    "ELU",
                    "SELU",
                    "PReLU",
                    "Hardtanh",
                }
                if len(params_list) > 0 or obj.__name__ in activation_whitelist:
                    torch_layers_dict[obj.__name__] = params_list

        self.LAYERS = torch_layers_dict
        # for Residual blocks

    def extract_torch_lossfunctions(self):
        torch_layers_names = dir(nn)
        torch_layers_dict = dict()

        for layer_name in torch_layers_names:
            obj = getattr(nn, layer_name)

            if (
                isinstance(obj, type)
                and obj is not nn.Module
                and issubclass(obj, nn.Module)
                and "Loss" in layer_name
            ):
                inspector = inspect.signature(obj).parameters
                params_list = list()

                for i in inspector:
                    if (
                        inspector[i].kind
                        == inspect._ParameterKind.POSITIONAL_OR_KEYWORD
                        and not isinstance(inspector[i].default, types.FunctionType)
                        and inspector[i].name not in self.unnecessary_loss_params
                    ):
                        params_list.append(
                            {
                                "name": inspector[i].name,
                                "defaultvalue": inspector[i].default,
                                "type": inspector[i].annotation,
                            },
                        )

                if len(params_list) > 0:
                    torch_layers_dict[obj.__name__] = params_list

        self.LOSSFUNC = torch_layers_dict

    def extract_torch_optimizers(self):
        torch_optimizers_names = dir(optim)
        torch_optimizers_dict = dict()
        if self.debug:
            print(torch_optimizers_names)
        for optimizer_name in torch_optimizers_names:
            obj = getattr(optim, optimizer_name)

            if isinstance(obj, type):
                inspector = inspect.signature(obj).parameters
                params_list = list()

                for i in inspector:
                    if (
                        inspector[i].kind
                        == inspect._ParameterKind.POSITIONAL_OR_KEYWORD
                        and inspector[i].name not in self.unnecessary_optimizer_params
                        and not isinstance(inspector[i].default, types.FunctionType)
                    ):
                        # print(inspector[i].annotation,
                        #       type(inspector[i].default))
                        params = None
                        if inspector[i].annotation == inspect._empty:
                            params = type(inspector[i].default)
                        else:
                            params = inspector[i].annotation

                        params_list.append(
                            {
                                "name": inspector[i].name,
                                "defaultvalue": inspector[i].default,
                                "type": params,
                            },
                        )

                if len(params_list) > 0:
                    torch_optimizers_dict[obj.__name__] = params_list

        self.OPTIMIZERS = torch_optimizers_dict

    def extract_pretrained_models(self):
        # Extract torchvision models
        self.PRETRAINED_MODELS = models.list_models()

        if self.debug:
            print(f"Extracted Pretrained Models: {self.PRETRAINED_MODELS}")

    def extract_Datasets(self):
        self.DATASETS = sorted(
            [
                dataset
                for dataset in datasets.__all__
                if dataset not in self.unnecessary_datasets
            ],
        )
        self.DATASETS.insert(0, "CustomDataset")

    def extract_scheduler_learning(self):
        scheduler_dict = dict()
        scheduler_names = dir(lr_scheduler)

        # Add a 'None' choice
        scheduler_dict["None"] = []

        for scheduler_name in scheduler_names:
            # Skip scheduler names that don't start with a letter
            if not scheduler_name[0].isalpha():
                continue

            # Skip 'Optimizer' and 'LRScheduler'
            if scheduler_name in ["Optimizer", "LRScheduler"]:
                continue

            obj = getattr(lr_scheduler, scheduler_name)

            if isinstance(obj, type) and not isinstance(obj, types.BuiltinFunctionType):
                try:
                    inspector = inspect.signature(obj).parameters
                except ValueError:
                    continue

                params_list = list()

                for param_name in inspector:
                    if (
                        inspector[param_name].kind
                        == inspect._ParameterKind.POSITIONAL_OR_KEYWORD
                        and inspector[param_name].name
                        not in self.unnecessary_schedulers_params
                        and not isinstance(
                            inspector[param_name].default,
                            types.FunctionType,
                        )
                    ):
                        param_type = None
                        if inspector[param_name].annotation == inspect._empty:
                            param_type = type(inspector[param_name].default)
                        else:
                            param_type = inspector[param_name].annotation

                        # Capture the original parameter name
                        original_param_name = param_name

                        # Change the parameter name if the scheduler is CyclicLR
                        if scheduler_name == "CyclicLR" and param_name == "mode":
                            param_name = "mode_CyclicLR"

                        params_list.append(
                            {
                                "name": param_name,
                                "defaultvalue": inspector[original_param_name].default,
                                "type": param_type,
                            },
                        )

                if len(params_list) > 0:
                    scheduler_dict[obj.__name__] = params_list

        self.SCHEDULERS = scheduler_dict
        if self.debug:
            print(scheduler_dict)

    def extract_complex_arch_models(self):
        self.COMPLEX_ARCH_MODELS = {
            "YOLOX-Nano",
            "YOLOX-Tiny",
            "YOLOX-Small",
            "YOLOX-Medium",
            "YOLOX-Large",
            "YOLOX-X-Large",
        }
        if self.debug:
            print(f"Extracted Complex Arch Models: {self.COMPLEX_ARCH_MODELS}")
