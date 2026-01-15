import importlib
import os

from src.utils.Singleton import Singleton


class Validation(metaclass=Singleton):
    def __init__(self) -> None:
        self.rule_map = dict()
        self.load()

    def load(self):
        rule_dir = os.path.join(os.path.dirname(__file__), "Validation_Rules")
        module_files = [
            f for f in os.listdir(rule_dir) if f.endswith(".py") and f != "__init__.py"
        ]
        for module_file in module_files:
            module_name = f"src.Tests.Validation_Rules.{module_file[:-3]}"
            module = importlib.import_module(module_name)
            class_name = module_file[:-3]
            cls = getattr(module, class_name, None)
            class_instance = cls()
            layer_name, callback_function = class_instance.definition()
            self.rule_map[layer_name] = callback_function

    def validate_layer(
        self,
        rule_func,
        layers,
        index,
        width,
        height,
        channels,
        features_after_1st_FC,
        flattened,
    ):
        return_val = rule_func(
            layers,
            index,
            width,
            height,
            channels,
            features_after_1st_FC,
            flattened,
        )

        return return_val

    def validate_and_correct_layers(self, layers, width, height, channels):
        features_after_1st_FC = None
        flattened = False
        index = 0
        while index < len(layers):
            self.layer_naming(layers)
            layer = layers[index]
            params = layer["params"]
            layer_type = layer["type"]
            if layer_type in self.rule_map:
                (
                    layers,
                    index,
                    width,
                    height,
                    channels,
                    features_after_1st_FC,
                    flattened,
                ) = self.validate_layer(
                    self.rule_map[layer_type],
                    layers,
                    index,
                    width,
                    height,
                    channels,
                    features_after_1st_FC,
                    flattened,
                )
            else:
                print(f"Validation rule not found for layer {layer_type}")
            index += 1
        return layers

    def layer_naming(self, layers):
        layer_freqs = dict()
        for layer in layers:
            self.add_layer_name(layer, layer_freqs)

    def add_layer_name(self, layer, layer_freqs):
        if layer["type"] in layer_freqs:
            layer_freqs[layer["type"]] += 1
        else:
            layer_freqs[layer["type"]] = 1
        layer["name"] = f"{layer['type'].lower()}_{layer_freqs[layer['type']]}"
