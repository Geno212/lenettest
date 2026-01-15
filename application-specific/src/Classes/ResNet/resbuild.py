import json

from PySide6.QtWidgets import QFileDialog


class ResBuildWindow:
    def __init__(self) -> None:
        pass

    def save_residual_json(self):
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Residual Block JSON File",
            self.SysPath.basedir,
            "JSON Files (*.json)",
        )
        self.Resarchitecture["layers"] = {"list": self.Resarchitecture["layers"]}
        with open(path, "w") as f:
            f.write(json.dumps(self.Resarchitecture, indent=4))
        if self.debug:
            print("Residual Block JSON file saved successfully.")

        return path

    def res_on_submit_residual_block_clicked(self):
        self.add_layers_names(self.Resarchitecture)
        with open(self.SysPath.ResJson, "w") as f:
            # f.write(json.dumps(self.Resarchitecture, indent=2))
            self.save_residual_json()
        self.ResCreation.close()

    def add_layers_names(self, architecture):
        layer_freqs = dict()
        for i in range(len(architecture["layers"])):
            layer = architecture["layers"][i]

            if layer["type"] in layer_freqs:
                layer_freqs[layer["type"]] += 1
            else:
                layer_freqs[layer["type"]] = 1
            layer["name"] = f"{layer['type'].lower()}_{layer_freqs[layer['type']]}"
