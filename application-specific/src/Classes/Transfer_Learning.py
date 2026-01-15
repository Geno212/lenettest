import json
import os
import sys

# import paths.SystemPaths as paths
from PySide6.QtCore import QProcess
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QFileDialog

basedir = os.path.dirname(__file__)
loader = QUiLoader()


class DataOfTransfer:
    def __init__(self):
        self.selected_pretrained_model = (
            self.Children.qt_pretrained_model_combobox.currentText()
        )

    def on_combobox_change(self):
        self.selected_pretrained_model = (
            self.Children.qt_pretrained_model_combobox.currentText()
        )

    def save_json_transfer(self):
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save JSON file",
            self.SysPath.jsondir,
            "JSON Files (*.json)",
        )

        self.architecture["transfer_model"] = self.selected_pretrained_model
        Height, Width = self.get_min_size(self.selected_pretrained_model)
        self.architecture["misc_params"]["height"] = max(
            self.architecture["misc_params"]["height"],
            Height,
        )
        self.architecture["misc_params"]["width"] = max(
            self.architecture["misc_params"]["width"],
            Width,
        )
        self.architecture["log_dir"] = self.SysPath.log_path
        try:
            self.architecture["misc_params"]["device_index"] = int(
                self.architecture["misc_params"]["device"].split(":")[1],
            )
        except:
            print("cpu")
        if path:
            self.SysPath.jsondir = path
            with open(path, "w") as f:
                f.write(json.dumps(self.architecture, indent=4))
            print("JSON file saved successfully.")

    # Render Json File Data

    def render_transfer_learning(self):
        self.selected_pretrained_model = (
            self.Children.qt_pretrained_model_combobox.currentText()
        )
        print(self.selected_pretrained_model)
        # Update paths based on the selected model
        self.SysPath.update_paths(self.selected_pretrained_model)

        path_output = self.Cookiecutter.render_cookiecutter_template(
            self.SysPath.transfer_jinja_json,
            self.SysPath.transfer_cookie_json,
            self.SysPath.transfer_template_dir,
        )
        # if self.selected_pretrained_model == "YOLOX-Small":
        #     self.SysPath.update_paths("yolox")
        if path_output:
            try:
                self.show_files(path_output)
            except:
                if self.debug:
                    print("ERRORRRRR")
            self.Pretrained_Process = QProcess()
            self.Pretrained_Process.readyReadStandardOutput.connect(
                self.handle_stdout_transfer_learning,
            )
            self.Pretrained_Process.readyReadStandardError.connect(
                self.handle_stderr_transfer_learning,
            )
            # Use the virtual environment Python executable
            python_executable = sys.executable
            self.Pretrained_Process.start(
                python_executable,
                [path_output + "/Pretrained_Output/main.py"],
            )

    def handle_stderr_transfer_learning(self):
        result = bytes(self.Pretrained_Process.readAllStandardError()).decode("utf8")
        print(result)

    def handle_stdout_transfer_learning(self):
        result = bytes(self.Pretrained_Process.readAllStandardOutput()).decode("utf8")
        print(result)
