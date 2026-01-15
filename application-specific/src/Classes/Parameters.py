import copy
import json

import torchvision
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QFileDialog

from src.Classes.Children import Children
from src.Classes.ModelLoader import ModelLoaderFactory
from src.Classes.Parameters_folder.Layers_System.Layers_System import Layers_System
from src.Classes.Parameters_folder.LossFunction import LossFunction
from src.Classes.Parameters_folder.Miscellaneous import Miscellaneous
from src.Classes.Parameters_folder.Optimizer import Optimizer
from src.Classes.Parameters_folder.Pretrained import Pretrained
from src.Classes.Parameters_folder.PretrainedComplex import PretrainedComplex
from src.Classes.Parameters_folder.Scheduler import Scheduler
from src.paths.SystemPaths import SystemPaths
from src.utils.Cookiecutter import Cookiecutter


class Parameters:
    def __init__(self) -> None:
        self.Cookiecutter = Cookiecutter()

        self.Children = Children()
        self.SysPath = SystemPaths()
        # optimizer
        self.Misc_params = Miscellaneous()

        self.Optim_params = Optimizer()
        # loss_func
        self.LossFunc_params = LossFunction()
        # scheduler
        self.Scheduler_params = Scheduler()
        # layers
        self.Layers_System = Layers_System()
        # pretrained
        self.Pretrained = Pretrained()
        # pretrained complex
        self.PretrainedComplex = PretrainedComplex()
        self.layers = []
        self.connections()

    def connections(self):
        self.Children.qt_actionLoad.triggered.connect(self.load_configs)
        self.Children.qt_Create_transfer_Model_QPushButton.clicked.connect(
            lambda submit_func=self.save_json_transfer,
            Template="Transfer Model": submit_func(Template),
        )

        self.Children.qt_Create_complex_Model_QPushButton.clicked.connect(
            lambda submit_func=self.save_json_transfer_complex,
            Template="Transfer Model": submit_func(Template),
        )

    def create_architecture(self):
        self.Layers_System.Validate_func()
        self.architecture = {
            "layers": self.Layers_System.layers,
            "misc_params": self.Misc_params.miscellaneous,
            "optimizer": self.Optim_params.optimizer,
            "loss_func": self.LossFunc_params.loss_function,
            "scheduler": self.Scheduler_params.scheduler,
            "pretrained": self.Pretrained.pretrained,
        }
        return self.architecture

    def create_architecture_complex(self):
        self.Layers_System.Validate_func()
        self.architecture = {
            "layers": self.Layers_System.layers,
            "misc_params": self.Misc_params.miscellaneous,
            "complex_misc_params": self.PretrainedComplex.cmplx_miscellaneous,
            "optimizer": self.Optim_params.optimizer,
            "loss_func": self.LossFunc_params.loss_function,
            "scheduler": self.Scheduler_params.scheduler,
            "pretrained": self.PretrainedComplex.pretrained,
            "pretrained_weights": self.PretrainedComplex.Children.weightsPathLineEdit.text(),
        }
        return self.architecture

    def load_configs(self):
        path_arch_json, _ = QFileDialog.getOpenFileName(
            None,
            "Load configuration file",
            self.SysPath.jsondir,
            "JSON Files (*.json)",
        )
        if path_arch_json:
            with open(path_arch_json) as json_file:
                temp = json.load(json_file)
                # misc
                self.Misc_params.load_from_config(temp)
                # optimizer
                self.Optim_params.load_from_config(temp)
                # loss_func
                self.LossFunc_params.load_from_config(temp)
                # scheduler
                self.Scheduler_params.load_from_config(temp)
                # layers
                self.Layers_System.load_from_config(temp)

    def save_json_transfer(self, Template=None):
        temp_arch = self.create_architecture()
        architecture = copy.deepcopy(temp_arch)
        print(architecture["misc_params"])
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save JSON file",
            self.SysPath.jsondir,
            "JSON Files (*.json)",
        )

        if Template == "Transfer Model":
            Height, Width = self.get_min_size(architecture["pretrained"]["value"])
            architecture["misc_params"]["height"] = max(
                architecture["misc_params"]["height"],
                Height,
            )
            architecture["misc_params"]["width"] = max(
                architecture["misc_params"]["width"],
                Width,
            )

            architecture["log_dir"] = self.SysPath.log_path

            # Use the Factory DP to get the correct model loader
            model_loader = ModelLoaderFactory.get_model_loader(architecture)
            model_loader.load_model()

        print(architecture["misc_params"])
        if path:
            self.SysPath.jsondir = path
            with open(path, "w") as f:
                f.write(json.dumps(architecture, indent=4))
            print("JSON file saved successfully.")

    def save_json_transfer_complex(self, Template=None):
        temp_arch = self.create_architecture_complex()
        architecture = copy.deepcopy(temp_arch)
        print(architecture["misc_params"])
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save JSON file",
            self.SysPath.jsondir,
            "JSON Files (*.json)",
        )

        if Template == "Transfer Model":
            Height, Width = self.get_min_size(architecture["pretrained"]["value"])
            architecture["misc_params"]["height"] = max(
                architecture["misc_params"]["height"],
                Height,
            )
            architecture["misc_params"]["width"] = max(
                architecture["misc_params"]["width"],
                Width,
            )

            architecture["log_dir"] = self.SysPath.log_path

            # Use the Factory DP to get the correct model loader
            # model_loader = ModelLoaderFactory.get_model_loader(architecture)
            # model_loader.load_model()

        print(architecture["misc_params"])
        if path:
            self.SysPath.jsondir = path
            with open(path, "w") as f:
                f.write(json.dumps(architecture, indent=4))
            print("JSON file saved successfully.")
            print(self.Children.qt_complex_model_ComboBox.currentText())
        self.SysPath.update_paths(
            model_name=self.Children.qt_complex_model_ComboBox.currentText(),
        )
        print("el path " + self.SysPath.transfer_cookie_json)
        path_output = self.Cookiecutter.render_cookiecutter_template(
            self.SysPath.transfer_jinja_json,
            self.SysPath.transfer_cookie_json,
            self.SysPath.transfer_template_dir,
        )
        print(path_output)
        if path_output is not None:
            try:
                self.show_files(path_output)
            except Exception as e:
                print("The exception is: " + str(e))
                print("ERRORRRRR")

            self.Pretrained_Process = QProcess()
            # self.Pretrained_Process.readyReadStandardOutput.connect(
            #     self.handle_stdout_transfer_learning)
            # self.Pretrained_Process.readyReadStandardError.connect(
            #     self.handle_stderr_transfer_learning)
            self.Pretrained_Process.start("python", [path_output + "/YOLOX/main.py"])

    def handle_stderr_transfer_learning(self):
        result = bytes(self.Pretrained_Process.readAllStandardError()).decode("utf8")
        print(result)

    def handle_stdout_transfer_learning(self):
        result = bytes(self.Pretrained_Process.readAllStandardOutput()).decode("utf8")
        print(result)

    def get_min_size(self, model_name):
        """Retrieves the minimum input size (height and width) for the specified model.

        Parameters
        ----------
            model_name (str): The name of the model (YOLOX or torchvision).

        Returns
        -------
            tuple: A tuple of (height, width).

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
            min_size = torchvision.models.get_model_weights(
                torchvision.models.__dict__[model_name],
            ).DEFAULT.meta["min_size"]
            return min_size
        except KeyError:
            raise ValueError(f"Unknown model: {model_name}")
