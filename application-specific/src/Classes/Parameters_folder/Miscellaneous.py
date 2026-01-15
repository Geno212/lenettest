from PySide6.QtWidgets import QComboBox, QLineEdit, QSpinBox

from src.Classes.Children import Children
from src.utils.Singleton import Singleton


class Miscellaneous(metaclass=Singleton):
    def __init__(self) -> None:
        self.Children = Children()
        self.miscellaneous = dict()
        self.tracked_data = []
        self.miscellaneous_params = {
            "device": self.Children.qt_selectedDevice_QComboBox,
            "width": self.Children.qt_inputWidth_QSpinBox,
            "height": self.Children.qt_inputHeight_QSpinBox,
            "channels": self.Children.qt_inputType_QSpinBox,
            "num_epochs": self.Children.qt_numEpochs_QSpinBox,
            "batch_size": self.Children.qt_batchSize_QSpinBox,
            "dataset": self.Children.qt_selectedDataset_QComboBox,
            "dataset_path": self.Children.qt_dataset_path_QLineEdit,
        }
        for param_name, widget in self.miscellaneous_params.items():
            self.fetch_data_params(param_name, widget)
            self.set_data_onChange(param_name, widget)
            self.tracked_data.append(param_name)

    def __setitem__(self, attribute, value):
        self.miscellaneous[attribute] = value
        print("ahmed")

    def load_from_config(self, json_data: dict):
        if len(json_data["misc_params"]) != 0:
            for key, value in json_data["misc_params"].items():
                self.set_data(key, value)

    def set_data_onChange(self, param_name, widget):
        if type(widget) == QSpinBox:
            widget.valueChanged.connect(
                lambda: self.fetch_data_params(param_name, widget),
            )
        elif type(widget) == QLineEdit:
            widget.textChanged.connect(
                lambda: self.fetch_data_params(param_name, widget),
            )

        elif type(widget) == QComboBox:
            widget.currentIndexChanged.connect(
                lambda: self.fetch_data_params(param_name, widget),
            )

    def fetch_data_params(self, param_name, widget):
        try:
            if type(widget) == QSpinBox:
                self.miscellaneous[param_name] = widget.value()
                if self.debug:
                    print(widget.value(), param_name)
            elif type(widget) == QLineEdit:
                self.miscellaneous[param_name] = widget.text()
                if self.debug:
                    print(widget.text(), param_name)
            elif type(widget) == QComboBox:
                data = widget.currentData()
                index = widget.currentIndex()
                self.miscellaneous[param_name] = {
                    "value": data,
                    "index": index,
                }

                if self.debug:
                    print(data, param_name)
        except:
            pass

    def set_data(self, key, value):
        print(key, value)
        try:
            widget = self.miscellaneous_params[key]
            if type(widget) == QSpinBox:
                widget.setValue(value)
            elif type(widget) == QLineEdit:
                widget.setText(value)
            elif type(widget) == QComboBox:
                widget.setCurrentIndex(value["index"])
        except:
            print("Error")

    def set_params_from_dict(self, params_dict):
        """Set miscellaneous parameters from a dictionary (CLI-friendly).

        Args:
            params_dict (dict): Dictionary containing parameter values

        """
        for key, value in params_dict.items():
            if key in ["device", "dataset"]:
                # For combo box values, store as dict with value and index
                if isinstance(value, dict):
                    self.miscellaneous[key] = value
                else:
                    # If it's a simple value, create the dict structure
                    self.miscellaneous[key] = {"value": value, "index": 0}
            else:
                # For simple values (int, str)
                self.miscellaneous[key] = value
