from PySide6.QtWidgets import QComboBox, QSpinBox

from src.Classes.Children import Children
from src.Classes.Parameters_folder.Miscellaneous import Miscellaneous
from src.Qt.Dialogue import LayerDialog
from src.Qt.WidgetData import WidgetData
from src.utils.AutoExtraction import AutoExtraction


class PretrainedComplex:
    def __init__(self) -> None:
        self.Children = Children()
        self.Miscellaneous = Miscellaneous()
        self.LayerDialog = LayerDialog()
        self.PRETRAINED_MODELS = AutoExtraction().PRETRAINED_MODELS
        self.COMPLEX_ARCH_MODELS = AutoExtraction().COMPLEX_ARCH_MODELS
        self.WidgetUtility = WidgetData()
        # self.fill_pretrained_model() -------------->RAFIK
        # self.fill_complex_arch_model()
        self.Children.qt_complex_model_ComboBox.currentIndexChanged.connect(
            self.on_complex_arch_combobox_change,
        )
        # self.Children.qt_numWorkers.valueChanged.connect(
        #     self.on_complex_misc_changed
        # )
        # self.Children.qt_evalInterval.valueChanged.connect(
        #     self.on_complex_misc_changed
        # )
        # self.Children.qt_warmupepochs.valueChanged.connect(
        #     self.on_complex_misc_changed
        # )
        # self.Children.qt_complexscheduler.currentIndexChanged.connect(
        #     self.on_complex_misc_changed
        # )
        # self.Children.qt_numclasses.valueChanged.connect(
        #     self.on_complex_misc_changed
        # )
        depth, width = self.get_model_dimensions(
            self.Children.qt_complex_model_ComboBox.currentText(),
        )
        self.cmplx_miscellaneous = dict()
        self.cmplx_tracked_data = []
        self.cmplx_miscellaneous_params = {
            "data_num_workers": self.Children.qt_numWorkers,
            "eval_interval": self.Children.qt_evalInterval,
            "warmup_epochs": self.Children.qt_warmupepochs,
            "scheduler": self.Children.qt_complexscheduler,
            "num_classes": self.Children.qt_numclasses,
        }

        for param_name, widget in self.cmplx_miscellaneous_params.items():
            self.fetch_cmplx_misc_param(param_name, widget)
            self.set_cmplx_misc_on_change(param_name, widget)
            self.cmplx_tracked_data.append(param_name)

        self.pretrained = {
            "value": self.Children.qt_complex_model_ComboBox.currentText(),
            "index": self.Children.qt_complex_model_ComboBox.currentIndex(),
            "depth": depth,
            "width": width,
        }

        self.selected_pretrained_model = (
            self.Children.qt_complex_model_ComboBox.currentText()
        )

    def fetch_cmplx_misc_param(self, param_name, widget):
        try:
            if isinstance(widget, QSpinBox):
                self.cmplx_miscellaneous[param_name] = widget.value()
            elif isinstance(widget, QComboBox):
                self.cmplx_miscellaneous[param_name] = widget.currentText()
        except Exception as e:
            print(f"[Error fetching {param_name}]: {e}")

    def set_cmplx_misc_on_change(self, param_name, widget):
        if isinstance(widget, QSpinBox):
            widget.valueChanged.connect(
                lambda: self.fetch_cmplx_misc_param(param_name, widget),
            )
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(
                lambda: self.fetch_cmplx_misc_param(param_name, widget),
            )

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

    def get_min_size(self, model_name):
        """Retrieves the minimum input size (height and width) for the specified model.

        Parameters
        ----------
            model_name (str): The name of the model (YOLOX or torchvision).

        Returns
        -------
            tuple: A tuple of (height, width).

        """
        # Define input sizes
        # Add rest of models to dictionary
        input_sizes = {
            "yolox-tiny": (416, 416),
            "yolox-small": (640, 640),
            "yolox-medium": (640, 640),
            "yolox-large": (640, 640),
            "yolox-x-large": (640, 640),
            "yolox-nano": (416, 416),
        }

        # Check if the model is a YOLOX model
        if model_name.lower() in input_sizes:
            return input_sizes[model_name.lower()]

    def on_complex_arch_combobox_change(self):
        selected_model = self.Children.qt_complex_model_ComboBox.currentText()
        depth, width = self.get_model_dimensions(selected_model)
        self.pretrained = {
            "value": self.Children.qt_complex_model_ComboBox.currentText(),
            "index": self.Children.qt_complex_model_ComboBox.currentIndex(),
            "depth": depth,
            "width": width,
        }
        Height, Width = self.get_min_size(self.pretrained["value"])
        if Height > self.Miscellaneous.miscellaneous["height"]:
            self.Miscellaneous.miscellaneous_params["height"].setValue(Height)
        if Width > self.Miscellaneous.miscellaneous["width"]:
            self.Miscellaneous.miscellaneous_params["width"].setValue(Width)
        print(f"Selected Complex Arch Model: {selected_model}")

    def load_from_config(self, json_data: dict):
        if len(json_data["pretrained"]) != 0:
            self.Children.qt_pretrained_model_combobox.setCurrentIndex(
                json_data["pretrained"]["index"],
            )

    def fill_pretrained_model(self):
        for i in self.PRETRAINED_MODELS:
            self.Children.qt_pretrained_model_combobox.addItem(i, i)

    def fill_complex_arch_model(self):
        for model in self.COMPLEX_ARCH_MODELS:
            self.Children.qt_complex_model_ComboBox.addItem(model)
