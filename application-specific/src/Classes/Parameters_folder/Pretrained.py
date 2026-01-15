import torchvision

from src.Classes.Children import Children
from src.Classes.Parameters_folder.Miscellaneous import Miscellaneous
from src.Qt.Dialogue import LayerDialog
from src.Qt.WidgetData import WidgetData
from src.utils.AutoExtraction import AutoExtraction


class Pretrained:
    def __init__(self) -> None:
        self.Children = Children()
        self.Miscellaneous = Miscellaneous()
        self.LayerDialog = LayerDialog()
        self.PRETRAINED_MODELS = AutoExtraction().PRETRAINED_MODELS
        self.COMPLEX_ARCH_MODELS = AutoExtraction().COMPLEX_ARCH_MODELS
        self.WidgetUtility = WidgetData()
        # self.fill_pretrained_model() -------------->RAFIK
        # self.fill_complex_arch_model()
        self.Children.qt_pretrained_model_combobox.currentIndexChanged.connect(
            self.on_combobox_change,
        )
        self.pretrained = {
            "value": self.Children.qt_pretrained_model_combobox.currentText(),
            "index": self.Children.qt_pretrained_model_combobox.currentIndex(),
        }

        self.selected_pretrained_model = (
            self.Children.qt_pretrained_model_combobox.currentText()
        )

    def get_min_size(self, model_name):
        """Retrieves the minimum input size (height and width) for the specified model.

        Parameters
        ----------
            model_name (str): The name of the model (YOLOX or torchvision).

        Returns
        -------
            tuple: A tuple of (height, width).

        """
        # Handle torchvision models
        try:
            min_size = torchvision.models.get_model_weights(
                torchvision.models.__dict__[model_name],
            ).DEFAULT.meta["min_size"]
            return min_size
        except KeyError:
            raise ValueError(f"Unknown model: {model_name}")

    def on_combobox_change(self):
        self.pretrained = {
            "value": self.Children.qt_pretrained_model_combobox.currentText(),
            "index": self.Children.qt_pretrained_model_combobox.currentIndex(),
        }

        Height, Width = self.get_min_size(self.pretrained["value"])
        if Height > self.Miscellaneous.miscellaneous["height"]:
            self.Miscellaneous.miscellaneous_params["height"].setValue(Height)
        if Width > self.Miscellaneous.miscellaneous["width"]:
            self.Miscellaneous.miscellaneous_params["width"].setValue(Width)

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
