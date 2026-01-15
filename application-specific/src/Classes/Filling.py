import torch


class FillingQt:
    def __init__(self) -> None:
        self.fill_placeholders(
            self.Children.qt_layersList_QVBoxLayout,
            self.LAYERS,
            self.Children.qt_addedLayers_QVBoxLayout,
        )

    def fill_placeholders(self, qt_layout, layers, q2_lay):
        # self.fill_layers(qt_layout, layers, q2_lay, self.architecture)
        self.fill_pretrained_model()
        self.fill_datasets()

    def fill_pretrained_model(self):
        # Populate the pretrained models combo box
        for i in self.PRETRAINED_MODELS:
            self.Children.qt_pretrained_model_combobox.addItem(i)

        # Populate the complex arch models combo box -------------->RAFIKK
        for model in self.COMPLEX_ARCH_MODELS:
            self.Children.qt_complex_model_ComboBox.addItem(model)

    def fill_datasets(self):
        for i in self.DATASETS:
            self.Children.qt_selectedDataset_QComboBox.addItem(i, i)

    def fill_cuda_devices(self, combo_box):
        device_names = []
        if torch.cuda.is_available():
            cuda_devices = torch.cuda.device_count()
            for i in range(cuda_devices):
                properties = torch.cuda.get_device_properties(i)
                device_name = properties.name
                device_names.append((f"cuda:{i}", device_name))

        combo_box.clear()
        combo_box.addItem("cpu", "cpu")
        for device_index, device_info in device_names:
            combo_box.addItem(device_info, device_index)
