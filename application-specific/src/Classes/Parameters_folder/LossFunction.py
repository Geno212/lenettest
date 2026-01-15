from PySide6.QtWidgets import QMessageBox, QPushButton

from src.Classes.Children import Children
from src.Qt.Dialogue import LayerDialog
from src.Qt.WidgetData import WidgetData
from src.utils.AutoExtraction import AutoExtraction


class LossFunction:
    def __init__(self) -> None:
        self.Children = Children()
        self.LayerDialog = LayerDialog()
        self.LOSSFUNC = AutoExtraction().LOSSFUNC
        self.WidgetUtility = WidgetData()
        self.loss_function = dict()
        self.fill_lossfunctions()

    def load_from_config(self, json_data: dict):
        if len(json_data["loss_func"]) != 0:
            self.loss_function = json_data["loss_func"]
            self.Children.qt_selectedLossFunc_QLineEdit.setText(
                json_data["loss_func"]["type"],
            )

    def fill_lossfunctions(self):
        for lossfunc in self.LOSSFUNC:
            selectLossFunc_QPushButton = QPushButton(lossfunc)
            selectLossFunc_QPushButton.clicked.connect(
                lambda i=lossfunc,
                j=self.LOSSFUNC,
                k=self.on_select_lossfunc_clicked: self.LayerDialog.on_torch_func_clicked(
                    i,
                    j,
                    k,
                    None,
                    None,
                ),
            )
            self.Children.qt_lossFuncsList_QVBoxLayout.addWidget(
                selectLossFunc_QPushButton,
            )

    def on_select_lossfunc_clicked(
        self,
        lossfunc_type,
        params_names,
        params_value_widgets,
        paramsWindow_QDialog,
        *args,
    ):
        self.selected_lossfunc = {"type": lossfunc_type, "params": dict()}
        for i in range(len(params_value_widgets)):
            param_value = self.WidgetUtility.get_widget_data(params_value_widgets[i])

            if param_value != "":
                # Special handling for AdaptiveLogSoftmaxWithLoss cutoffs parameter
                if (
                    lossfunc_type == "AdaptiveLogSoftmaxWithLoss"
                    and params_names[i] == "cutoffs"
                ):
                    # Convert single integer to list
                    if isinstance(param_value, (int, float)) or (isinstance(param_value, str) and param_value.isdigit()):
                        param_value = [int(param_value)]
                    elif isinstance(param_value, str):
                        # Try to parse as comma-separated list
                        try:
                            param_value = [
                                int(x.strip())
                                for x in param_value.split(",")
                                if x.strip()
                            ]
                        except ValueError:
                            param_value = [2]  # Default fallback

                self.selected_lossfunc["params"][params_names[i]] = param_value

        self.Children.qt_selectedLossFunc_QLineEdit.setText(lossfunc_type)

        # Validate loss function selection
        self._validate_loss_function_selection(lossfunc_type)

        self.loss_function = self.selected_lossfunc
        paramsWindow_QDialog.close()

    def _validate_loss_function_selection(self, lossfunc_type):
        """Validate if the selected loss function is appropriate for the dataset/task"""
        try:
            # Get the current dataset selection if available
            dataset_widget = self.Children.qt_dataset_combobox
            if dataset_widget and hasattr(dataset_widget, "currentText"):
                current_dataset = dataset_widget.currentText()

                # Define multi-class datasets that should use CrossEntropyLoss
                multiclass_datasets = [
                    "MNIST",
                    "CIFAR10",
                    "CIFAR100",
                    "FashionMNIST",
                    "ImageNet",
                ]

                # Check for common mismatches
                if (
                    current_dataset in multiclass_datasets
                    and lossfunc_type == "BCELoss"
                ):
                    reply = QMessageBox.question(
                        None,
                        "Loss Function Warning",
                        f"You selected BCELoss for {current_dataset} dataset.\n\n"
                        f"{current_dataset} is a multi-class classification dataset with multiple classes.\n"
                        "BCELoss is typically used for binary classification or multi-label tasks.\n\n"
                        "For multi-class classification, CrossEntropyLoss is usually more appropriate.\n\n"
                        "Do you want to continue with BCELoss anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )

                    if reply == QMessageBox.StandardButton.No:
                        # User chose not to continue, they can select a different loss function
                        return False

                elif (
                    current_dataset in multiclass_datasets
                    and lossfunc_type == "BCEWithLogitsLoss"
                ):
                    QMessageBox.information(
                        None,
                        "Loss Function Info",
                        f"You selected BCEWithLogitsLoss for {current_dataset} dataset.\n\n"
                        "This loss function is typically used for multi-label classification.\n"
                        f"For {current_dataset} (multi-class), CrossEntropyLoss is usually preferred.\n\n"
                        "Consider using CrossEntropyLoss for better results.",
                    )

        except Exception as e:
            # If validation fails, just continue without blocking the user
            print(f"Loss function validation warning: {e}")

        return True
