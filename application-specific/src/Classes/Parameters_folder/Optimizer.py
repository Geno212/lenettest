from PySide6.QtWidgets import QPushButton

from src.Classes.Children import Children
from src.Qt.Dialogue import LayerDialog
from src.Qt.WidgetData import WidgetData
from src.utils.AutoExtraction import AutoExtraction


class Optimizer:
    def __init__(self) -> None:
        self.Children = Children()
        self.LayerDialog = LayerDialog()
        self.OPTIMIZERS = AutoExtraction().OPTIMIZERS
        self.WidgetUtility = WidgetData()
        self.optimizer = dict()
        self.fill_optimizers()

    def load_from_config(self, json_data: dict):
        if len(json_data["optimizer"]) != 0:
            self.optimizer = json_data["optimizer"]
            self.Children.qt_selectedOptimizer_QLineEdit.setText(
                json_data["optimizer"]["type"],
            )

    def fill_optimizers(self):
        for one_optimizer in self.OPTIMIZERS:
            selectOptimizer_QPushButton = QPushButton(one_optimizer)
            selectOptimizer_QPushButton.clicked.connect(
                lambda i=one_optimizer,
                j=self.OPTIMIZERS,
                k=self.on_select_optimizer_clicked: self.LayerDialog.on_torch_func_clicked(
                    i,
                    j,
                    k,
                    None,
                    None,
                ),
            )
            self.Children.qt_optimizersList_QVBoxLayout.addWidget(
                selectOptimizer_QPushButton,
            )

    def on_select_optimizer_clicked(
        self,
        optimizer_type,
        params_names,
        params_value_widgets,
        paramsWindow_QDialog,
        *args,
    ):
        self.selected_optimizer = {"type": optimizer_type, "params": dict()}
        for i in range(len(params_value_widgets)):
            param_value = self.WidgetUtility.get_widget_data(
                params_value_widgets[i],
            )
            if param_value != "":
                self.selected_optimizer["params"][params_names[i]] = param_value

            self.Children.qt_selectedOptimizer_QLineEdit.setText(optimizer_type)
        self.optimizer = self.selected_optimizer
        paramsWindow_QDialog.close()
