from PySide6.QtWidgets import QPushButton

from src.Classes.Children import Children
from src.Qt.Dialogue import LayerDialog
from src.Qt.WidgetData import WidgetData
from src.utils.AutoExtraction import AutoExtraction


class Scheduler:
    def __init__(self) -> None:
        self.Children = Children()
        self.LayerDialog = LayerDialog()
        self.SCHEDULERS = AutoExtraction().SCHEDULERS
        self.WidgetUtility = WidgetData()
        self.scheduler = dict()
        self.fill_schedulers()

    def load_from_config(self, json_data: dict):
        if len(json_data["scheduler"]) != 0:
            self.scheduler = json_data["scheduler"]
            self.Children.qt_selectedScheduler_QLineEdit.setText(
                json_data["scheduler"]["type"],
            )

    def fill_schedulers(self):
        for scheduler in self.SCHEDULERS:
            selectScheduler_QPushButton = QPushButton(scheduler)
            selectScheduler_QPushButton.clicked.connect(
                lambda i=scheduler,
                j=self.SCHEDULERS,
                k=self.on_select_scheduler_clicked: self.LayerDialog.on_torch_func_clicked(
                    i,
                    j,
                    k,
                    None,
                    None,
                ),
            )

            self.Children.qt_schedulersList_QVBoxLayout.addWidget(
                selectScheduler_QPushButton,
            )

    def on_select_scheduler_clicked(
        self,
        scheduler_type,
        params_names,
        params_value_widgets,
        paramsWindow_QDialog,
        *args,
    ):
        self.selected_scheduler = {"type": scheduler_type, "params": dict()}
        for i in range(len(params_value_widgets)):
            param_value = self.WidgetUtility.get_widget_data(params_value_widgets[i])

            if param_value != "":
                self.selected_scheduler["params"][params_names[i]] = param_value

        self.Children.qt_selectedScheduler_QLineEdit.setText(scheduler_type)
        self.scheduler = self.selected_scheduler
        paramsWindow_QDialog.close()

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
                self.selected_lossfunc["params"][params_names[i]] = param_value

        self.Children.qt_selectedLossFunc_QLineEdit.setText(lossfunc_type)
        self.loss_function = self.selected_lossfunc
        paramsWindow_QDialog.close()
