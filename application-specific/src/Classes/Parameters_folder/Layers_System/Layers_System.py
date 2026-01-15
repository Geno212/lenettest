import copy

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from src.Classes.Children import Children
from src.Classes.Parameters_folder.Layers_System.Residual_System import Residual_System
from src.Classes.Parameters_folder.Miscellaneous import Miscellaneous
from src.paths.SystemPaths import SystemPaths
from src.Qt.Buttons import QTButtons
from src.Qt.Dialogue import LayerDialog
from src.Qt.WidgetData import WidgetData
from src.Tests.StaticAnalysis import StaticAnalysis
from src.Tests.Validation import Validation
from src.utils.AutoExtraction import AutoExtraction


class Layers_System:
    def __init__(self) -> None:
        self.Residual_System = Residual_System(self.on_submit_layer_clicked)
        self.Children = Children()
        self.SysPath = SystemPaths()
        self.StaticAnalysis = StaticAnalysis()
        self.Misc_params = Miscellaneous()
        self.LayerDialog = LayerDialog()
        self.LAYERS = AutoExtraction().LAYERS
        self.WidgetUtility = WidgetData()
        self.Resarchitecture = {"layers": []}
        self.layers = []
        self.Qtbtn = QTButtons()
        self.Layer_Validation = Validation()
        self.fill_layers()

    def load_from_config(self, json_data: dict):
        if len(json_data["layers"]["list"]) != 0:
            temp_layers = copy.deepcopy(json_data["layers"]["list"])
            self.layers = []
            layout = self.Children.qt_addedLayers_QVBoxLayout
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            for layer in temp_layers:
                self.create_layer_node(
                    layer,
                    -1,
                    self.Children.qt_addedLayers_QVBoxLayout,
                )

    def fill_layers(self):
        for layer in self.LAYERS:
            layer_btn = QPushButton(layer)
            if layer == "Residual_Block":
                pass
            else:
                layer_btn.clicked.connect(
                    lambda i=layer,
                    j=self.LAYERS,
                    k=self.on_submit_layer_clicked,
                    q_layout=self.Children.qt_addedLayers_QVBoxLayout: self.LayerDialog.on_torch_func_clicked(
                        i,
                        j,
                        k,
                        q_layout,
                    ),
                )
                self.Children.qt_layersList_QVBoxLayout.addWidget(layer_btn)

    def on_submit_layer_clicked(
        self,
        layer_type,
        params_names,
        params_value_widgets,
        paramsWindow_QDialog,
        qt_layout,
    ):
        dlg = QMessageBox()
        dlg.setWindowTitle("error!")
        dlg.setStandardButtons(QMessageBox.Yes)
        dlg.setIcon(QMessageBox.Critical)
        layer = {
            "type": layer_type,
            "params": dict(),
        }
        for i in range(len(params_value_widgets)):
            param_value = self.WidgetUtility.get_widget_data(params_value_widgets[i])
            if param_value != "":
                layer["params"][params_names[i]] = param_value
        cond = True
        if cond:
            self.create_layer_node(layer, -1, qt_layout)
            paramsWindow_QDialog.close()

    def create_layer_node(self, layer, index, qt_layout):
        print(layer)
        addedLayerRow_QHBoxLayout = QHBoxLayout()

        border_QFrame = QFrame()

        border_QFrame.setMaximumHeight(70)

        border_QFrame.setLayout(addedLayerRow_QHBoxLayout)

        border_QFrame.setStyleSheet(
            """
            QFrame{
                border: 1px solid black;
            }
            QLabel{
                border: none;
                font-weight: bold;
                font-size: 15px;
            }
            """,
        )
        delete_QPushButton = self.Qtbtn.qbutton_layer_manager(
            self.SysPath.delete_icon_path,
            qt_layout,
            border_QFrame,
            self.on_delete_layer_clicked,
        )
        up_QPushButton = self.Qtbtn.qbutton_layer_manager(
            self.SysPath.up_icon_path,
            qt_layout,
            border_QFrame,
            self.on_move_buttons_clicked,
            "up",
        )
        down_QPushButton = self.Qtbtn.qbutton_layer_manager(
            self.SysPath.down_icon_path,
            qt_layout,
            border_QFrame,
            self.on_move_buttons_clicked,
            "down",
        )

        moveableArrows_QVBoxLayout = QVBoxLayout()
        moveableArrows_QVBoxLayout.addWidget(up_QPushButton)
        moveableArrows_QVBoxLayout.addWidget(down_QPushButton)

        addedLayerRow_QHBoxLayout.addWidget(QLabel(layer["type"]))
        addedLayerRow_QHBoxLayout.addLayout(moveableArrows_QVBoxLayout)
        addedLayerRow_QHBoxLayout.addWidget(delete_QPushButton)

        if index == -1:
            qt_layout.addWidget(border_QFrame)
            self.layers.append(layer)
        else:
            qt_layout.insertWidget(index, border_QFrame)
            self.layers.insert(index, layer)
        self.Analyze()

    def Validate_func(self):
        temp_length = len(self.layers)

        temp_dict = self.Layer_Validation.validate_and_correct_layers(
            self.layers,
            self.Misc_params.miscellaneous["width"],
            self.Misc_params.miscellaneous["height"],
            self.Misc_params.miscellaneous["channels"],
        )

        if temp_length != len(temp_dict):
            temp_dict = {"layers": {"list": self.layers}}
            self.load_from_config(temp_dict)
            self.Layer_Validation.layer_naming(self.layers)

    def Analyze(self):
        self.violations_list = self.StaticAnalysis.analyze(self.layers)
        self.Children.qt_violations_text_edit.clear()
        for i in self.violations_list:
            escaped_text = i.replace("<", "&lt;").replace(">", "&gt;")
            self.Children.qt_violations_text_edit.append(
                rf'<span style="color:#ff0000">{escaped_text}</span>',
            )

    def on_delete_layer_clicked(self, border_QFrame, qt_layout):
        for i in range(len(self.layers)):
            if border_QFrame == qt_layout.itemAt(i).widget():
                layer_widget = qt_layout.itemAt(i).widget()

                self.layers.pop(i)
                layer_widget.deleteLater()
                qt_layout.removeWidget(layer_widget)
                break
        self.Analyze()

    def on_move_buttons_clicked(self, border_QFrame, qt_layout, direction):
        size = len(self.layers)
        for i in range(size):
            if border_QFrame == qt_layout.itemAt(i).widget():
                if (i == 0 and direction == "up") or (
                    i == size - 1 and direction == "down"
                ):
                    break

                layer_widget = qt_layout.itemAt(i).widget()
                if direction == "up":
                    new_idx = i - 1
                elif direction == "down":
                    new_idx = i + 1

                layer = self.layers.pop(i)
                self.layers.insert(new_idx, layer)

                qt_layout.removeWidget(layer_widget)
                qt_layout.insertWidget(new_idx, layer_widget)
                break
        self.Analyze()

    def on_res_block_clicked(self, func_name, torch_funcs, on_submit_func, qt_layout):
        if len(self.Resarchitecture["layers"]) > 0:
            self.LayerDialog.on_torch_func_clicked(
                func_name,
                torch_funcs,
                on_submit_func,
                qt_layout,
            )
        else:
            self.Children.ResCreation.show()
