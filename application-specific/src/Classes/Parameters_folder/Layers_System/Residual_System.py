import copy
import json

from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from src.Classes.Children import Children
from src.paths.SystemPaths import SystemPaths
from src.Qt.Buttons import QTButtons
from src.Qt.Dialogue import LayerDialog
from src.Qt.WidgetData import WidgetData
from src.Tests.Validation import Validation
from src.utils.AutoExtraction import AutoExtraction


class Residual_System:
    def __init__(self, callback) -> None:
        self.callback = callback
        self.Children = Children()
        self.setup()
        self.SysPath = SystemPaths()
        self.LayerDialog = LayerDialog()
        self.RES_LAYERS = AutoExtraction().LAYERS_WITHOUT_RES
        self.WidgetUtility = WidgetData()
        self.res_layers = []
        self.Resarchitecture = {"layers": []}
        self.Qtbtn = QTButtons()
        self.Layer_Validation = Validation()
        self.Children.qt_submitRes_QPushButton.clicked.connect(
            self.res_on_submit_residual_block_clicked,
        )
        self.fill_layers()

    def setup(self):
        layer_btn = QPushButton("Residual_Block")
        layer_btn.clicked.connect(
            lambda i="Residual_Block",
            j=AutoExtraction().LAYERS,
            k=self.on_submit_layer_clicked,
            q_layout=self.Children.qt_addedLayers_QVBoxLayout: self.on_res_block_clicked(
                i,
                j,
                k,
                q_layout,
            ),
        )
        self.Children.qt_layersList_QVBoxLayout.addWidget(layer_btn)

    def load_from_config(self, json_data: dict):
        if len(json_data["layers"]["list"]) != 0:
            temp_layers = copy.deepcopy(json_data["layers"]["list"])
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
        for layer in self.RES_LAYERS:
            layer_btn = QPushButton(layer)
            layer_btn.clicked.connect(
                lambda i=layer,
                j=self.RES_LAYERS,
                k=self.on_submit_layer_clicked,
                q_layout=self.Children.qt_res_addedLayers_QVBoxLayout: self.LayerDialog.on_torch_func_clicked(
                    i,
                    j,
                    k,
                    q_layout,
                ),
            )
            self.Children.qt_res_layersList_QVBoxLayout.addWidget(layer_btn)

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
            self.res_layers.append(layer)
        else:
            qt_layout.insertWidget(index, border_QFrame)
            self.res_layers.insert(index, layer)

    def on_delete_layer_clicked(self, border_QFrame, qt_layout):
        for i in range(len(self.res_layers)):
            if border_QFrame == qt_layout.itemAt(i).widget():
                layer_widget = qt_layout.itemAt(i).widget()

                self.res_layers.pop(i)
                layer_widget.deleteLater()
                qt_layout.removeWidget(layer_widget)
                break

    def on_move_buttons_clicked(self, border_QFrame, qt_layout, direction):
        size = len(self.res_layers)
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

                layer = self.res_layers.pop(i)
                self.res_layers.insert(new_idx, layer)

                qt_layout.removeWidget(layer_widget)
                qt_layout.insertWidget(new_idx, layer_widget)
                break

    def res_on_submit_residual_block_clicked(self):
        self.add_layers_names(self.res_layers)
        with open(self.SysPath.ResJson, "w") as f:
            self.save_residual_json()
        self.Children.ResCreation.close()

    def save_residual_json(self):
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Residual Block JSON File",
            self.SysPath.basedir,
            "JSON Files (*.json)",
        )
        self.Resarchitecture["layers"] = {"list": self.res_layers}
        with open(path, "w") as f:
            f.write(json.dumps(self.Resarchitecture, indent=4))

        return path

    def add_layers_names(self, architecture):
        layer_freqs = dict()
        for i in self.res_layers:
            layer = i
            if layer["type"] in layer_freqs:
                layer_freqs[layer["type"]] += 1
            else:
                layer_freqs[layer["type"]] = 1
            layer["name"] = f"{layer['type'].lower()}_{layer_freqs[layer['type']]}"

    def on_res_block_clicked(self, func_name, torch_funcs, on_submit_func, qt_layout):
        print("el marady hena")
        if len(self.res_layers) > 0:
            self.LayerDialog.on_torch_func_clicked(
                "Residual_Block",
                AutoExtraction().LAYERS,
                self.callback,
                self.Children.qt_addedLayers_QVBoxLayout,
            )
        else:
            self.Children.ResCreation.show()
