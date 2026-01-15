from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout


class LayerNodeManager:
    def __init__(self) -> None:
        self.misc_params = dict()
        self.architecture = {
            "layers": [],
            "misc_params": {
                "width": 1,
                "height": 1,
                "channels": 1,
                "num_epochs": 1,
                "batch_size": 0,
                "device": "cpu",
            },
        }
        self.Resarchitecture = {
            "layers": [],
        }

    def create_layer_node(self, layer, index, qt_layout, arch_dict):
        addedLayerRow_QHBoxLayout = QHBoxLayout()
        border_QFrame = QFrame()
        border_QFrame.setMaximumHeight(70)
        border_QFrame.setLayout(addedLayerRow_QHBoxLayout)
        border_QFrame.setStyleSheet("""
            QFrame{
                border: 1px solid black;
            }
            QLabel{
                border: none;
                font-weight: bold;
                font-size: 15px;
            }
        """)
        delete_QPushButton = self.Qtbtn.qbutton_layer_manager(
            self.SysPath.delete_icon_path,
            qt_layout,
            border_QFrame,
            self.on_delete_layer_clicked,
            arch_dict,
        )
        up_QPushButton = self.Qtbtn.qbutton_layer_manager(
            self.SysPath.up_icon_path,
            qt_layout,
            border_QFrame,
            self.on_move_buttons_clicked,
            "up",
            arch_dict,
        )
        down_QPushButton = self.Qtbtn.qbutton_layer_manager(
            self.SysPath.down_icon_path,
            qt_layout,
            border_QFrame,
            self.on_move_buttons_clicked,
            "down",
            arch_dict,
        )

        moveableArrows_QVBoxLayout = QVBoxLayout()
        moveableArrows_QVBoxLayout.addWidget(up_QPushButton)
        moveableArrows_QVBoxLayout.addWidget(down_QPushButton)

        addedLayerRow_QHBoxLayout.addWidget(QLabel(layer["type"]))
        addedLayerRow_QHBoxLayout.addLayout(moveableArrows_QVBoxLayout)
        addedLayerRow_QHBoxLayout.addWidget(delete_QPushButton)

        if index == -1:
            qt_layout.addWidget(border_QFrame)
            arch_dict["layers"].append(layer)
        else:
            qt_layout.insertWidget(index, border_QFrame)
            arch_dict["layers"].insert(index, layer)

    def on_delete_layer_clicked(self, border_QFrame, qt_layout, arch_dict):
        for i in range(len(arch_dict["layers"])):
            if border_QFrame == qt_layout.itemAt(i).widget():
                layer_widget = qt_layout.itemAt(i).widget()

                arch_dict["layers"].pop(i)
                layer_widget.deleteLater()
                qt_layout.removeWidget(layer_widget)
                break

    def on_move_buttons_clicked(self, border_QFrame, qt_layout, direction, arch_dict):
        size = len(arch_dict["layers"])
        print(size, "Size", direction)
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

                layer = arch_dict["layers"].pop(i)
                arch_dict["layers"].insert(new_idx, layer)

                qt_layout.removeWidget(layer_widget)
                qt_layout.insertWidget(new_idx, layer_widget)
                break
