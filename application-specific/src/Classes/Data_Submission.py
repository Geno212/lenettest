import json
import os

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

basedir = os.path.dirname(__file__)


class DataSubmission:
    def __init__(self) -> None:
        self.connections = dict()
        self.violations_list = []

    def on_submit_params_clicked(self):
        print("submit params clicked")
        self.architecture = self.Parameters.create_architecture()
        print(self.architecture)
        self.save_json()

    def on_submit_arch_clicked(self):
        self.validate_and_correct_layers(
            self.Children.qt_addedLayers_QVBoxLayout,
            self.architecture,
        )
        arch_json_file_path = self.save_json()
        self.violations_list = self.StaticAnalysis.analyze(self.architecture["layers"])
        self.Children.qt_violations_text_edit.clear()
        for i in self.violations_list:
            escaped_text = i.replace("<", "&lt;").replace(">", "&gt;")
            self.Children.qt_violations_text_edit.append(
                rf'<span style="color:#ff0000">{escaped_text}</span>',
            )

    def handle_stderr(self):
        result = bytes(self.Manual_Process.readAllStandardError()).decode("utf8")
        if self.debug:
            print(result)

    def handle_stdout(self):
        result = bytes(self.Manual_Process.readAllStandardOutput()).decode("utf8")
        if self.debug:
            print(result)

    def generate_manual_project(self):
        path_output = self.Cookiecutter.render_cookiecutter_template(
            self.SysPath.manual_jinja_json,
            self.SysPath.manual_cookie_json,
            self.SysPath.manual_template_dir,
        )

        if path_output:
            try:
                self.show_files(path_output)
            except:
                if self.debug:
                    print("ERRORRRRR")
            self.Manual_Process = QProcess()
            self.Manual_Process.readyReadStandardOutput.connect(self.handle_stdout)
            self.Manual_Process.readyReadStandardError.connect(self.handle_stderr)
            self.Manual_Process.start(
                "python",
                [path_output + "/Manual_Output/main.py"],
            )

    def on_submit_layer_clicked(
        self,
        layer_type,
        params_names,
        params_value_widgets,
        paramsWindow_QDialog,
        qt_layout,
        arch_dict,
    ):
        # Initialize message error box
        dlg = QMessageBox()
        dlg.setWindowTitle("error!")
        dlg.setStandardButtons(QMessageBox.Yes)
        dlg.setIcon(QMessageBox.Critical)
        layer = {
            "type": layer_type,
            "params": dict(),
        }

        for i in range(len(params_value_widgets)):
            param_value = self.get_widget_data(params_value_widgets[i])
            if param_value != "":
                layer["params"][params_names[i]] = param_value
        # cond=self.test_layer(layer_type, params_names, params_value_widgets)
        cond = True
        if cond:
            self.create_layer_node(layer, -1, qt_layout, arch_dict)
            paramsWindow_QDialog.close()

    # save json for manual arch
    def save_json(self):
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Architecture JSON File",
            self.SysPath.jsondir,
            "JSON Files (*.json)",
        )
        if path:
            self.SysPath.jsondir = path
            self.architecture["log_dir"] = self.SysPath.log_path
            # test for deep and shallow to avoid errors
            architecture = self.architecture.copy()

            architecture["layers"] = {"list": self.architecture["layers"]}

            with open(path, "w") as f:
                f.write(json.dumps(architecture, indent=4))

            # with open("D:\Graduation Project\Siemens-System-Level-Modelling-of-ASDLA-Graduation-Project\src\our jsons\json1.json","w")as f:
            #     f.write(json.dumps(architecture, indent=4))
            if self.debug:
                print("JSON file saved successfully.")

        return path
