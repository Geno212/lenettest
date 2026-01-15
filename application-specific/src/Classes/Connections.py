import json
import os
import sys

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QComboBox, QFileDialog, QLineEdit, QMessageBox, QSpinBox

message = ""


class Connections:
    def __init__(self) -> None:
        self.Children.qt_submitParams_QPushButton.clicked.connect(
            self.on_submit_params_clicked,
        )

        self.Children.qt_submitArch_QPushButton.clicked.connect(
            self.on_submit_params_clicked,
        )

        self.Children.qt_manual_generate.clicked.connect(self.generate_manual_project)

        self.Children.qt_Create_transfer_learning_model_QPushButton.clicked.connect(
            self.render_transfer_learning,
        )
        # self.Children.qt_Create_complex_Model_QPushButton.currentIndexChanged.connect(
        #     self.on_combobox_change#createfunction
        # )

        self.Children.qt_selectedDevice_QComboBox.currentIndexChanged.connect(
            self.fill_cuda_devices(self.Children.qt_selectedDevice_QComboBox),
        )
        self.Children.qt_dataset_path_QPushButton.clicked.connect(
            self.on_dataset_path_clicked,
        )
        self.Children.qt_Log_Directory_btn.clicked.connect(self.on_log_dir_clicked)

        self.Children.browseWeightsButton.clicked.connect(self.pick_weights_file)
        self.Children.browseImageButton.clicked.connect(self.pick_image_file)
        self.Children.browseHGDCkptButton.clicked.connect(self.pick_hgd_ckpt_file)
        self.Children.testImageButton.clicked.connect(self.handle_test_image)
        self.Children.testBrowseWeightsButton.clicked.connect(
            self.pick_test_weights_file,
        )
        self.Children.browseClassNamesButton.clicked.connect(self.pick_class_names_file)

    def set_data_load(self, param_name, widget, new_value):
        self.connections[param_name] = widget
        if type(widget) == QSpinBox:
            widget.valueChanged.connect(
                lambda: self.fetch_data_params(param_name, widget),
            )
        elif type(widget) == QLineEdit:
            widget.textChanged.connect(
                lambda: self.fetch_data_params(param_name, widget),
            )

        elif type(widget) == QComboBox:
            widget.currentIndexChanged.connect(
                lambda: self.fetch_data_params(param_name, widget),
            )

    def on_log_dir_clicked(self):
        path = QFileDialog.getExistingDirectory(None, "Select a Directory")
        if path:
            self.Children.qt_logdirlineedit.setText(path)

    def on_dataset_path_clicked(self):
        path = QFileDialog.getExistingDirectory(None, "Select a Directory")
        if path:
            self.Children.qt_dataset_path_QLineEdit.setText(path)

    def pick_weights_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Pretrained Weights",
            "",
            "All Files (*.*)",
        )
        if file_path:
            self.Children.weightsPathLineEdit.setText(file_path)

    def pick_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Image File",
            "",
            "Images (*.png *.jpg *.jpeg)",
        )
        if file_path:
            self.Children.imagePathLineEdit.setText(file_path)

    def pick_hgd_ckpt_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select HGD Checkpoint File",
            "",
            "All Files (*.*)",
        )
        if file_path:
            self.Children.hgdCkptPathLineEdit.setText(file_path)
        else:
            self.Children.hgdCkptPathLineEdit.setText(r"weights\Defense.pt")

    def pick_test_weights_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Weights File",
            "",
            "All Files (*.*)",
        )
        if file_path:
            self.Children.testWeightsPathLineEdit.setText(file_path)

    def pick_class_names_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Class Names File",
            "",
            "All Files (*.*)",
        )
        if file_path:
            self.Children.classNamesPathLineEdit.setText(file_path)

    def handle_test_image(self):
        # Step 1: Collect paths from textboxes
        image_path = self.Children.imagePathLineEdit.text().replace("file:///", "")
        weights_path = self.Children.testWeightsPathLineEdit.text().replace(
            "file:///",
            "",
        )
        hgd_ckpt_path = self.Children.hgdCkptPathLineEdit.text().replace("file:///", "")
        class_names_path = self.Children.classNamesPathLineEdit.text().replace(
            "file:///",
            "",
        )

        if (
            not image_path
            or not weights_path
            or not hgd_ckpt_path
            or not class_names_path
        ):
            QMessageBox.warning(
                None,
                "Missing Inputs",
                "Please provide all required inputs, including class names.",
            )
            return

        # Step 2: Save paths to a JSON file using cookiecutter
        json_data = {
            "project_name": "GeneratedTestModel",
            "image_path": image_path,
            "weights_path": weights_path,
            "hgd_ckpt_path": hgd_ckpt_path,
            "class_names_path": class_names_path,
        }
        json_file_path = os.path.join(
            self.SysPath.testing_template_dir,
            "cookiecutter.json",
        )
        if not os.path.exists(self.SysPath.testing_template_dir):
            os.makedirs(
                self.SysPath.testing_template_dir,
            )  # Ensure the directory exists
        with open(json_file_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        # Step 3: Generate the new script using cookiecutter
        template_dir = (
            self.SysPath.testing_template_dir
        )  # Path to the cookiecutter template directory
        output_dir = (
            self.SysPath.testing_output_dir
        )  # Path to save the generated script
        try:
            self.Cookiecutter.generate_project(template_dir, output_dir)
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to generate script: {e!s}")
            return

        # Step 4: Run the generated script using QProcess
        generated_script_path = os.path.join(
            output_dir,
            "GeneratedTestModel",
            "testFullModel.py",
        )
        if not os.path.exists(generated_script_path):
            QMessageBox.critical(None, "Error", "Generated script not found.")
            return

        self.test_process = QProcess()
        self.test_process.readyReadStandardOutput.connect(self.handle_test_stdout)
        self.test_process.readyReadStandardError.connect(self.handle_test_stderr)
        # Use the virtual environment Python executable
        python_executable = sys.executable
        self.test_process.start(python_executable, [generated_script_path])

    def handle_test_stdout(self):
        result = bytes(self.test_process.readAllStandardOutput()).decode("utf8")
        print(result)

    def handle_test_stderr(self):
        result = bytes(self.test_process.readAllStandardError()).decode("utf8")
        print(result)
