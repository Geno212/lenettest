import json
import os
import sys

import cookiecutter.main
import cookiecutter.prompt
from cookiecutter.main import cookiecutter
from jinja2 import Environment, FileSystemLoader
from PySide6.QtCore import QProcess
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QFileDialog, QMessageBox

from src.paths.SystemPaths import SystemPaths
from src.utils.Singleton import Singleton

basedir = os.path.dirname(__file__)
loader = QUiLoader()


class Cookiecutter(metaclass=Singleton):
    def __init__(self, jinja_templates, debug) -> None:
        self.SysPath = SystemPaths()
        self.debug = debug
        self.SysPath.jinja_templates_path = jinja_templates
        self.jinja_template_filename = "cookiecutter.json.jinja"

    def render_cookiecutter_template(
        self,
        src_cookie_json_path,
        output_cookie_json_path,
        template_dir,
    ):
        path_arch_json, _ = QFileDialog.getOpenFileName(
            None,
            "Load Architecture JSON file",
            self.SysPath.jsondir,
            "JSON Files (*.json)",
        )
        path_output = None
        if path_arch_json:
            data = None
            with open(path_arch_json) as json_file:
                data = json.load(json_file)

            if "list" in data["layers"]:
                data["residual"] = {"layers": {"list": []}}
                for layer in data["layers"]["list"]:
                    if layer["type"] == "Residual_Block":
                        path_residual_json, _ = QFileDialog.getOpenFileName(
                            None,
                            "Load Residual Block JSON file",
                            self.SysPath.jsondir,
                            "JSON Files (*.json)",
                        )
                        with open(path_residual_json) as json_file:
                            data["residual"] = json.load(json_file)
                        break

            path_output = QFileDialog.getExistingDirectory(
                None,
                "Pick a folder to save the output",
                self.SysPath.jsondir,
            )
            self.SysPath.jsondir = path_output
            self.cookicutterpreproccess(
                data,
                src_cookie_json_path,
                output_cookie_json_path,
            )
            self.generate_project(template_dir, path_output)
        return path_output

    def generate_project(self, template_path, output_path):
        cookiecutter(
            template_path,
            output_dir=output_path,
            no_input=True,
            overwrite_if_exists=True,
        )

    def cookicutterpreproccess(
        self,
        data,
        src_cookie_json_path,
        output_cookie_json_path,
    ):
        env = Environment(loader=FileSystemLoader(src_cookie_json_path))
        if self.debug:
            print(
                src_cookie_json_path,
                output_cookie_json_path,
                self.SysPath.jinja_templates_path,
            )
        template = env.get_template(self.jinja_template_filename)

        data = self.remove_empty_arrays(data)
        result_file = template.render(my_dict=json.dumps(data, indent=4))
        with open(output_cookie_json_path, "w") as json_file:
            json_file.write(str(result_file))

    def remove_empty_arrays(self, d):
        return {k: v for k, v in d.items() if v != []}

    def render_cookiecutter_template_cli(
        self,
        data,
        src_cookie_json_path,
        output_cookie_json_path,
        template_dir,
        output_path,
    ):
        """CLI-friendly version of render_cookiecutter_template that doesn't use Qt dialogs.

        Args:
            data: Architecture data dictionary
            src_cookie_json_path: Path to the jinja template directory
            output_cookie_json_path: Path where cookiecutter.json will be written
            template_dir: Path to the cookiecutter template directory
            output_path: Path where the generated project will be created

        Returns:
            str: Path to the generated output directory

        """
        # Handle residual blocks if present
        if "list" in data.get("layers", {}):
            data["residual"] = {"layers": {"list": []}}
            for layer in data["layers"]["list"]:
                if layer.get("type") == "Residual_Block":
                    residual_layers = layer.get("layers", [])
                    data["residual"] = {"layers": {"list": residual_layers}}
                    break

        # Process the cookiecutter template
        self.cookicutterpreproccess(
            data,
            src_cookie_json_path,
            output_cookie_json_path,
        )

        # Generate the project
        self.generate_project(
            template_dir,
            output_path,
        )

        return output_path

    @staticmethod
    def test_image(
        image_path,
        weights_path,
        hgd_ckpt_path,
        sys_path,
        cookiecutter_instance,
    ):
        # Validate paths
        if not image_path or not weights_path or not hgd_ckpt_path:
            QMessageBox.warning(
                None,
                "Missing Paths",
                "Please provide all required paths.",
            )
            return

        # Prepare JSON data for cookiecutter
        json_data = {
            "image_path": image_path,
            "weights_path": weights_path,
            "hgd_ckpt_path": hgd_ckpt_path,
        }

        # Save JSON data to the desired directory
        temp_json_path = os.path.join(
            sys_path.testing_template_dir,
            "cookiecutter.json",
        )
        try:
            with open(temp_json_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to save JSON file: {e!s}")
            return

        # Define paths for cookiecutter template and output
        template_dir = sys_path.testing_template_dir
        output_dir = sys_path.testing_output_dir

        try:
            # Render the cookiecutter template
            path_output = cookiecutter_instance.render_cookiecutter_template(
                src_cookie_json_path=temp_json_path,
                output_cookie_json_path=sys_path.transfer_cookie_json,
                template_dir=template_dir,
            )
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to render template: {e!s}")
            return

        # Check if the generated script exists
        generated_script_path = os.path.join(output_dir, "testFullModel.py")
        if not os.path.exists(generated_script_path):
            QMessageBox.critical(None, "Error", "Generated script not found.")
            return

        # Run the generated script using QProcess
        try:
            test_process = QProcess()
            test_process.readyReadStandardOutput.connect(
                lambda: print(
                    bytes(test_process.readAllStandardOutput()).decode("utf8"),
                ),
            )
            test_process.readyReadStandardError.connect(
                lambda: print(
                    bytes(test_process.readAllStandardError()).decode("utf8"),
                ),
            )
            # Use the virtual environment Python executable
            python_executable = sys.executable
            test_process.start(python_executable, [generated_script_path])
        except Exception as e:
            QMessageBox.critical(
                None,
                "Error",
                f"Failed to execute the script: {e!s}",
            )
