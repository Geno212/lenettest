import json


class FileGenerator:
    def file_read_json(self):
        with open(self.SysPath.arch_json_path) as f:
            self.architecture = json.load(f)
            self.layers = self.architecture["layers"]
            self.misc_params = self.architecture["misc_params"]

    def generate_model(self):
        jinja_input_path = self.SysPath.model_jinja_path
        python_output_path = self.SysPath.model_py_path
        self.convert_jinja_to_code(
            jinja_input_path,
            python_output_path,
            {
                "layers": self.layers,
            },
        )

    def generate_train(self):
        self.convert_jinja_to_code(
            self.SysPath.train_jinja_path,
            self.SysPath.train_py_path,
            self.misc_params,
        )
