import os

from src.paths import PathsFactory
from src.utils.Singleton import Singleton


class SystemPaths(metaclass=Singleton):
    def __init__(self) -> None:
        print("System paths Class")

        # base directories
        self.basedir = os.path.dirname(__file__)
        self.jsondir = self.basedir
        self.publicdir = os.path.normpath(os.path.join(self.basedir, "../../public/"))
        self.srcdir = os.path.normpath(os.path.join(self.basedir, "./../"))
        self.datadir = os.path.normpath(os.path.join(self.basedir, "./../../data"))
        # log dir tensorboard
        self.Tensor_logs_dir = os.path.normpath(
            os.path.join(self.datadir, "./tensorboardlogs"),
        )

        # GUI Paths and UI
        self.GUI_path = os.path.normpath(
            os.path.join(self.publicdir, "./GUI/mainwindow.ui"),
        )
        self.main_ui_path = os.path.normpath(
            os.path.join(self.publicdir, "./GUI/mainwindow.ui"),
        )
        self.res_build_ui_path = os.path.normpath(
            os.path.join(self.publicdir, "./GUI/resbuild.ui"),
        )

        # Icons Paths
        self.delete_icon_path = os.path.normpath(
            os.path.join(self.publicdir, "./icons/delete.png"),
        )
        self.up_icon_path = os.path.normpath(
            os.path.join(self.publicdir, "./icons/up.png"),
        )
        self.down_icon_path = os.path.normpath(
            os.path.join(self.publicdir, "./icons/down.png"),
        )
        self.siemens_logo = os.path.normpath(
            os.path.join(self.publicdir, "./icons/siemens_logo.png"),
        )
        self.siemens_icon = os.path.normpath(
            os.path.join(self.publicdir, "./icons/siemens_icon.png"),
        )

        # JSON paths
        self.ResJson = os.path.normpath(
            os.path.join(self.publicdir, "json_files/res.json"),
        )
        self.arch_json_path = os.path.normpath(
            os.path.join(self.publicdir, "json_files/arch.json"),
        )

        # Template paths
        self.jinja_templates = os.path.normpath(
            os.path.join(self.publicdir, "jinja_templates"),
        )

        self.manual_jinja_json = os.path.normpath(
            os.path.join(self.publicdir, "jinja_templates/Manual"),
        )
        self.manual_template_dir = os.path.normpath(
            os.path.join(self.publicdir, "./Cookiecutter/Manual"),
        )
        self.manual_cookie_json = os.path.normpath(
            os.path.join(self.publicdir, "./Cookiecutter/Manual/cookiecutter.json"),
        )

        # YOLOX Pretrained Paths
        self.yolox_pretrained_jinja_json = os.path.normpath(
            os.path.join(self.publicdir, "jinja_templates/YOLOX"),
        )
        self.yolox_pretrained_template_dir = os.path.normpath(
            os.path.join(self.publicdir, "./Cookiecutter/YOLOX"),
        )
        self.yolox_pretrained_cookie_json = os.path.normpath(
            os.path.join(self.publicdir, "./Cookiecutter/YOLOX/cookiecutter.json"),
        )

        # Initialize paths based on default model
        self.update_paths("default")

    def update_paths(self, model_name):
        # get path through factory DP
        (
            self.transfer_jinja_json,
            self.transfer_template_dir,
            self.transfer_cookie_json,
        ) = PathsFactory.PathFactory.get_paths(self, model_name)
        print(
            self.transfer_jinja_json,
            self.transfer_template_dir,
            self.transfer_cookie_json,
        )
        # Dataset paths
        self.dataset_path = os.path.normpath(os.path.join(self.datadir, "./dataset"))
        # Log Path
        self.log_path = os.path.normpath(
            os.path.join(self.datadir, "./tensorboardlogs"),
        )
        print(self.log_path)
        # Rules Static Analysis
        self.warning_rules_path = os.path.normpath(
            os.path.join(self.publicdir, "./Rules/warning_rules.txt"),
        )
        self.css_path = "public/GUI/NewUI.qss"

        # to work on later
        self.model_py_path = "python_files/model.py"
        self.train_py_path = "python_files/train.py"
        self.model_jinja_path = "jinja_templates/model.py.jinja"
        self.train_jinja_path = "jinja_templates/train.py.jinja"
        # Testing template and output paths
        self.testing_template_dir = os.path.normpath(
            os.path.join(self.publicdir, "Cookiecutter", "TestingCmplx", "Template"),
        )
        self.testing_output_dir = os.path.normpath(
            os.path.join(self.publicdir, "Cookiecutter", "TestingCmplx", "Output"),
        )
