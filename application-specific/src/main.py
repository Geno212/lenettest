import sys

from PySide6 import QtCore
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication

from src.Classes.Initializer import Initializer
from src.paths.SystemPaths import SystemPaths
from src.Tests.StaticAnalysis import StaticAnalysis
from src.utils.AutoExtraction import AutoExtraction


def main():
    app = QApplication(sys.argv)
    window = MainUI()
    window.load_style()
    app.exec()


class MainUI(QtCore.QObject, Initializer):
    def __init__(self):
        self.debug = False
        self.SysPath = SystemPaths()
        self.style_file_path = self.SysPath.css_path

        self.AutoExtraction = AutoExtraction(self.debug)
        self.StaticAnalysis = StaticAnalysis(
            self.SysPath.warning_rules_path,
            self.debug,
        )

        self.loader = QUiLoader()
        (
            self.LAYERS,
            self.LOSSFUNC,
            self.OPTIMIZERS,
            self.SCHEDULERS,
            self.PRETRAINED_MODELS,
            self.LAYERS_WITHOUT_RES,
            self.DATASETS,
            self.COMPLEX_ARCH_MODELS,
        ) = self.AutoExtraction.extracted_data()

        Initializer.__init__(self)

        self.ui.setWindowIcon(QIcon(self.SysPath.siemens_icon))
        self.Children.Logo_placeholder.setPixmap(QPixmap(self.SysPath.siemens_logo))
        self.ui.show()

    def load_style(self):
        with open(self.style_file_path) as file:
            style = file.read()
            self.ui.setStyleSheet(style)


if __name__ == "__main__":
    main()
