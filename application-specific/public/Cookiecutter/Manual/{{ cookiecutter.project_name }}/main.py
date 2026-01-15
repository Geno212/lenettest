import os
import shutil
import sys

from PySide6.QtCore import QProcess, QThread, Signal
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
)
from python.manual import train

basedir = os.path.dirname(__file__)
model_output = os.path.normpath(os.path.join(basedir, "./SystemC/Pt/model.pt"))
build_output = os.path.normpath(os.path.join(basedir, "./SystemC/build"))
source_output = os.path.normpath(os.path.join(basedir, "./SystemC"))


loader = QUiLoader()


class Worker(QThread):
    progressChanged = Signal(int)

    def run(self):
        train(callback=self.update_progress, logdir=self.logdir)

    def update_progress(self, value):
        self.progressChanged.emit(value)


class MainUI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.logdir = r"{{cookiecutter.log_dir}}"
        self.window = loader.load(os.path.join(basedir, "wrapper.ui"), None)
        self.window.setWindowTitle("Train & Wrap")
        self.train_btn = self.window.findChild(QPushButton, "Train")
        self.wrap_btn = self.window.findChild(QPushButton, "Wrap")
        self.progress_bar = self.window.findChild(QProgressBar, "progressBar")
        self.lineEdit = self.window.findChild(QLineEdit, "lineEdit")
        self.pushButton = self.window.findChild(QPushButton, "pushButton")
        self.lineEdit.setText(self.logdir)
        self.pushButton.clicked.connect(self.log_dir)
        self.lineEdit.textChanged.connect(self.line_dir)
        self.wrap_btn.setEnabled(False)
        self.train_btn.clicked.connect(self.train)
        self.wrap_btn.clicked.connect(self.cmake_wrap)
        self.worker = Worker()
        self.worker.progressChanged.connect(self.update_progress)
        self.worker.finished.connect(self.on_train_finished)
        self.window.show()

    def line_dir(self):
        self.logdir = self.lineEdit.text()

    def log_dir(self):
        path_output = QFileDialog.getExistingDirectory(
            None,
            "Pick a folder to save the output",
        )
        if path_output:
            self.lineEdit.setText(path_output)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_train_finished(self):
        self.progress_bar.setValue(100)
        self.wrap_btn.setEnabled(True)

    def train(self):
        self.worker.logdir = self.logdir
        try:
            self.worker.start()
        except:
            pass

    def cmake_wrap(self):
        self.count = 0
        path = os.path.isfile(model_output)
        if path == False:
            self.wrap_btn.setEnabled(False)
        else:
            self.test = QProcess()
            self.test.readyReadStandardOutput.connect(self.handle_stdout)
            self.test.readyReadStandardError.connect(self.handle_stderr)
            self.test.stateChanged.connect(self.handle_state)

            if os.path.exists(build_output):
                shutil.rmtree(build_output)
            # print( build_output)
            self.test.start("cmake", ["-S", source_output, "-B", build_output])

    def handle_stderr(self):
        result = bytes(self.test.readAllStandardError()).decode("utf8")
        print(result)

    def handle_stdout(self):
        result = bytes(self.test.readAllStandardOutput()).decode("utf8")
        print(result)

    def handle_state(self, state):
        self.count += 1
        if self.count == 1:
            self.test.start("cmake", ["--build", build_output, "--clean-first"])
        else:
            print("done")


def main():
    app = QApplication(sys.argv)
    window = MainUI()
    app.exec()


if __name__ == "__main__":
    main()
