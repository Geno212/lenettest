"""This file is main.py for a PySide6 application that trains a YOLOX model."""

import os
import shutil
import subprocess
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

basedir = os.path.dirname(__file__)
model_output = os.path.normpath(os.path.join(basedir, "./SystemC/Pt/model.pt"))
build_output = os.path.normpath(os.path.join(basedir, "./SystemC/build"))
source_output = os.path.normpath(os.path.join(basedir, "./SystemC"))


loader = QUiLoader()


class Worker(QThread):
    """Worker thread to run the training process in the background."""

    progressChanged = Signal(int)

    def run(self):
        """Run the training process."""
        # train(callback=self.update_progress, logdir=self.logdir)
        # Get the base directory dynamically
        basedir = os.path.dirname(__file__)

        # Dynamically construct file paths based on the base directory
        train_script = os.path.join(basedir, r"python\train.py")
        yolox_custom = os.path.join(basedir, r"python\yolox_custom.py")
        weight_file = r"{{cookiecutter.pretrained_weights}}"

        print("Training started...")

        # Build the command dynamically
        command = [
            "start",
            "cmd",
            "/k",
            "python",
            train_script,
            "-f",
            yolox_custom,
            "-d",
            "1",
            "-b",
            "2",
            "--fp16",
            "-o",
            "-c",
            weight_file,
        ]

        # Open a new cmd window and run the command
        process = subprocess.Popen(command, shell=True)

        # Read the output and error streams
        stdout, stderr = process.communicate()

        print("Output:\n", stdout)
        print("Errors:\n", stderr)

    def update_progress(self, value):
        """Update the progress bar value."""
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
        self.pushButton.clicked.connect(self.log_dir)
        self.lineEdit.setText(self.logdir)
        self.lineEdit.textChanged.connect(self.line_dir)
        self.wrap_btn.setEnabled(False)
        self.train_btn.clicked.connect(self.train)
        self.wrap_btn.clicked.connect(self.cmake_wrap)

        self.worker = Worker()
        self.worker.progressChanged.connect(self.update_progress)
        self.worker.finished.connect(self.on_train_finished)

        self.window.show()

    def line_dir(self):
        """Update the log directory when the text in the line edit changes."""
        self.logdir = self.lineEdit.text()

    def log_dir(self):
        """Open a dialog to select the log directory."""
        path_output = QFileDialog.getExistingDirectory(
            None,
            "Pick a folder to save the output",
        )
        if path_output:
            self.lineEdit.setText(path_output)

    def update_progress(self, value):
        """Update the progress bar value."""
        self.progress_bar.setValue(value)

    def on_train_finished(self):
        """Handle the completion of the training process."""
        self.progress_bar.setValue(0)
        self.wrap_btn.setEnabled(False)

    def train(self):
        """Start the training process in a separate thread."""
        self.worker.logdir = self.logdir
        try:
            self.worker.start()
            self.window.hide()
        except:
            pass

    def cmake_wrap(self):
        """Run the CMake build process."""
        self.count = 0
        path = os.path.isfile(model_output)
        if path is False:
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
        """Handle the standard error output from the CMake process."""
        result = bytes(self.test.readAllStandardError()).decode("utf8")
        print(result)

    def handle_stdout(self):
        """Handle the standard output from the CMake process."""
        result = bytes(self.test.readAllStandardOutput()).decode("utf8")
        print(result)

    def handle_state(self, state):
        """Handle the state changes of the CMake process."""
        self.count += 1
        if self.count == 1:
            self.test.start("cmake", ["--build", build_output, "--clean-first"])
        else:
            print("done")


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    window = MainUI()
    app.exec()


if __name__ == "__main__":
    main()
