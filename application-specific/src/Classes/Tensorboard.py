from PySide6.QtCore import QProcess, QTimer, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QProgressBar, QVBoxLayout


class TensorView:
    def __init__(self) -> None:
        # get the tensorboard object from the ui
        self.tensorQt = self.ui.findChild(QVBoxLayout, "TensorView")
        # create and add the webview object qt
        self.tensorWeb = QWebEngineView()
        self.tensorQt.addWidget(self.tensorWeb)
        self.tensorWeb.setUrl(QUrl("http://localhost:6006/"))
        self.tensorWeb.hide()
        self.Children.qt_logdirlineedit.textChanged.connect(self.Tensorboard_run)
        self.reload_timer = QTimer()
        self.progress_timer = QTimer()
        self.setup_timers()
        # Initialize progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }

            QProgressBar::chunk {
                background-color: orange;
                width: 20px;
            }
        """,
        )
        self.tensorQt.addWidget(self.progress_bar)
        self.Children.qt_logdirlineedit.setText(self.SysPath.log_path)

    def Tensorboard_run(self):
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.Tensorboard_Process = QProcess()
        if self.Children.qt_logdirlineedit.text() != "":
            self.Tensorboard_Process.start(
                "tensorboard",
                ["--logdir", self.Children.qt_logdirlineedit.text()],
            )
            self.progress_timer.start()

    def setup_timers(self):
        self.reload_timer.setSingleShot(True)
        self.reload_timer.start(10000)  # Reload once after 10 seconds

        self.progress_timer.setInterval(
            250,
        )  # Set interval to 500 milliseconds (0.5 seconds)
        self.progress_timer.timeout.connect(self.increment_progress_bar)

    def increment_progress_bar(self):
        current_value = self.progress_bar.value()

        if current_value < 100:
            self.progress_bar.setValue(current_value + 1)
            if current_value > 95:
                self.tensorWeb.reload()
        else:
            self.progress_timer.stop()
            self.progress_bar.hide()
            self.tensorWeb.show()
