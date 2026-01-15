from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QPushButton

from src.utils.Singleton import Singleton


class QTButtons(metaclass=Singleton):
    def qbutton_layer_manager(
        self,
        icon_path,
        qt_layout,
        border_QFrame,
        function,
        *args,
    ):
        """Creates a QPushButton with a specified icon, maximum width, and click event handler.

        Parameters
        ----------
            icon_path (str): Path to the icon image to be displayed on the button.
            qt_layout (QLayout): The Qt layout to be passed to the callback function.
            border_QFrame (QFrame): The QFrame to be passed to the callback function.
            function (callable): The function to be called when the button is clicked.
            *args: Additional arguments to be passed to the callback function.

        Returns
        -------
            QPushButton: The configured QPushButton instance.

        """
        Button = QPushButton()
        Button.setMaximumWidth(30)
        Button.setIcon(QIcon(icon_path))
        Button.clicked.connect(
            lambda func=function, i=border_QFrame, q_layout=qt_layout: func(
                i,
                q_layout,
                *args,
            ),
        )

        return Button
