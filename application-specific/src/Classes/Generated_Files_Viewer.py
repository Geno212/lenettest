import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidgetItem


class GeneratedFilesViewer:
    def __init__(self) -> None:
        self.Children.qt_output_tree_viewer.itemClicked.connect(self.on_file_clicked)

    def show_files(self, project_path):
        for i in range(self.Children.qt_output_tree_viewer.topLevelItemCount()):
            top_level_item = self.Children.qt_output_tree_viewer.topLevelItem(i)
            if top_level_item.data(0, Qt.ItemDataRole.UserRole) == project_path:
                self.Children.qt_output_tree_viewer.takeTopLevelItem(i)
                break
        root = QTreeWidgetItem([os.path.basename(project_path)])
        root.setData(0, Qt.ItemDataRole.UserRole, project_path)
        self.Children.qt_output_tree_viewer.addTopLevelItem(root)
        self.create_children(root, project_path)

    def create_children(self, root, project_path):
        for item in os.listdir(project_path):
            item_path = os.path.join(project_path, item)
            child_item = QTreeWidgetItem([item])
            child_item.setData(0, Qt.ItemDataRole.UserRole, item_path)
            root.addChild(child_item)
            if os.path.isdir(item_path):
                self.create_children(child_item, item_path)

    def on_file_clicked(self, file, column):
        file_path = file.data(0, Qt.ItemDataRole.UserRole)
        if os.path.isfile(file_path):
            try:
                with open(file_path) as file:
                    content = file.read()
                    self.Children.qt_output_file_viewer.setPlainText(content)
            except:
                pass
