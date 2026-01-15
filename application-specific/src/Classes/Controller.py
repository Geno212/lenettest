from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout


class Controller:
    def __init__(self) -> None:
        self.res_block_built = False

    def on_generate_files_clicked(self):
        generator = self.file_read_json()
        self.generate_model()
        self.generate_train()

    def on_res_block_clicked(
        self,
        func_name,
        torch_funcs,
        on_submit_func,
        qt_layout,
        arch_dict,
    ):
        if len(self.Resarchitecture["layers"]) > 0:
            self.on_torch_func_clicked(
                func_name,
                torch_funcs,
                on_submit_func,
                qt_layout,
                arch_dict,
            )
        else:
            self.ResCreation.show()

    def on_torch_func_clicked(self, func_name, torch_funcs, on_submit_func, *args):
        # initialize QDialog and lists
        paramsWindow_QDialog = QDialog()
        paramsWindow_QDialog.setMinimumWidth(330)
        allParamsColumn_QVBoxLayout = QVBoxLayout()
        params_value_widgets = []
        params_names = []

        self.LayerDialog.create_dialogue_controller(
            torch_funcs,
            func_name,
            params_names,
            params_value_widgets,
            allParamsColumn_QVBoxLayout,
        )
        ################################

        allParamsColumn_QVBoxLayout.addWidget(QLabel())
        submitLayer_QPushButton = QPushButton("Submit Layer")

        ##############################
        submitLayer_QPushButton.clicked.connect(
            lambda submit_func=on_submit_func,
            i=func_name,
            j=params_names,
            k=params_value_widgets,
            l=paramsWindow_QDialog,
            q_layout=args[0],
            arch=args[1]: submit_func(
                i,
                j,
                k,
                l,
                q_layout,
                arch,
            ),
        )

        allParamsColumn_QVBoxLayout.addWidget(submitLayer_QPushButton)
        paramsWindow_QDialog.setLayout(allParamsColumn_QVBoxLayout)
        paramsWindow_QDialog.setWindowTitle(f"{func_name}")
        paramsWindow_QDialog.exec()
