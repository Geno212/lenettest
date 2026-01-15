from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QTextEdit,
    QTreeWidget,
    QVBoxLayout,
)

from src.utils.Singleton import Singleton


class Children(metaclass=Singleton):
    def __init__(self, ResPath, GUIPath, loader) -> None:
        self.loader = loader
        self.ResCreation = self.loader.load(
            ResPath,
            None,
        )
        self.ui = self.loader.load(
            GUIPath,
            None,
        )

        self.find_children()

    def find_children(self) -> None:
        """Locate all qt elements"""
        self.Logo_placeholder = self.ui.findChild(
            QLabel,
            "Logo_placeholder",
        )
        self.qt_layers_scroll_box = self.ui.findChild(
            QVBoxLayout,
            "Scrollbox",
        )
        self.qt_layersList_QVBoxLayout = self.ui.findChild(
            QVBoxLayout,
            "layersList_QVBoxLayout",
        )
        self.qt_inputWidth_QSpinBox = self.ui.findChild(QSpinBox, "inputWidth_QSpinBox")
        self.qt_inputHeight_QSpinBox = self.ui.findChild(
            QSpinBox,
            "inputHeight_QSpinBox",
        )
        self.qt_batchSize_QSpinBox = self.ui.findChild(QSpinBox, "batchSize_QSpinBox")
        self.qt_numEpochs_QSpinBox = self.ui.findChild(
            QSpinBox,
            "numberEpochs_QSpinBox",
        )
        self.qt_selectedOptimizer_QLineEdit = self.ui.findChild(
            QLineEdit,
            "selectedOptimizer_QLineEdit",
        )
        self.qt_selectedLossFunc_QLineEdit = self.ui.findChild(
            QLineEdit,
            "selectedLossFunc_QLineEdit",
        )
        self.qt_logdirlineedit = self.ui.findChild(
            QLineEdit,
            "logdirlineedit",
        )

        self.qt_selectedScheduler_QLineEdit = self.ui.findChild(
            QLineEdit,
            "selectedScheduler_QLineEdit",
        )
        self.qt_selectedDevice_QComboBox = self.ui.findChild(
            QComboBox,
            "deviceComboBox",
        )
        self.qt_selectedDataset_QComboBox = self.ui.findChild(
            QComboBox,
            "datasetComboBox",
        )

        # replacment

        self.qt_inputType_QSpinBox = self.ui.findChild(QSpinBox, "inputChannelsSpinBox")
        self.qt_addedLayers_QVBoxLayout = self.ui.findChild(
            QVBoxLayout,
            "addedLayers_QVBoxLayout",
        )
        self.qt_optimizersList_QVBoxLayout = self.ui.findChild(
            QVBoxLayout,
            "optimizersList_QVBoxLayout",
        )
        self.qt_lossFuncsList_QVBoxLayout = self.ui.findChild(
            QVBoxLayout,
            "lossFuncsList_QVBoxLayout",
        )
        self.qt_schedulersList_QVBoxLayout = self.ui.findChild(
            QVBoxLayout,
            "schedulersList_QVBoxLayout",
        )

        self.qt_dataset_path_QPushButton = self.ui.findChild(
            QPushButton,
            "dataset_path_QPushButton",
        )
        self.qt_dataset_path_QLineEdit = self.ui.findChild(
            QLineEdit,
            "dataset_path_QLineEdit",
        )
        self.qt_submitParams_QPushButton = self.ui.findChild(
            QPushButton,
            "submitParams_QPushButton",
        )
        self.qt_submitArch_QPushButton = self.ui.findChild(
            QPushButton,
            "submitArch_QPushButton",
        )
        self.qt_generateFiles_QPushButton = self.ui.findChild(
            QPushButton,
            "generateFiles_QPushButton",
        )
        self.qt_trainModel_QPushButton = self.ui.findChild(
            QPushButton,
            "trainPyModel_btn",
        )
        self.qt_Log_Directory_btn = self.ui.findChild(
            QPushButton,
            "Log_Directory_btn",
        )

        ####################################################
        self.qt_res_layersList_QVBoxLayout = self.ResCreation.findChild(
            QVBoxLayout,
            "res_layersList_QVBoxLayout",
        )
        self.qt_res_addedLayers_QVBoxLayout = self.ResCreation.findChild(
            QVBoxLayout,
            "res_addedLayers_QVBoxLayout",
        )
        self.qt_submitRes_QPushButton = self.ResCreation.findChild(
            QPushButton,
            "submitRes_QPushButton",
        )
        ##########################################################
        self.qt_pretrained_model_combobox = self.ui.findChild(
            QComboBox,
            "Pretrained_model_ComboBox",
        )
        self.qt_complex_model_ComboBox = self.ui.findChild(
            QComboBox,
            "Complex_model_ComboBox",
        )
        self.qt_Create_transfer_learning_model_QPushButton = self.ui.findChild(
            QPushButton,
            "Train_transfer_Model_QPushButton",
        )
        self.qt_Create_transfer_Model_QPushButton = self.ui.findChild(
            QPushButton,
            "Create_transfer_Model_QPushButton",
        )
        self.qt_Run_transfer_Model_QPushButton = self.ui.findChild(
            QPushButton,
            "Run_transfer_Model_QPushButton",
        )
        # new buttons ----------->Karim
        # to save parameters in json file
        self.qt_Create_complex_Model_QPushButton = self.ui.findChild(
            QPushButton,
            "Create_complex_Model_QPushButton",
        )
        # to generate the code
        self.qt_Generate_complex_Model_QPushButton = self.ui.findChild(
            QPushButton,
            "Generate_complex_Model_QPushButton",
        )
        self.browseWeightsButton = self.ui.findChild(
            QPushButton,
            "browseWeightsButton",
        )
        self.weightsPathLineEdit = self.ui.findChild(
            QLineEdit,
            "weightsPathLineEdit",
        )
        # testing generating manual
        self.qt_manual_generate = self.ui.findChild(
            QPushButton,
            "Generate_manual_QPushButton",
        )

        self.qt_output_tree_viewer = self.ui.findChild(
            QTreeWidget,
            "files_tree",
        )

        self.qt_output_file_viewer = self.ui.findChild(
            QTextBrowser,
            "codeSample",
        )
        # complex misc parameters
        self.qt_evalInterval = self.ui.findChild(
            QSpinBox,
            "evalIntervalSpinBox",
        )
        self.qt_warmupepochs = self.ui.findChild(
            QSpinBox,
            "warmupEpochsSpinBox",
        )
        self.qt_numWorkers = self.ui.findChild(
            QSpinBox,
            "numWorkersSpinBox",
        )
        self.qt_complexscheduler = self.ui.findChild(
            QComboBox,
            "schedulerComboBox",
        )
        self.qt_numclasses = self.ui.findChild(
            QSpinBox,
            "numclassesSpinBox",
        )
        # Violation Layer List
        self.qt_violations_text_edit = self.ui.findChild(
            QTextEdit,
            "Violation_QTextEdit",
        )
        # Actions
        self.qt_actionLoad = self.ui.findChild(
            QAction,
            "actionLoad",
        )

        # Test Tab Buttons
        self.browseImageButton = self.ui.findChild(
            QPushButton,
            "browseImageButton",
        )

        self.imagePathLineEdit = self.ui.findChild(
            QLineEdit,
            "imagePathLineEdit",
        )

        self.browseHGDCkptButton = self.ui.findChild(
            QPushButton,
            "browseHGDCkptButton",
        )

        self.hgdCkptPathLineEdit = self.ui.findChild(
            QLineEdit,
            "hgdCkptPathLineEdit",
        )

        self.testImageButton = self.ui.findChild(
            QPushButton,
            "testImageButton",
        )

        self.testWeightsPathLabel = self.ui.findChild(
            QLabel,
            "testWeightsPathLabel",
        )
        self.testWeightsPathLineEdit = self.ui.findChild(
            QLineEdit,
            "testWeightsPathLineEdit",
        )
        self.testBrowseWeightsButton = self.ui.findChild(
            QPushButton,
            "testBrowseWeightsButton",
        )

        self.TestingComplexArch_TextBox = self.ui.findChild(
            QTextEdit,
            "TestingComplexArch_TextBox",
        )
        self.classNamesPathLineEdit = self.ui.findChild(
            QLineEdit,
            "classNamesPathLineEdit",
        )
        self.browseClassNamesButton = self.ui.findChild(
            QPushButton,
            "browseClassNamesButton",
        )
