from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import sys
import os

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")

        # for graph
        self.graphicsView = QGraphicsView(Form) # tag_mask
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView.setFixedSize(512, 512)
        self.graphicsView_2 = QGraphicsView(Form) # orient map
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.graphicsView_2.setFixedSize(512, 512)
        self.graphicsView_3 = QGraphicsView(Form) # result
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.graphicsView_3.setFixedSize(512, 512)
        self.graphicsView_4 = QGraphicsView(Form) # tag image
        self.graphicsView_4.setFixedSize(256, 256)
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.graphicsView_5 = QGraphicsView(Form) # ref image
        self.graphicsView_5.setFixedSize(256, 256)
        self.graphicsView_5.setObjectName("graphicsView_5")

        # self.grid1 = QGridLayout()
        # self.grid1.addWidget(self.graphicsView, 0,0,1,1)
        # self.grid1.addWidget(self.graphicsView_2, 0, 1, 1, 1)
        # self.grid1.addWidget(self.graphicsView_3, 0, 3, 1, 1)

        # self.grid0 = QHBoxLayout()
        # self.grid0.addWidget(self.graphicsView_5)
        # self.grid0.addWidget(self.graphicsView_4)

        # for Buttons
        self.button_W = 107
        self.button_H = 37

        self.pushButton0 = QPushButton(Form)
        self.pushButton0.setObjectName("Save")
        self.pushButton0.setFixedSize(self.button_W, self.button_H)
        self.pushButton = QPushButton(Form)
        self.pushButton.setObjectName("Edit")
        self.pushButton.setFixedSize(self.button_W, self.button_H)
        self.pushButton_2 = QPushButton(Form)
        self.pushButton_2.setFixedSize(self.button_W, self.button_H)
        self.pushButton_2.setObjectName("open_ref_img")
        self.pushButton_3 = QPushButton(Form)
        self.pushButton_3.setFixedSize(self.button_W, self.button_H)
        self.pushButton_3.setObjectName("open_tag_img")
        self.pushButton_4 = QPushButton(Form)
        self.pushButton_4.setFixedSize(self.button_W, self.button_H)
        self.pushButton_4.setObjectName("open_mask")
        self.pushButton_5 = QPushButton(Form)
        self.pushButton_5.setFixedSize(self.button_W, self.button_H)
        self.pushButton_5.setObjectName("open_orient")
        self.pushButton_6 = QPushButton(Form)
        self.pushButton_6.setFixedSize(self.button_W, self.button_H)
        self.pushButton_6.setObjectName("hair")
        self.pushButton_7 = QPushButton(Form)
        self.pushButton_7.setFixedSize(self.button_W, self.button_H)
        self.pushButton_7.setObjectName("background")
        self.pushButton_8 = QPushButton(Form)
        self.pushButton_8.setFixedSize(self.button_W, self.button_H)
        self.pushButton_8.setObjectName("mask_+")
        self.pushButton_9 = QPushButton(Form)
        self.pushButton_9.setFixedSize(self.button_W, self.button_H)
        self.pushButton_9.setObjectName("mask_-")
        self.pushButton_10 = QPushButton(Form)
        self.pushButton_10.setFixedSize(self.button_W, self.button_H)
        self.pushButton_10.setObjectName("clear")
        self.pushButton_11 = QPushButton(Form)
        self.pushButton_11.setFixedSize(self.button_W, self.button_H)
        self.pushButton_11.setObjectName("brush")
        # self.pushButton_12 = QPushButton(Form)
        # self.pushButton_12.setFixedSize(self.button_W, self.button_H)
        # self.pushButton_12.setObjectName("background")
        self.pushButton_13 = QPushButton(Form)
        self.pushButton_13.setFixedSize(self.button_W, self.button_H)
        self.pushButton_13.setObjectName("orient_+")
        self.pushButton_14 = QPushButton(Form)
        self.pushButton_14.setFixedSize(self.button_W, self.button_H)
        self.pushButton_14.setObjectName("orient_-")
        # self.pushButton_15 = QPushButton(Form)
        # self.pushButton_15.setFixedSize(self.button_W, self.button_H)
        # self.pushButton_15.setObjectName("erase")

        _translate = QCoreApplication.translate
        self.pushButton0.setText(_translate("Form", "Save"))
        self.pushButton.setText(_translate("Form", "Edit"))
        self.pushButton_2.setText(_translate("Form", "Open Ref"))
        self.pushButton_3.setText(_translate("Form", "Open Tag"))
        self.pushButton_4.setText(_translate("Form", "Open Mask"))
        self.pushButton_5.setText(_translate("Form", "Open Orient"))
        self.pushButton_6.setText(_translate("Form", "Hair"))
        self.pushButton_7.setText(_translate("Form", "BackGround"))
        self.pushButton_8.setText(_translate("Form", "+"))
        self.pushButton_9.setText(_translate("Form", "-"))
        self.pushButton_10.setText(_translate("Form", "Clear"))
        self.pushButton_11.setText(_translate("Form", "Brush"))
        # self.pushButton_12.setText(_translate("Form", "Orient Edit"))
        self.pushButton_13.setText(_translate("Form", "+"))
        self.pushButton_14.setText(_translate("Form", "-"))
        # self.pushButton_15.setText(_translate("Form", "Erase"))

        self.pushButton0.clicked.connect(Form.save)
        self.pushButton.clicked.connect(Form.edit)
        self.pushButton_2.clicked.connect(Form.open_ref)
        self.pushButton_3.clicked.connect(Form.open_tag)
        self.pushButton_4.clicked.connect(Form.open_mask)
        self.pushButton_5.clicked.connect(Form.open_orient)
        self.pushButton_6.clicked.connect(Form.hair_mode)
        self.pushButton_7.clicked.connect(Form.bg_mode)
        self.pushButton_8.clicked.connect(Form.increase)
        self.pushButton_9.clicked.connect(Form.decrease)
        self.pushButton_10.clicked.connect(Form.clear)
        self.pushButton_11.clicked.connect(Form.orient_mode)
        # self.pushButton_12.clicked.connect(Form.orient_edit)
        self.pushButton_13.clicked.connect(Form.orient_increase)
        self.pushButton_14.clicked.connect(Form.orient_decrease)
        # self.pushButton_15.clicked.connect(Form.erase_mode)

        self.grid2 = QGridLayout()
        self.grid2.addWidget(self.pushButton0,0,1,1,1)
        self.grid2.addWidget(self.pushButton,0,0,1,1)
        self.grid2.addWidget(self.pushButton_2,1,0,1,1)
        self.grid2.addWidget(self.pushButton_3,1,1,1,1)
        # self.grid2.addWidget(self.pushButton_4)
        # self.grid2.addWidget(self.pushButton_5)

        self.grid3 = QGridLayout()
        self.grid3.addWidget(self.pushButton_4,0,0,1,1)
        self.grid3.addWidget(self.pushButton_6,0,1,1,1)
        self.grid3.addWidget(self.pushButton_7,1,1,1,1)
        self.grid3.addWidget(self.pushButton_8,0,2,1,1)
        self.grid3.addWidget(self.pushButton_9,1,2,1,1)
        self.grid3.addWidget(self.pushButton_10,1,0,1,1)

        self.grid4 = QGridLayout()
        self.grid4.addWidget(self.pushButton_5,0,0,1,1)
        self.grid4.addWidget(self.pushButton_11,0,1,1,1)
        # self.grid4.addWidget(self.pushButton_15)
        self.grid4.addWidget(self.pushButton_13,1,0,1,1)
        self.grid4.addWidget(self.pushButton_14,1,1,1,1)
        # self.grid4.addWidget(self.pushButton_12)


        # for radioButton
        self.clickButtion1 = QRadioButton(Form)
        self.clickButtion1.setText('Reference')
        self.clickButtion1.setChecked(True)
        # self.clickButtion1.clicked.connect(Form.selectM)
        self.clickButtion2 = QRadioButton(Form)
        self.clickButtion2.setText('Edited')
        # self.clickButtion2.clicked.connect(Form.selectM)

        self.grid6 = QHBoxLayout()
        self.grid6_1 = QGridLayout()
        self.grid6_1.addWidget(self.clickButtion1,0,0,1,1)
        self.grid6_1.addWidget(self.clickButtion2,1,0,1,1)
        self.grid6.addLayout(self.AddLayout(self.grid6_1, 'Hair Mask'))

        self.clickButtion3 = QRadioButton(Form)
        self.clickButtion3.setText('Reference')
        self.clickButtion3.setChecked(True)
        # self.clickButtion3.clicked.connect(Form.selectO)
        self.clickButtion4 = QRadioButton(Form)
        self.clickButtion4.setText('Edited')
        # self.clickButtion4.clicked.connect(Form.selectO)
        self.grid6_2 = QGridLayout()
        self.grid6_2.addWidget(self.clickButtion3,0,0,1,1)
        self.grid6_2.addWidget(self.clickButtion4,1,0,1,1)

        self.grid6.addLayout(self.AddLayout(self.grid6_2, 'Hair Orientation'))

        # for Layout setting
        mainLayout = QVBoxLayout()
        Form.setLayout(mainLayout)
        Form.resize(1616, 808)

        subLayout = QHBoxLayout()
        subLayout.addLayout(self.AddWidgt(self.graphicsView, 'Hair Mask'))
        subLayout.addLayout(self.AddWidgt(self.graphicsView_2, 'Hair Orientation'))
        subLayout.addLayout(self.AddWidgt(self.graphicsView_3, 'Result'))

        subLayout2_1 = QHBoxLayout()
        # subLayout2_1.addLayout(self.AddLayout(self.grid2, 'Main Buttons'))
        subLayout2_1.addLayout(self.AddLayout(self.grid3, 'Mask Edit'))
        subLayout2_2 = QVBoxLayout()
        subLayout2_2.addLayout(self.AddLayout(self.grid6, 'State'))
        subLayout2_2.addLayout(subLayout2_1)

        subLayout2_3 = QVBoxLayout()
        subLayout2_3.addLayout(self.AddLayout(self.grid2, 'Main Buttons'))
        subLayout2_3.addLayout(self.AddLayout(self.grid4, 'Orient Edit'))

        subLayout2 = QHBoxLayout()
        subLayout2.addLayout(self.AddWidgt(self.graphicsView_4, 'Tagert Image'))
        subLayout2.addLayout(self.AddWidgt(self.graphicsView_5, 'Reference Image'))
        subLayout2.addLayout(subLayout2_2)
        subLayout2.addLayout(subLayout2_3)

        mainLayout.addLayout(subLayout)
        mainLayout.addLayout(subLayout2)

    def setButtonColor(self, button, path, H, W):
        pixmap = QPixmap(path)
        fitPixmap = pixmap.scaled(W, H, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        icon = QIcon(fitPixmap)
        button.setIcon(icon)
        # button.setIconSize(W,H)

    def AddLayout(self, widget, title=''):
        widgetLayout = QVBoxLayout()
        widgetBox = QGroupBox()
        if title != '':
            widgetBox.setTitle(title)
        widgetBox.setAlignment(Qt.AlignCenter)
        widgetBox.setLayout(widget)
        widgetLayout.addWidget(widgetBox)

        return widgetLayout


    def AddWidgt(self, widget, title):
        widgetLayout = QVBoxLayout()
        widgetBox = QGroupBox()
        widgetBox.setTitle(title)
        widgetBox.setAlignment(Qt.AlignCenter)
        vbox_t = QGridLayout()
        vbox_t.addWidget(widget,0,0,1,1)
        widgetBox.setLayout(vbox_t)
        widgetLayout.addWidget(widgetBox)

        return widgetLayout


if __name__=="__main__":
    app=QApplication(sys.argv)
    Form = QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

