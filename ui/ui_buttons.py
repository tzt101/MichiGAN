from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import sys


class GUIButton(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self.pushButton = QPushButton()
        self.pushButton.setObjectName("Edit")
        self.pushButton_2 = QPushButton()
        self.pushButton_2.setObjectName("open_ref_img")
        self.pushButton_3 = QPushButton()
        self.pushButton_3.setObjectName("open_tag_img")
        self.pushButton_4 = QPushButton()
        self.pushButton_4.setObjectName("open_mask")
        self.pushButton_5 = QPushButton()
        self.pushButton_5.setObjectName("open_orient")
        self.pushButton_6 = QPushButton()
        self.pushButton_6.setObjectName("hair")
        self.pushButton_7 = QPushButton()
        self.pushButton_7.setObjectName("background")
        self.pushButton_8 = QPushButton()
        self.pushButton_8.setObjectName("mask_+")
        self.pushButton_9 = QPushButton()
        self.pushButton_9.setObjectName("mask_-")
        self.pushButton_10 = QPushButton()
        self.pushButton_10.setObjectName("brush")
        self.pushButton_11 = QPushButton()
        self.pushButton_11.setObjectName("background")
        self.pushButton_12 = QPushButton()
        self.pushButton_12.setObjectName("orient_+")
        self.pushButton_13 = QPushButton()
        self.pushButton_13.setObjectName("orient_-")

        self.pushButton.clicked.connect(self.edit)
        self.pushButton_2.clicked.connect(self.open_ref)
        self.pushButton_3.clicked.connect(open_tag)
        self.pushButton_4.clicked.connect(Form.open_mask)
        self.pushButton_5.clicked.connect(Form.open_orient)
        self.pushButton_6.clicked.connect(Form.save_img)
        self.pushButton_7.clicked.connect(Form.bg_mode)
        self.pushButton_8.clicked.connect(Form.hair_mode)
        self.pushButton_9.clicked.connect(Form.clear)
        self.pushButton_10.clicked.connect(Form.increase)
        self.pushButton_11.clicked.connect(Form.decrease)


        self.grid1 = QGridLayout()
        self.setLayout(self.grid)
        self.resize(60, 100)
        self.grid1.addWidget(self.pushButton_2, 0,0,1,1)
        self.grid1.addWidget(self.pushButton_3, 1, 0, 1, 1)
        self.grid1.addWidget(self.pushButton_4, 1, 0, 1, 1)
        self.grid1.addWidget(self.pushButton_5, 1, 1, 1, 1)


if __name__=="__main__":
    app=QApplication(sys.argv)
    win=GUIButton()
    win.show()
    sys.exit(app.exec_())