#! /usr/bin/python
#-*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd() + "robocode/GUI")
sys.path.append(os.getcwd() + "robocode/Objects")
sys.path.append(os.getcwd() + "robocode/robotImages")
sys.path.append(os.getcwd() + "/Robots")
from robocode.GUI.window import MainWindow
from PyQt5.QtWidgets import QApplication


if __name__ == "__main__":

   app = QApplication(sys.argv)
   app.setApplicationName("Python-Robocode")
   myapp = MainWindow()
   myapp.show()
   sys.exit(app.exec_())
