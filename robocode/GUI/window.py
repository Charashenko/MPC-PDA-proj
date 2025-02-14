# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""

import os, pickle, sys

from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QHeaderView, QTableWidgetItem
from PyQt5.QtCore import pyqtSlot, QTimer, QSizeF

from ..Objects.graph import Graph
from .Ui_window import Ui_MainWindow
from .battle import Battle
from ..Objects.robot import Robot
from .RobotInfo import RobotInfo
from ..Objects.statistic import statistic

from tf_agents.environments import tf_py_environment

sys.path.append(os.getcwd() + "/model/")

from model.utils import DataProcessor
from model.net import Net
from model.game_env import GameEnv

import matplotlib
import matplotlib.pyplot as plt


ROUND_LIMIT = 1500


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """

    def __init__(self, parent=None):
        """
        Constructor
        """
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.countBattle = 0
        self.timer = QTimer()
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.hide()
        self.nns = []
        self.current_round = 0

    @pyqtSlot()
    def on_pushButton_clicked(self):
        """
        Start the last battle
        """

        if os.path.exists(os.getcwd() + "/.datas/lastArena"):
            with open(os.getcwd() + "/.datas/lastArena", "rb") as file:
                unpickler = pickle.Unpickler(file)
                dico = unpickler.load()
            file.close()
        else:
            print("No last arena found.")

        self.setUpBattle(dico["width"], dico["height"], dico["botList"])

    def setUpBattle(self, width, height, botList):
        self.countBattle = 0
        self.tableWidget.clearContents()
        self.tableWidget.hide()
        self.graphicsView.show()
        self.width = width
        self.height = height
        self.botList = botList
        self.statisticDico = {}
        num_of_opps = len(botList) - 1
        instances = []
        idx = 0
        for bot in botList:
            self.statisticDico[self.repres(bot)] = statistic()
            robot = bot(QSizeF(width, height), None, str(bot))
            instances.append(robot)
            # Setup networks, agents and environments for each bot
            if "AI" in str(bot):
                idx += 1
                self.setup_nn(robot, num_of_opps, idx)
        self.botList = instances
        self.startBattle(self.countBattle)

    def startBattle(self, battle_count):
        try:
            self.timer.timeout.disconnect(self.tick)
            del self.timer
            del self.scene
            del self.sceneMenu
        except:
            pass

        self.timer = QTimer()
        self.countBattle += 1
        self.sceneMenu = QGraphicsScene()
        self.graphicsView_2.setScene(self.sceneMenu)
        self.scene = Graph(self, self.width, self.height, battle_count)
        for bot in self.botList:
            bot.set_parent(self.scene)
        self.graphicsView.setScene(self.scene)
        self.scene.AddRobots(self.botList)
        self.timer.timeout.connect(self.tick)
        self.timer.start(self.horizontalSlider.value())
        self.resizeEvent()
        self.current_round = 0

    def tick(self):
        for nn in self.nns:
            if nn.bot in self.scene.aliveBots:
                nn.predict()
        if self.current_round >= ROUND_LIMIT:
            print("round limit reached")
            self.scene.battleFinished()
            self.current_round = 0
            return
        self.scene.advance()
        self.current_round += 1

    @pyqtSlot(int)
    def on_horizontalSlider_valueChanged(self, value):
        """
        Slot documentation goes here.
        """
        print(value)
        self.timer.setInterval(value)

    @pyqtSlot()
    def on_actionNew_triggered(self):
        """
        Battle Menu
        """
        self.battleMenu = Battle(self)
        self.battleMenu.show()

    @pyqtSlot()
    def on_actionNew_2_triggered(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        print("Not Implemented Yet")

    @pyqtSlot()
    def on_actionOpen_triggered(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        print("Not Implemented Yet")

    def resizeEvent(self, evt=None):
        try:
            self.graphicsView.fitInView(self.scene.sceneRect(), 4)
        except:
            pass

    def addRobotInfo(self, robot):
        self.sceneMenu.setSceneRect(0, 0, 170, 800)
        rb = RobotInfo()
        rb.pushButton.setText(str(robot))
        rb.progressBar.setValue(100)
        rb.robot = robot
        robot.info = rb
        robot.progressBar = rb.progressBar
        robot.icon = rb.toolButton
        robot.icon2 = rb.toolButton_2
        p = self.sceneMenu.addWidget(rb)
        l = len(self.scene.aliveBots)
        self.sceneMenu.setSceneRect(0, 0, 170, l * 80)
        p.setPos(0, (l - 1) * 80)

    def chooseAction(self):
        if self.countBattle >= self.spinBox.value():
            "Menu Statistic"
            self.graphicsView.hide()
            self.tableWidget.show()
            self.tableWidget.setRowCount(len(self.statisticDico))
            i = 0
            for key, value in self.statisticDico.items():
                self.tableWidget.setItem(i, 0, QTableWidgetItem(key))
                self.tableWidget.setItem(i, 1, QTableWidgetItem(str(value.first)))
                self.tableWidget.setItem(i, 2, QTableWidgetItem(str(value.second)))
                self.tableWidget.setItem(i, 3, QTableWidgetItem(str(value.third)))
                self.tableWidget.setItem(i, 4, QTableWidgetItem(str(value.points)))

                i += 1

            self.timer.stop()
            #for nn in self.nns:
            #print(f"{nn.bot}: {nn.rewards}")
            self.countBattle = 0
        else:
            print(f"{self.countBattle + 1}. battle started")
            self.startBattle(self.countBattle)

    def repres(self, bot):
        repres = repr(bot).split(".")
        return repres[1].replace("'>", "")

    def setup_nn(self, bot, num_of_opps, idx):
        env = tf_py_environment.TFPyEnvironment(
            GameEnv(bot=bot, init_num_of_opponents=num_of_opps)
        )
        nn = Net(env, idx)
        self.nns.append(nn)
        bot.set_nn(nn)
        nn.set_bot(bot)
