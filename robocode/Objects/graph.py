#! /usr/bin/python
# -*- coding: utf-8 -*-

import time, os, random

from PyQt5.QtWidgets import QGraphicsScene, QMessageBox, QGraphicsRectItem
from PyQt5.QtGui import QPixmap, QColor, QBrush
from PyQt5.QtCore import QPointF, QRectF

from .robot import Robot
from ..GUI.outPrint import outPrint


class Graph(QGraphicsScene):
    def __init__(self, parent, width, height, battle_count):
        QGraphicsScene.__init__(self, parent)
        self.setSceneRect(0, 0, width, height)
        self.Parent = parent

        # self.Parent.graphicsView.centerOn(250, 250)
        self.width = width
        self.height = height
        self.grid = self.getGrid()
        self.setTiles()
        self.battle_count = battle_count

    def AddRobots(self, botList):
        self.aliveBots = []
        self.deadBots = []
        try:
            posList = random.sample(self.grid, len(botList))
            for bot in botList:
                try:
                    # robot = botList[i](self.sceneRect().size(), self, str(botList[i]))
                    # if "AI" in str(botList[i]):
                    #     # nns[i].env.set_bot_instance(robot)
                    #     robot.set_nn(nns[i])
                    self.aliveBots.append(bot)
                    self.addItem(bot)
                    bot.setPos(posList.pop())
                    self.Parent.addRobotInfo(bot)
                except Exception as e:
                    print("graph: Problem with bot file '{}': {}".format(bot, str(e)))

            self.Parent.battleMenu.close()
        except ValueError:
            QMessageBox.about(self.Parent, "Alert", "Too many Bots for the map's size!")
        except AttributeError:
            pass

    def battleFinished(self):
        print(f"{self.battle_count+1}. battle terminated")
        try:
            for robot in self.aliveBots:
                if robot.ai:
                    robot.robot_dead = True
                    print(f"override: {robot}")
            self.deadBots.append(robot)
            self.removeItem(robot)
        except IndexError:
            pass
        j = len(self.deadBots)

        for i in range(j):
            print("N° {}:{}".format(j - i, self.deadBots[i]))
            self.deadBots[i].reset_health()
            if j - i == 1:  # first place
                self.Parent.statisticDico[repr(self.deadBots[i])].first += 1
            if j - i == 2:  # 2nd place
                self.Parent.statisticDico[repr(self.deadBots[i])].second += 1
            if j - i == 3:  # 3rd place
                self.Parent.statisticDico[repr(self.deadBots[i])].third += 1

            self.Parent.statisticDico[repr(self.deadBots[i])].points += i

        for robot in self.deadBots:
            if robot.ai:
                if robot.robot_dead:
                    robot.nn.predict()

        self.Parent.chooseAction()

    def setTiles(self):
        # background
        brush = QBrush()
        pix = QPixmap(os.getcwd() + "/robocode/robotImages/tile.png")
        brush.setTexture(pix)
        brush.setStyle(24)
        self.setBackgroundBrush(brush)

        # wall
        # left
        left = QGraphicsRectItem()
        pix = QPixmap(os.getcwd() + "/robocode/robotImages/tileVert.png")
        left.setRect(QRectF(0, 0, pix.width(), self.height))
        brush.setTexture(pix)
        brush.setStyle(24)
        left.setBrush(brush)
        left.name = "left"
        self.addItem(left)
        # right
        right = QGraphicsRectItem()
        right.setRect(self.width - pix.width(), 0, pix.width(), self.height)
        right.setBrush(brush)
        right.name = "right"
        self.addItem(right)
        # top
        top = QGraphicsRectItem()
        pix = QPixmap(os.getcwd() + "/robocode/robotImages/tileHori.png")
        top.setRect(QRectF(0, 0, self.width, pix.height()))
        brush.setTexture(pix)
        brush.setStyle(24)
        top.setBrush(brush)
        top.name = "top"
        self.addItem(top)
        # bottom
        bottom = QGraphicsRectItem()
        bottom.setRect(0, self.height - pix.height(), self.width, pix.height())
        bottom.setBrush(brush)
        bottom.name = "bottom"
        self.addItem(bottom)

    def getGrid(self):
        w = int(self.width / 80)
        h = int(self.height / 80)
        l = []
        for i in range(w):
            for j in range(h):
                l.append(QPointF((i + 0.5) * 80, (j + 0.5) * 80))
        return l
