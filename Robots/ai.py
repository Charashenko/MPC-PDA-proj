#! /usr/bin/python
# -*- coding: utf-8 -*-

from robocode.Objects.robot import Robot
import sys, os
import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.drivers import py_driver
from tf_agents.metrics import py_metrics
from tf_agents.policies import random_py_policy

sys.path.append(os.getcwd() + "/model/")

from model.utils import DataProcessor
from model.net import Net
from model.game_env import GameEnv

import random


class AI(Robot):
    def init(self):
        self.name = "abcdefghjk"[random.randint(0, 9)].upper()
        self.ai = True
        self.robot_dead = False
        self.data_processor = DataProcessor(self.getMapSize())

        self.event_buffer = {
            "onhitbyrobot": False,
            "onhitwall": False,
            "onrobothit": False,
            "onhitbybullet": False,
            "onbullethit": False,
            "onbulletmiss": False,
            "ontargetspotted": False,
        }

        self.setColor(250, 10, 20)
        self.setGunColor(0, 0, 0)
        self.setRadarColor(200, 100, 0)
        self.setBulletsColor(100, 150, 250)
        self.radarVisible(True)
        self.lockRadar("gun")
        self.setRadarField("thin")
        self.inTheCorner = False

    def set_nn(self, nn):
        self.nn = nn

    def run(self):
        pass

    def sensors(self):
        # self.nn.predict()
        pass

    def onHitWall(self):
        self.event_buffer["onhitwall"] = True

    def onRobotHit(self, robotId, robotName):
        self.event_buffer["onrobothit"] = True

    def onHitByRobot(self, robotId, robotName):
        self.event_buffer["onhitbyrobot"] = True

    def onHitByBullet(self, bulletBotId, bulletBotName, bulletPower):
        self.event_buffer["onhitbybullet"] = True

    def onBulletHit(self, botId, bulletId):
        self.event_buffer["onbullethit"] = True

    def onBulletMiss(self, bulletId):
        self.event_buffer["onbulletmiss"] = True

    def onTargetSpotted(self, botId, botName, botPos):
        self.event_buffer["ontargetspotted"] = True

    def onRobotDeath(self):
        self.robot_dead = True

    def get_num_of_opps(self):
        return len(self.getEnemiesLeft())

    def get_state(self):
        pos = self.getPosition()
        state = [
            self.data_processor.pos_x(pos.x()),
            self.data_processor.pos_y(pos.y()),
            self.data_processor.health(self.getHealth()),
            self.data_processor.gun_heading(self.getGunHeading()),
            self.data_processor.tank_heading(self.getHeading()),
            self.data_processor.radar_heading(self.getRadarHeading()),
            self.data_processor.notif(self.event_buffer["onhitbyrobot"]),
            self.data_processor.notif(self.event_buffer["onhitwall"]),
            self.data_processor.notif(self.event_buffer["onrobothit"]),
            self.data_processor.notif(self.event_buffer["onhitbybullet"]),
            self.data_processor.notif(self.event_buffer["onbullethit"]),
            self.data_processor.notif(self.event_buffer["onbulletmiss"]),
            self.data_processor.notif(self.event_buffer["ontargetspotted"]),
        ]
        self.clear_event_buffer()
        return state

    def action_exec(self, action):
        action = action.tolist()
        if type(action) is list:
            action = action[0]
        actions = {
            0: self.move,
            1: self.move,
            2: self.turn,
            3: self.turn,
            4: self.radarTurn,
            5: self.radarTurn,
            6: self.gunTurn,
            7: self.gunTurn,
            8: self.fire,
        }
        if action == 8:
            actions[action](5)
            return
        action_heading = 1 if action % 2 == 0 else -1
        actions[action](action_heading)

    def clear_event_buffer(self):
        for event in self.event_buffer.keys():
            self.event_buffer[event] = False

    def death_ack(self):
        self.robot_dead = False

    def __str__(self):
        return self.name
