#! /usr/bin/python
#-*- coding: utf-8 -*-

from robocode.Objects.robot import Robot
import math

class AI(Robot):
    def init(self):
        self.setColor(250, 10, 20)
        self.setGunColor(0, 0, 0)
        self.setRadarColor(200, 100, 0)
        self.setBulletsColor(100, 150, 250)
        self.radarVisible(True)
        size = self.getMapSize()
        self.lockRadar("gun")
        self.setRadarField("thin")
        self.inTheCorner = False
    
    def run(self):
        pass        

    def onHitWall(self):
        pass

    def sensors(self): 
        pass
        
    def onRobotHit(self, robotId, robotName):
        pass
        
    def onHitByRobot(self, robotId, robotName):
        pass

    def onHitByBullet(self, bulletBotId, bulletBotName, bulletPower):
        pass
        
    def onBulletHit(self, botId, bulletId):
        pass
        
    def onBulletMiss(self, bulletId):
        pass
        
    def onRobotDeath(self):
        pass
        
    def onTargetSpotted(self, botId, botName, botPos):
        pass
           
