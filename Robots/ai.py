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
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import q_policy

sys.path.append(os.getcwd() + "/model/")

from model.utils import DataProcessor
from model.net import Net
from model.game_env import GameEnv


class AI(Robot):
    def init(self):
        self.robot_dead = False
        self.data_processor = DataProcessor(self.getMapSize())

        self.env = GameEnv(
            state_getter=self.get_state,
            num_of_opponents_getter=self.get_num_of_opps,
            on_robot_death_getter=self.get_robot_death,
        )

        self.nn = Net(self.env)
        self.time_step = self.nn.time_step

        self.buffer = []

        self.setColor(250, 10, 20)
        self.setGunColor(0, 0, 0)
        self.setRadarColor(200, 100, 0)
        self.setBulletsColor(100, 150, 250)
        self.radarVisible(True)
        self.lockRadar("gun")
        self.setRadarField("thin")
        self.inTheCorner = False

    def run(self):
        pass

    def onHitWall(self):
        pass

    def sensors(self):
        self.tick()

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
        self.robot_dead = True

    def onTargetSpotted(self, botId, botName, botPos):
        pass

    def get_num_of_opps(self):
        return len(self.getEnemiesLeft())

    def get_state(
        self,
        on_hit_by_bot=0,
        on_hit_wall=0,
        on_robot_hit=0,
        on_hit_by_bullet=0,
        on_bullet_hit=0,
        on_bullet_miss=0,
    ):
        pos = self.getPosition()
        state = [
            self.data_processor.pos_x(pos.x()),
            self.data_processor.pos_y(pos.y()),
            self.data_processor.health(self.getHealth()),
            self.data_processor.gun_heading(self.getGunHeading()),
            self.data_processor.tank_heading(self.getHeading()),
            self.data_processor.radar_heading(self.getRadarHeading()),
            on_hit_by_bot,
            on_hit_wall,
            on_robot_hit,
            on_hit_by_bullet,
            on_bullet_hit,
            on_bullet_miss,
        ]
        return tf.convert_to_tensor(np.array(state))

    def get_robot_death(self):
        return self.robot_dead

    def tick(self):
        # self.nn.collect_episode(self.env, self.agent.collect_policy)

        # iterator = iter(self.nn.replay_buffer.as_dataset(sample_batch_size=1))
        # trajectories, _ = next(iterator)
        # train_loss = self.nn.agent

        # step = self.nn.agent.train_step_counter.numpy()

        # print("step = {0}: loss = {1}".format(step, train_loss.loss))
        metric = py_metrics.AverageReturnMetric()
        observers = [metric]

        print(self.nn.agent.time_step_spec.observation)
        batch_size = 1
        observation = tf.ones(self.nn.agent.time_step_spec.observation)
        time_steps = ts.restart(observation, batch_size=batch_size)

        my_q_policy = q_policy.QPolicy(
            self.nn.agent.time_step_spec,
            self.nn.agent.action_spec,
            q_network=self.nn.model,
        )
        action_step = my_q_policy.action(time_steps)
        distribution_step = my_q_policy.distribution(time_steps)

        print("Action:")
        print(action_step.action)

        print("Action distribution:")
        print(distribution_step.action)

        self.driver = py_driver.PyDriver(
            self.env,
            my_q_policy,
            observers,
            max_steps=10,
            max_episodes=1,
        )

        print(metric.result())
