from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import random

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.specs import from_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from utils import get_action_mapping

OBSERVATION_SPEC_SIZE = 13
ACTION_SPEC_SIZE = 8

RANDOM_FIRE_PROBABILITY = 0.2
RANDOM_ACTION_PROBABILITY = 0.2
# DEBUG
OPPS = 10
# -----


class GameEnv(py_environment.PyEnvironment):
    def __init__(self, bot, init_num_of_opponents):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=ACTION_SPEC_SIZE,
            name="action",
        )

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(OBSERVATION_SPEC_SIZE,),
            dtype=np.float64,
            minimum=-1,
            maximum=1,
            name="observation",
        )
        self._state = self._reset_state()
        self._episode_ended = False
        self._bot = bot
        self._init_num_of_opponents = init_num_of_opponents
        self.current_round = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset_state(self):
        return np.zeros((1, OBSERVATION_SPEC_SIZE))

    def _reset(self):
        self._state = self._reset_state()
        self._episode_ended = False
        return ts.restart(self._state, 1)

    def _step(self, action):
        # DEBUG
        global OPPS
        # -----
        reward = 0
        if self._episode_ended:
            return self.reset()

        self._state = self._bot.get_state()

        if self._bot.get_num_of_opps() == 0 or self._bot.robot_dead:
            self._episode_ended = True
        else:
            if random.randint(0, 100) / 100 < RANDOM_ACTION_PROBABILITY:
                action = random.randint(0, 8)
            self._bot.action_exec(action)

        if self._episode_ended:
            self._bot.death_ack()
            reward = (self._init_num_of_opponents - self._bot.get_num_of_opps()) * 10
            return ts.termination(
                np.array([self._state], dtype=np.float64),
                reward,
            )
        else:
            # DEBUG
            OPPS -= 1
            # -----
            reward = self._calculate_reward(self._state[-7:], action)
            return ts.transition(
                np.array([self._state], dtype=np.float64),
                reward=reward,
                discount=1.0,
            )

    def _calculate_reward(self, events, action):
        if type(action) is np.ndarray:
            action = action.tolist()
        if type(action) is list:
            action = action[0]
        reward = 0
        action_mappings = get_action_mapping()
        # on hit by robot
        if events[0] == 1:
            reward -= 2
            if action_mappings.get(action) not in ["move", "turn"]:
                reward -= 2
        # on hit wall
        if events[1] == 1:
            reward -= 5
            if action_mappings.get(action) not in ["turn"]:
                reward -= 5
            else:
                reward += 5
        # on robot hit
        if events[2] == 1:
            reward += 5
            if action_mappings.get(action) not in ["fire", "move"]:
                reward -= 3
            else:
                reward += 3
        # on hit by bullet
        if events[3] == 1:
            reward -= 3
            if action_mappings.get(action) not in ["move"]:
                reward -= 5
        # on bullet hit
        if events[4] == 1:
            reward += 50
            if action_mappings.get(action) not in ["fire"]:
                reward -= 10
            else:
                reward += 50
        # on bullet miss
        if events[5] == 1:
            reward -= 10
            if action_mappings.get(action) not in ["radarTurn", "gunTurn"]:
                reward -= 5
        # on target spotted
        if events[6] == 1:
            if random.randint(0, 100) / 100 < RANDOM_FIRE_PROBABILITY:
                self._bot.fire(3)
                reward += 50
            reward += 10
            if action_mappings.get(action) in ["fire"]:
                reward += 100
            else:
                reward -= 10
        # firing when no detection
        if action_mappings.get(action) == "fire":
            if events[6] == -1:
                reward -= 10
            else:
                reward += 50
        # move
        if action_mappings.get(action) == "move":
            reward += 1

        if action_mappings.get(action) == "radarTurn":
            reward += 1

        return reward

    def set_bot_instance(self, bot):
        self._bot = bot


# DEBUG
def state_getter():
    arr = np.zeros([OBSERVATION_SPEC_SIZE])
    return arr


# DEBUG
def num_of_opponents_getter():
    global OPPS
    return OPPS


# DEBUG
def on_robot_death_getter():
    return False


# DEBUG
# environment = GameEnv(state_getter, num_of_opponents_getter, on_robot_death_getter)
# utils.validate_py_environment(environment, episodes=5)
# -----
