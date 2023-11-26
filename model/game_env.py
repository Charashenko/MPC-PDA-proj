from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

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

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset_state(self):
        return np.zeros([OBSERVATION_SPEC_SIZE])
        # return np.expand_dims(np.zeros(([OBSERVATION_SPEC_SIZE])), axis=-1)

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
            self._bot.action_exec(action)

        if self._episode_ended:
            reward = self._init_num_of_opponents - self._bot.get_num_of_opps()
            return ts.termination(
                np.array([self._state], dtype=np.float64),
                # np.expand_dims(np.array([self._state], dtype=np.float64), axis=-1),
                reward,
            )
        else:
            # DEBUG
            OPPS -= 1
            # -----
            reward = self._calculate_reward(self._state[-7:], action.tolist())
            return ts.transition(
                np.array([self._state], dtype=np.float64),
                # np.expand_dims(np.array([self._state], dtype=np.float64), axis=-1),
                reward=reward,
                discount=1.0,
            )

    def _calculate_reward(self, events, action):
        reward = 0
        action_mappings = get_action_mapping()
        # on hit by robot
        if events[0] == 1:
            reward -= 1
            if action_mappings.get(action) not in ["move", "turn"]:
                reward -= 1
        # on hit wall
        if events[1] == 1:
            reward -= 1
            if action_mappings.get(action) not in ["turn"]:
                reward -= 1
        # on robot hit
        if events[2] == 1:
            reward += 1
            if action_mappings.get(action) not in ["fire", "move"]:
                reward -= 1
            else:
                reward += 1
        # on hit by bullet
        if events[3] == 1:
            reward -= 1
            if action_mappings.get(action) not in ["move", "turn"]:
                reward -= 1
        # on bullet hit
        if events[4] == 1:
            reward += 1
            if action_mappings.get(action) not in ["radarTurn", "gunTurn", "fire"]:
                reward -= 1
            else:
                reward += 1
        # on bullet miss
        if events[5] == 1:
            reward -= 3
            if action_mappings.get(action) not in ["radarTurn", "gunTurn", "fire"]:
                reward -= 1
        # on target spotted
        if events[6] == 1:
            reward += 1
            if action_mappings.get(action) in ["fire"]:
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
