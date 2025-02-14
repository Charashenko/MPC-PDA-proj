import tensorflow as tf
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.policies import py_tf_eager_policy
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_rnn_network
from tf_agents.networks import q_network
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import q_policy
from tf_agents.networks import sequential
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.trajectories import from_transition
from tf_agents.trajectories import Trajectory

import random
import numpy as np
import pickle
import os
import time

INPUT_SIZE = 13
OUTPUT_SIZE = 9
LAYER_PARAMS = (
    100,
    50,
    25,
)
LEARNING_RATE = 0.3
REPLAY_BUFFER_CAPACITY = 1000
COLLECT_STEPS_PER_ITERATION = 200
BATCH_SIZE = 1
CHECKPOINT_DIR = "checkpoints"
REWARDS_DIR = "rewards"
WEIGHTS_FILE = "weights"


class Net:
    def __init__(self, env, number):
        self.number = number
        self.env = env
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.train_step_counter = tf.Variable(0)
        self.rewards = []
        self.episode_reward = 0
        self.num_of_steps_in_episode = 0
        self.losses = []

        try:
            os.mkdir(CHECKPOINT_DIR)
        except:
            pass
        try:
            os.mkdir(REWARDS_DIR)
        except:
            pass

        self.model = q_network.QNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            fc_layer_params=LAYER_PARAMS,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03,
                maxval=0.03,
                seed=random.randint(0, 1000),
            ),
        )

        self.agent = dqn_agent.DqnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            q_network=self.model,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter,
            n_step_update=COLLECT_STEPS_PER_ITERATION-1,
        )

        self.load_model()
        self.agent.initialize()
        self._conf_replay_buffer()
        self._init_policy()

    def _conf_replay_buffer(self):
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.training_data_spec,
            batch_size=BATCH_SIZE,
            max_length=REPLAY_BUFFER_CAPACITY,
        )

    def _init_policy(self):
        self.agent.train_step_counter.assign(0)
        self.time_step = self.env.reset()
        self.policy = q_policy.QPolicy(
            self.agent.time_step_spec,
            self.agent.action_spec,
            q_network=self.model,
        )

    def predict(self):
        action_step = self.policy.action(self.time_step)
        new_time_step = self.env.step(action_step.action)
        trajectory = self.collect_step(self.time_step, action_step, new_time_step)
        self.time_step = new_time_step
        if self.time_step.is_last():
            self.episode_reward += self.time_step.reward
            #self.num_of_steps_in_episode += 1
            self.train()
        elif self.time_step.is_first():
            self.num_of_steps_in_episode = 0
            self.episode_reward = 0
        else:
            self.replay_buffer.add_batch(trajectory)
            self.episode_reward += self.time_step.reward
            self.num_of_steps_in_episode += 1

    def train(self):
        dataset = self.replay_buffer.as_dataset(
            sample_batch_size=1,
            num_steps=COLLECT_STEPS_PER_ITERATION,
            single_deterministic_pass=False,
        )
        iterator = iter(dataset)
        # print(dataset)
        # try:
        #     loss = self.agent.train(experience=trajectories)
        # except:
        #     print('failed to train')
        #     pass
        for _ in range(int(self.num_of_steps_in_episode/COLLECT_STEPS_PER_ITERATION)):
            try:
                trajectories, _ = next(iterator)
                loss = self.agent.train(experience=trajectories)
            except:
                print('failed to train')
                pass
            # self.losses.append(loss)
        self.save_model()
        self.save_avg_reward()
        #self.replay_buffer.clear()

    def save_avg_reward(self):
        self.episode_reward = tf.keras.backend.get_value(self.episode_reward).tolist()[
            0
        ]
        if self.num_of_steps_in_episode == 0:
            val = 0
        else:
            val = self.episode_reward / self.num_of_steps_in_episode
        try:
            with open(f"{REWARDS_DIR}/reward_{self.number}", "a") as f:
                f.write(str(val) + "\n")
        except:
            with open(f"{REWARDS_DIR}/reward_{self.number}", "x") as f:
                f.write(str(val) + "\n")

    def save_model(self):
        try:
            pickle.dump(
                self.model.get_weights(),
                open(f"{CHECKPOINT_DIR}/{WEIGHTS_FILE}_{self.number}", "wb"),
            )
        except:
            pickle.dump(
                self.model.get_weights(),
                open(f"{CHECKPOINT_DIR}/{WEIGHTS_FILE}_{self.number}", "xb"),
            )

    def load_model(self):
        try:
            self.model.set_weights(
                pickle.load(
                    open(f"{CHECKPOINT_DIR}/{WEIGHTS_FILE}_{self.number}", "rb")
                )
            )
        except:
            print(
                f"Can't load weight file {CHECKPOINT_DIR}/{WEIGHTS_FILE}_{self.number}"
            )

    def set_bot(self, bot):
        self.bot = bot

    def collect_step(self, time_step, action_step, next_time_step):
        trajectory = from_transition(time_step, action_step, next_time_step)
        action = self._keras_backend(trajectory.action, 2)
        discount = self._keras_backend(trajectory.discount, 1)
        next_step_type = self._keras_backend(trajectory.next_step_type, 1)
        observation = self._keras_backend(trajectory.observation, 2)
        policy_info = trajectory.policy_info
        reward = self._keras_backend(trajectory.reward, 1)
        step_type = self._keras_backend(trajectory.step_type, 2)

        action = tf.constant(action)
        discount = tf.constant(discount)
        next_step_type = tf.constant(next_step_type)
        observation = tf.constant(observation)
        reward = tf.constant(reward)
        step_type = tf.constant(step_type)

        # print(f"{self.bot}{action}")

        values = Trajectory(
            step_type,
            observation,
            action,
            policy_info,
            next_step_type,
            reward,
            discount,
        )

        values_batched = tf.nest.map_structure(lambda t: tf.stack([t]), values)

        return values_batched

    def _keras_backend(self, tensor, iterations):
        data = tf.keras.backend.get_value(tensor)
        if iterations == 0:
            return data
        for i in range(0, iterations):
            if type(data) is np.ndarray:
                data = data[0]
        return data
