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

import random

# import reverb

INPUT_SIZE = 13
OUTPUT_SIZE = 9
LAYER_PARAMS = (
    20,
    10,
)
LEARNING_RATE = 0.01
REPLAY_BUFFER_CAPACITY = 1000
COLLECT_STEPS_PER_ITERATION = 10
BATCH_SIZE = 1


class Net:
    def __init__(self, env):
        self.env = env
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.train_step_counter = tf.Variable(0)

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
        )

        self.agent.initialize()
        self._conf_replay_buffer()
        self._init_policy()

    def _conf_replay_buffer(self):
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=BATCH_SIZE,
            max_length=REPLAY_BUFFER_CAPACITY,
        )

        self.replay_observer = [self.replay_buffer.add_batch]

        self.collect_op = dynamic_step_driver.DynamicStepDriver(
            self.env,
            self.agent.collect_policy,
            observers=self.replay_observer,
            num_steps=COLLECT_STEPS_PER_ITERATION,
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
        self.time_step = self.env.step(action_step.action)

    def train(self):
        dataset = self.replay_buffer.as_dataset(sample_batch_size=1, num_steps=10)
        print(dataset)
