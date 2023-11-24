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

# import reverb

INPUT_SIZE = 12
NUM_ACTIONS = 5
LAYER_PARAMS = (INPUT_SIZE,)
LEARNING_RATE = 0.01


class Net:
    def __init__(self, env):
        self.env = env
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.train_step_counter = tf.Variable(0)

        self.model = q_network.QNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
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
        # self._conf_reverb()
        self._init_policy()

    def _dense_layer(self, num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode="fan_in", distribution="truncated_normal"
            ),
        )

    def _conf_reverb(self):
        table_name = "uniform_table"
        replay_buffer_signature = tensor_spec.from_spec(self.agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

        table = reverb.Table(
            table_name,
            max_size=replay_buffer_max_length,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=replay_buffer_signature,
        )

        self.reverb_server = reverb.Server([table])

        self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            self.agent.collect_data_spec,
            table_name=table_name,
            sequence_length=2,
            local_server=self.reverb_server,
        )

        self.rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            self.replay_buffer.py_client, table_name, sequence_length=2
        )

    def _init_policy(self):
        self.agent.train_step_counter.assign(0)
        self.time_step = self.env.reset()
        self.policy = q_policy.QPolicy(
            self.agent.time_step_spec,
            self.agent.action_spec,
            q_network=self.model,
        )
