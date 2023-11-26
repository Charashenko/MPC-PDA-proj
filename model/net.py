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

# import reverb

INPUT_SIZE = 13
OUTPUT_SIZE = 9
LAYER_PARAMS = (
    100,
    50,
)
LEARNING_RATE = 0.01


class Net:
    def __init__(self, env):
        self.env = env
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.train_step_counter = tf.Variable(0)

        self.tf_env = tf_py_environment.TFPyEnvironment(self.env)
        action_tensor_spec = tensor_spec.from_spec(env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        # dense_layers = [self._dense_layer(num_units) for num_units in LAYER_PARAMS]
        # q_values_layer = tf.keras.layers.Dense(
        #     num_actions,
        #     activation=None,
        #     kernel_initializer=tf.keras.initializers.RandomUniform(
        #         minval=-0.03, maxval=0.03
        #     ),
        #     bias_initializer=tf.keras.initializers.Constant(-0.2),
        #     input_shape=(OUTPUT_SIZE,),
        #     input_dim=1,
        # )
        # input_layer = tf.keras.layers.Dense(
        #     INPUT_SIZE,
        #     activation=tf.keras.activations.relu,
        #     kernel_initializer=tf.keras.initializers.VarianceScaling(
        #         scale=2.0, mode="fan_in", distribution="truncated_normal"
        #     ),
        # )
        # self.model = tf.keras.models.Sequential(dense_layers + [q_values_layer])

        self.model = q_rnn_network.QRnnNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            lstm_size=(INPUT_SIZE,),
        )

        # self.model = self.create_q_model()

        self.agent = dqn_agent.DqnAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            q_network=self.model,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter,
        )

        self.agent.initialize()
        # self._conf_reverb()
        self._init_policy()
        print(self.model.summary())

    def create_q_model(self):
        # Network defined by the Deepmind paper

        model = sequential.Sequential(
            input_spec=self.tf_env.observation_spec(),
            layers=[
                tf.keras.layers.Dense(
                    INPUT_SIZE,
                    input_shape=(INPUT_SIZE,),
                    input_dim=1,
                ),
                tf.keras.layers.Dense(
                    INPUT_SIZE,
                    activation="relu",
                    input_dim=1,
                ),
                tf.keras.layers.Dense(
                    OUTPUT_SIZE,
                    input_dim=1,
                ),
            ],
        )

        return model

    def _dense_layer(self, num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode="fan_in", distribution="truncated_normal"
            ),
            input_dim=1,
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
        self.policy_state = self.policy.get_initial_state(self.tf_env.batch_size)

    def predict(self):
        print(self.time_step)

        action_step = self.policy.action(self.time_step, self.policy_state)
        self.time_step = self.tf_env.step(action_step.action)
        self.policy_state = self.policy_step.state

        # action_step = self.policy.action(self.time_step)
        # self.time_step = self.env.step(action_step.action)
