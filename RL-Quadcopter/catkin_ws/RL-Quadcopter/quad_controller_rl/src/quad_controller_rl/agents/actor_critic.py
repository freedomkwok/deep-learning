"""Policy search agent."""

import numpy as np
import tensorflow as tf
import tflearn
from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.stackMemory import Memory
import quad_controller_rl.util as util
import pandas as pd
import os

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_low, action_space, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_low = action_low
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient, name='actor_gradient')
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 8, name='actor_fully_connected_8')
        net = tflearn.layers.normalization.batch_normalization(net, name='actor_batch_normalization_1')
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 6, name='actor_fully_connected_6')
        net = tflearn.layers.normalization.batch_normalization(net, name='actor_batch_normalization_2')
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 4, name='actor_fully_connected_4')
        net = tflearn.layers.normalization.batch_normalization(net, name='actor_batch_normalization_2')
        net = tflearn.activations.relu(net)

        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)

        out = tflearn.fully_connected(
            net, self.a_dim, activation='sigmoid', weights_init=w_init, name='actor_out')

        scaled_out = tf.add(tf.multiply(out, self.action_space), self.action_low)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 4, name='critic_fully_connected_4')
        net = tflearn.layers.normalization.batch_normalization(net, name='critic_batch_normalization_1')
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 3, name='critic_fully_connected_3')
        net = tflearn.layers.normalization.batch_normalization(net, name='critic_batch_normalization_1')
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 3, name='t1_fully_connected')
        t2 = tflearn.fully_connected(action, 3, name='t2_fully_connected')

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init, name='critic_out')
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.mu = mu if mu is not None else np.zeros(1)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(1) * self.mu
        self.reset()

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def reset(self):
        self.state = self.mu

class GredientPolicySearch(BaseAgent):
    """Sample agent that searches for optimal policy randomly."""

    def __init__(self, task):
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space
        self.state_size = 1# np.prod(self.task.observation_space.shape)
        self.state_range = (self.task.observation_space.high - self.task.observation_space.low)[2:3]
        self.action_size = 1# np.prod(self.task.action_space.shape)
        self.action_range = (self.task.action_space.high - self.task.action_space.low)[2:3]

        # Policy parameters
        # self.w = np.random.normal(
        #     size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
        #     scale=(self.action_range / (2 * self.state_size)).reshape(1,
        #                                                               -1))  # start producing actions in a decent range

        self.stats_filename = os.path.join(
            util.get_param('out'),
            "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_size))

        # Episode variables
        self.reset_episode_vars()
        self.actor_learning_rate = 0.0001
        self.tau = 0.99
        self.mini_batch_size = 64
        self.buffer_size = 1000000
        self.critic_learning_rate = 0.001
        self.gamma = 0.9
        self.episode_num = 0;

        self.memory = Memory(self.buffer_size)
        self.checkpoint = '/home/robond/catkin_ws/src/RL-Quadcopter/quad_controller_rl/src/checkpoints/actor_critic.ckpt'
        self.sess = tf.Session()
        self.savor = tf.train.Saver()
        self.actor = ActorNetwork(self.sess, self.state_size, self.action_size, self.task.action_space.low[2:3], \
                                  self.action_range, self.actor_learning_rate, self.tau, self.mini_batch_size)


        self.critic = CriticNetwork(self.sess, self.state_size, self.action_size, self.critic_learning_rate, self.tau,
                               self.gamma,
                               self.actor.get_num_trainable_vars())

        self.sess.run(tf.initialize_all_variables())

        self.actor.update_target_network()
        self.critic.update_target_network()

    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
                        header=not os.path.isfile(self.stats_filename))  # write header first time only

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0

    def postprocess_action(self, action):
        complete_action = np.zeros(self.task.action_space.shape)  # shape: (6,)
        complete_action[2:3] = action  # linear force only
        return complete_action

    def step(self, state, reward, done, pi):
        # Transform state vector
        old_height = state[2:3]
        state = (old_height - self.task.observation_space.low[2:3]) / self.state_range  # scale to [0.0, 1.0]
        state = state.reshape(1, -1)  # convert to row vector

        ep_reward = 0
        ep_ave_max_q = 0

        # Choose an action
        action = self.act(state)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done, pi)
            self.total_reward += reward
            self.count += 1

        if len(self.memory) > self.mini_batch_size:
            s_batch, a_batch, r_batch, s2_batch, d_batch, pi_batch = self.memory.sample(self.mini_batch_size)

            target_q = self.critic.predict_target(  # critic create feedfoward
                s2_batch, self.actor.predict_target(s2_batch))  # actor create feedforward

            y_i = []
            for k in range(self.mini_batch_size):
                if d_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.gamma * target_q[k])  ## what is gamma

            # Update the critic given the targets
            predicted_q_value, _ = self.critic.train(
                s_batch, a_batch, np.reshape(y_i, (self.mini_batch_size, 1)))

            ep_ave_max_q += np.amax(predicted_q_value)  # getting max value out

            # Update the actor policy using the sampled gradient
            a_outs = self.actor.predict(s_batch)
            grads = self.critic.action_gradients(s_batch, a_outs)
            self.actor.train(s_batch, grads[0])

            # Update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()

            # Learn, if at end of episode
        if done:
            print('reward', self.total_reward)
            print("height", old_height)
            self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1
            self.reset_episode_vars()

        self.last_state = state
        self.last_action = action

        final_action = self.actor.predict(state)
        return self.postprocess_action(final_action)

    def act(self, state):
        # Choose action based on given state and policy
        # action = np.dot(state, self.w)  # simple linear policy
        # print(state.shape, action.shape)
        # return action

        action = self.actor.predict(state)
        nosie = self.actor_noise.sample()
        return action + nosie
