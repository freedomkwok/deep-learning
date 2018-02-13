"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
import tensorflow as tf
import tflearn
from quad_controller_rl.tasks.base_task import BaseTask

class Hover(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))

        self.max_duration = 3.0  # secs
        self.target_z = 10.0  # target height (z position) to reach for successful takeoff
        self.sess = tf.session()

    def reset(self):
        # Nothing to reset; just return initial condition
        print("reset")
        return Pose(
                position=Point(0.0, 0.0, 9),  # drop off from a slight random height np.random.normal(0.5, 0.1)
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):

        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        state = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # Compute reward / penalty and check if this episode is complete
        done = False
        pi = abs(self.target_z - pose.position.z); ## for better sorting
        reward = -min(abs(pi), 30.0)  # reward = zero for matching target z, -ve as you go farther, upto -20
        if pose.position.z >= self.target_z:  # agent has crossed the target height
            reward += 20.0  # bonus reward
            done = True
        if pose.position.x == 0.0 or pose.position.y == 0.0:  # agent has crossed the target height
            reward += 10.0  # bonus reward
        elif timestamp > self.max_duration:  # agent has run out of time
            reward -= 10.0  # extra penalty
            done = True

        action = self.getTrainAction()

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done

    def getTrainAction(self):

    def initialNetwork(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400, name='actor_fully_connected_400')
        net = tflearn.layers.normalization.batch_normalization(net, name='actor_batch_normalization_1')
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300, name='actor_fully_connected_300')
        net = tflearn.layers.normalization.batch_normalization(net, name='actor_batch_normalization_2')
        net = tflearn.activations.relu(net)

        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)

        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init, name='actor_out')

        scaled_out = tf.divide(tf.subtract(out, self.action_low), self.action_space)

