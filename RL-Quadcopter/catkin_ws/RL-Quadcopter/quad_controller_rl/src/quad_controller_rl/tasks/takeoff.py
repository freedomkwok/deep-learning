"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask


class Takeoff(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 25.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))

        self.target_z = 10.0  # target height (z position) to reach for successful takeoff

        self.max_duration = 5.0  # secs
        self.max_error_position = 9  # distance units
        self.target_position = np.array([0.0, 0.0, 10.0])  # target position to hover at
        self.weight_position = 0.99
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # target orientation quaternion (upright)
        self.weight_orientation = 0.3
        self.target_velocity = np.array([0.0, 0.0, 4.0])  # target velocity (ideally should stay in place)
        self.weight_velocity = 0.01

    def reset(self):
        p = self.target_position + np.random.normal(0.5, 0.1, size=3)
        return Pose(
                position=Point(*p),  # drop off from a slight random height np.random.normal(0.5, 0.1)
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0)
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        state = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        done = False
        position = np.sum(np.abs(self.target_position - np.array([pose.position.x, pose.position.y,
                                                                       pose.position.z]))) ## for better sorting
        volocity = np.sum(np.abs(self.target_velocity - np.array([angular_velocity.x, angular_velocity.y,
                                                                  angular_velocity.z])))  ## for better sorting
        reward = -min(position, self.max_error_position) * self.weight_position - (min(volocity, 5) * self.weight_velocity) # reward = zero for matching target z, -ve as you go farther, upto -20

        if position >= self.max_error_position:  # agent has crossed the target height
            reward -= 50.0  # bonus reward
            done = True
        elif timestamp > self.max_duration:  # agent has run out of time
            reward += 50.0  # extra penalty
            done = True

        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        if done:
            print("action", action[2:3])

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
