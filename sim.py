import irsim
import numpy as np
import random

import shapely
from irsim.lib.handler.geometry_handler import GeometryFactory

class SIM_ENV:
    def __init__(self, world_file="robot_world.yaml", save_ani=False):
        self.env = irsim.make(world_file, save_ani=save_ani)
        self.scan_history = []  # Keep track of scan history for more context
        self.prev_min_distance = float('inf')  # For tracking improvement in obstacle avoidance
        self.cumulative_distance = 0  # Track total distance moved for reward

    def step(self, lin_velocity=0.0, steering_angle=0.0):
        # print("Step with linear velocity:", lin_velocity, "and steering angle:", steering_angle)
        # Apply actions (note: using steering_angle instead of angular velocity for Ackermann)
        self.env.step(action_id=0, action=np.array([[lin_velocity], [steering_angle]]))
        self.env.render()

        # Get laser scan
        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]
        self.scan_history.append(min(latest_scan))  # Track minimum distance to obstacles
        if len(self.scan_history) > 10:  # Keep history limited
            self.scan_history.pop(0)

        # Get robot state
        robot_state = self.env.get_robot_state()
        
        # Check for collision
        collision = self.env.robot.collision
        
        # Calculate distance moved since last step (for rewarding forward progress)
        if hasattr(self, 'last_position'):
            dx = robot_state[0].item() - self.last_position[0]
            dy = robot_state[1].item() - self.last_position[1]
            step_distance = np.sqrt(dx**2 + dy**2)
            self.cumulative_distance += step_distance
        else:
            step_distance = 0
        
        # Store current position for next step
        self.last_position = [robot_state[0].item(), robot_state[1].item()]
        
        # Get heading direction vector
        heading_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        
        # Calculate current minimum distance to obstacles
        min_distance = min(latest_scan)
        min_distance_improvement = self.prev_min_distance - min_distance
        self.prev_min_distance = min_distance
        
        # Actions for reference in reward calculation
        action = [lin_velocity, steering_angle]
        
        # Calculate reward
        reward = self.get_reward(collision, action, latest_scan, step_distance, min_distance_improvement)

        return latest_scan, min_distance, heading_vector[0], heading_vector[1], collision, action, reward

    def reset(self, robot_state=None, random_obstacles=True):
        # Reset robot to a random position if not specified
        if robot_state is None:
            robot_state = [[random.uniform(5, 45)], [random.uniform(5, 45)], [random.uniform(-3.14, 3.14)], [0]]

        self.env.robot.set_state(
            state=np.array(robot_state),
            init=True,
        )

        # Random obstacle positions
        if random_obstacles:
            self.env.random_obstacle_position(
                range_low=[5, 5, -3.14],
                range_high=[45, 45, 3.14],
                ids=[i + 1 for i in range(20)],  # Adjusted for 20 obstacles per your YAML
                non_overlapping=True,
            )

        self.env.reset()
        
        # Reset tracking variables
        self.scan_history = []
        self.prev_min_distance = float('inf')
        self.cumulative_distance = 0
        
        # Initial action (stationary)
        action = [0.0, 0.0]
        
        # Take first step to get initial observations
        latest_scan, min_distance, heading_x, heading_y, collision, action, reward = self.step(
            lin_velocity=action[0], steering_angle=action[1]
        )
        
        # Set initial position for distance calculation
        self.last_position = [self.env.get_robot_state()[0].item(), self.env.get_robot_state()[1].item()]
        
        return latest_scan, min_distance, heading_x, heading_y, collision, action, reward

    @staticmethod
    def get_reward(collision, action, laser_scan, distance_moved, min_distance_improvement):
        """
        Reward function focused on obstacle avoidance and exploration:
        - Large negative reward for collisions
        - Positive reward for moving forward (higher speed)
        - Penalty for large steering angles (to promote smoother paths)
        - Penalty for being too close to obstacles
        - Reward for improving distance to obstacles (moving away from them)
        """
        if collision:
            return -100.0  # Large penalty for collision
        
        # Forward movement reward
        movement_reward = 2.0 * action[0] * distance_moved
        
        # Steering penalty (less steering is better)
        steering_penalty = -0.5 * abs(action[1])
        
        # Obstacle proximity penalty (larger penalty when closer to obstacles)
        min_scan = min(laser_scan)
        proximity_penalty = 0
        if min_scan < 3.0:  # Only apply when obstacles are close
            proximity_penalty = -2.0 * (3.0 - min_scan) if min_scan > 0.5 else -5.0
        
        # Reward for increasing distance to obstacles
        avoidance_reward = 1.0 * min_distance_improvement if min_distance_improvement > 0 else 0
        
        return movement_reward + steering_penalty + proximity_penalty + avoidance_reward