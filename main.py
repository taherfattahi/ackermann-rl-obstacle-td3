from models.TD3.TD3 import TD3

import torch
import numpy as np
from sim import SIM_ENV
from utils import get_buffer


def main(args=None):
    """Main training function for Ackermann robot obstacle avoidance"""
    # Configuration
    action_dim = 2  # [linear_velocity, steering_angle]
    max_action = 1.0  # Maximum absolute value of actions (will be scaled appropriately)
    state_dim = 23  # 20 lidar scans + min_distance + heading_x + heading_y
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training parameters
    nr_eval_episodes = 10
    max_epochs = 100  # Increased for better learning
    epoch = 0
    episodes_per_epoch = 50
    episode = 0
    train_every_n = 1  # Train every episode for faster learning
    training_iterations = 100  # More iterations per training cycle
    batch_size = 128  # Larger batch size
    max_steps = 500  # Longer episodes
    steps = 0
    load_saved_buffer = False
    pretrain = False
    pretraining_iterations = 10
    save_every = 5  # Save more frequently

    print("Initializing TD3 model...")
    # Initialize model
    model = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=False,
    )

    print("Initializing simulation environment...")
    # Initialize environment
    sim = SIM_ENV()
    
    print("Setting up replay buffer...")
    # Initialize replay buffer
    replay_buffer = get_buffer(
        model,
        sim,
        load_saved_buffer,
        pretrain,
        pretraining_iterations,
        training_iterations,
        batch_size,
    )

    # Get initial state
    latest_scan, min_distance, heading_x, heading_y, collision, actions, reward = sim.step(
        lin_velocity=0.0, steering_angle=0.0
    )

    print("Starting training...")
    
    while epoch < max_epochs:
        # Prepare state representation
        state, terminal = prepare_state(latest_scan, min_distance, heading_x, heading_y, actions)
        
        # Get action from the model
        action = model.get_action(np.array(state), True)
        
        # Scale actions for the environment:
        # - Linear velocity: scale from [-1, 1] to [0, 4]
        # - Steering angle: keep in [-1, 1] range
        a_in = [
            (action[0] + 1) * 2.0,  # Scale to [0, 4]
            action[1],  # Keep steering in [-1, 1] range
        ]

        # Take a step in the environment
        latest_scan, min_distance, heading_x, heading_y, collision, actions, reward = sim.step(
            lin_velocity=a_in[0], steering_angle=a_in[1]
        )
        
        # Prepare next state representation
        next_state, next_terminal = prepare_state(latest_scan, min_distance, heading_x, heading_y, actions)
        
        # Determine if the episode is terminal
        terminal = collision or (steps >= max_steps)
        
        # Add experience to replay buffer
        replay_buffer.add(state, action, reward, terminal, next_state)

        # If episode ends, reset the environment
        if terminal:
            latest_scan, min_distance, heading_x, heading_y, collision, actions, reward = sim.reset()
            episode += 1
            
            # Train the model
            if episode % train_every_n == 0:
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )
            
            steps = 0
            
            print(f"Episode {episode}/{episodes_per_epoch} completed")
        else:
            steps += 1

        # Evaluate performance at the end of each epoch
        if (episode + 1) % episodes_per_epoch == 0:
            episode = 0
            epoch += 1
            evaluate(model, epoch, sim, eval_episodes=nr_eval_episodes)
            
            # Save agent after evaluation
            # model.save()
            
            print(f"Epoch {epoch}/{max_epochs} completed\n")


def prepare_state(latest_scan, min_distance, heading_x, heading_y, actions):
    """
    Prepare state representation for the model.
    
    State includes:
    - Downsampled laser scan (20 values)
    - Minimum distance to obstacles
    - Robot heading (x and y components)
    """
    # Downsample lidar data to 20 points
    scan_step = len(latest_scan) // 20
    downsampled_scan = [latest_scan[i*scan_step] for i in range(20)]
    
    # Normalize scan values (assuming max range is 20 based on YAML)
    normalized_scan = [min(s / 20.0, 1.0) for s in downsampled_scan]
    
    # Combine all state components
    state = normalized_scan + [min_distance / 20.0, heading_x, heading_y]
    
    # Check for terminal state (collision)
    terminal = min_distance < 0.5
    
    return state, terminal


def evaluate(model, epoch, sim, eval_episodes=10):
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating robot performance")
    avg_reward = 0.0
    collisions = 0
    distances_traveled = []
    min_obstacle_distances = []
    
    for i in range(eval_episodes):
        print(f"Evaluation episode {i+1}/{eval_episodes}")
        episode_reward = 0.0
        steps = 0
        
        # Reset environment
        latest_scan, min_distance, heading_x, heading_y, collision, actions, reward = sim.reset()
        done = False
        
        # Track distance traveled
        start_position = sim.last_position.copy()
        
        # Track minimum distance to obstacles in this episode
        episode_min_distance = float('inf')
        
        while not done and steps < 500:
            # Get state and action
            state, _ = prepare_state(latest_scan, min_distance, heading_x, heading_y, actions)
            action = model.get_action(np.array(state), False)  # No exploration during evaluation
            
            # Scale actions
            a_in = [
                (action[0] + 1) * 2.0,  # Scale to [0, 4]
                action[1],  # Keep steering in [-1, 1] range
            ]
            
            # Take step
            latest_scan, min_distance, heading_x, heading_y, collision, actions, reward = sim.step(
                lin_velocity=a_in[0], steering_angle=a_in[1]
            )
            
            # Update tracking variables
            episode_reward += reward
            if min_distance < episode_min_distance:
                episode_min_distance = min_distance
            
            steps += 1
            if collision:
                collisions += 1
                done = True
        
        # Calculate final distance traveled
        end_position = sim.last_position
        distance_traveled = np.sqrt(
            (end_position[0] - start_position[0])**2 + 
            (end_position[1] - start_position[1])**2
        )
        distances_traveled.append(distance_traveled)
        min_obstacle_distances.append(episode_min_distance)
        
        avg_reward += episode_reward

    # Calculate averages
    avg_reward /= eval_episodes
    avg_collision_rate = collisions / eval_episodes
    avg_distance = sum(distances_traveled) / eval_episodes
    avg_min_distance = sum(min_obstacle_distances) / eval_episodes
    
    # Print results
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Collision rate: {avg_collision_rate:.2f}")
    print(f"Average Distance Traveled: {avg_distance:.2f}")
    print(f"Average Minimum Distance to Obstacles: {avg_min_distance:.2f}")
    print("..............................................")
    
    # Log to tensorboard
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/collision_rate", avg_collision_rate, epoch)
    model.writer.add_scalar("eval/distance_traveled", avg_distance, epoch)
    model.writer.add_scalar("eval/min_obstacle_distance", avg_min_distance, epoch)


if __name__ == "__main__":
    main()