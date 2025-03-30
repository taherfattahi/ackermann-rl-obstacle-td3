import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import time

from models.TD3.TD3 import TD3
from sim import SIM_ENV


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


def evaluate(model, sim, num_episodes=10, max_steps=500, render=False, save_plots=False, verbose=True):
    """
    Evaluate a trained model in the simulation environment.
    
    Args:
        model: Trained TD3 model
        sim: Simulation environment
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        render: Whether to render the environment during evaluation
        save_plots: Whether to save trajectory plots
        verbose: Whether to print detailed evaluation information
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    total_rewards = []
    total_steps = []
    collision_count = 0
    distances_traveled = []
    min_obstacle_distances = []
    trajectories = []
    
    if save_plots and not os.path.exists('evaluation_plots'):
        os.makedirs('evaluation_plots')
    
    if verbose:
        print("Starting evaluation...")
    
    for episode in range(num_episodes):
        # Reset environment
        latest_scan, min_distance, heading_x, heading_y, collision, actions, reward = sim.reset()
        
        episode_reward = 0
        steps = 0
        done = False
        
        # Track trajectory
        trajectory = [(sim.last_position[0], sim.last_position[1])]
        
        # Track minimum distance to obstacles in this episode
        episode_min_distance = min_distance
        
        while not done and steps < max_steps:
            # Get state and action
            state, _ = prepare_state(latest_scan, min_distance, heading_x, heading_y, actions)
            action = model.get_action(np.array(state), False)  # No exploration during evaluation
            
            # Scale actions for the environment
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
            steps += 1
            
            # Record position for trajectory
            trajectory.append((sim.last_position[0], sim.last_position[1]))
            
            # Update minimum distance
            if min_distance < episode_min_distance:
                episode_min_distance = min_distance
            
            if collision:
                collision_count += 1
                done = True
            
            if render:
                # Add a small delay to visualize better
                time.sleep(0.05)
        
        # Calculate distance traveled
        start_pos = trajectory[0]
        end_pos = trajectory[-1]
        distance_traveled = np.sqrt(
            (end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2
        )
        
        # Store results
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        distances_traveled.append(distance_traveled)
        min_obstacle_distances.append(episode_min_distance)
        trajectories.append(trajectory)
        
        if verbose:
            print(f"Episode {episode+1}: Reward: {episode_reward:.2f}, Steps: {steps}, "
                  f"Distance: {distance_traveled:.2f}m, Min Obstacle Distance: {episode_min_distance:.2f}m, "
                  f"Collision: {'Yes' if collision else 'No'}")
        
        # Create trajectory plot
        if save_plots:
            create_trajectory_plot(trajectory, episode, sim)
    
    # Calculate average metrics
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    avg_distance = np.mean(distances_traveled)
    avg_min_distance = np.mean(min_obstacle_distances)
    collision_rate = collision_count / num_episodes
    success_rate = 1 - collision_rate
    
    # Print summary
    if verbose:
        print("\nEvaluation Summary:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Steps: {avg_steps:.2f}")
        print(f"Average Distance Traveled: {avg_distance:.2f}m")
        print(f"Average Minimum Obstacle Distance: {avg_min_distance:.2f}m")
        print(f"Collision Rate: {collision_rate:.2f}")
        print(f"Success Rate: {success_rate:.2f}")
    
    # Create summary plot
    if save_plots:
        create_summary_plot(trajectories, sim)
    
    # Return metrics as a dictionary
    return {
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_distance": avg_distance,
        "avg_min_distance": avg_min_distance,
        "collision_rate": collision_rate,
        "success_rate": success_rate,
        "total_rewards": total_rewards,
        "total_steps": total_steps,
        "distances_traveled": distances_traveled,
        "min_obstacle_distances": min_obstacle_distances,
        "trajectories": trajectories
    }


def create_trajectory_plot(trajectory, episode_num, sim):
    """Create and save a plot of the robot's trajectory for a single episode"""
    plt.figure(figsize=(10, 10))
    
    # Plot trajectory
    traj_x, traj_y = zip(*trajectory)
    plt.plot(traj_x, traj_y, 'b-', linewidth=2, label='Robot Path')
    
    # Mark start and end
    plt.plot(traj_x[0], traj_y[0], 'go', markersize=10, label='Start')
    plt.plot(traj_x[-1], traj_y[-1], 'ro', markersize=10, label='End')
    
    # Plot obstacles
    for obstacle in sim.env.obstacle_list:
        if hasattr(obstacle, '_state'):
            x, y = obstacle._state[0][0], obstacle._state[1][0]
            # Assuming circular obstacles with radius=1 (from your YAML)
            radius = 1.0
            plt.gca().add_patch(Circle((x, y), radius, color='gray', alpha=0.5))
    
    # Set plot limits based on world size from YAML
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    
    plt.title(f'Robot Trajectory - Episode {episode_num+1}')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(f'evaluation_plots/trajectory_episode_{episode_num+1}.png')
    plt.close()


def create_summary_plot(trajectories, sim):
    """Create and save a summary plot with all trajectories"""
    plt.figure(figsize=(12, 12))
    
    # Plot all trajectories
    for i, trajectory in enumerate(trajectories):
        traj_x, traj_y = zip(*trajectory)
        plt.plot(traj_x, traj_y, '-', linewidth=1, alpha=0.7, label=f'Episode {i+1}')
        
        # Mark start and end points
        plt.plot(traj_x[0], traj_y[0], 'go', markersize=6, alpha=0.7)
        plt.plot(traj_x[-1], traj_y[-1], 'ro', markersize=6, alpha=0.7)
    
    # Plot obstacles from the last environment state
    for obstacle in sim.env.obstacle_list:
        if hasattr(obstacle, '_state'):
            x, y = obstacle._state[0][0], obstacle._state[1][0]
            # Assuming circular obstacles with radius=1 (from your YAML)
            radius = 1.0
            plt.gca().add_patch(Circle((x, y), radius, color='gray', alpha=0.5))
    
    # Set plot limits based on world size from YAML
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    
    plt.title('All Robot Trajectories')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    
    # Add custom legend elements for start and end points
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Start Points'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='End Points')
    ]
    plt.legend(handles=custom_lines, loc='upper right')
    
    plt.savefig('evaluation_plots/all_trajectories.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained TD3 model for Ackermann robot obstacle avoidance')
    parser.add_argument('--model_dir', type=str, default='saved_models', help='Directory containing saved model files')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--render', action='store_true', help='Render the environment during evaluation')
    parser.add_argument('--save_plots', action='store_true', help='Save trajectory plots')
    parser.add_argument('--world_file', type=str, default='robot_world.yaml', help='YAML file for world configuration')
    args = parser.parse_args()
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    state_dim = 23  # 20 lidar scans + min_distance + heading_x + heading_y
    action_dim = 2  # [linear_velocity, steering_angle]
    max_action = 1.0
    
    model = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        load_model=True  # Load the saved model
    )
    
    # Check if model files exist
    actor_path = f"{args.model_dir}/td3_actor.pth"
    critic_path = f"{args.model_dir}/td3_critic.pth"
    
    if not os.path.exists(actor_path) or not os.path.exists(critic_path):
        print(f"Error: Model files not found in {args.model_dir}")
        return
    
    # Initialize simulation environment
    sim = SIM_ENV(world_file=args.world_file)
    
    # Run evaluation
    results = evaluate(
        model=model,
        sim=sim,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        save_plots=args.save_plots,
        verbose=True
    )
    
    # Save results to file
    with open('evaluation_results.txt', 'w') as f:
        f.write("Evaluation Results:\n")
        f.write(f"Average Reward: {results['avg_reward']:.2f}\n")
        f.write(f"Average Steps: {results['avg_steps']:.2f}\n")
        f.write(f"Average Distance Traveled: {results['avg_distance']:.2f}m\n")
        f.write(f"Average Minimum Obstacle Distance: {results['avg_min_distance']:.2f}m\n")
        f.write(f"Collision Rate: {results['collision_rate']:.2f}\n")
        f.write(f"Success Rate: {results['success_rate']:.2f}\n")
        
        f.write("\nDetailed Episode Results:\n")
        for i in range(len(results['total_rewards'])):
            f.write(f"Episode {i+1}: Reward: {results['total_rewards'][i]:.2f}, "
                   f"Steps: {results['total_steps'][i]}, "
                   f"Distance: {results['distances_traveled'][i]:.2f}m, "
                   f"Min Obstacle Distance: {results['min_obstacle_distances'][i]:.2f}m\n")
    
    print("Evaluation complete. Results saved to evaluation_results.txt")
    if args.save_plots:
        print("Trajectory plots saved to evaluation_plots/ directory")


if __name__ == "__main__":
    main()