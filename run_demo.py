import argparse
import time
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from models.TD3.TD3 import TD3
from sim import SIM_ENV


def prepare_state(latest_scan, min_distance, heading_x, heading_y, actions):
    """
    Prepare state representation for the model.
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


def visualize_lidar(ax, robot_pos, heading_vector, lidar_scan, max_range=20):
    """Visualize lidar scan data on the plot"""
    # Get robot position and orientation
    x, y = robot_pos
    heading_x, heading_y = heading_vector
    heading_angle = np.arctan2(heading_y, heading_x)
    
    # Plot robot as a triangle
    robot_size = 1.0
    dx = robot_size * np.cos(heading_angle)
    dy = robot_size * np.sin(heading_angle)
    
    # Plot robot body (rectangle approximated by a wedge)
    robot = plt.Circle((x, y), 0.8, color='blue', alpha=0.7)
    ax.add_artist(robot)
    
    # Plot robot direction indicator
    ax.arrow(x, y, dx, dy, head_width=0.5, head_length=0.5, fc='blue', ec='blue')
    
    # Plot lidar scan points
    num_points = len(lidar_scan)
    angle_increment = 2 * np.pi / num_points
    
    for i, dist in enumerate(lidar_scan):
        if dist >= max_range:  # Skip if max range or no detection
            continue
            
        # Calculate angle for this lidar point
        # Assuming lidar scan is centered around the robot's heading
        angle = heading_angle - np.pi / 2 + i * angle_increment
        
        # Calculate endpoint
        end_x = x + dist * np.cos(angle)
        end_y = y + dist * np.sin(angle)
        
        # Draw line for this scan
        ax.plot([x, end_x], [y, end_y], 'r-', alpha=0.1)
        
        # Draw point at detection
        ax.plot(end_x, end_y, 'ro', markersize=2, alpha=0.5)


def run_demo(model, sim, save_video=False, max_steps=500, delay=0.1):
    """
    Run a live demonstration of the trained model
    
    Args:
        model: Trained TD3 model
        sim: Simulation environment
        save_video: Whether to save a video of the demonstration
        max_steps: Maximum steps to run
        delay: Delay between steps (for visualization)
    """
    # Reset environment
    latest_scan, min_distance, heading_x, heading_y, collision, actions, reward = sim.reset()
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.ion()  # Turn on interactive mode
    
    # Track trajectory
    trajectory = [(sim.last_position[0], sim.last_position[1])]
    traj_line, = ax.plot([], [], 'b-', linewidth=2)
    
    # Plot obstacles
    for obstacle in sim.env.obstacle_list:
        if hasattr(obstacle, '_state'):
            x, y = obstacle._state[0][0], obstacle._state[1][0]
            # Assuming circular obstacles with radius=1 (from your YAML)
            radius = 1.0
            obstacle_circle = plt.Circle((x, y), radius, color='gray', alpha=0.5)
            ax.add_artist(obstacle_circle)
    
    # Set plot limits based on world size from YAML
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_title('Ackermann Robot Navigation Demo')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True)
    
    # Initialize status text
    status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top')
    
    # Video frames if saving video
    frames = []
    
    steps = 0
    done = False
    total_reward = 0
    
    while not done and steps < max_steps:
        # Clear previous robot visualization
        for collection in ax.collections[:]:
            if isinstance(collection, plt.Circle) and collection.get_facecolor()[0][0] == 0:
                collection.remove()

        
        # Get state and action
        state, _ = prepare_state(latest_scan, min_distance, heading_x, heading_y, actions)
        action = model.get_action(np.array(state), False)  # No exploration during demo
        
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
        total_reward += reward
        steps += 1
        
        # Record position for trajectory
        trajectory.append((sim.last_position[0], sim.last_position[1]))
        
        # Update trajectory line
        traj_x, traj_y = zip(*trajectory)
        traj_line.set_data(traj_x, traj_y)
        
        # Visualize lidar scan
        heading_vector = [heading_x, heading_y]
        visualize_lidar(ax, sim.last_position, heading_vector, latest_scan)
        
        # Update status text
        status = (f"Step: {steps}  Reward: {total_reward:.1f}  "
                 f"Speed: {a_in[0]:.2f} m/s  Steering: {a_in[1]:.2f} rad  "
                 f"Min Distance: {min_distance:.2f}m")
        status_text.set_text(status)
        
        # Draw and pause
        plt.draw()
        plt.pause(delay)
        
        if save_video:
            # Save frame for video
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
        
        if collision:
            print("Collision detected! Ending demo.")
            status_text.set_text(status + " - COLLISION!")
            plt.draw()
            plt.pause(1.0)  # Pause longer on collision
            done = True
    
    # Final pause to show the end result
    plt.pause(2.0)
    
    # Save video if requested
    if save_video and frames:
        try:
            import cv2
            print("Saving video...")
            output_file = 'demo_video.mp4'
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(output_file, fourcc, 10, (width, height))
            
            for frame in frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            video.release()
            print(f"Video saved to {output_file}")
        except ImportError:
            print("Could not save video - OpenCV (cv2) not installed")
            print("Install with: pip install opencv-python")
    
    plt.ioff()  # Turn off interactive mode
    plt.show()
    
    # Print summary
    print(f"\nDemo Summary:")
    print(f"Steps: {steps}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Collision: {'Yes' if collision else 'No'}")
    
    # Calculate distance traveled
    start_pos = trajectory[0]
    end_pos = trajectory[-1]
    distance_traveled = np.sqrt(
        (end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2
    )
    print(f"Distance Traveled: {distance_traveled:.2f}m")


def main():
    parser = argparse.ArgumentParser(description='Run a demonstration of the trained TD3 model')
    parser.add_argument('--model_dir', type=str, default='models/TD3/checkpoint', help='Directory containing saved model files')
    parser.add_argument('--save_video', action='store_true', help='Save a video of the demonstration')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum steps to run')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between steps (seconds)')
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
    
    # Run demo
    print("Starting demo...")
    run_demo(
        model=model,
        sim=sim,
        save_video=args.save_video,
        max_steps=args.max_steps,
        delay=args.delay
    )


if __name__ == "__main__":
    main()