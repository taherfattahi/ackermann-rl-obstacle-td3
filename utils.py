from typing import List
from tqdm import tqdm
import yaml

from replay_buffer import ReplayBuffer, RolloutReplayBuffer

class Pretraining:
    def __init__(
        self,
        file_names: List[str],
        model: object,
        replay_buffer: object,
        reward_function,
    ):
        self.file_names = file_names
        self.model = model
        self.replay_buffer = replay_buffer
        self.reward_function = reward_function

    def load_buffer(self):
        for file_name in self.file_names:
            print("Loading file: ", file_name)
            with open(file_name, "r") as file:
                samples = yaml.full_load(file)
                for i in tqdm(range(1, len(samples) - 1)):
                    sample = samples[i]
                    # Modified to handle Ackermann robot without goal
                    latest_scan = sample["latest_scan"]
                    min_distance = min(latest_scan) if "min_distance" not in sample else sample["min_distance"]
                    
                    # Get heading vector components or default to cos/sin if present
                    heading_x = sample.get("heading_x", sample.get("cos", 0))
                    heading_y = sample.get("heading_y", sample.get("sin", 0))
                    
                    collision = sample["collision"]
                    action = sample["action"]

                    # Prepare state using the updated method for Ackermann robot
                    state, terminal = self.prepare_state_for_pretraining(
                        latest_scan, min_distance, heading_x, heading_y, action
                    )

                    if terminal:
                        continue

                    next_sample = samples[i + 1]
                    next_latest_scan = next_sample["latest_scan"]
                    next_min_distance = min(next_latest_scan) if "min_distance" not in next_sample else next_sample["min_distance"]
                    
                    # Get next heading vector components
                    next_heading_x = next_sample.get("heading_x", next_sample.get("cos", 0))
                    next_heading_y = next_sample.get("heading_y", next_sample.get("sin", 0))
                    
                    next_collision = next_sample["collision"]
                    next_action = next_sample["action"]
                    
                    # Prepare next state
                    next_state, next_terminal = self.prepare_state_for_pretraining(
                        next_latest_scan,
                        next_min_distance,
                        next_heading_x,
                        next_heading_y,
                        next_action,
                    )
                    
                    # Calculate reward using the current reward function
                    # We use 0 as step_distance and min_distance_improvement for pretraining data
                    reward = self.reward_function(
                        next_collision, action, next_latest_scan, 0, 0
                    )
                    
                    # Add to replay buffer
                    self.replay_buffer.add(
                        state, action, reward, next_terminal, next_state
                    )

        return self.replay_buffer

    def prepare_state_for_pretraining(self, latest_scan, min_distance, heading_x, heading_y, actions):
        """
        Prepare state for pretraining - similar to what model.prepare_state would do
        but compatible with the data format from the files
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

    def train(
        self,
        pretraining_iterations,
        replay_buffer,
        iterations,
        batch_size,
    ):
        print("Running Pretraining")
        for _ in tqdm(range(pretraining_iterations)):
            self.model.train(
                replay_buffer=replay_buffer,
                iterations=iterations,
                batch_size=batch_size,
            )
        print("Model Pretrained")


def get_buffer(
    model,
    sim,
    load_saved_buffer,
    pretrain,
    pretraining_iterations,
    training_iterations,
    batch_size,
    buffer_size=50000,
    random_seed=666,
    file_names=["assets/data.yml"],
    history_len=10,
):
    """
    Initialize and potentially load a replay buffer.
    
    Can either use standard ReplayBuffer or RolloutReplayBuffer based on preference.
    Using standard ReplayBuffer by default as it's simpler and works well with TD3.
    """
    # Choose buffer type - RolloutReplayBuffer is useful if you want to use history
    # but standard ReplayBuffer is simpler and works well with TD3
    use_rollout_buffer = False
    
    if use_rollout_buffer:
        replay_buffer = RolloutReplayBuffer(
            buffer_size=buffer_size, 
            random_seed=random_seed,
            history_len=history_len
        )
    else:
        replay_buffer = ReplayBuffer(buffer_size=buffer_size, random_seed=random_seed)

    if pretrain:
        assert (
            load_saved_buffer
        ), "To pre-train model, load_saved_buffer must be set to True"

    if load_saved_buffer:
        pretraining = Pretraining(
            file_names=file_names,
            model=model,
            replay_buffer=replay_buffer,
            reward_function=sim.get_reward,
        )  # instantiate pre-training
        replay_buffer = (
            pretraining.load_buffer()
        )  # fill buffer with experiences from the data.yml file
        if pretrain:
            pretraining.train(
                pretraining_iterations=pretraining_iterations,
                replay_buffer=replay_buffer,
                iterations=training_iterations,
                batch_size=batch_size,
            )  # run pre-training

    return replay_buffer