import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
# Fix for Scipy/Pennylane attribute error: Explicitly import sparse.linalg and constants
import scipy.sparse.linalg 
import scipy.constants
import pennylane as qml
import pygame
import random
import copy
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import deque
import os

# ==========================================
# 1. Environment Configuration
# ==========================================
@dataclass
class EnvConfig:
    screen_width: int = 800
    screen_height: int = 600
    robot_radius: int = 15
    goal_radius: int = 20
    hazard_radius: int = 30
    num_hazards: int = 8 # Increased slightly for difficulty
    robot_speed: float = 5.0
    max_steps: int = 500
    collision_penalty: float = -10.0
    goal_reward: float = 100.0
    step_penalty: float = -0.01
    distance_reward_scale: float = 0.1
    shield_enabled: bool = True
    rollback_enabled: bool = True
    # Non-IID Setting: 'all', 'top', 'bottom', 'left', 'right'
    hazard_zone: str = 'all' 

class SafetyGymEnv:
    """Safety-aware robot navigation environment"""
    def __init__(self, config: EnvConfig, render_mode='human', client_id=0):
        self.config = config
        self.render_mode = render_mode
        self.client_id = client_id
        self.screen = None
        self.font = None
        
        if render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((config.screen_width, config.screen_height))
            pygame.display.set_caption(f"Federated Quantum RL")
            self.font = pygame.font.SysFont("Arial", 24)
        
        self.reset()
    
    def reset(self):
        self.robot_pos = np.array([self.config.screen_width / 2, self.config.screen_height - 50], dtype=np.float32)
        # Goal is always roughly opposite to start or random, keeping it simple
        self.goal_pos = np.array([random.randint(100, self.config.screen_width - 100), random.randint(50, 150)], dtype=np.float32)
        
        self.hazards = []
        self.hazard_shields = []
        self._generate_hazards()
        self.steps = 0
        self.done = False
        self.previous_distance = self._distance_to_goal()
        return self._get_observation()
    
    def _generate_hazards(self):
        """Generates hazards based on zone (Non-IID) and ensures no overlap"""
        attempts = 0
        while len(self.hazards) < self.config.num_hazards and attempts < 1000:
            attempts += 1
            
            # 1. Define Zone Constraints (Non-IID Data Generation)
            if self.config.hazard_zone == 'top':
                # Hazards only in top half
                y_min, y_max = 50, self.config.screen_height // 2
                x_min, x_max = 30, self.config.screen_width - 30
            elif self.config.hazard_zone == 'bottom':
                # Hazards only in bottom half (near start)
                y_min, y_max = self.config.screen_height // 2, self.config.screen_height - 50
                x_min, x_max = 30, self.config.screen_width - 30
            elif self.config.hazard_zone == 'left':
                y_min, y_max = 50, self.config.screen_height - 50
                x_min, x_max = 30, self.config.screen_width // 2
            else: # 'all' or random
                y_min, y_max = 50, self.config.screen_height - 50
                x_min, x_max = 30, self.config.screen_width - 30

            pos = np.array([random.randint(x_min, x_max), random.randint(y_min, y_max)], dtype=np.float32)
            
            # 2. Safety Checks
            # Check distance from robot
            if np.linalg.norm(pos - self.robot_pos) < 100: continue
            # Check distance from goal
            if np.linalg.norm(pos - self.goal_pos) < 100: continue
            
            # 3. Hazard Overlap Check (New Feature)
            overlap = False
            for h in self.hazards:
                # Ensure at least 2.5x radius distance between hazards
                if np.linalg.norm(pos - h) < (self.config.hazard_radius * 2.5):
                    overlap = True
                    break
            
            if not overlap:
                self.hazards.append(pos)
                self.hazard_shields.append(random.random() < 0.5)

    def _get_observation(self):
        # Normalize coords
        obs = [
            self.robot_pos[0] / self.config.screen_width,
            self.robot_pos[1] / self.config.screen_height,
            (self.goal_pos[0] - self.robot_pos[0]) / self.config.screen_width,
            (self.goal_pos[1] - self.robot_pos[1]) / self.config.screen_height,
            self._distance_to_goal() / np.sqrt(800**2 + 600**2)
        ]
        # 3 Nearest hazards
        hazard_info = []
        for h in self.hazards:
            diff = h - self.robot_pos
            dist = np.linalg.norm(diff)
            hazard_info.append((dist, diff[0]/800, diff[1]/600))
        hazard_info.sort(key=lambda x: x[0])
        for i in range(3):
            if i < len(hazard_info):
                obs.extend([hazard_info[i][1], hazard_info[i][2]])
            else:
                obs.extend([0, 0])
        return np.array(obs, dtype=np.float32) # Size: 5 + 6 = 11

    def _distance_to_goal(self):
        return np.linalg.norm(self.goal_pos - self.robot_pos)

    def step(self, action):
        if self.done: return self._get_observation(), 0, True, {}
        
        # Simple movement logic
        action = np.clip(action, -1, 1)
        self.robot_pos += action * self.config.robot_speed
        self.robot_pos = np.clip(self.robot_pos, 0, [800, 600])
        
        dist = self._distance_to_goal()
        reward = self.config.step_penalty + (self.previous_distance - dist) * self.config.distance_reward_scale
        self.previous_distance = dist
        
        # Check collision
        for h in self.hazards:
            if np.linalg.norm(h - self.robot_pos) < (self.config.robot_radius + self.config.hazard_radius):
                reward += self.config.collision_penalty
                self.done = True
                return self._get_observation(), reward, self.done, {'goal': False}

        if dist < (self.config.robot_radius + self.config.goal_radius):
            reward += self.config.goal_reward
            self.done = True
            return self._get_observation(), reward, self.done, {'goal': True}
            
        self.steps += 1
        if self.steps >= self.config.max_steps: self.done = True
        
        return self._get_observation(), reward, self.done, {'goal': False}
    
    def render(self):
        if self.render_mode != 'human' or self.screen is None:
            return
        
        # Fill background
        self.screen.fill((20, 20, 30)) # Dark blue-ish background
        
        # Draw hazards (Red circles)
        for i, hazard in enumerate(self.hazards):
            color = (200, 50, 50) if not self.hazard_shields[i] else (200, 100, 50)
            pygame.draw.circle(self.screen, color, hazard.astype(int), self.config.hazard_radius)
            if self.hazard_shields[i]:
                pygame.draw.circle(self.screen, (0, 255, 0), hazard.astype(int), self.config.hazard_radius, 2)
            
        # Draw goal (Green circle)
        pygame.draw.circle(self.screen, (50, 255, 50), self.goal_pos.astype(int), self.config.goal_radius)
        
        # Draw robot (Blue circle)
        pygame.draw.circle(self.screen, (100, 150, 255), self.robot_pos.astype(int), self.config.robot_radius)
        
        # UI Information
        info_text = f"Client: {self.client_id} | Zone: {self.config.hazard_zone.upper()} | Steps: {self.steps}"
        text_surface = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Minimal delay to allow viewing
        pygame.time.wait(2)

    def close(self):
        if self.render_mode == 'human':
            pygame.quit()

# ==========================================
# 2. Hybrid Quantum-Classical Actor
# ==========================================
n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Encode classical data (compressed to n_qubits) into rotation angles
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    # Variational Layer
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Measure expectation
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridQuantumActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Classical Pre-processing (Compress 11 dims -> 4 dims)
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, n_qubits) # Reduce to number of qubits
        
        # Quantum Layer
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Classical Post-processing (4 dims -> Action dim)
        self.action_mean = nn.Linear(n_qubits, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = torch.tanh(self.fc2(x)) * np.pi # Scale to [-pi, pi] for embedding
        
        # Pass through Quantum Circuit
        x = self.q_layer(x)
        
        # Map quantum features to action mean
        mean = torch.tanh(self.action_mean(x))
        return mean, self.log_std

# Standard Classical Critic (Value Function)
class ClassicalCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, state):
        return self.net(state)

# ==========================================
# 3. Federated PPO Client
# ==========================================
class PPOClient:
    def __init__(self, client_id, state_dim, action_dim, lr=3e-4, render=False, hazard_zone='all'):
        self.id = client_id
        self.render_enabled = render
        # Only initialize Pygame for the rendering client
        render_mode = 'human' if render else 'none'
        
        # Setup specific env config for this client (Non-IID)
        client_config = EnvConfig(hazard_zone=hazard_zone)
        
        self.env = SafetyGymEnv(client_config, render_mode=render_mode, client_id=client_id)
        
        # Local Models
        self.actor = HybridQuantumActor(state_dim, action_dim)
        self.critic = ClassicalCritic(state_dim)
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.memory = []
    
    def sync_weights(self, global_actor_state, global_critic_state):
        """Download global model weights"""
        self.actor.load_state_dict(global_actor_state)
        self.critic.load_state_dict(global_critic_state)

    def collect_experience(self, min_steps=500):
        """Run policy in local env to collect data"""
        self.memory = []
        state = self.env.reset()
        steps = 0
        while steps < min_steps:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                mean, log_std = self.actor(state_t)
                std = log_std.exp()
                dist = Normal(mean, std)
                action = dist.sample()
                action_np = action.numpy().flatten()
            
            next_state, reward, done, _ = self.env.step(action_np)
            
            # === Visualization Logic ===
            if self.render_enabled:
                self.env.render()
            
            self.memory.append((state, action, reward, next_state, done, dist.log_prob(action).sum().item()))
            
            state = next_state
            steps += 1
            if done: state = self.env.reset()
            
    def train_local(self, epochs=4, batch_size=64, gamma=0.99):
        """PPO Update on local data"""
        if len(self.memory) < batch_size: return 0, 0
        
        states, actions, rewards, next_states, dones, old_log_probs = zip(*self.memory)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.stack(actions).squeeze() # Shape fix might be needed
        rewards = torch.FloatTensor(rewards)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # Calculate Returns (Monte Carlo for simplicity)
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Optimization Loop
        for _ in range(epochs):
            # Calculate Value & Advantage
            values = self.critic(states).squeeze()
            advantage = returns - values.detach()
            
            # Actor Loss
            mean, log_std = self.actor(states)
            dist = Normal(mean, log_std.exp())
            new_log_probs = dist.log_prob(actions).sum(dim=1) if actions.dim() > 1 else dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss
            critic_loss = F.mse_loss(self.critic(states).squeeze(), returns)
            
            # Updates
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            
        return actor_loss.item(), critic_loss.item()

# ==========================================
# 4. Federated Learning Orchestrator
# ==========================================
def plot_training_curves(episode_rewards):
    """Plot training curves with moving average"""
    plt.figure(figsize=(12, 5))
    
    # Episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='Raw Reward')
    plt.title('Federated Global Rewards')
    plt.xlabel('Round')
    plt.ylabel('Avg Reward')
    plt.grid(True)
    
    # Moving average
    plt.subplot(1, 2, 2)
    window = 10 if len(episode_rewards) > 20 else 1
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg, color='red', label=f'Moving Avg ({window})')
    plt.title(f'Smoothed Learning Curve')
    plt.xlabel('Round')
    plt.ylabel('Smoothed Reward')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('federated_training_curves.png')
    print("Training curves updated: 'federated_training_curves.png'")
    plt.close() # Close plot to free memory

def federated_training_loop():
    print("Initializing Federated Quantum PPO...")
    
    state_dim = 11
    action_dim = 2
    n_clients = 3
    n_rounds = 500
    
    # Global Server Models
    global_actor = HybridQuantumActor(state_dim, action_dim)
    global_critic = ClassicalCritic(state_dim)
    
    # Initialize Clients with Non-IID Zones
    # Client 0: Hazards Top
    # Client 1: Hazards Bottom
    # Client 2: Hazards Left
    zones = ['top', 'bottom', 'left']
    
    # Enable render=True for ALL clients to see them sequentially
    clients = [PPOClient(i, state_dim, action_dim, render=True, hazard_zone=zones[i]) for i in range(n_clients)]
    
    rewards_history = []
    
    for round_idx in range(n_rounds):
        # 1. Broadcast Weights
        actor_state = global_actor.state_dict()
        critic_state = global_critic.state_dict()
        
        local_actor_weights = []
        local_critic_weights = []
        round_rewards = []
        
        # 2. Client Local Training (Sequential visualization)
        for client in clients:
            client.sync_weights(actor_state, critic_state)
            
            # This will open/update the pygame window for this specific client
            # The label on screen will change to show which client it is
            client.collect_experience(min_steps=300) 
            a_loss, c_loss = client.train_local()
            
            avg_rew = sum([x[2] for x in client.memory])
            round_rewards.append(avg_rew)
            
            local_actor_weights.append(copy.deepcopy(client.actor.state_dict()))
            local_critic_weights.append(copy.deepcopy(client.critic.state_dict()))
            
        # 3. Global Aggregation (FedAvg)
        avg_actor = copy.deepcopy(local_actor_weights[0])
        avg_critic = copy.deepcopy(local_critic_weights[0])
        
        for k in avg_actor.keys():
            for i in range(1, n_clients):
                avg_actor[k] += local_actor_weights[i][k]
            avg_actor[k] = torch.div(avg_actor[k], n_clients)
            
        for k in avg_critic.keys():
            for i in range(1, n_clients):
                avg_critic[k] += local_critic_weights[i][k]
            avg_critic[k] = torch.div(avg_critic[k], n_clients)
            
        global_actor.load_state_dict(avg_actor)
        global_critic.load_state_dict(avg_critic)
        
        avg_round_reward = np.mean(round_rewards)
        rewards_history.append(avg_round_reward)
        
        # Logging & Visualization Updates
        if (round_idx + 1) % 10 == 0:
            print(f"Round {round_idx+1}/{n_rounds} | Avg Global Reward: {avg_round_reward:.2f}")
            plot_training_curves(rewards_history)
            
        # Model Saving
        if (round_idx + 1) % 100 == 0:
            save_path = f"fed_qppo_checkpoint_round_{round_idx+1}.pth"
            torch.save({
                'actor': global_actor.state_dict(),
                'critic': global_critic.state_dict(),
                'round': round_idx,
            }, save_path)
            print(f"Saved Checkpoint: {save_path}")

    print("\nTraining Finished.")

if __name__ == "__main__":
    federated_training_loop()