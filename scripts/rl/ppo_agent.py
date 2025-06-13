#!/usr/bin/env python3
"""
PPO Agent for Kubernetes Resource Management
Based on baseline experimental findings and optimized for GPU/CPU scheduling decisions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """Configuration for PPO agent"""
    # Network architecture
    hidden_size: int = 256
    num_layers: int = 3

    # PPO hyperparameters (optimized for our use case)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training parameters
    ppo_epochs: int = 4
    batch_size: int = 64
    buffer_size: int = 2048

    # Environment specific
    observation_dim: int = 10
    action_dims: List[int] = None  # [5, 5, 3] for our action space

    def __post_init__(self):
        if self.action_dims is None:
            self.action_dims = [5, 5, 3]  # GPU, CPU, placement actions

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network optimized for Kubernetes resource management

    Architecture designed based on baseline findings:
    - Handles multi-dimensional action space (GPU/CPU scaling + placement)
    - Incorporates load pattern awareness
    - Optimized for resource efficiency decisions
    """

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config

        # Shared feature extraction layers
        self.shared_layers = nn.ModuleList([
            nn.Linear(config.observation_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        ])

        # Actor heads for each action dimension
        self.actor_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, action_dim)
            for action_dim in config.action_dims
        ])

        # Critic head
        self.critic_head = nn.Linear(config.hidden_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, observations: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the network

        Args:
            observations: Batch of observations [batch_size, obs_dim]

        Returns:
            action_logits: List of logits for each action dimension
            values: State values [batch_size, 1]
        """

        # Shared feature extraction
        x = observations
        for layer in self.shared_layers:
            if isinstance(layer, nn.ReLU):
                x = layer(x)
            else:
                x = layer(x)

        # Actor outputs (action logits)
        action_logits = [head(x) for head in self.actor_heads]

        # Critic output (state value)
        values = self.critic_head(x)

        return action_logits, values

    def get_action_and_value(self, observations: torch.Tensor, actions: Optional[torch.Tensor] = None):
        """
        Get actions and values for given observations

        Args:
            observations: Batch of observations
            actions: Optional actions for evaluation (during training)

        Returns:
            actions: Sampled actions
            log_probs: Log probabilities of actions
            values: State values
            entropy: Action entropy
        """

        action_logits, values = self.forward(observations)

        # Create categorical distributions for each action dimension
        action_dists = [torch.distributions.Categorical(logits=logits) for logits in action_logits]

        if actions is None:
            # Sample actions during rollout
            sampled_actions = [dist.sample() for dist in action_dists]
            actions = torch.stack(sampled_actions, dim=1)
        else:
            # Use provided actions during training
            sampled_actions = [actions[:, i] for i in range(len(action_dists))]

        # Calculate log probabilities and entropy
        log_probs = torch.stack([
            dist.log_prob(action) for dist, action in zip(action_dists, sampled_actions)
        ], dim=1).sum(dim=1)

        entropy = torch.stack([dist.entropy() for dist in action_dists], dim=1).sum(dim=1)

        return actions, log_probs, values.squeeze(-1), entropy

class PPOBuffer:
    """Experience buffer for PPO training"""

    def __init__(self, buffer_size: int, observation_dim: int, action_dims: List[int]):
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.action_dims = action_dims

        # Initialize buffers
        self.observations = np.zeros((buffer_size, observation_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, len(action_dims)), dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)

        self.ptr = 0
        self.size = 0

    def store(self, obs: np.ndarray, action: np.ndarray, reward: float,
              value: float, log_prob: float, done: bool):
        """Store a transition in the buffer"""

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get a random batch of experiences"""

        indices = np.random.choice(self.size, batch_size, replace=False)

        return {
            'observations': torch.FloatTensor(self.observations[indices]),
            'actions': torch.LongTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'values': torch.FloatTensor(self.values[indices]),
            'log_probs': torch.FloatTensor(self.log_probs[indices]),
            'dones': torch.BoolTensor(self.dones[indices])
        }

    def compute_gae(self, next_value: float, gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation"""

        advantages = np.zeros_like(self.rewards)
        returns = np.zeros_like(self.rewards)

        gae = 0
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_val = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_val * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[step] = gae
            returns[step] = gae + self.values[step]

        return advantages, returns

    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0

class PPOAgent:
    """
    PPO Agent for Kubernetes Resource Management

    Optimized based on baseline experimental findings:
    - Reward function incorporates load pattern sensitivity
    - Action space designed for GPU/CPU scaling decisions
    - Training optimized for resource efficiency
    """

    def __init__(self, config: PPOConfig, device: str = "cpu"):
        self.config = config
        self.device = device

        # Initialize network
        self.network = ActorCriticNetwork(config).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

        # Initialize buffer
        self.buffer = PPOBuffer(
            buffer_size=config.buffer_size,
            observation_dim=config.observation_dim,
            action_dims=config.action_dims
        )

        # Training metrics
        self.training_metrics = {
            'episode_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'policy_losses': deque(maxlen=100),
            'value_losses': deque(maxlen=100),
            'entropy_losses': deque(maxlen=100)
        }

        logger.info(f"Initialized PPO agent with {sum(p.numel() for p in self.network.parameters())} parameters")

    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Get action for given observation

        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy

        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value
        """

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

            if deterministic:
                # Use deterministic policy (for evaluation)
                action_logits, value = self.network(obs_tensor)
                actions = torch.stack([torch.argmax(logits, dim=1) for logits in action_logits], dim=1)
                log_prob = torch.tensor(0.0)  # Not used in deterministic mode
            else:
                # Use stochastic policy (for training)
                actions, log_prob, value, _ = self.network.get_action_and_value(obs_tensor)

            return actions.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()

    def store_transition(self, obs: np.ndarray, action: np.ndarray, reward: float,
                        value: float, log_prob: float, done: bool):
        """Store a transition in the buffer"""
        self.buffer.store(obs, action, reward, value, log_prob, done)

    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """
        Update the policy using PPO

        Args:
            next_value: Value of the next state (for GAE computation)

        Returns:
            training_metrics: Dictionary of training metrics
        """

        if self.buffer.size < self.config.batch_size:
            return {}

        # Compute advantages and returns
        advantages, returns = self.buffer.compute_gae(
            next_value, self.config.gamma, self.config.gae_lambda
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        observations = torch.FloatTensor(self.buffer.observations[:self.buffer.size]).to(self.device)
        actions = torch.LongTensor(self.buffer.actions[:self.buffer.size]).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs[:self.buffer.size]).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []

        # PPO training epochs
        for epoch in range(self.config.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(self.buffer.size)

            for start in range(0, self.buffer.size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                # Get batch data
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Forward pass
                _, new_log_probs, values, entropy = self.network.get_action_and_value(
                    batch_obs, batch_actions
                )

                # Calculate ratios
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                # Policy loss (clipped)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.config.clip_epsilon,
                                  1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = (policy_loss +
                            self.config.value_coef * value_loss +
                            self.config.entropy_coef * entropy_loss)

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Store metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        # Clear buffer
        self.buffer.clear()

        # Update training metrics
        metrics = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'total_loss': np.mean(policy_losses) + self.config.value_coef * np.mean(value_losses) +
                         self.config.entropy_coef * np.mean(entropy_losses)
        }

        # Store in history
        self.training_metrics['policy_losses'].append(metrics['policy_loss'])
        self.training_metrics['value_losses'].append(metrics['value_loss'])
        self.training_metrics['entropy_losses'].append(metrics['entropy_loss'])

        return metrics

    def save_model(self, filepath: str):
        """Save the model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_metrics': dict(self.training_metrics)
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load the model"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load training metrics if available
        if 'training_metrics' in checkpoint:
            for key, values in checkpoint['training_metrics'].items():
                self.training_metrics[key] = deque(values, maxlen=100)

        logger.info(f"Model loaded from {filepath}")

    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        stats = {}

        for key, values in self.training_metrics.items():
            if len(values) > 0:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_recent'] = values[-1] if values else 0

        return stats

    def set_eval_mode(self):
        """Set network to evaluation mode"""
        self.network.eval()

    def set_train_mode(self):
        """Set network to training mode"""
        self.network.train()

def create_ppo_config(
    learning_rate: float = 3e-4,
    hidden_size: int = 256,
    num_layers: int = 3,
    buffer_size: int = 2048,
    batch_size: int = 64,
    ppo_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5
) -> PPOConfig:
    """Create PPO configuration with sensible defaults"""

    return PPOConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        buffer_size=buffer_size,
        batch_size=batch_size,
        ppo_epochs=ppo_epochs,
        observation_dim=10,  # Based on our environment
        action_dims=[5, 5, 3]  # GPU scaling, CPU scaling, placement preference
    )
