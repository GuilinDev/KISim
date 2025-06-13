#!/usr/bin/env python3
"""
Main training script for RL-based Kubernetes resource management
Based on baseline experimental findings and optimized for GPU/CPU scheduling.
"""

import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from kubernetes_env import KubernetesRLEnvironment, EnvironmentConfig, LoadPattern
from ppo_agent import PPOAgent, create_ppo_config
from env_utils import LoadPatternGenerator, create_environment_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RLTrainer:
    """
    RL Trainer for Kubernetes Resource Management

    Implements training loop with:
    - Load pattern variation based on baseline findings
    - Performance tracking and comparison with baselines
    - Model checkpointing and evaluation
    """

    def __init__(self, config: Dict):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory
        self.results_dir = f"/home/guilin/allProjects/ecrl/experiments/rl/training_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize environment
        env_config = EnvironmentConfig(**config['environment'])
        self.env = KubernetesRLEnvironment(env_config)

        # Initialize agent
        ppo_config = create_ppo_config(**config['agent'])
        device = "cuda" if torch.cuda.is_available() and config.get('use_gpu', True) else "cpu"
        self.agent = PPOAgent(ppo_config, device=device)

        # Training tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_summaries = []
        self.training_metrics = []

        logger.info(f"Initialized RL trainer with device: {device}")
        logger.info(f"Results will be saved to: {self.results_dir}")

    def train(self, num_episodes: int = 100, save_interval: int = 10, eval_interval: int = 20):
        """
        Main training loop

        Args:
            num_episodes: Number of training episodes
            save_interval: Episodes between model saves
            eval_interval: Episodes between evaluations
        """

        logger.info(f"Starting RL training for {num_episodes} episodes")

        best_reward = float('-inf')

        for episode in range(num_episodes):
            episode_start_time = time.time()

            # Run training episode
            episode_reward, episode_length, episode_summary = self._run_episode(episode, training=True)

            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_summaries.append(episode_summary)

            # Update agent
            if episode > 0:  # Skip first episode for buffer warmup
                training_metrics = self.agent.update()
                self.training_metrics.append(training_metrics)

            # Logging
            episode_time = time.time() - episode_start_time
            avg_reward = np.mean(self.episode_rewards[-10:])  # Last 10 episodes

            logger.info(f"Episode {episode + 1}/{num_episodes} - "
                       f"Reward: {episode_reward:.3f}, "
                       f"Length: {episode_length}, "
                       f"Avg Reward (10): {avg_reward:.3f}, "
                       f"Time: {episode_time:.1f}s, "
                       f"Pattern: {episode_summary.get('load_pattern', 'unknown')}")

            # Save model periodically
            if (episode + 1) % save_interval == 0:
                model_path = os.path.join(self.results_dir, f"model_episode_{episode + 1}.pt")
                self.agent.save_model(model_path)

                # Save best model
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    best_model_path = os.path.join(self.results_dir, "best_model.pt")
                    self.agent.save_model(best_model_path)
                    logger.info(f"New best model saved with reward: {best_reward:.3f}")

            # Evaluation
            if (episode + 1) % eval_interval == 0:
                self._evaluate_agent(episode + 1)

            # Save training progress
            self._save_training_progress()

        logger.info("Training completed!")

        # Final evaluation and analysis
        self._final_evaluation()

    def _run_episode(self, episode: int, training: bool = True) -> tuple:
        """
        Run a single episode

        Args:
            episode: Episode number
            training: Whether this is a training episode

        Returns:
            episode_reward: Total reward for episode
            episode_length: Number of steps in episode
            episode_summary: Episode summary statistics
        """

        # Reset environment
        observation = self.env.reset()

        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Get action from agent
            if training:
                action, log_prob, value = self.agent.get_action(observation, deterministic=False)
            else:
                action, log_prob, value = self.agent.get_action(observation, deterministic=True)

            # Take step in environment
            next_observation, reward, done, info = self.env.step(action)

            # Store transition (only during training)
            if training:
                self.agent.store_transition(observation, action, reward, value, log_prob, done)

            # Update tracking
            episode_reward += reward
            episode_length += 1
            observation = next_observation

            # Log step info
            if episode_length % 5 == 0:  # Log every 5 steps
                logger.debug(f"Episode {episode + 1}, Step {episode_length}: "
                           f"Reward: {reward:.3f}, "
                           f"Action: {action}, "
                           f"Latency: {info.get('performance_metrics', {}).get('latency_p95', 0):.1f}ms")

        # Get episode summary
        episode_summary = self.env.get_episode_summary()
        episode_summary['episode_reward'] = episode_reward
        episode_summary['training'] = training

        return episode_reward, episode_length, episode_summary

    def _evaluate_agent(self, episode: int):
        """Evaluate agent performance"""
        logger.info(f"Evaluating agent at episode {episode}")

        # Set agent to evaluation mode
        self.agent.set_eval_mode()

        eval_rewards = []
        eval_summaries = []

        # Run evaluation episodes (one for each load pattern)
        load_patterns = ['ramp', 'spike', 'periodic', 'random']

        for pattern in load_patterns:
            # Force specific load pattern for evaluation
            self.env.current_load_pattern = LoadPattern(pattern)

            eval_reward, eval_length, eval_summary = self._run_episode(episode, training=False)
            eval_rewards.append(eval_reward)
            eval_summaries.append(eval_summary)

            logger.info(f"Eval {pattern}: Reward={eval_reward:.3f}, "
                       f"Avg Latency={eval_summary.get('avg_latency', 0):.1f}ms, "
                       f"Avg Throughput={eval_summary.get('avg_throughput', 0):.1f} req/s")

        # Save evaluation results
        eval_results = {
            'episode': episode,
            'timestamp': time.time(),
            'rewards': eval_rewards,
            'summaries': eval_summaries,
            'avg_reward': np.mean(eval_rewards)
        }

        eval_file = os.path.join(self.results_dir, f"evaluation_episode_{episode}.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)

        # Set agent back to training mode
        self.agent.set_train_mode()

    def _save_training_progress(self):
        """Save training progress and metrics"""

        progress = {
            'timestamp': time.time(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_summaries': self.episode_summaries,
            'training_metrics': self.training_metrics,
            'agent_stats': self.agent.get_training_stats()
        }

        progress_file = os.path.join(self.results_dir, "training_progress.json")
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2, default=str)

        # Generate training plots
        self._generate_training_plots()

    def _generate_training_plots(self):
        """Generate training visualization plots"""

        if len(self.episode_rewards) < 2:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Episode rewards
        episodes = range(1, len(self.episode_rewards) + 1)
        ax1.plot(episodes, self.episode_rewards, alpha=0.7, label='Episode Reward')
        if len(self.episode_rewards) >= 10:
            # Moving average
            window = min(10, len(self.episode_rewards) // 2)
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(self.episode_rewards) + 1), moving_avg,
                    color='red', linewidth=2, label=f'Moving Avg ({window})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Episode lengths
        ax2.plot(episodes, self.episode_lengths, alpha=0.7, color='green')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths')
        ax2.grid(True, alpha=0.3)

        # Training losses (if available)
        if self.training_metrics:
            policy_losses = [m.get('policy_loss', 0) for m in self.training_metrics if m]
            value_losses = [m.get('value_loss', 0) for m in self.training_metrics if m]

            if policy_losses:
                ax3.plot(policy_losses, label='Policy Loss', alpha=0.7)
                ax3.plot(value_losses, label='Value Loss', alpha=0.7)
                ax3.set_xlabel('Update')
                ax3.set_ylabel('Loss')
                ax3.set_title('Training Losses')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

        # Performance by load pattern
        if self.episode_summaries:
            pattern_rewards = {}
            for summary in self.episode_summaries:
                pattern = summary.get('load_pattern', 'unknown')
                reward = summary.get('episode_reward', 0)
                if pattern not in pattern_rewards:
                    pattern_rewards[pattern] = []
                pattern_rewards[pattern].append(reward)

            patterns = list(pattern_rewards.keys())
            avg_rewards = [np.mean(pattern_rewards[p]) for p in patterns]

            ax4.bar(patterns, avg_rewards, alpha=0.7)
            ax4.set_xlabel('Load Pattern')
            ax4.set_ylabel('Average Reward')
            ax4.set_title('Performance by Load Pattern')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, "training_progress.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _final_evaluation(self):
        """Perform comprehensive final evaluation"""
        logger.info("Performing final evaluation...")

        # Load best model
        best_model_path = os.path.join(self.results_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            self.agent.load_model(best_model_path)

        # Comprehensive evaluation across all patterns
        final_results = {}

        for pattern in ['ramp', 'spike', 'periodic', 'random']:
            pattern_results = []

            # Run multiple evaluation episodes for each pattern
            for run in range(5):
                self.env.current_load_pattern = LoadPattern(pattern)
                reward, length, summary = self._run_episode(0, training=False)
                pattern_results.append(summary)

            final_results[pattern] = pattern_results

        # Save final results
        final_file = os.path.join(self.results_dir, "final_evaluation.json")
        with open(final_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        logger.info(f"Final evaluation completed. Results saved to {final_file}")

def main():
    parser = argparse.ArgumentParser(description='Train RL agent for Kubernetes resource management')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--config', type=str, default=None, help='Configuration file path')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    # Default configuration
    config = {
        'environment': {
            'namespace': 'workloads',
            'prometheus_url': 'http://localhost:9090',
            'locust_url': 'http://localhost:8089',
            'episode_duration': 300,
            'action_interval': 30
        },
        'agent': {
            'learning_rate': 3e-4,
            'hidden_size': 256,
            'buffer_size': 2048,
            'batch_size': 64,
            'ppo_epochs': 4
        },
        'use_gpu': args.gpu
    }

    # Load custom configuration if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)

    # Initialize and run trainer
    trainer = RLTrainer(config)
    trainer.train(num_episodes=args.episodes)

if __name__ == "__main__":
    main()
