#!/usr/bin/env python3
"""
RL-integrated load controller for training episodes
Combines dynamic load generation with RL agent training.
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dynamic_load_controller import DynamicLoadController, LoadPattern

logger = logging.getLogger(__name__)

@dataclass
class RLLoadConfig:
    """Configuration for RL-integrated load controller"""
    base_url: str = "http://localhost:8089"
    patterns: List[str] = None
    episode_duration: int = 300
    step_interval: int = 30
    ramp_up_time: int = 30
    results_dir: str = "/home/guilin/allProjects/ecrl/experiments/rl"
    
    def __post_init__(self):
        if self.patterns is None:
            self.patterns = ["ramp", "spike", "periodic", "random"]

class RLLoadController:
    """
    RL-integrated load controller
    
    Manages dynamic load patterns during RL training episodes:
    - Coordinates with RL agent training steps
    - Provides load pattern variation for robust training
    - Collects performance metrics for reward calculation
    """
    
    def __init__(self, config: RLLoadConfig):
        self.config = config
        self.load_controller = DynamicLoadController(config.base_url)
        
        # Episode state
        self.current_episode = 0
        self.current_pattern = None
        self.episode_start_time = None
        self.episode_metrics = []
        
        # Load pattern rotation
        self.pattern_index = 0
        
        logger.info("Initialized RL load controller")
        
    def start_episode(self, episode: int, pattern: Optional[str] = None) -> str:
        """
        Start a new training episode with specified or random load pattern
        
        Args:
            episode: Episode number
            pattern: Specific pattern to use (if None, rotates through patterns)
            
        Returns:
            selected_pattern: The pattern that was started
        """
        
        self.current_episode = episode
        self.episode_start_time = time.time()
        self.episode_metrics = []
        
        # Select load pattern
        if pattern is None:
            # Rotate through patterns for diverse training
            selected_pattern = self.config.patterns[self.pattern_index % len(self.config.patterns)]
            self.pattern_index += 1
        else:
            selected_pattern = pattern
            
        self.current_pattern = selected_pattern
        
        logger.info(f"Starting episode {episode} with pattern: {selected_pattern}")
        
        # Start the load pattern
        self._start_load_pattern(selected_pattern)
        
        return selected_pattern
        
    def _start_load_pattern(self, pattern: str):
        """Start the specified load pattern"""
        
        try:
            if pattern == "ramp":
                self.load_controller.start_ramp_pattern(
                    duration=self.config.episode_duration,
                    min_users=14,
                    max_users=100
                )
            elif pattern == "spike":
                self.load_controller.start_spike_pattern(
                    duration=self.config.episode_duration,
                    base_users=10,
                    spike_users=100,
                    spike_duration=30
                )
            elif pattern == "periodic":
                self.load_controller.start_periodic_pattern(
                    duration=self.config.episode_duration,
                    min_users=10,
                    max_users=100,
                    period=60
                )
            elif pattern == "random":
                self.load_controller.start_random_pattern(
                    duration=self.config.episode_duration,
                    min_users=13,
                    max_users=96
                )
            else:
                logger.error(f"Unknown pattern: {pattern}")
                return
                
            logger.info(f"Started {pattern} load pattern")
            
        except Exception as e:
            logger.error(f"Error starting load pattern {pattern}: {e}")
            
    def get_current_metrics(self) -> Dict:
        """Get current load and performance metrics"""
        
        try:
            # Get current load metrics from Locust
            stats = self.load_controller.get_current_stats()
            
            # Calculate episode progress
            elapsed_time = time.time() - self.episode_start_time if self.episode_start_time else 0
            episode_progress = min(1.0, elapsed_time / self.config.episode_duration)
            
            metrics = {
                'episode': self.current_episode,
                'pattern': self.current_pattern,
                'elapsed_time': elapsed_time,
                'episode_progress': episode_progress,
                'timestamp': time.time()
            }
            
            # Add Locust stats if available
            if stats:
                metrics.update({
                    'current_users': stats.get('user_count', 0),
                    'current_rps': stats.get('current_rps', 0),
                    'avg_response_time': stats.get('avg_response_time', 0),
                    'p95_response_time': stats.get('current_response_time_percentile_95', 0),
                    'total_requests': stats.get('num_requests', 0),
                    'failed_requests': stats.get('num_failures', 0)
                })
                
            # Store metrics for episode summary
            self.episode_metrics.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {
                'episode': self.current_episode,
                'pattern': self.current_pattern,
                'error': str(e),
                'timestamp': time.time()
            }
            
    def end_episode(self) -> Dict:
        """End current episode and return summary"""
        
        try:
            # Stop load generation
            self.load_controller.stop_load()
            
            # Calculate episode summary
            episode_summary = self._calculate_episode_summary()
            
            # Save episode results
            self._save_episode_results(episode_summary)
            
            logger.info(f"Episode {self.current_episode} completed. "
                       f"Pattern: {self.current_pattern}, "
                       f"Duration: {episode_summary.get('duration', 0):.1f}s")
            
            return episode_summary
            
        except Exception as e:
            logger.error(f"Error ending episode: {e}")
            return {'error': str(e)}
            
    def _calculate_episode_summary(self) -> Dict:
        """Calculate summary statistics for the episode"""
        
        if not self.episode_metrics:
            return {}
            
        # Extract metrics
        response_times = [m.get('avg_response_time', 0) for m in self.episode_metrics if 'avg_response_time' in m]
        p95_times = [m.get('p95_response_time', 0) for m in self.episode_metrics if 'p95_response_time' in m]
        rps_values = [m.get('current_rps', 0) for m in self.episode_metrics if 'current_rps' in m]
        user_counts = [m.get('current_users', 0) for m in self.episode_metrics if 'current_users' in m]
        
        summary = {
            'episode': self.current_episode,
            'pattern': self.current_pattern,
            'duration': time.time() - self.episode_start_time if self.episode_start_time else 0,
            'total_metrics_collected': len(self.episode_metrics),
            'start_time': self.episode_start_time,
            'end_time': time.time()
        }
        
        # Performance statistics
        if response_times:
            summary.update({
                'avg_response_time_mean': np.mean(response_times),
                'avg_response_time_std': np.std(response_times),
                'avg_response_time_min': np.min(response_times),
                'avg_response_time_max': np.max(response_times)
            })
            
        if p95_times:
            summary.update({
                'p95_response_time_mean': np.mean(p95_times),
                'p95_response_time_std': np.std(p95_times),
                'p95_response_time_min': np.min(p95_times),
                'p95_response_time_max': np.max(p95_times)
            })
            
        if rps_values:
            summary.update({
                'throughput_mean': np.mean(rps_values),
                'throughput_std': np.std(rps_values),
                'throughput_min': np.min(rps_values),
                'throughput_max': np.max(rps_values)
            })
            
        if user_counts:
            summary.update({
                'user_count_mean': np.mean(user_counts),
                'user_count_std': np.std(user_counts),
                'user_count_min': np.min(user_counts),
                'user_count_max': np.max(user_counts)
            })
            
        # Final metrics from last measurement
        if self.episode_metrics:
            final_metrics = self.episode_metrics[-1]
            summary.update({
                'final_total_requests': final_metrics.get('total_requests', 0),
                'final_failed_requests': final_metrics.get('failed_requests', 0),
                'final_success_rate': (
                    (final_metrics.get('total_requests', 0) - final_metrics.get('failed_requests', 0)) /
                    max(final_metrics.get('total_requests', 1), 1) * 100
                )
            })
            
        return summary
        
    def _save_episode_results(self, summary: Dict):
        """Save episode results to file"""
        
        try:
            # Create episode-specific directory
            episode_dir = os.path.join(self.config.results_dir, f"episode_{self.current_episode}")
            os.makedirs(episode_dir, exist_ok=True)
            
            # Save episode summary
            summary_file = os.path.join(episode_dir, "episode_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
            # Save detailed metrics
            metrics_file = os.path.join(episode_dir, "episode_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.episode_metrics, f, indent=2, default=str)
                
            logger.debug(f"Episode results saved to {episode_dir}")
            
        except Exception as e:
            logger.error(f"Error saving episode results: {e}")
            
    def get_pattern_for_episode(self, episode: int) -> str:
        """Get the pattern that should be used for a specific episode"""
        
        # Cycle through patterns to ensure balanced training
        pattern_index = episode % len(self.config.patterns)
        return self.config.patterns[pattern_index]
        
    def is_episode_complete(self) -> bool:
        """Check if current episode should be completed"""
        
        if not self.episode_start_time:
            return False
            
        elapsed_time = time.time() - self.episode_start_time
        return elapsed_time >= self.config.episode_duration
        
    def get_episode_progress(self) -> float:
        """Get current episode progress (0.0 to 1.0)"""
        
        if not self.episode_start_time:
            return 0.0
            
        elapsed_time = time.time() - self.episode_start_time
        return min(1.0, elapsed_time / self.config.episode_duration)
        
    def cleanup(self):
        """Clean up resources"""
        
        try:
            # Stop any running load
            self.load_controller.stop_load()
            logger.info("RL load controller cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def create_rl_load_config(
    episode_duration: int = 300,
    step_interval: int = 30,
    patterns: List[str] = None,
    results_dir: str = "/home/guilin/allProjects/ecrl/experiments/rl"
) -> RLLoadConfig:
    """Create RL load configuration with sensible defaults"""
    
    if patterns is None:
        patterns = ["ramp", "spike", "periodic", "random"]
        
    return RLLoadConfig(
        episode_duration=episode_duration,
        step_interval=step_interval,
        patterns=patterns,
        results_dir=results_dir
    )

# Example usage for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RL load controller')
    parser.add_argument('--pattern', type=str, default='ramp', help='Load pattern to test')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and test controller
    config = create_rl_load_config(episode_duration=args.duration)
    controller = RLLoadController(config)
    
    try:
        # Start test episode
        pattern = controller.start_episode(0, args.pattern)
        print(f"Started test episode with pattern: {pattern}")
        
        # Monitor for duration
        start_time = time.time()
        while time.time() - start_time < args.duration:
            metrics = controller.get_current_metrics()
            print(f"Progress: {controller.get_episode_progress():.2f}, "
                  f"Users: {metrics.get('current_users', 0)}, "
                  f"RPS: {metrics.get('current_rps', 0):.1f}")
            time.sleep(10)
            
        # End episode
        summary = controller.end_episode()
        print(f"Episode completed: {summary}")
        
    finally:
        controller.cleanup()
