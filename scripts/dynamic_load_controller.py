#!/usr/bin/env python3
"""
Dynamic Load Controller for Locust

This script controls Locust via its REST API to generate dynamic load patterns:
1. Sudden spike: Rapid increase from 10 to 100 users, then back to 10
2. Gradual ramp: Linear increase from 1 to 100 users over time
3. Periodic pattern: Sinusoidal variation between 10-100 users
4. Random fluctuation: Random user count within defined bounds

Usage:
  python dynamic_load_controller.py --pattern [spike|ramp|periodic|random] --duration <minutes> --host <locust-host>
"""

import argparse
import requests
import time
import math
import random
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocustController:
    """Controller for Locust load testing via REST API"""

    def __init__(self, host="http://localhost:8089", target_host=None):
        """
        Initialize the controller

        Args:
            host: Locust web interface URL
            target_host: Target system URL for Locust to test
        """
        self.host = host.rstrip('/')
        self.target_host = target_host
        self.stats_history = []
        self.user_count_history = []
        self.timestamps = []

    def start_test(self, user_count=10, spawn_rate=10):
        """
        Start a new Locust test

        Args:
            user_count: Initial number of users
            spawn_rate: Rate at which to spawn users

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Starting test with {user_count} users at spawn rate {spawn_rate}")

        payload = {
            'user_count': user_count,
            'spawn_rate': spawn_rate
        }

        if self.target_host:
            payload['host'] = self.target_host

        try:
            response = requests.post(f"{self.host}/swarm", data=payload)
            if response.status_code == 200:
                logger.info("Test started successfully")
                return True
            else:
                logger.error(f"Failed to start test: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error starting test: {str(e)}")
            return False

    def stop_test(self):
        """
        Stop the current Locust test

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Stopping test")

        try:
            response = requests.get(f"{self.host}/stop")
            if response.status_code == 200:
                logger.info("Test stopped successfully")
                return True
            else:
                logger.error(f"Failed to stop test: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error stopping test: {str(e)}")
            return False

    def update_user_count(self, user_count, spawn_rate=None, record_history=True):
        """
        Update the number of users during a test

        Args:
            user_count: New number of users
            spawn_rate: Rate at which to spawn users (optional)
            record_history: Whether to record this change in history (optional)

        Returns:
            bool: True if successful, False otherwise
        """
        if spawn_rate is None:
            # Default spawn rate is the same as user count for faster adjustment
            spawn_rate = max(1, user_count)

        logger.info(f"Updating user count to {user_count} with spawn rate {spawn_rate}")

        payload = {
            'user_count': user_count,
            'spawn_rate': spawn_rate
        }

        try:
            response = requests.post(f"{self.host}/swarm", data=payload)
            if response.status_code == 200:
                if record_history:
                    self.user_count_history.append(user_count)
                    self.timestamps.append(datetime.now())
                logger.info(f"User count updated to {user_count}")
                return True
            else:
                logger.error(f"Failed to update user count: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error updating user count: {str(e)}")
            return False

    def get_current_stats(self):
        """
        Get current test statistics

        Returns:
            dict: Current statistics or None if failed
        """
        try:
            response = requests.get(f"{self.host}/stats/requests")
            if response.status_code == 200:
                stats = response.json()
                self.stats_history.append(stats)
                return stats
            else:
                logger.error(f"Failed to get stats: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return None

    def save_results(self, filename_prefix):
        """
        Save test results to files

        Args:
            filename_prefix: Prefix for the output files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = f"{filename_prefix}_{timestamp}_stats.json"
        users_file = f"{filename_prefix}_{timestamp}_users.json"
        plot_file = f"{filename_prefix}_{timestamp}_plot.png"

        # Save stats history
        with open(stats_file, 'w') as f:
            json.dump(self.stats_history, f, indent=2)
        logger.info(f"Stats saved to {stats_file}")

        # Save user count history
        user_data = {
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'user_counts': self.user_count_history
        }
        with open(users_file, 'w') as f:
            json.dump(user_data, f, indent=2)
        logger.info(f"User count history saved to {users_file}")

        # Create and save plot
        self.create_plot(plot_file)
        logger.info(f"Plot saved to {plot_file}")

    def create_plot(self, filename):
        """
        Create a plot of user count and response times

        Args:
            filename: Output file name
        """
        if not self.timestamps or not self.stats_history:
            logger.warning("No data to plot")
            return

        # Extract response times from stats history
        response_times = []
        for stats in self.stats_history:
            if 'total' in stats and 'current_response_time_percentile_95' in stats['total']:
                response_times.append(stats['total']['current_response_time_percentile_95'])
            else:
                response_times.append(None)

        # Create relative time labels (t1, t2, t3, ...)
        time_labels = [f't{i+1}' for i in range(len(self.timestamps))]

        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot user count on left y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Time Points')
        ax1.set_ylabel('User Count', color=color)
        ax1.plot(time_labels, self.user_count_history, color=color, marker='o', linewidth=2, markersize=6)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        # Plot response time on right y-axis
        if any(rt is not None for rt in response_times):
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('P95 Response Time (ms)', color=color)
            ax2.plot(time_labels, response_times, color=color, marker='x', linewidth=2, markersize=6)
            ax2.tick_params(axis='y', labelcolor=color)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add title and adjust layout
        plt.title('User Count and Response Time Over Time')
        fig.tight_layout()

        # Save plot
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def run_spike_pattern(controller, duration_minutes=15, min_users=10, max_users=100):
    """
    Run a sudden spike load pattern

    Args:
        controller: LocustController instance
        duration_minutes: Total test duration in minutes
        min_users: Minimum number of users
        max_users: Maximum number of users during spike
    """
    logger.info(f"Running spike pattern: {min_users}->{max_users}->{min_users} users over {duration_minutes} minutes")

    # Calculate timings
    total_seconds = duration_minutes * 60
    baseline_seconds = total_seconds * 0.3   # 30% of time at baseline
    peak_seconds = total_seconds * 0.4       # 40% of time at peak
    final_seconds = total_seconds * 0.3      # 30% of time back at baseline

    # Start with min_users
    controller.start_test(user_count=min_users, spawn_rate=min_users)

    # Record baseline phase with regular intervals
    baseline_steps = 6
    step_duration = baseline_seconds / baseline_steps
    for i in range(baseline_steps):
        controller.user_count_history.append(min_users)
        controller.timestamps.append(datetime.now())
        controller.get_current_stats()
        time.sleep(step_duration)

    # Rapid ramp up to max_users
    controller.update_user_count(max_users, spawn_rate=max_users, record_history=False)
    time.sleep(2)  # Brief pause for user count to stabilize

    # Record peak phase with regular intervals
    peak_steps = 8
    step_duration = peak_seconds / peak_steps
    for i in range(peak_steps):
        controller.user_count_history.append(max_users)
        controller.timestamps.append(datetime.now())
        controller.get_current_stats()
        time.sleep(step_duration)

    # Rapid ramp down to min_users
    controller.update_user_count(min_users, spawn_rate=max_users, record_history=False)
    time.sleep(2)  # Brief pause for user count to stabilize

    # Record final phase with regular intervals
    final_steps = 6
    step_duration = final_seconds / final_steps
    for i in range(final_steps):
        controller.user_count_history.append(min_users)
        controller.timestamps.append(datetime.now())
        controller.get_current_stats()
        time.sleep(step_duration)

    # Get final stats
    controller.get_current_stats()
    controller.stop_test()

def run_gradual_ramp_pattern(controller, duration_minutes=20, min_users=1, max_users=100):
    """
    Run a gradual ramp load pattern

    Args:
        controller: LocustController instance
        duration_minutes: Total test duration in minutes
        min_users: Starting number of users
        max_users: Maximum number of users
    """
    logger.info(f"Running gradual ramp pattern: {min_users}->{max_users} users over {duration_minutes} minutes")

    # Calculate timings
    total_seconds = duration_minutes * 60
    step_count = 20  # Number of steps to increase users
    step_seconds = total_seconds / step_count

    # Start with min_users
    controller.start_test(user_count=min_users, spawn_rate=min_users)
    time.sleep(5)  # Short delay to ensure test starts

    # Gradually increase users
    for i in range(1, step_count + 1):
        user_count = min_users + int((max_users - min_users) * (i / step_count))
        controller.update_user_count(user_count)
        controller.get_current_stats()
        time.sleep(step_seconds)

    # Get final stats
    controller.get_current_stats()
    controller.stop_test()

def run_periodic_pattern(controller, duration_minutes=30, min_users=10, max_users=100, cycles=3):
    """
    Run a periodic (sinusoidal) load pattern

    Args:
        controller: LocustController instance
        duration_minutes: Total test duration in minutes
        min_users: Minimum number of users
        max_users: Maximum number of users
        cycles: Number of complete sine wave cycles
    """
    logger.info(f"Running periodic pattern: {min_users}-{max_users} users with {cycles} cycles over {duration_minutes} minutes")

    # Calculate timings
    total_seconds = duration_minutes * 60
    step_count = 60  # Number of steps (adjust for smoother curve)
    step_seconds = total_seconds / step_count

    # Start with min_users
    controller.start_test(user_count=min_users, spawn_rate=min_users)
    time.sleep(5)  # Short delay to ensure test starts

    # Run sinusoidal pattern
    for i in range(step_count):
        # Calculate user count using sine wave
        progress = i / step_count
        angle = progress * cycles * 2 * math.pi
        user_count = min_users + int((max_users - min_users) * (0.5 + 0.5 * math.sin(angle)))

        controller.update_user_count(user_count)
        controller.get_current_stats()
        time.sleep(step_seconds)

    # Get final stats
    controller.get_current_stats()
    controller.stop_test()

def run_random_pattern(controller, duration_minutes=25, min_users=10, max_users=100, change_interval_seconds=30):
    """
    Run a random fluctuation load pattern

    Args:
        controller: LocustController instance
        duration_minutes: Total test duration in minutes
        min_users: Minimum number of users
        max_users: Maximum number of users
        change_interval_seconds: Seconds between user count changes
    """
    logger.info(f"Running random pattern: {min_users}-{max_users} users changing every {change_interval_seconds}s over {duration_minutes} minutes")

    # Calculate timings
    total_seconds = duration_minutes * 60
    steps = int(total_seconds / change_interval_seconds)

    # Start with random user count
    initial_users = random.randint(min_users, max_users)
    controller.start_test(user_count=initial_users, spawn_rate=initial_users)
    time.sleep(5)  # Short delay to ensure test starts

    # Run random pattern
    for _ in range(steps):
        user_count = random.randint(min_users, max_users)
        controller.update_user_count(user_count)
        controller.get_current_stats()
        time.sleep(change_interval_seconds)

    # Get final stats
    controller.get_current_stats()
    controller.stop_test()

def main():
    """Main function to parse arguments and run the selected load pattern"""
    parser = argparse.ArgumentParser(description='Dynamic Load Controller for Locust')
    parser.add_argument('--pattern', type=str, required=True, choices=['spike', 'ramp', 'periodic', 'random'],
                        help='Load pattern to generate')
    parser.add_argument('--duration', type=int, default=15,
                        help='Test duration in minutes')
    parser.add_argument('--min-users', type=int, default=10,
                        help='Minimum number of users')
    parser.add_argument('--max-users', type=int, default=100,
                        help='Maximum number of users')
    parser.add_argument('--host', type=str, default='http://localhost:8089',
                        help='Locust web interface URL')
    parser.add_argument('--target-host', type=str,
                        default='http://mobilenetv4-triton-svc.workloads.svc.cluster.local:8000',
                        help='Target system URL for Locust to test')
    parser.add_argument('--output-dir', type=str, default='../../results/dynamic',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create controller
    controller = LocustController(host=args.host, target_host=args.target_host)

    # Run selected pattern
    try:
        if args.pattern == 'spike':
            run_spike_pattern(controller, args.duration, args.min_users, args.max_users)
        elif args.pattern == 'ramp':
            run_gradual_ramp_pattern(controller, args.duration, args.min_users, args.max_users)
        elif args.pattern == 'periodic':
            run_periodic_pattern(controller, args.duration, args.min_users, args.max_users)
        elif args.pattern == 'random':
            run_random_pattern(controller, args.duration, args.min_users, args.max_users)

        # Save results
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        output_prefix = os.path.join(args.output_dir, f"{args.pattern}")
        controller.save_results(output_prefix)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        controller.stop_test()
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        controller.stop_test()

if __name__ == "__main__":
    main()
