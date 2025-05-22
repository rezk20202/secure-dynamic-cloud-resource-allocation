#!/usr/bin/env python3
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.environment import CloudEnvironment
from src.drl_agent import DRLAgent
from src.ids import IntrusionDetectionSystem

# Create output directories
model_dir = 'models'
viz_dir = 'output/visualizations'
for directory in [model_dir, viz_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Configuration
num_vms = 10
num_resources = 4
security_weight = 0.3
episodes = 50

print("Starting integrated training mode with IDS...")

# Create environment
env = CloudEnvironment(num_vms, num_resources, security_weight)

# Create DRL agent
state_size = env.observation_space.shape[0]
agent = DRLAgent(state_size, env.action_space)

# Create IDS but don't start monitoring thread
print("Training IDS...")
ids = IntrusionDetectionSystem(env)

# Manually train the IDS without starting threads
print("Generating training data for IDS...")
X, y = ids.generate_training_data(num_samples=500)

# Train all the IDS models
print("Training IDS models...")
ids._train_ddos_detector(ids._generate_normal_data(300), 100)
ids._train_cryptojacking_detector(ids._generate_normal_data(300), 100)
ids._train_malware_detector(ids._generate_normal_data(300), 100)
ids._train_insider_detector(ids._generate_normal_data(300), 100)

print("IDS training completed")

# Track metrics for visualization
all_rewards = []
all_security_incidents = []
all_resource_utils = []
all_availability = []

# Train the DRL agent
print(f"Training DRL agent for {episodes} episodes...")
for episode in range(episodes):
    # Reset environment
    state = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    
    # Run episode
    while not done:
        # Choose action
        action = agent.act(state)
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Check for security manually using rule-based detection only
        if steps % 5 == 0:  # Check every 5 steps
            vm_features = ids._extract_features()
            threats = ids._rule_based_detection(vm_features)
            
            if threats:
                # Update environment with detected threats
                vm_indices = [threat['vm_idx'] for threat in threats]
                threat_types = [threat['type'] for threat in threats]
                severity_scores = [threat['severity'] for threat in threats]
                env.update_security_metrics(vm_indices, threat_types, severity_scores)
        
        # Remember experience
        agent.remember(state, action, reward, next_state, done)
        
        # Update state
        state = next_state
        episode_reward += reward
        steps += 1
        
        # Train agent
        if len(agent.memory) > agent.batch_size:
            agent.replay()
    
    # Track metrics for visualization
    all_rewards.append(episode_reward)
    all_security_incidents.append(info['security_incidents'])
    all_resource_utils.append(info['resource_utilization'])
    all_availability.append(info['availability'])
    
    # Print progress
    if (episode + 1) % 10 == 0:
        print(f"Episode: {episode+1}/{episodes}, Reward: {episode_reward:.2f}, "
              f"Security Incidents: {info['security_incidents']}, "
              f"Resource Utilization: {info['resource_utilization']:.2f}, "
              f"Availability: {info['availability']:.2f}")

# Save the model
agent.save_model(os.path.join(model_dir, 'drl_model.h5'))
print(f"Model saved to {os.path.join(model_dir, 'drl_model.h5')}")

# Create visualizations
print("Generating training visualizations...")
plt.figure(figsize=(15, 10))

# Plot rewards
plt.subplot(2, 2, 1)
plt.plot(range(1, episodes+1), all_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episode Rewards')

# Plot security incidents
plt.subplot(2, 2, 2)
plt.plot(range(1, episodes+1), all_security_incidents)
plt.xlabel('Episode')
plt.ylabel('Security Incidents')
plt.title('Security Incidents per Episode')

# Plot resource utilization
plt.subplot(2, 2, 3)
plt.plot(range(1, episodes+1), all_resource_utils)
plt.xlabel('Episode')
plt.ylabel('Resource Utilization')
plt.title('Average Resource Utilization')

# Plot availability
plt.subplot(2, 2, 4)
plt.plot(range(1, episodes+1), all_availability)
plt.xlabel('Episode')
plt.ylabel('Availability')
plt.title('System Availability')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'training_metrics.png'))
print(f"Visualizations saved to {os.path.join(viz_dir, 'training_metrics.png')}")

print("Training completed successfully.")