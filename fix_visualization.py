#!/usr/bin/env python3
import os
import sys
import time
from src.secure_manager import SecureCloudManager
from src.visualizer import Visualizer

# Create the secure cloud manager
print("Creating secure cloud manager...")
manager = SecureCloudManager(num_vms=10, num_resources=4, security_weight=0.3)

# Load the trained model
print("Loading trained model...")
try:
    manager.load_models(drl_path="models/drl_model.h5")
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("Will train a small model")
    manager._train_ids()
    manager.train(episodes=5)

# Create visualizer
visualizer = Visualizer(output_dir='output/visualizations')

# Start the manager
print("Starting secure cloud manager...")
manager.start(train_first=False)

# Allow time for system to initialize
print("Waiting for system initialization...")
time.sleep(5)

# Get initial status and update visualizer
print("Getting initial system status...")
status = manager.get_system_status()
visualizer.update_history(status)

# Replace the attack simulation section with this more diverse approach
# Create a mix of different attacks with varying severities
print("Simulating various attacks with different severities...")

# Multiple DDoS attacks with different severities
print("Simulating DDoS attacks...")
for i, severity in enumerate([0.9, 0.7, 0.5]):
    manager.simulate_attack('ddos', vm_idx=(i % 10), severity=severity)
    print(f"Simulated DDoS attack on VM {i % 10} with severity {severity}")
    time.sleep(2)  # Wait between attacks
    
    status = manager.get_system_status()
    visualizer.update_history(status)

# Fewer cryptojacking attacks with high severity
print("Simulating cryptojacking attack...")
manager.simulate_attack('cryptojacking', vm_idx=6, severity=0.85)
time.sleep(2)

status = manager.get_system_status()
visualizer.update_history(status)

# Several malware attacks
print("Simulating malware attacks...")
for i, severity in enumerate([0.8, 0.6, 0.75, 0.9]):
    manager.simulate_attack('malware', vm_idx=(i+2) % 10, severity=severity)
    print(f"Simulated malware attack on VM {(i+2) % 10} with severity {severity}")
    time.sleep(2)
    
    status = manager.get_system_status()
    visualizer.update_history(status)

# Just one insider threat
print("Simulating insider threat...")
manager.simulate_attack('insider', vm_idx=1, severity=0.95)
time.sleep(2)

# Get final status and update visualizer
status = manager.get_system_status()
visualizer.update_history(status)

# Generate threat visualization
print("Generating threat visualization...")
visualizer.plot_threat_analysis(status, show=False, save=True)
print("Visualization saved to output/visualizations/")

# Clean shutdown
print("Stopping secure cloud manager...")
manager.stop()
print("Done")