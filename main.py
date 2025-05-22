#!/usr/bin/env python3
import argparse
import time
import os
import sys
from src.secure_manager import SecureCloudManager
from src.visualizer import Visualizer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Secure Cloud Resource Allocation System')
    
    # System configuration
    parser.add_argument('--num-vms', type=int, default=10,
                        help='Number of VMs to manage (default: 10)')
    parser.add_argument('--num-resources', type=int, default=4,
                        help='Number of resource types (default: 4)')
    parser.add_argument('--security-weight', type=float, default=0.3,
                        help='Weight of security in resource allocation (default: 0.3)')
    
    # Training parameters
    parser.add_argument('--training-episodes', type=int, default=200,
                        help='Number of episodes for training (default: 200)')
    parser.add_argument('--evaluation-episodes', type=int, default=50,
                        help='Number of episodes for evaluation (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for DRL agent training (default: 32)')
    
    # Monitoring parameters
    parser.add_argument('--monitoring-interval', type=float, default=5,
                        help='Interval for security monitoring in seconds (default: 5)')
    parser.add_argument('--visualization-interval', type=float, default=30,
                        help='Interval for visualization updates in seconds (default: 30)')
    
    # Logging and output
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for logs and visualizations (default: output)')
    
    # Operation mode
    parser.add_argument('--mode', type=str, default='run',
                        choices=['train', 'evaluate', 'run'],
                        help='Operation mode (default: run)')
    parser.add_argument('--model-path', type=str, default='models/drl_model.h5',
                        help='Path to save/load model (default: models/drl_model.h5)')
    parser.add_argument('--simulation-time', type=int, default=600,
                        help='Simulation time in seconds for run mode (default: 600)')
    parser.add_argument('--attack-simulation', action='store_true',
                        help='Enable attack simulation during run')
    
    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    return parser.parse_args()

def create_output_directories(args):
    """Create output directories"""
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create visualization directory
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Create log directory
    log_dir = os.path.join(args.output_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create model directory
    model_dir = os.path.dirname(args.model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

def simulate_attack(manager):
    """Simulate a random attack"""
    attack_types = ['ddos', 'cryptojacking', 'malware', 'insider']
    attack_type = attack_types[int(time.time()) % len(attack_types)]
    severity = 0.5 + (time.time() % 10) / 20  # 0.5 to 1.0
    manager.simulate_attack(attack_type, severity=severity)

def train_mode(args):
    """Run in training mode"""
    print("Starting training mode...")
    
    # Create the secure cloud manager
    manager = SecureCloudManager(
        num_vms=args.num_vms,
        num_resources=args.num_resources,
        security_weight=args.security_weight,
        monitoring_interval=args.monitoring_interval,
        log_level=args.log_level
    )
    
    # Create visualizer
    import os.path
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    visualizer = Visualizer(output_dir=viz_dir)
    
    try:
        # Train IDS and DRL agent
        manager._train_ids()
        manager.train(episodes=args.training_episodes)
        
        # Plot training metrics
        visualizer.plot_training_metrics(manager, save=True)
        
        # Save the model
        manager.save_models(drl_path=args.model_path)
        print(f"Model saved to {args.model_path}")
        
        print("Training completed successfully.")
    finally:
        # Ensure all threads are properly terminated
        print("Cleaning up resources...")
        if hasattr(manager, 'ids'):
            # Stop IDS monitoring if it's running
            if hasattr(manager.ids, 'stop_monitoring'):
                manager.ids.stop_monitoring()
            
        # Force exit to ensure all threads terminate
        print("Training mode complete. Exiting.")
        import os
        os._exit(0)  # This will force termination of all threads
 

def evaluate_mode(args):
    """Run in evaluation mode"""
    print("Starting evaluation mode...")
    
    # Create the secure cloud manager
    manager = SecureCloudManager(
        num_vms=args.num_vms,
        num_resources=args.num_resources,
        security_weight=args.security_weight,
        monitoring_interval=args.monitoring_interval,
        log_level=args.log_level
    )
    
    # Load the model
    manager.load_models(drl_path=args.model_path)
    print(f"Model loaded from {args.model_path}")
    
    # Create visualizer
    visualizer = Visualizer(output_dir=os.path.join(args.output_dir, 'visualizations'))
    
    # Start the manager
    manager.start(train_first=False)
    
    try:
        # Run for a number of steps
        print(f"Evaluating for {args.evaluation_episodes} episodes...")
        for i in range(args.evaluation_episodes):
            # Get system status
            status = manager.get_system_status()
            
            # Update visualizer history
            visualizer.update_history(status)
            
            # Plot every 10 episodes
            if (i + 1) % 10 == 0:
                visualizer.plot_system_overview(status, show=False, save=True)
                visualizer.plot_threat_analysis(status, show=False, save=True)
                print(f"Completed {i+1}/{args.evaluation_episodes} episodes")
            
            # Simulate attacks occasionally
            if args.attack_simulation and i % 5 == 0:
                simulate_attack(manager)
            
            # Sleep to simulate episode
            time.sleep(1)
    
    finally:
        # Stop the manager
        manager.stop()
        
        # Export results
        visualizer.export_data()
        print("Evaluation completed successfully.")

def run_mode(args):
    """Run in deployment mode"""
    print("Starting deployment mode...")
    
    # Create the secure cloud manager
    manager = SecureCloudManager(
        num_vms=args.num_vms,
        num_resources=args.num_resources,
        security_weight=args.security_weight,
        monitoring_interval=args.monitoring_interval,
        log_level=args.log_level
    )
    
    # Create visualizer
    visualizer = Visualizer(output_dir=os.path.join(args.output_dir, 'visualizations'))
    
    # Check if model exists
    if os.path.exists(args.model_path):
        # Load existing model
        manager.load_models(drl_path=args.model_path)
        print(f"Model loaded from {args.model_path}")
        train_first = False
    else:
        # Train new model
        print(f"Model not found at {args.model_path}. Training a new model...")
        train_first = True
    
    # Start the manager
    manager.start(train_first=train_first)
    
    try:
        # Interactive mode
        if args.interactive:
            run_interactive_mode(manager, visualizer)
        else:
            # Run for specified time
            start_time = time.time()
            end_time = start_time + args.simulation_time
            
            print(f"Running simulation for {args.simulation_time} seconds...")
            
            # Visualization interval counter
            last_viz_time = start_time
            
            while time.time() < end_time:
                # Get system status
                status = manager.get_system_status()
                
                # Update visualizer history
                visualizer.update_history(status)
                
                # Generate visualizations periodically
                current_time = time.time()
                if current_time - last_viz_time >= args.visualization_interval:
                    visualizer.plot_system_overview(status, show=False, save=True)
                    visualizer.plot_threat_analysis(status, show=False, save=True)
                    
                    # Print progress
                    elapsed = current_time - start_time
                    remaining = end_time - current_time
                    print(f"Simulation progress: {elapsed:.1f}/{args.simulation_time} seconds "
                          f"({elapsed/args.simulation_time*100:.1f}%), "
                          f"{remaining:.1f} seconds remaining")
                    
                    last_viz_time = current_time
                
                # Simulate attacks occasionally
                if args.attack_simulation and int(time.time()) % 60 == 0:
                    simulate_attack(manager)
                
                # Sleep to avoid excessive processing
                time.sleep(1)
            
            print("Simulation completed successfully.")
    
    finally:
        # Stop the manager
        manager.stop()
        
        # Export results
        export_path = visualizer.export_data()
        print(f"Results exported to {export_path}")

def run_interactive_mode(manager, visualizer):
    """Run in interactive mode"""
    print("\nInteractive Mode Commands:")
    print("  status    - Display system status")
    print("  vis       - Show system visualization")
    print("  threat    - Show threat analysis")
    print("  inject    - Inject workload to a VM")
    print("  attack    - Simulate an attack")
    print("  export    - Export data to CSV")
    print("  help      - Show this help message")
    print("  exit      - Exit the program")
    
    while True:
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'exit':
                break
            
            elif command == 'help':
                print("\nAvailable Commands:")
                print("  status    - Display system status")
                print("  vis       - Show system visualization")
                print("  threat    - Show threat analysis")
                print("  inject    - Inject workload to a VM")
                print("  attack    - Simulate an attack")
                print("  export    - Export data to CSV")
                print("  help      - Show this help message")
                print("  exit      - Exit the program")
            
            elif command == 'status':
                status = manager.get_system_status()
                visualizer.update_history(status)
                
                # Display system status
                print("\nSystem Status:")
                print(f"  Uptime: {status['system']['uptime']}")
                print(f"  Availability: {status['system']['availability']:.2f}")
                print(f"  Active VMs: {status['system']['active_vms']}/{status['system']['vm_count']}")
                print(f"  Avg Workload: {status['system']['avg_workload']:.2f}")
                print(f"  Avg Resource Util: {status['system']['avg_resource_util']:.2f}")
                print(f"  Security Incidents: {status['system']['security_incidents']}")
                
                # Display VM status
                print("\nVM Status:")
                for vm in status['vms']:
                    if vm['status'] == 'Running':
                        print(f"  VM {vm['id']}: Workload={vm['workload']:.2f}, "
                              f"Response={vm['response_time']:.2f}, "
                              f"Security={vm['security_risk']:.2f}")
            
            elif command == 'vis':
                status = manager.get_system_status()
                visualizer.update_history(status)
                visualizer.plot_system_overview(status, show=True, save=False)
            
            elif command == 'threat':
                status = manager.get_system_status()
                visualizer.update_history(status)
                visualizer.plot_threat_analysis(status, show=True, save=False)
            
            elif command == 'inject':
                try:
                    vm_idx = int(input("VM index: "))
                    workload = float(input("Workload increase (0-1): "))
                    manager.inject_workload(vm_idx, workload)
                    print(f"Injected workload {workload} to VM {vm_idx}")
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
            
            elif command == 'attack':
                attack_types = ['ddos', 'cryptojacking', 'malware', 'insider']
                
                print("\nAttack Types:")
                for i, attack_type in enumerate(attack_types):
                    print(f"  {i+1}. {attack_type}")
                
                try:
                    attack_idx = int(input("Select attack type (1-4): ")) - 1
                    if 0 <= attack_idx < len(attack_types):
                        attack_type = attack_types[attack_idx]
                        vm_input = input("VM index (leave empty for random): ")
                        vm_idx = int(vm_input) if vm_input.strip() else None
                        severity = float(input("Severity (0.1-1.0): "))
                        
                        manager.simulate_attack(attack_type, vm_idx, severity)
                        print(f"Simulated {attack_type} attack on " + 
                              f"{'random VM' if vm_idx is None else f'VM {vm_idx}'} " +
                              f"with severity {severity}")
                    else:
                        print("Invalid attack type.")
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
            
            elif command == 'export':
                filename = visualizer.export_data()
                print(f"Data exported to {filename}")
            
            else:
                print("Unknown command. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directories
    create_output_directories(args)
    
    # Run in specified mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'evaluate':
        evaluate_mode(args)
    elif args.mode == 'run':
        run_mode(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == '__main__':
    main()