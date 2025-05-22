import numpy as np
import threading
import time
from datetime import datetime
from src.environment import CloudEnvironment
from src.drl_agent import DRLAgent
from src.ids import IntrusionDetectionSystem
from src.logger import Logger
from src.visualizer import Visualizer

class SecureCloudManager:
    """
    Integrated system that combines DRL-based resource allocation with IDS
    
    This class manages the cooperation between the resource allocation system
    and the intrusion detection system, providing a unified interface.
    """
    def __init__(self, num_vms=10, num_resources=4, security_weight=0.3, 
                 monitoring_interval=5, log_level='INFO'):
        """
        Initialize the secure cloud manager
        
        Args:
            num_vms: Number of VMs to manage
            num_resources: Number of resource types (CPU, memory, etc.)
            security_weight: Weight of security in reward calculation
            monitoring_interval: IDS monitoring interval in seconds
            log_level: Logging level
        """
        # Create logger
        self.logger = Logger(log_level=log_level)
        self.logger.info("Initializing Secure Cloud Resource Manager")
        
        # Create environment
        self.env = CloudEnvironment(num_vms, num_resources, security_weight)
        
        # Create DRL agent
        state_size = self.env.observation_space.shape[0]
        self.agent = DRLAgent(state_size, self.env.action_space)
        
        # Create and train IDS
        self.ids = IntrusionDetectionSystem(self.env, monitoring_interval)
        
        # Performance metrics
        self.episode_rewards = []
        self.episode_security_incidents = []
        self.episode_resource_util = []
        self.episode_availability = []
        self.steps_taken = 0
        
        # Resource allocation thread
        self.allocation_thread = None
        self.allocation_running = False
        self.allocation_interval = 1.0  # seconds
        
        # Current system state
        self.current_state = None
        self.last_action = None
        self.training_mode = True
        
        # Lock for synchronizing access to environment
        self.env_lock = threading.Lock()
    
    def _train_ids(self):
        """Generate data and train the IDS"""
        self.logger.info("Training Intrusion Detection System")
    
        # Stop any existing monitoring threads
        if hasattr(self.ids, 'running') and self.ids.running:
            self.ids.stop_monitoring()
            # Give time for threads to close
            time.sleep(1)
        
        self.ids.generate_training_data()
        self.logger.info("IDS training completed")
    
    def train(self, episodes=200, max_steps=100):
        """
        Train the DRL agent
        
        Args:
            episodes: Number of episodes to train
            max_steps: Maximum steps per episode
        """
        self.logger.info(f"Starting DRL agent training for {episodes} episodes")
        
        self.training_mode = True
        for episode in range(episodes):
            # Reset environment
            with self.env_lock:
                state = self.env.reset()
            
            self.current_state = state
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Run episode
            while not done and episode_steps < max_steps:
                # Choose action
                action = self.agent.act(state)
                self.last_action = action
                
                # Take action
                with self.env_lock:
                    next_state, reward, done, info = self.env.step(action)
                
                # Remember experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                self.current_state = state
                episode_reward += reward
                episode_steps += 1
                self.steps_taken += 1
                
                # Train agent with experiences
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.replay()
                    if episode_steps % 10 == 0:
                        self.logger.debug(f"Training loss: {loss:.4f}")
            
            # Log metrics
            self.episode_rewards.append(episode_reward)
            self.episode_security_incidents.append(info['security_incidents'])
            self.episode_resource_util.append(info['resource_utilization'])
            self.episode_availability.append(info['availability'])
            
            # Print progress
            if (episode + 1) % 10 == 0:
                self.logger.info(
                    f"Episode: {episode+1}/{episodes}, "
                    f"Reward: {episode_reward:.2f}, "
                    f"Security Incidents: {info['security_incidents']}, "
                    f"Resource Utilization: {info['resource_utilization']:.2f}, "
                    f"Availability: {info['availability']:.2f}"
                )
        
        self.logger.info("DRL agent training completed")
    
    def start(self, train_first=True):
        """
        Start the secure cloud manager
        
        Args:
            train_first: If True, train the IDS and DRL agent first
        """
        self.logger.info("Starting Secure Cloud Manager")
        
        # Reset environment
        with self.env_lock:
            self.current_state = self.env.reset()
        
        # Train IDS if needed
        if train_first:
            self._train_ids()
        
        # Start IDS monitoring
        self.ids.start_monitoring()
        
        # Start resource allocation thread
        self.allocation_running = True
        self.allocation_thread = threading.Thread(target=self._allocation_loop)
        self.allocation_thread.daemon = True
        self.allocation_thread.start()
        
        self.logger.info("Secure Cloud Manager started successfully")
    
    def stop(self):
        """Stop the secure cloud manager"""
        self.logger.info("Stopping Secure Cloud Manager")
        
        # Stop resource allocation
        self.allocation_running = False
        if self.allocation_thread:
            self.allocation_thread.join(timeout=2)
        
        # Stop IDS monitoring
        self.ids.stop_monitoring()
        
        self.logger.info("Secure Cloud Manager stopped")
    
    def _allocation_loop(self):
        """Main resource allocation loop"""
        self.logger.info("Resource allocation loop started")
        
        self.training_mode = False
        step_count = 0

        # Create visualizer instance if not already present
        if not hasattr(self, 'visualizer'):
            self.visualizer = Visualizer(output_dir='output/visualizations')
        
        while self.allocation_running:
            # Get current state
            with self.env_lock:
                state = self.current_state
            
            # Choose action (without exploration in deployment)
            action = self.agent.act(state, evaluate=True)
            self.last_action = action
            
            # Take action
            with self.env_lock:
                next_state, reward, done, info = self.env.step(action)
                self.current_state = next_state
            
            # Update visualizer history AFTER each action (add this line)
            status = self.get_system_status()
            self.visualizer.update_history(status)
            
            # Log action and result
            step_count += 1
            if step_count % 10 == 0:
                self.logger.info(
                    f"Step {step_count}: Action={action}, "
                    f"Reward={reward:.2f}, "
                    f"Resource Util={info['resource_utilization']:.2f}, "
                    f"VM Util={info['vm_utilization']:.2f}, "
                    f"Workload={info['workload']:.2f}"
                )
                
                # Log security status
                self.logger.info(
                    f"Security Status: Incidents={info['security_incidents']}, "
                    f"Availability={info['availability']:.2f}, "
                    f"Error Rate={info['error_rate']:.2f}"
                )
                
                # Log detailed threat information
                if info['security_incidents'] > 0:
                    threat_types = ', '.join([f"{k}:{v}" for k, v in info['security_threats'].items() if v > 0])
                    self.logger.warning(f"Detected Threats: {threat_types}")
            
            # Reset if environment is done
            if done:
                self.logger.info("Environment reset due to terminal state")
                with self.env_lock:
                    self.current_state = self.env.reset()
            
            # Sleep for allocation interval
            time.sleep(self.allocation_interval)
    
    def inject_workload(self, vm_idx, workload_increase):
        """
        Inject additional workload to a VM (for testing)
        
        Args:
            vm_idx: VM index
            workload_increase: Amount to increase workload (0-1)
        """
        with self.env_lock:
            if 0 <= vm_idx < self.env.num_vms and self.env.vm_status[vm_idx] == 1:
                self.env.workload[vm_idx] = min(1.0, self.env.workload[vm_idx] + workload_increase)
                self.logger.info(f"Injected workload to VM {vm_idx}, new workload: {self.env.workload[vm_idx]:.2f}")
    
    def simulate_attack(self, attack_type, vm_idx=None, severity=0.7):
        """
        Simulate an attack for testing response
        
        Args:
            attack_type: Type of attack ('ddos', 'cryptojacking', 'malware', 'insider')
            vm_idx: VM to attack (random if None)
            severity: Attack severity (0-1)
        """
        if vm_idx is None:
            # Choose a random active VM
            active_vms = np.where(self.env.vm_status == 1)[0]
            if len(active_vms) == 0:
                self.logger.warning("No active VMs to attack")
                return
            vm_idx = np.random.choice(active_vms)
        
        # Validate VM index
        if vm_idx < 0 or vm_idx >= self.env.num_vms or self.env.vm_status[vm_idx] == 0:
            self.logger.warning(f"Invalid VM index {vm_idx} for attack simulation")
            return
        
        # Validate attack type
        valid_types = ['ddos', 'cryptojacking', 'malware', 'insider']
        if attack_type not in valid_types:
            self.logger.warning(f"Invalid attack type {attack_type}")
            return
        
        # Simulate attack
        severity = max(0.1, min(1.0, severity))  # Clamp to 0.1-1.0
        
        with self.env_lock:
            # Update environment to simulate attack effects
            if attack_type == 'ddos':
                # DDoS: High bandwidth usage, high response time
                self.env.resources[vm_idx][2] = min(1.0, self.env.resources[vm_idx][2] + severity * 0.5)  # Bandwidth
                self.env.response_time[vm_idx] = min(1.0, self.env.response_time[vm_idx] + severity * 0.4)
            
            elif attack_type == 'cryptojacking':
                # Cryptojacking: High CPU usage with normal workload
                self.env.resources[vm_idx][0] = min(1.0, self.env.resources[vm_idx][0] + severity * 0.7)  # CPU
            
            elif attack_type == 'malware':
                # Malware: Variable resource usage
                self.env.resources[vm_idx] = np.minimum(
                    self.env.resources[vm_idx] + np.random.uniform(0.1, 0.3, self.env.num_resources),
                    1.0
                )
            
            elif attack_type == 'insider':
                # Insider: High storage and bandwidth
                self.env.resources[vm_idx][2] = min(1.0, self.env.resources[vm_idx][2] + severity * 0.4)  # Bandwidth
                self.env.resources[vm_idx][3] = min(1.0, self.env.resources[vm_idx][3] + severity * 0.6)  # Storage
            
            # Update security metrics directly
            self.env.update_security_metrics([vm_idx], [attack_type], [severity])
        
        self.logger.warning(
            f"Simulated {attack_type} attack on VM {vm_idx} with severity {severity:.2f}"
        )
    
    def get_system_status(self):
        """Get the current system status for monitoring/visualization"""
        with self.env_lock:
            vm_details = self.env.get_vm_details()
            system_info = {
                'uptime': self.env.uptime,
                'availability': self.env.availability,
                'error_rate': self.env.error_rate,
                'operational_cost': self.env.operational_cost,
                'security_incidents': self.env.security_incidents,
                'security_threats': self.env.security_threats,
                'vm_count': self.env.num_vms,
                'active_vms': sum(self.env.vm_status),
                'avg_workload': np.mean(self.env.workload),
                'avg_resource_util': np.mean(self.env.resources),
                'timestamp': datetime.now().isoformat()
            }
        
        # Get IDS status
        ids_status = self.ids.get_threat_summary()
        
        # Get agent metrics
        agent_metrics = self.agent.get_metrics()
        
        return {
            'system': system_info,
            'vms': vm_details,
            'ids': ids_status,
            'agent': agent_metrics,
            'last_action': self.last_action
        }
    
    def save_models(self, drl_path="models/drl_model.h5"):
        """Save trained models"""
        self.logger.info("Saving models")
        self.agent.save_model(drl_path)
        self.logger.info(f"DRL model saved to {drl_path}")
    
    def load_models(self, drl_path="models/drl_model.h5"):
        """Load trained models"""
        self.logger.info("Loading models")
        self.agent.load_model(drl_path)
        self.logger.info(f"DRL model loaded from {drl_path}")