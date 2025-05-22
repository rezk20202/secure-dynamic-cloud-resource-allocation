import numpy as np
import gym
from gym import spaces

class CloudEnvironment(gym.Env):
    """
    Custom environment for cloud resource allocation with security integration
    
    This environment simulates a cloud infrastructure with multiple VMs, each with
    various resources (CPU, memory, bandwidth, storage) and workloads. It tracks
    system performance and security metrics.
    """
    def __init__(self, num_vms=10, num_resources=4, security_weight=0.3):
        super(CloudEnvironment, self).__init__()
        
        # Environment parameters
        self.num_vms = num_vms  # Number of virtual machines
        self.num_resources = num_resources  # CPU, memory, bandwidth, storage, etc.
        self.security_weight = security_weight  # Weight for security in reward function
        
        # Resource types (for logging and visualization)
        self.resource_types = ['CPU', 'Memory', 'Bandwidth', 'Storage'][:num_resources]
        
        # Define action space
        # Actions: [VM index, Action type, Action value]
        # Action types: 0=allocate, 1=reduce, 2=migrate, 3=new VM, 4=shutdown
        self.action_space = spaces.MultiDiscrete([
            num_vms,       # VM index
            5,             # Action type
            10             # Action value (0-9, scaled based on action type)
        ])
        
        # Define observation space
        # State includes:
        # - Resource allocation for each VM (num_vms * num_resources)
        # - Workload for each VM (num_vms)
        # - Security metrics for each VM (num_vms)
        # - System health metrics (4: uptime, error rate, availability, cost)
        total_state_size = (num_vms * num_resources) + (num_vms * 3) + 4
        
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(total_state_size,), 
            dtype=np.float32
        )
        
        # Security incidents tracking
        self.security_incidents = 0
        self.security_threats = {
            'ddos': 0,
            'cryptojacking': 0,
            'malware': 0,
            'insider': 0,
            'other': 0
        }
       
        # Initialize state
        self.reset()
        
        
        # System health tracking
        self.uptime = 0
        self.error_rate = 0
        self.availability = 1.0
        self.operational_cost = 0.5  # Normalized between 0-1
        
    def reset(self):
        """Reset the environment state"""
        # Initialize resource utilization (normalized between 0-1)
        self.resources = np.random.uniform(0.2, 0.6, (self.num_vms, self.num_resources))
        
        # Initialize workload for each VM (normalized between 0-1)
        self.workload = np.random.uniform(0.3, 0.7, self.num_vms)
        
        # Initialize security metrics (0 = secure, higher values indicate potential threats)
        self.security_metrics = np.zeros(self.num_vms)
        
        # Initialize VM status (1 = running, 0 = off)
        self.vm_status = np.ones(self.num_vms)
        
        # Initialize VM response time (normalized, lower is better)
        self.response_time = np.random.uniform(0.2, 0.4, self.num_vms)
        
        # Reset security incident counter
        self.security_incidents = 0
        for key in self.security_threats:
            self.security_threats[key] = 0
        
        # Reset system health metrics
        self.uptime = 0
        self.error_rate = 0
        self.availability = 1.0
        self.operational_cost = 0.5
        
        return self._get_state()
    
    def _get_state(self):
        """Convert the environment state to the observation vector"""
        # Flatten resources
        resources_flat = self.resources.flatten()
        
        # Create workload state
        workload_state = self.workload * self.vm_status  # Only active VMs have workload
        
        # Create security state
        security_state = self.security_metrics
        
        # Create response time state
        response_time_state = self.response_time
        
        # Create system health state
        system_health = np.array([
            self.uptime/100.0,  # Normalized uptime
            self.error_rate,    # Error rate
            self.availability,  # Availability
            self.operational_cost  # Cost
        ])
        
        # Combine all state components
        state = np.concatenate([
            resources_flat,
            workload_state,
            security_state,
            response_time_state,
            system_health
        ])
        
        return state
    
    def step(self, action):
        """
        Execute one step in the environment
        action: [VM index, Action type, Action value]
        """
        vm_idx, action_type, action_value = action
        
        # Handle case where VM is shut down but not for action_type=3 (new VM)
        if self.vm_status[vm_idx] == 0 and action_type != 3:
            # Can't perform actions on shut down VMs except starting new ones
            reward = -5
            done = False
            info = self._get_info()
            return self._get_state(), reward, done, info
        
        # Execute action based on action type
        if action_type == 0:  # Allocate more resources
            self._allocate_resources(vm_idx, action_value)
        elif action_type == 1:  # Reduce resources
            self._reduce_resources(vm_idx, action_value)
        elif action_type == 2:  # Migrate workload
            target_vm = action_value % self.num_vms
            self._migrate_workload(vm_idx, target_vm)
        elif action_type == 3:  # Spin up new VM
            self._spin_up_vm(vm_idx)
        elif action_type == 4:  # Shutdown VM
            self._shutdown_vm(vm_idx)
        
        # Update workload
        self._update_workload()
        
        # Update system health
        self._update_system_health()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = False
        if self.uptime >= 100 or self.availability < 0.5 or self.security_incidents > 10:
            done = True
        
        # Get updated state
        state = self._get_state()
        
        # Additional info
        info = self._get_info()
        
        return state, reward, done, info
    
    def _get_info(self):
        """Return current environment information"""
        return {
            'resource_utilization': np.mean(self.resources),
            'vm_utilization': np.sum(self.vm_status) / self.num_vms,
            'workload': np.mean(self.workload),
            'security_incidents': self.security_incidents,
            'security_threats': self.security_threats,
            'availability': self.availability,
            'uptime': self.uptime,
            'operational_cost': self.operational_cost,
            'error_rate': self.error_rate,
            'response_time': np.mean(self.response_time)
        }
    
    def _allocate_resources(self, vm_idx, action_value):
        """Allocate more resources to the selected VM"""
        # Scale action value from 0-9 to 0.1-0.5 (10% to 50% increase)
        increase_amt = (action_value + 1) / 20.0
        
        # Increase resources for selected VM (with upper limit)
        self.resources[vm_idx] = np.minimum(self.resources[vm_idx] * (1 + increase_amt), 1.0)
        
        # Increase operational cost
        self.operational_cost = min(1.0, self.operational_cost + 0.02)
    
    def _reduce_resources(self, vm_idx, action_value):
        """Reduce resources for the selected VM"""
        # Scale action value from 0-9 to 0.1-0.5 (10% to 50% decrease)
        decrease_amt = (action_value + 1) / 20.0
        
        # Decrease resources for selected VM (with lower limit)
        self.resources[vm_idx] = np.maximum(self.resources[vm_idx] * (1 - decrease_amt), 0.1)
        
        # Decrease operational cost
        self.operational_cost = max(0.1, self.operational_cost - 0.01)
    
    def _migrate_workload(self, source_vm, target_vm):
        """Migrate workload from source VM to target VM"""
        # Ensure target VM is active
        if self.vm_status[target_vm] == 0:
            return  # Can't migrate to inactive VM
        
        # Calculate amount to migrate (50% of source workload)
        migrate_amount = self.workload[source_vm] * 0.5
        
        # Transfer workload
        self.workload[source_vm] -= migrate_amount
        self.workload[target_vm] = min(1.0, self.workload[target_vm] + migrate_amount)
        
        # Migration temporarily increases response time
        self.response_time[source_vm] = min(1.0, self.response_time[source_vm] + 0.1)
        self.response_time[target_vm] = min(1.0, self.response_time[target_vm] + 0.1)
    
    def _spin_up_vm(self, vm_idx):
        """Spin up a new VM or restart an existing one"""
        if self.vm_status[vm_idx] == 0:
            # Restart existing VM
            self.vm_status[vm_idx] = 1
            self.resources[vm_idx] = np.random.uniform(0.2, 0.4, self.num_resources)
            self.workload[vm_idx] = 0.1
            self.response_time[vm_idx] = 0.3
            self.security_metrics[vm_idx] = 0.0
            
            # Increase operational cost
            self.operational_cost = min(1.0, self.operational_cost + 0.05)
    
    def _shutdown_vm(self, vm_idx):
        """Shutdown a VM"""
        if self.vm_status[vm_idx] == 1:
            # Only shutdown if there are at least 2 active VMs
            if np.sum(self.vm_status) > 1:
                self.vm_status[vm_idx] = 0
                self.workload[vm_idx] = 0
                # Distribute workload to other active VMs
                active_vms = np.where(self.vm_status == 1)[0]
                if len(active_vms) > 0:
                    workload_to_distribute = self.workload[vm_idx] / len(active_vms)
                    for vm in active_vms:
                        self.workload[vm] = min(1.0, self.workload[vm] + workload_to_distribute)
                
                # Decrease operational cost
                self.operational_cost = max(0.1, self.operational_cost - 0.03)
    
    def _update_workload(self):
        """Update workload with some random fluctuation"""
        for i in range(self.num_vms):
            if self.vm_status[i] == 1:  # Only update active VMs
                # Random workload change
                delta = np.random.uniform(-0.05, 0.08)
                self.workload[i] = np.clip(self.workload[i] + delta, 0.1, 1.0)
                
                # Update response time based on workload and resources
                resource_avg = np.mean(self.resources[i])
                if resource_avg > 0:
                    # Response time improves with more resources and degrades with higher workload
                    target_response = self.workload[i] / resource_avg
                    # Smooth change in response time
                    self.response_time[i] = 0.8 * self.response_time[i] + 0.2 * target_response
                    self.response_time[i] = np.clip(self.response_time[i], 0.1, 1.0)
    
    def _update_system_health(self):
        """Update system health metrics"""
        # Increment uptime
        self.uptime += 1
        
        # Calculate resource balance
        active_vms = np.where(self.vm_status == 1)[0]
        if len(active_vms) > 0:
            resource_balance = np.std([np.mean(self.resources[i]) for i in active_vms])
        else:
            resource_balance = 1.0  # Maximum imbalance
        
        # Update error rate based on resource balance and workload
        workload_avg = np.mean(self.workload)
        self.error_rate = 0.8 * self.error_rate + 0.2 * (resource_balance * workload_avg)
        self.error_rate = np.clip(self.error_rate, 0, 1)
        
        # Update availability based on error rate
        self.availability = 1.0 - (self.error_rate * 0.5)
    
    def update_security_metrics(self, vm_indices, threat_types, severity_scores):
        """
        Update security metrics based on IDS findings
        
        Args:
            vm_indices: List of VM indices with detected threats
            threat_types: List of threat types corresponding to vm_indices
            severity_scores: List of severity scores for each threat
        """
        for vm_idx, threat_type, severity in zip(vm_indices, threat_types, severity_scores):
            if 0 <= vm_idx < self.num_vms:
                # Update security metric for the VM
                self.security_metrics[vm_idx] = min(1.0, self.security_metrics[vm_idx] + severity)
                
                # Track security incidents
                self.security_incidents += 1
                
                # Track specific threat types
                if threat_type in self.security_threats:
                    self.security_threats[threat_type] += 1
                else:
                    self.security_threats['other'] += 1
    
    def _calculate_reward(self, action):
        """Calculate reward based on performance and security"""
        # Get active VMs
        active_vms = np.where(self.vm_status == 1)[0]
        num_active_vms = len(active_vms)
        
        if num_active_vms == 0:
            return -100  # Major penalty if all VMs are down
        
        # 1. Performance reward components
        # Resource utilization efficiency
        if num_active_vms > 0:
            # Average resource utilization across active VMs
            resource_util = np.mean([np.mean(self.resources[i]) for i in active_vms])
            # Average workload across active VMs
            workload_avg = np.mean([self.workload[i] for i in active_vms])
            # Resource-workload balance reward
            resource_workload_match = 1.0 - abs(resource_util - workload_avg)
        else:
            resource_workload_match = 0
            
        # Response time reward (lower is better)
        response_time_reward = 1.0 - np.mean([self.response_time[i] for i in active_vms])
        
        # SLA compliance reward
        sla_compliance = 1.0 - self.error_rate
        
        # Combine performance metrics
        performance_reward = (
            0.4 * resource_workload_match + 
            0.3 * response_time_reward + 
            0.3 * sla_compliance
        ) * 10
        
        # 2. Security reward components
        # Security incident penalty
        security_penalty = self.security_incidents * 2
        
        # Average security risk across VMs
        security_risk = np.mean(self.security_metrics)
        
        # Combined security metrics
        security_reward = -5 * security_risk - security_penalty
        
        # 3. Operational cost penalty
        cost_penalty = self.operational_cost * 5
        
        # Calculate total reward with weights
        total_reward = (
            (1 - self.security_weight) * performance_reward + 
            self.security_weight * security_reward - 
            0.2 * cost_penalty
        )
        
        return total_reward

    def get_vm_details(self):
        """Return detailed information about each VM for visualization"""
        vm_details = []
        for i in range(self.num_vms):
            vm_details.append({
                'id': i,
                'status': 'Running' if self.vm_status[i] == 1 else 'Offline',
                'workload': self.workload[i],
                'security_risk': self.security_metrics[i],
                'response_time': self.response_time[i],
                'resources': {
                    rtype: self.resources[i][j] 
                    for j, rtype in enumerate(self.resource_types)
                }
            })
        return vm_details