import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from datetime import datetime

class Visualizer:
    """
    Visualizer for the secure cloud management system
    
    This class provides visualization capabilities for system metrics,
    security incidents, and performance trends.
    """
    def __init__(self, output_dir='visualizations'):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Directory to save visualization images
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.output_dir = output_dir
        self.history = {
            'timestamps': [],
            'resource_util': [],
            'workload': [],
            'security_incidents': [],
            'availability': [],
            'active_vms': [],
            'response_time': [],
            'operational_cost': [],
            'threat_counts': {
                'ddos': [],
                'cryptojacking': [],
                'malware': [],
                'insider': [],
                'other': []
            }
        }
    
    def update_history(self, status):
        """
        Update historical data with current status
        
        Args:
            status: System status dictionary from SecureCloudManager.get_system_status()
        """
        # Extract metrics
        system = status['system']
        
        # Calculate system-wide metrics
        active_vms = [vm for vm in status['vms'] if vm['status'] == 'Running']
        avg_response_time = np.mean([vm['response_time'] for vm in active_vms]) if active_vms else 0
        
        # Add timestamp
        self.history['timestamps'].append(datetime.now())
        
        # Add system metrics
        self.history['resource_util'].append(system['avg_resource_util'])
        self.history['workload'].append(system['avg_workload'])
        self.history['security_incidents'].append(system['security_incidents'])
        self.history['availability'].append(system['availability'])
        self.history['active_vms'].append(system['active_vms'])
        self.history['response_time'].append(avg_response_time)
        self.history['operational_cost'].append(system['operational_cost'])
        
        # Add threat counts - Fix this section to handle missing data
        for threat_type in self.history['threat_counts']:
        # Get threat count from system or default to 0 if not present
            count = system.get('security_threats', {}).get(threat_type, 0)
            self.history['threat_counts'][threat_type].append(count)
        
        # Add this debug print to check if security data is being recorded
        print(f"DEBUG: Updated history with security incidents: {system['security_incidents']}")
        if system['security_incidents'] > 0:
            print(f"DEBUG: Threat types: {system.get('security_threats', {})}")
    
    def plot_system_overview(self, status, show=True, save=False):
        """
        Plot system overview visualization
        
        Args:
            status: System status dictionary from SecureCloudManager.get_system_status()
            show: Whether to display the plot
            save: Whether to save the plot to file
        """
        plt.figure(figsize=(16, 10))
        
        # Extract data
        system = status['system']
        vms = status['vms']
        
        # 1. Resource utilization subplot
        plt.subplot(2, 3, 1)
        active_vms = [vm for vm in vms if vm['status'] == 'Running']
        vm_ids = [vm['id'] for vm in active_vms]
        
        # Extract resource types
        if active_vms:
            resource_types = list(active_vms[0]['resources'].keys())
            
            # Plot resource utilization
            x = np.arange(len(active_vms))
            width = 0.8 / len(resource_types)
            
            for i, rtype in enumerate(resource_types):
                values = [vm['resources'][rtype] for vm in active_vms]
                plt.bar(x + (i - len(resource_types)/2 + 0.5) * width, values, 
                        width=width, label=rtype)
            
            plt.xlabel('VM ID')
            plt.ylabel('Resource Utilization')
            plt.title('VM Resource Utilization')
            plt.xticks(x, vm_ids)
            plt.ylim(0, 1.1)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No active VMs', horizontalalignment='center',
                    verticalalignment='center', transform=plt.gca().transAxes)
        
        # 2. Workload subplot
        plt.subplot(2, 3, 2)
        if active_vms:
            workloads = [vm['workload'] for vm in active_vms]
            plt.bar(vm_ids, workloads, color='green')
            plt.xlabel('VM ID')
            plt.ylabel('Workload')
            plt.title('VM Workload')
            plt.ylim(0, 1.1)
        else:
            plt.text(0.5, 0.5, 'No active VMs', horizontalalignment='center',
                    verticalalignment='center', transform=plt.gca().transAxes)
        
        # 3. Security risk subplot
        plt.subplot(2, 3, 3)
        if active_vms:
            security_risks = [vm['security_risk'] for vm in active_vms]
            bars = plt.bar(vm_ids, security_risks)
            
            # Color bars based on risk level
            for i, bar in enumerate(bars):
                if security_risks[i] < 0.2:
                    bar.set_color('green')
                elif security_risks[i] < 0.5:
                    bar.set_color('yellow')
                elif security_risks[i] < 0.8:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            plt.xlabel('VM ID')
            plt.ylabel('Security Risk')
            plt.title('VM Security Risk')
            plt.ylim(0, 1.1)
        else:
            plt.text(0.5, 0.5, 'No active VMs', horizontalalignment='center',
                    verticalalignment='center', transform=plt.gca().transAxes)
        
        # 4. System metrics over time
        plt.subplot(2, 3, 4)
        if len(self.history['timestamps']) > 1:
            plt.plot(self.history['timestamps'], self.history['resource_util'], 
                    'b-', label='Resource Util')
            plt.plot(self.history['timestamps'], self.history['workload'], 
                    'g-', label='Workload')
            plt.plot(self.history['timestamps'], self.history['availability'], 
                    'r-', label='Availability')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('System Metrics Over Time')
            plt.ylim(0, 1.1)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Insufficient history data', horizontalalignment='center',
                    verticalalignment='center', transform=plt.gca().transAxes)
        
        # 5. Security incidents over time
        plt.subplot(2, 3, 5)
        if len(self.history['timestamps']) > 1:
            # Stack plot of different threat types
            threat_types = list(self.history['threat_counts'].keys())
            threat_data = [self.history['threat_counts'][t] for t in threat_types]
            
            plt.stackplot(self.history['timestamps'], threat_data, 
                        labels=threat_types, alpha=0.7)
            plt.xlabel('Time')
            plt.ylabel('Incidents')
            plt.title('Security Incidents by Type')
            plt.legend(loc='upper left')
        else:
            plt.text(0.5, 0.5, 'Insufficient history data', horizontalalignment='center',
                    verticalalignment='center', transform=plt.gca().transAxes)
        
        # 6. System health metrics
        plt.subplot(2, 3, 6)
        health_metrics = ['Availability', 'Error Rate', 'Cost']
        health_values = [system['availability'], system['error_rate'], system['operational_cost']]
        
        colors = ['green', 'red', 'blue']
        bars = plt.bar(health_metrics, health_values, color=colors)
        
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('System Health')
        plt.ylim(0, 1.1)
        
        # Add title and adjust layout
        plt.suptitle(f'Secure Cloud System Overview - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                    fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.output_dir, f'system_overview_{timestamp}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        # Show figure
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_training_metrics(self, manager, show=True, save=False):
        """
        Plot training metrics visualization
        
        Args:
            manager: SecureCloudManager instance
            show: Whether to display the plot
            save: Whether to save the plot to file
        """
        plt.figure(figsize=(16, 10))
        
        # Extract data
        episodes = range(1, len(manager.episode_rewards) + 1)
        
        # 1. Episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(episodes, manager.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards')
        
        # 2. Security incidents
        plt.subplot(2, 2, 2)
        plt.plot(episodes, manager.episode_security_incidents)
        plt.xlabel('Episode')
        plt.ylabel('Security Incidents')
        plt.title('Security Incidents per Episode')
        
        # 3. Resource utilization
        plt.subplot(2, 2, 3)
        plt.plot(episodes, manager.episode_resource_util)
        plt.xlabel('Episode')
        plt.ylabel('Resource Utilization')
        plt.title('Average Resource Utilization')
        
        # 4. System availability
        plt.subplot(2, 2, 4)
        plt.plot(episodes, manager.episode_availability)
        plt.xlabel('Episode')
        plt.ylabel('Availability')
        plt.title('System Availability')
        
        # Add title and adjust layout
        plt.suptitle('DRL Agent Training Metrics', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.output_dir, f'training_metrics_{timestamp}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        # Show figure
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_threat_analysis(self, status, history_length=50, show=True, save=False):
        """
        Plot threat analysis visualization
        
        Args:
            status: System status dictionary from SecureCloudManager.get_system_status()
            history_length: Number of historical points to include
            show: Whether to display the plot
            save: Whether to save the plot to file
        """
        plt.figure(figsize=(16, 10))
        
        # Extract data
        system = status['system']
        threat_summary = status['ids'] if 'ids' in status else None
        
        # 1. Threat type distribution
        plt.subplot(2, 2, 1)
        # Use the security_threats from system status directly
        if 'security_threats' in system and any(system['security_threats'].values()):
            # Get threat counts directly from system
            threat_types = list(system['security_threats'].keys())
            threat_counts = list(system['security_threats'].values())
        
            # Only show non-zero counts
            non_zero = [(t, c) for t, c in zip(threat_types, threat_counts) if c > 0]
            if non_zero:
                types, counts = zip(*non_zero)
                plt.pie(counts, labels=types, autopct='%1.1f%%')
                plt.title('Threat Distribution by Type')
            else:
                plt.text(0.5, 0.5, 'No threats detected', horizontalalignment='center',
                        verticalalignment='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, 'No threat data available', horizontalalignment='center',
                    verticalalignment='center', transform=plt.gca().transAxes)
            
        # 2. Threat severity
        plt.subplot(2, 2, 2)
        # Calculate highest severity by type from VM security metrics
        if 'vms' in status:
            # Create a severity dictionary from VM data
            threat_severity = {}
            for vm in status['vms']:
                if vm['security_risk'] > 0:
                    # Try to match VM to threat type (simplified approach)
                    for threat_type, count in system.get('security_threats', {}).items():
                        if count > 0 and threat_type not in threat_severity:
                            threat_severity[threat_type] = vm['security_risk']
                            break
            
            # Only show non-zero severities
            if threat_severity:
                threat_types = list(threat_severity.keys())
                severities = list(threat_severity.values())
                
                bars = plt.bar(threat_types, severities)
                
                # Color bars based on severity
                for i, bar in enumerate(bars):
                    if severities[i] < 0.3:
                        bar.set_color('green')
                    elif severities[i] < 0.6:
                        bar.set_color('yellow')
                    else:
                        bar.set_color('red')
                
                plt.xlabel('Threat Type')
                plt.ylabel('Highest Severity')
                plt.title('Threat Severity by Type')
                plt.ylim(0, 1.1)
            else:
                plt.text(0.5, 0.5, 'No threats detected', horizontalalignment='center',
                        verticalalignment='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, 'No threat data available', horizontalalignment='center',
                    verticalalignment='center', transform=plt.gca().transAxes)
        
        # 3. Threat history
        plt.subplot(2, 2, 3)
        history_len = min(history_length, len(self.history['timestamps']))
        if history_len > 1:
            # Get last N points
            timestamps = self.history['timestamps'][-history_len:]
            threat_data = {}
            
            for t_type in self.history['threat_counts']:
                threat_data[t_type] = self.history['threat_counts'][t_type][-history_len:]
            
            # Stack plot of different threat types
            threat_types = list(threat_data.keys())
            data_to_plot = [threat_data[t] for t in threat_types]
            
            plt.stackplot(timestamps, data_to_plot, labels=threat_types, alpha=0.7)
            plt.xlabel('Time')
            plt.ylabel('Cumulative Incidents')
            plt.title('Threat History')
            plt.legend(loc='upper left')
        else:
            plt.text(0.5, 0.5, 'Insufficient history data', horizontalalignment='center',
                    verticalalignment='center', transform=plt.gca().transAxes)
        
        # 4. Affected VMs
        plt.subplot(2, 2, 4)
        vm_risks = {}
        for vm in status['vms']:
            vm_risks[f"VM {vm['id']}"] = vm['security_risk']
        
        # Sort by risk level
        vm_risks = {k: v for k, v in sorted(vm_risks.items(), key=lambda item: item[1], reverse=True)}
        
        # Plot VMs with non-zero risk
        non_zero = {k: v for k, v in vm_risks.items() if v > 0}
        if non_zero:
            vms = list(non_zero.keys())
            risks = list(non_zero.values())
            
            bars = plt.bar(vms, risks)
            
            # Color bars based on risk level
            for i, bar in enumerate(bars):
                if risks[i] < 0.2:
                    bar.set_color('green')
                elif risks[i] < 0.5:
                    bar.set_color('yellow')
                elif risks[i] < 0.8:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            plt.xlabel('VM')
            plt.ylabel('Security Risk')
            plt.title('VM Security Risk Levels')
            plt.ylim(0, 1.1)
        else:
            plt.text(0.5, 0.5, 'No VMs with security risk', horizontalalignment='center',
                    verticalalignment='center', transform=plt.gca().transAxes)
        
        # Add title and adjust layout
        plt.suptitle('Security Threat Analysis', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.output_dir, f'threat_analysis_{timestamp}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        # Show figure
        if show:
            plt.show()
        else:
            plt.close()
            
    def export_data(self, filename=None):
        """
        Export historical data to CSV
        
        Args:
            filename: Output filename (default: auto-generated)
        
        Returns:
            str: Path to the exported CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.output_dir, f'security_data_{timestamp}.csv')
        
        # Prepare dataframe
        data = {
            'timestamp': self.history['timestamps'],
            'resource_utilization': self.history['resource_util'],
            'workload': self.history['workload'],
            'security_incidents': self.history['security_incidents'],
            'availability': self.history['availability'],
            'active_vms': self.history['active_vms'],
            'response_time': self.history['response_time'],
            'operational_cost': self.history['operational_cost']
        }
        
        # Add threat counts
        for threat_type in self.history['threat_counts']:
            data[f'threat_{threat_type}'] = self.history['threat_counts'][threat_type]
        
        # Create dataframe and export
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
        return filename