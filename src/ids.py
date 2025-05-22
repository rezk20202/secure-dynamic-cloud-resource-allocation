import numpy as np
import pandas as pd
import threading
import time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

class IntrusionDetectionSystem:
    """
    Ensemble-based intrusion detection system for cloud environments
    
    This system runs periodic security checks to detect various types of threats
    including DDoS, cryptojacking, malware, and insider threats.
    """
    def __init__(self, environment, monitoring_interval=5):
        """
        Initialize the IDS
        
        Args:
            environment: CloudEnvironment instance
            monitoring_interval: Interval in seconds between security checks
        """
        self.env = environment
        self.monitoring_interval = monitoring_interval
        self.running = False
        self.monitor_thread = None
        
        # Initialize the classifiers for different threat types
        self.ddos_model = self._create_ensemble_classifier()
        self.cryptojacking_model = self._create_ensemble_classifier()
        self.malware_model = self._create_ensemble_classifier()
        self.insider_model = self._create_ensemble_classifier()
        
        # Feature scalers
        self.ddos_scaler = StandardScaler()
        self.cryptojacking_scaler = StandardScaler()
        self.malware_scaler = StandardScaler()
        self.insider_scaler = StandardScaler()
        
        # Track training status
        self.is_trained = {
            'ddos': False,
            'cryptojacking': False,
            'malware': False,
            'insider': False
        }
        
        # Logs for detected threats
        self.threat_logs = []
        
        # Historical feature data for anomaly detection
        self.history_size = 50
        self.feature_history = []
    
    def _create_ensemble_classifier(self):
        """Create an ensemble classifier using RandomForest, SVM, and XGBoost"""
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        svm_clf = SVC(probability=True, random_state=42)
        xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        
        return VotingClassifier(
            estimators=[('rf', rf_clf), ('svm', svm_clf), ('xgb', xgb_clf)],
            voting='hard'
        )
    
    def start_monitoring(self):
        """Start the periodic security monitoring thread"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("Security monitoring started.")
    
    def stop_monitoring(self):
        """Stop the periodic security monitoring"""
        if hasattr(self, 'running'):
            self.running = False
            if hasattr(self, 'monitor_thread') and self.monitor_thread:
                try:
                    self.monitor_thread.join(timeout=2)
                except:
                    pass  # Ignore any errors during thread termination
                print("Security monitoring stopped.")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs periodically"""
        while self.running:
            self._perform_security_check()
            time.sleep(self.monitoring_interval)
    
    def _perform_security_check(self):
        """Perform a comprehensive security check"""
        # Extract features for all active VMs
        vm_features = self._extract_features()
        
        if not vm_features:
            return  # No active VMs to check
        
        # Update feature history
        self.feature_history.append(vm_features)
        if len(self.feature_history) > self.history_size:
            self.feature_history.pop(0)
        
        # Detect different types of threats
        detected_threats = []
        
        # Only perform model-based detection if models are trained
        if self.is_trained['ddos']:
            ddos_threats = self._detect_ddos_threats(vm_features)
            detected_threats.extend(ddos_threats)
        
        if self.is_trained['cryptojacking']:
            cryptojacking_threats = self._detect_cryptojacking_threats(vm_features)
            detected_threats.extend(cryptojacking_threats)
        
        if self.is_trained['malware']:
            malware_threats = self._detect_malware_threats(vm_features)
            detected_threats.extend(malware_threats)
        
        if self.is_trained['insider']:
            insider_threats = self._detect_insider_threats(vm_features)
            detected_threats.extend(insider_threats)
        
        # Always perform rule-based detection as a fallback
        rule_based_threats = self._rule_based_detection(vm_features)
        detected_threats.extend(rule_based_threats)

         # Add this debug print to check if threats are detected
        if detected_threats:
            print(f"DEBUG: Detected {len(detected_threats)} threats: {[t['type'] for t in detected_threats]}")
        
        # Update environment with detected threats
        if detected_threats:
            vm_indices = [threat['vm_idx'] for threat in detected_threats]
            threat_types = [threat['type'] for threat in detected_threats]
            severity_scores = [threat['severity'] for threat in detected_threats]
            
            # Update security metrics in the environment
            self.env.update_security_metrics(vm_indices, threat_types, severity_scores)
            
            # Log threats
            for threat in detected_threats:
                self.threat_logs.append({
                    'timestamp': time.time(),
                    'vm_idx': threat['vm_idx'],
                    'threat_type': threat['type'],
                    'severity': threat['severity'],
                    'details': threat['details']
                })

            # Add this debug print to check if threat_logs is being updated
            print(f"DEBUG: Total threats in log: {len(self.threat_logs)}")
    
    def _extract_features(self):
        """Extract security-relevant features from the environment"""
        features = []
        vm_details = self.env.get_vm_details()
        
        for vm in vm_details:
            if vm['status'] == 'Running':  # Only check active VMs
                vm_idx = vm['id']
                
                # Extract basic VM metrics
                cpu_usage = vm['resources'].get('CPU', 0)
                memory_usage = vm['resources'].get('Memory', 0)
                bandwidth_usage = vm['resources'].get('Bandwidth', 0)
                storage_usage = vm['resources'].get('Storage', 0)
                workload = vm['workload']
                response_time = vm['response_time']
                
                # Calculate derived metrics
                resource_avg = np.mean([r for r in vm['resources'].values()])
                resource_std = np.std([r for r in vm['resources'].values()])
                resource_workload_ratio = resource_avg / max(workload, 0.01)
                
                # Get historical data for this VM if available
                if self.feature_history:
                    # Extract historical data for this VM
                    historical_features = [
                        hist_features[vm_idx] for hist_features in self.feature_history 
                        if vm_idx in hist_features
                    ]
                    
                    if historical_features:
                        # Calculate rate of change metrics
                        cpu_change_rate = self._calculate_change_rate([f['cpu_usage'] for f in historical_features], cpu_usage)
                        memory_change_rate = self._calculate_change_rate([f['memory_usage'] for f in historical_features], memory_usage)
                        bandwidth_change_rate = self._calculate_change_rate([f['bandwidth_usage'] for f in historical_features], bandwidth_usage)
                        workload_change_rate = self._calculate_change_rate([f['workload'] for f in historical_features], workload)
                    else:
                        cpu_change_rate = memory_change_rate = bandwidth_change_rate = workload_change_rate = 0
                else:
                    cpu_change_rate = memory_change_rate = bandwidth_change_rate = workload_change_rate = 0
                
                # Combine all features
                vm_feature = {
                    'vm_idx': vm_idx,
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'bandwidth_usage': bandwidth_usage,
                    'storage_usage': storage_usage,
                    'workload': workload,
                    'response_time': response_time,
                    'resource_avg': resource_avg,
                    'resource_std': resource_std,
                    'resource_workload_ratio': resource_workload_ratio,
                    'cpu_change_rate': cpu_change_rate,
                    'memory_change_rate': memory_change_rate,
                    'bandwidth_change_rate': bandwidth_change_rate,
                    'workload_change_rate': workload_change_rate,
                }
                features.append(vm_feature)
        
        return features
    
    def _calculate_change_rate(self, historical_values, current_value):
        """Calculate the rate of change of a metric"""
        if not historical_values:
            return 0
        
        # Calculate the trend over the last few values
        n = min(5, len(historical_values))
        recent_values = historical_values[-n:]
        
        if n > 1:
            # Calculate slope of recent trend
            x = np.arange(n)
            y = np.array(recent_values)
            slope = np.polyfit(x, y, 1)[0]
            
            # Calculate deviation from trend
            expected = recent_values[-1] + slope
            deviation = current_value - expected
            
            return deviation
        else:
            # Simple difference if not enough history
            return current_value - recent_values[-1]
    
    def _detect_ddos_threats(self, vm_features):
        """Detect potential DDoS attacks"""
        detected_threats = []
        
        # Extract features relevant to DDoS detection
        ddos_features = []
        vm_indices = []
        
        for vm in vm_features:
            vm_indices.append(vm['vm_idx'])
            ddos_features.append([
                vm['bandwidth_usage'],
                vm['bandwidth_change_rate'],
                vm['cpu_usage'],
                vm['response_time'],
                vm['workload'],
                vm['resource_workload_ratio']
            ])
        
        # Transform features
        X = np.array(ddos_features)
        X_scaled = self.ddos_scaler.transform(X)
        
        # Predict with model
        predictions = self.ddos_model.predict(X_scaled)
        
        # Process predictions
        for i, prediction in enumerate(predictions):
            if prediction == 1:
                # Calculate severity based on feature values
                severity = min(1.0, (
                    ddos_features[i][0] * 0.4 +  # bandwidth_usage
                    abs(ddos_features[i][1]) * 0.3 +  # bandwidth_change_rate
                    ddos_features[i][3] * 0.3  # response_time
                ))
                
                detected_threats.append({
                    'vm_idx': vm_indices[i],
                    'type': 'ddos',
                    'severity': severity,
                    'details': 'Potential DDoS attack detected based on bandwidth and response time patterns'
                })
        
        return detected_threats
    
    def _detect_cryptojacking_threats(self, vm_features):
        """Detect potential cryptojacking attacks"""
        detected_threats = []
        
        # Extract features relevant to cryptojacking detection
        cryptojacking_features = []
        vm_indices = []
        
        for vm in vm_features:
            vm_indices.append(vm['vm_idx'])
            cryptojacking_features.append([
                vm['cpu_usage'],
                vm['cpu_change_rate'],
                vm['memory_usage'],
                vm['bandwidth_usage'],
                vm['workload'],
                vm['resource_workload_ratio']
            ])
        
        # Transform features
        X = np.array(cryptojacking_features)
        X_scaled = self.cryptojacking_scaler.transform(X)
        
        # Predict with model
        predictions = self.cryptojacking_model.predict(X_scaled)
        
        # Process predictions
        for i, prediction in enumerate(predictions):
            if prediction == 1:
                # Calculate severity based on feature values
                severity = min(1.0, (
                    cryptojacking_features[i][0] * 0.6 +  # cpu_usage
                    cryptojacking_features[i][2] * 0.2 +  # memory_usage
                    (1.0 - cryptojacking_features[i][5]) * 0.2  # inverse of resource_workload_ratio
                ))
                
                detected_threats.append({
                    'vm_idx': vm_indices[i],
                    'type': 'cryptojacking',
                    'severity': severity,
                    'details': 'Potential cryptojacking attack detected based on CPU usage patterns'
                })
        
        return detected_threats
    
    def _detect_malware_threats(self, vm_features):
        """Detect potential malware infections"""
        detected_threats = []
        
        # Extract features relevant to malware detection
        malware_features = []
        vm_indices = []
        
        for vm in vm_features:
            vm_indices.append(vm['vm_idx'])
            malware_features.append([
                vm['cpu_usage'],
                vm['memory_usage'],
                vm['storage_usage'],
                vm['bandwidth_usage'],
                vm['cpu_change_rate'],
                vm['memory_change_rate'],
                vm['resource_std']
            ])
        
        # Transform features
        X = np.array(malware_features)
        X_scaled = self.malware_scaler.transform(X)
        
        # Predict with model
        predictions = self.malware_model.predict(X_scaled)
        
        # Process predictions
        for i, prediction in enumerate(predictions):
            if prediction == 1:
                # Calculate severity based on feature values
                severity = min(1.0, (
                    malware_features[i][0] * 0.3 +  # cpu_usage
                    malware_features[i][1] * 0.2 +  # memory_usage
                    malware_features[i][3] * 0.3 +  # bandwidth_usage
                    malware_features[i][6] * 0.2  # resource_std
                ))
                
                detected_threats.append({
                    'vm_idx': vm_indices[i],
                    'type': 'malware',
                    'severity': severity,
                    'details': 'Potential malware infection detected based on resource usage patterns'
                })
        
        return detected_threats
    
    def _detect_insider_threats(self, vm_features):
        """Detect potential insider threats"""
        detected_threats = []
        
        # Extract features relevant to insider threat detection
        insider_features = []
        vm_indices = []
        
        for vm in vm_features:
            vm_indices.append(vm['vm_idx'])
            insider_features.append([
                vm['storage_usage'],
                vm['bandwidth_usage'],
                vm['workload'],
                vm['resource_workload_ratio'],
                vm['storage_usage'] / max(vm['workload'], 0.01)  # storage to workload ratio
            ])
        
        # Transform features
        X = np.array(insider_features)
        X_scaled = self.insider_scaler.transform(X)
        
        # Predict with model
        predictions = self.insider_model.predict(X_scaled)
        
        # Process predictions
        for i, prediction in enumerate(predictions):
            if prediction == 1:
                # Calculate severity based on feature values
                severity = min(1.0, (
                    insider_features[i][0] * 0.4 +  # storage_usage
                    insider_features[i][1] * 0.3 +  # bandwidth_usage
                    insider_features[i][4] * 0.3  # storage to workload ratio
                ))
                
                detected_threats.append({
                    'vm_idx': vm_indices[i],
                    'type': 'insider',
                    'severity': severity,
                    'details': 'Potential insider threat detected based on unusual storage and data transfer patterns'
                })
        
        return detected_threats
    
    def _rule_based_detection(self, vm_features):
        """
        Rule-based detection as a fallback and for initial detection
        before models are trained
        """
        detected_threats = []
        
        for vm in vm_features:
            vm_idx = vm['vm_idx']
            
            # DDoS rule: High bandwidth with high response time
            if (vm['bandwidth_usage'] > 0.8 and 
                vm['response_time'] > 0.7 and 
                abs(vm['bandwidth_change_rate']) > 0.2):
                
                severity = min(1.0, (
                    vm['bandwidth_usage'] * 0.5 + 
                    vm['response_time'] * 0.3 +
                    abs(vm['bandwidth_change_rate']) * 0.2
                ))
                
                detected_threats.append({
                    'vm_idx': vm_idx,
                    'type': 'ddos',
                    'severity': severity,
                    'details': 'Potential DDoS attack detected by rule-based system'
                })
            
            # Cryptojacking rule: High CPU with low workload
            if (vm['cpu_usage'] > 0.8 and 
                vm['workload'] < 0.4 and 
                vm['resource_workload_ratio'] > 2.0):
                
                severity = min(1.0, (
                    vm['cpu_usage'] * 0.6 + 
                    (1.0 - vm['workload']) * 0.2 +
                    min(vm['resource_workload_ratio'] / 5.0, 0.2)
                ))
                
                detected_threats.append({
                    'vm_idx': vm_idx,
                    'type': 'cryptojacking',
                    'severity': severity,
                    'details': 'Potential cryptojacking attack detected by rule-based system'
                })
            
            # Malware rule: Unusual resource usage patterns
            if (vm['resource_std'] > 0.3 and 
                vm['cpu_change_rate'] > 0.15 and
                vm['memory_change_rate'] > 0.15):
                
                severity = min(1.0, (
                    vm['resource_std'] * 0.4 + 
                    vm['cpu_change_rate'] * 0.3 +
                    vm['memory_change_rate'] * 0.3
                ))
                
                detected_threats.append({
                    'vm_idx': vm_idx,
                    'type': 'malware',
                    'severity': severity,
                    'details': 'Potential malware detected by rule-based system'
                })
            
            # Insider threat rule: High storage and bandwidth with moderate workload
            if (vm['storage_usage'] > 0.7 and 
                vm['bandwidth_usage'] > 0.6 and
                vm['workload'] < 0.6):
                
                storage_workload_ratio = vm['storage_usage'] / max(vm['workload'], 0.01)
                if storage_workload_ratio > 2.0:
                    severity = min(1.0, (
                        vm['storage_usage'] * 0.4 + 
                        vm['bandwidth_usage'] * 0.3 +
                        min(storage_workload_ratio / 10.0, 0.3)
                    ))
                    
                    detected_threats.append({
                        'vm_idx': vm_idx,
                        'type': 'insider',
                        'severity': severity,
                        'details': 'Potential insider threat detected by rule-based system'
                    })
        
        return detected_threats
    
    def generate_training_data(self, num_samples=1000):
        """Generate synthetic training data for each threat type"""
        # Generate normal data
        normal_data = self._generate_normal_data(num_samples // 2)
        
        # Generate attack data for each type and combine with normal data
        self._train_ddos_detector(normal_data, num_samples // 8)
        self._train_cryptojacking_detector(normal_data, num_samples // 8)
        self._train_malware_detector(normal_data, num_samples // 8)
        self._train_insider_detector(normal_data, num_samples // 8)
    
    def _generate_normal_data(self, num_samples):
        """Generate synthetic normal behavior data"""
        normal_data = []
        
        for _ in range(num_samples):
            # Generate normal resource usage patterns
            cpu = np.random.uniform(0.2, 0.7)
            memory = np.random.uniform(0.2, 0.7)
            bandwidth = np.random.uniform(0.1, 0.6)
            storage = np.random.uniform(0.1, 0.6)
            
            # Generate normal workload
            workload = np.random.uniform(0.2, 0.7)
            
            # Generate derived metrics
            resource_avg = np.mean([cpu, memory, bandwidth, storage])
            resource_std = np.std([cpu, memory, bandwidth, storage])
            resource_workload_ratio = resource_avg / max(workload, 0.01)
            
            # Generate rates of change
            cpu_change = np.random.uniform(-0.1, 0.1)
            memory_change = np.random.uniform(-0.1, 0.1)
            bandwidth_change = np.random.uniform(-0.1, 0.1)
            workload_change = np.random.uniform(-0.1, 0.1)
            
            # Generate response time correlated with workload and resources
            response_time = min(1.0, workload / max(resource_avg, 0.1) * np.random.uniform(0.8, 1.2))
            
            normal_data.append({
                'cpu_usage': cpu,
                'memory_usage': memory,
                'bandwidth_usage': bandwidth,
                'storage_usage': storage,
                'workload': workload,
                'response_time': response_time,
                'resource_avg': resource_avg,
                'resource_std': resource_std,
                'resource_workload_ratio': resource_workload_ratio,
                'cpu_change_rate': cpu_change,
                'memory_change_rate': memory_change,
                'bandwidth_change_rate': bandwidth_change,
                'workload_change_rate': workload_change,
            })
        
        return normal_data
    
    def _train_ddos_detector(self, normal_data, num_attack_samples):
        """Train the DDoS detector with synthetic data"""
        # Generate DDoS attack data
        attack_data = []
        
        for _ in range(num_attack_samples):
            # DDoS has high bandwidth, high response time
            cpu = np.random.uniform(0.6, 0.9)
            memory = np.random.uniform(0.3, 0.7)
            bandwidth = np.random.uniform(0.8, 1.0)
            storage = np.random.uniform(0.1, 0.6)
            
            # DDoS often has normal or low workload but high resource usage
            workload = np.random.uniform(0.1, 0.5)
            
            # High response time during attack
            response_time = np.random.uniform(0.7, 1.0)
            
            # Derived metrics
            resource_avg = np.mean([cpu, memory, bandwidth, storage])
            resource_std = np.std([cpu, memory, bandwidth, storage])
            resource_workload_ratio = resource_avg / max(workload, 0.01)
            
            # Sharp increase in bandwidth
            bandwidth_change = np.random.uniform(0.2, 0.5)
            cpu_change = np.random.uniform(0.1, 0.3)
            memory_change = np.random.uniform(-0.1, 0.1)
            workload_change = np.random.uniform(-0.1, 0.1)
            
            attack_data.append({
                'cpu_usage': cpu,
                'memory_usage': memory,
                'bandwidth_usage': bandwidth,
                'storage_usage': storage,
                'workload': workload,
                'response_time': response_time,
                'resource_avg': resource_avg,
                'resource_std': resource_std,
                'resource_workload_ratio': resource_workload_ratio,
                'cpu_change_rate': cpu_change,
                'memory_change_rate': memory_change,
                'bandwidth_change_rate': bandwidth_change,
                'workload_change_rate': workload_change,
            })
        
        # Prepare datasets
        X_normal = np.array([[
            d['bandwidth_usage'],
            d['bandwidth_change_rate'],
            d['cpu_usage'],
            d['response_time'],
            d['workload'],
            d['resource_workload_ratio']
        ] for d in normal_data])
        
        X_attack = np.array([[
            d['bandwidth_usage'],
            d['bandwidth_change_rate'],
            d['cpu_usage'],
            d['response_time'],
            d['workload'],
            d['resource_workload_ratio']
        ] for d in attack_data])
        
        X = np.vstack([X_normal, X_attack])
        y = np.hstack([np.zeros(len(normal_data)), np.ones(len(attack_data))])
        
        # Scale features
        self.ddos_scaler.fit(X)
        X_scaled = self.ddos_scaler.transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        # Train the model
        self.ddos_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.ddos_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"DDoS detector trained with accuracy: {accuracy:.4f}")
        
        self.is_trained['ddos'] = True
    
    def _train_cryptojacking_detector(self, normal_data, num_attack_samples):
        """Train the cryptojacking detector with synthetic data"""
        # Generate cryptojacking attack data
        attack_data = []
        
        for _ in range(num_attack_samples):
            # Cryptojacking has high CPU, moderate memory
            cpu = np.random.uniform(0.8, 1.0)
            memory = np.random.uniform(0.4, 0.8)
            bandwidth = np.random.uniform(0.1, 0.4)
            storage = np.random.uniform(0.1, 0.5)
            
            # Low workload despite high CPU
            workload = np.random.uniform(0.1, 0.4)
            
            # Normal to high response time
            response_time = np.random.uniform(0.4, 0.8)
            
            # Derived metrics
            resource_avg = np.mean([cpu, memory, bandwidth, storage])
            resource_std = np.std([cpu, memory, bandwidth, storage])
            resource_workload_ratio = resource_avg / max(workload, 0.01)
            
            # Steady high CPU usage
            cpu_change = np.random.uniform(-0.05, 0.1)
            memory_change = np.random.uniform(-0.1, 0.1)
            bandwidth_change = np.random.uniform(-0.1, 0.1)
            workload_change = np.random.uniform(-0.1, 0.1)
            
            attack_data.append({
                'cpu_usage': cpu,
                'memory_usage': memory,
                'bandwidth_usage': bandwidth,
                'storage_usage': storage,
                'workload': workload,
                'response_time': response_time,
                'resource_avg': resource_avg,
                'resource_std': resource_std,
                'resource_workload_ratio': resource_workload_ratio,
                'cpu_change_rate': cpu_change,
                'memory_change_rate': memory_change,
                'bandwidth_change_rate': bandwidth_change,
                'workload_change_rate': workload_change,
            })
        
        # Prepare datasets
        X_normal = np.array([[
            d['cpu_usage'],
            d['cpu_change_rate'],
            d['memory_usage'],
            d['bandwidth_usage'],
            d['workload'],
            d['resource_workload_ratio']
        ] for d in normal_data])
        
        X_attack = np.array([[
            d['cpu_usage'],
            d['cpu_change_rate'],
            d['memory_usage'],
            d['bandwidth_usage'],
            d['workload'],
            d['resource_workload_ratio']
        ] for d in attack_data])
        
        X = np.vstack([X_normal, X_attack])
        y = np.hstack([np.zeros(len(normal_data)), np.ones(len(attack_data))])
        
        # Scale features
        self.cryptojacking_scaler.fit(X)
        X_scaled = self.cryptojacking_scaler.transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        # Train the model
        self.cryptojacking_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.cryptojacking_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Cryptojacking detector trained with accuracy: {accuracy:.4f}")
        
        self.is_trained['cryptojacking'] = True
    
    def _train_malware_detector(self, normal_data, num_attack_samples):
        """Train the malware detector with synthetic data"""
        # Generate malware attack data
        attack_data = []
        
        for _ in range(num_attack_samples):
            # Malware has variable resource usage patterns
            cpu = np.random.uniform(0.4, 0.9)
            memory = np.random.uniform(0.4, 0.9)
            bandwidth = np.random.uniform(0.3, 0.8)
            storage = np.random.uniform(0.3, 0.8)
            
            # Variable workload
            workload = np.random.uniform(0.2, 0.6)
            
            # Variable response time
            response_time = np.random.uniform(0.3, 0.8)
            
            # Derived metrics
            resource_avg = np.mean([cpu, memory, bandwidth, storage])
            resource_std = np.random.uniform(0.3, 0.5)  # High variability
            resource_workload_ratio = resource_avg / max(workload, 0.01)
            
            # Rapid changes in resource usage
            cpu_change = np.random.uniform(0.15, 0.4) * (1 if np.random.random() > 0.5 else -1)
            memory_change = np.random.uniform(0.15, 0.4) * (1 if np.random.random() > 0.5 else -1)
            bandwidth_change = np.random.uniform(0.1, 0.3) * (1 if np.random.random() > 0.5 else -1)
            workload_change = np.random.uniform(-0.1, 0.1)
            
            attack_data.append({
                'cpu_usage': cpu,
                'memory_usage': memory,
                'bandwidth_usage': bandwidth,
                'storage_usage': storage,
                'workload': workload,
                'response_time': response_time,
                'resource_avg': resource_avg,
                'resource_std': resource_std,
                'resource_workload_ratio': resource_workload_ratio,
                'cpu_change_rate': cpu_change,
                'memory_change_rate': memory_change,
                'bandwidth_change_rate': bandwidth_change,
                'workload_change_rate': workload_change,
            })
        
        # Prepare datasets
        X_normal = np.array([[
            d['cpu_usage'],
            d['memory_usage'],
            d['storage_usage'],
            d['bandwidth_usage'],
            d['cpu_change_rate'],
            d['memory_change_rate'],
            d['resource_std']
        ] for d in normal_data])
        
        X_attack = np.array([[
            d['cpu_usage'],
            d['memory_usage'],
            d['storage_usage'],
            d['bandwidth_usage'],
            d['cpu_change_rate'],
            d['memory_change_rate'],
            d['resource_std']
        ] for d in attack_data])
        
        X = np.vstack([X_normal, X_attack])
        y = np.hstack([np.zeros(len(normal_data)), np.ones(len(attack_data))])
        
        # Scale features
        self.malware_scaler.fit(X)
        X_scaled = self.malware_scaler.transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        # Train the model
        self.malware_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.malware_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Malware detector trained with accuracy: {accuracy:.4f}")
        
        self.is_trained['malware'] = True
    
    def _train_insider_detector(self, normal_data, num_attack_samples):
        """Train the insider threat detector with synthetic data"""
        # Generate insider threat attack data
        attack_data = []
        
        for _ in range(num_attack_samples):
            # Insider threats have high storage and bandwidth usage
            cpu = np.random.uniform(0.2, 0.7)
            memory = np.random.uniform(0.2, 0.7)
            bandwidth = np.random.uniform(0.6, 0.9)
            storage = np.random.uniform(0.7, 1.0)
            
            # Normal workload
            workload = np.random.uniform(0.2, 0.6)
            
            # Normal response time
            response_time = np.random.uniform(0.2, 0.6)
            
            # Derived metrics
            resource_avg = np.mean([cpu, memory, bandwidth, storage])
            resource_std = np.std([cpu, memory, bandwidth, storage])
            resource_workload_ratio = resource_avg / max(workload, 0.01)
            
            # Changes focused on storage and bandwidth
            cpu_change = np.random.uniform(-0.1, 0.1)
            memory_change = np.random.uniform(-0.1, 0.1)
            bandwidth_change = np.random.uniform(0.05, 0.2)
            workload_change = np.random.uniform(-0.1, 0.1)
            
            attack_data.append({
                'cpu_usage': cpu,
                'memory_usage': memory,
                'bandwidth_usage': bandwidth,
                'storage_usage': storage,
                'workload': workload,
                'response_time': response_time,
                'resource_avg': resource_avg,
                'resource_std': resource_std,
                'resource_workload_ratio': resource_workload_ratio,
                'cpu_change_rate': cpu_change,
                'memory_change_rate': memory_change,
                'bandwidth_change_rate': bandwidth_change,
                'workload_change_rate': workload_change,
            })
        
        # Prepare datasets
        X_normal = np.array([[
            d['storage_usage'],
            d['bandwidth_usage'],
            d['workload'],
            d['resource_workload_ratio'],
            d['storage_usage'] / max(d['workload'], 0.01)  # storage to workload ratio
        ] for d in normal_data])
        
        X_attack = np.array([[
            d['storage_usage'],
            d['bandwidth_usage'],
            d['workload'],
            d['resource_workload_ratio'],
            d['storage_usage'] / max(d['workload'], 0.01)  # storage to workload ratio
        ] for d in attack_data])
        
        X = np.vstack([X_normal, X_attack])
        y = np.hstack([np.zeros(len(normal_data)), np.ones(len(attack_data))])
        
        # Scale features
        self.insider_scaler.fit(X)
        X_scaled = self.insider_scaler.transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        # Train the model
        self.insider_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.insider_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Insider threat detector trained with accuracy: {accuracy:.4f}")
        
        self.is_trained['insider'] = True
    
    def get_threat_summary(self):
        """Get a summary of detected threats"""
        if not self.threat_logs:
            return "No threats detected yet."
        
        threat_counts = {
            'ddos': 0,
            'cryptojacking': 0,
            'malware': 0,
            'insider': 0,
            'other': 0
            }
        
        affected_vms = set()
        highest_severity = {
                'ddos': 0,
                'cryptojacking': 0,
                'malware': 0,
                'insider': 0,
                'other': 0
        }
        
        # Count threats by type
        for threat in self.threat_logs:
            threat_type = threat['threat_type']
            if threat_type in threat_counts:
                threat_counts[threat_type] += 1
            else:
                threat_counts['other'] += 1
            
            affected_vms.add(threat['vm_idx'])
            
            # Track highest severity by type
            if threat_type in highest_severity:
                highest_severity[threat_type] = max(highest_severity[threat_type], threat['severity'])
            else:
                highest_severity['other'] = max(highest_severity['other'], threat['severity'])
        
        # Format summary
        summary = {
            'total_threats': len(self.threat_logs),
            'affected_vms': len(affected_vms),
            'threat_counts': threat_counts,
            'highest_severity': highest_severity,
            'recent_threats': self.threat_logs[-5:] if self.threat_logs else []
        }
        
        return summary