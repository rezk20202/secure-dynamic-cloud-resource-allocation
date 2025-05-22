import numpy as np
import tensorflow as tf
import random
from collections import deque

class DRLAgent:
    """
    Deep Reinforcement Learning agent for cloud resource allocation
    
    This agent uses Deep Q-Learning to make optimal resource allocation
    decisions based on the system state, balancing performance and security.
    """
    def __init__(self, state_size, action_space):
        # Environment parameters
        self.state_size = state_size
        self.action_space = action_space
        
        # Action space components
        self.num_vms = action_space.nvec[0]
        self.num_action_types = action_space.nvec[1]
        self.action_values = action_space.nvec[2]
        
        # Learning parameters
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_freq = 10  # Update target network every N steps
        self.batch_size = 32
        
        # Create main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        
        # Training metrics
        self.train_step_counter = 0
        self.loss_history = []
    
    def _build_model(self):
        """Build a neural network model for deep Q-learning"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            # Output: [vm_idx, action_type, action_value]
            tf.keras.layers.Dense(self.num_vms * self.num_action_types * self.action_values)
        ])
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_network(self):
        """Update target network with weights from main network"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, evaluate=False):
        """
        Choose an action based on the current state
        
        Args:
            state: Current state vector
            evaluate: If True, use greedy policy (no exploration)
        
        Returns:
            action: [vm_idx, action_type, action_value]
        """
        if not evaluate and np.random.rand() <= self.epsilon:
            # Random action during training
            vm_idx = random.randrange(self.num_vms)
            action_type = random.randrange(self.num_action_types)
            action_value = random.randrange(self.action_values)
            return [vm_idx, action_type, action_value]
        
        # Reshape state for prediction
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1))
        
        # Get Q-values
        q_values = self.model(state_tensor).numpy()[0]
        
        # Reshape Q-values to match action space
        q_values = q_values.reshape(self.num_vms, self.num_action_types, self.action_values)
        
        # Get best action
        vm_idx, action_type, action_value = np.unravel_index(np.argmax(q_values), q_values.shape)
        
        return [vm_idx, action_type, action_value]
    
    def replay(self):
        """Train the model with experiences from memory"""
        if len(self.memory) < self.batch_size:
            return 0  # Not enough samples
        
        # Sample a batch of experiences
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare data for training
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict Q-values for current states
        q_values = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values for taken actions
        for i in range(self.batch_size):
            # Convert action to indices
            vm_idx, action_type, action_value = actions[i]
            
            # Calculate target Q-value
            if dones[i]:
                # Terminal state - only immediate reward
                target_q = rewards[i]
            else:
                # Non-terminal state - reward + discounted max future Q-value
                next_q_values_reshaped = next_q_values[i].reshape(
                    self.num_vms, self.num_action_types, self.action_values
                )
                max_next_q = np.max(next_q_values_reshaped)
                target_q = rewards[i] + self.gamma * max_next_q
            
            # Update Q-value for the action taken
            action_idx = vm_idx * (self.num_action_types * self.action_values) + \
                         action_type * self.action_values + \
                         action_value
            q_values[i][action_idx] = target_q
        
        # Train the model
        history = self.model.fit(states, q_values, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # Update target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_freq == 0:
            self.update_target_network()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def save_model(self, filepath):
        """Save the model to disk"""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load the model from disk"""
        try:
            # Try loading the full model first
            self.model = tf.keras.models.load_model(filepath)
        except (TypeError, ValueError) as e:
            print(f"Warning: Could not load complete model: {e}")
            print("Attempting to load just the weights...")
            
            # Create a fresh model with the same architecture
            self.model = self._build_model()
            
            # Try to load just the weights
            try:
                self.model.load_weights(filepath)
                print("Successfully loaded model weights")
            except:
                print("ERROR: Failed to load model weights. Using untrained model.")
        
        self.update_target_network()
    
    def get_metrics(self):
        """Return agent metrics for monitoring"""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'average_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0
        }