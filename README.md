# secure-dynamic-cloud-resource-allocation

This project integrates Deep Reinforcement Learning (DRL) for efficient cloud resource allocation with an Intrusion Detection System (IDS) for enhanced security. The system optimizes resource allocation while monitoring for potential security threats.

## Features

- DRL-based resource allocation for optimal VM placement and resource distribution
- Ensemble-based intrusion detection (Random Forest, SVM, XGBoost)
- Real-time security monitoring running periodically
- Detection of anomalous patterns including potential DDoS, cryptojacking, malware, and insider threats
- Comprehensive logging and visualization

## System Requirements

- macOS (for iOS laptop)
- Python 3.8+
- 4GB+ RAM
- 2GB+ free disk space

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/secure-cloud-allocation.git
cd secure-cloud-allocation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the System

To start the system with default parameters:

```bash
python main.py
```

For custom configuration:

```bash
python main.py --num-vms 20 --security-weight 0.4 --training-episodes 300
```

### Demo Notebook

A demonstration notebook is provided to help understand the system:

```bash
jupyter notebook notebooks/demo.ipynb
```

### Configuration Options

- `--num-vms`: Number of VMs to manage (default: 10)
- `--num-resources`: Number of resource types (default: 3)
- `--security-weight`: Weight of security in resource allocation (default: 0.3)
- `--training-episodes`: Number of episodes for training (default: 200)
- `--evaluation-episodes`: Number of episodes for evaluation (default: 50)
- `--batch-size`: Batch size for DRL agent training (default: 32)
- `--monitoring-interval`: Interval for security monitoring in seconds (default: 5)
- `--log-level`: Logging level (default: INFO)

## System Architecture

The system consists of several key components:

1. **Cloud Environment**: Simulates a cloud environment with multiple VMs and resources
2. **DRL Agent**: Learns optimal resource allocation policies
3. **Intrusion Detection System**: Monitors for security threats
4. **Secure Cloud Manager**: Integrates DRL and IDS components
5. **Logger**: Handles real-time logging of system events
6. **Visualizer**: Provides visualization of system performance

## Extending the System

### Adding New Threat Detection

To add detection for new types of threats, extend the `IntrusionDetectionSystem` class in `src/ids.py` with new detection methods.

### Custom Resource Allocation Policies

To implement custom resource allocation policies, modify the reward function in `src/environment.py` or create a new DRL model in `src/drl_agent.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project builds upon:
- [Deep-Reinforcement-Learning-for-cloud](https://github.com/Shahid-Mohammed-Shaikbepari/Deep-Reinforcement-Learning-for-cloud)
- Ensemble-based intrusion detection techniques
