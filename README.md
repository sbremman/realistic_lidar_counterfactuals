# Realistic LiDAR Counterfactuals

This repository provides an implementation of generating realistic counterfactual explanations for Machine Learning (ML) agents that control mobile robots using raw 2D LiDAR data as input. The project explores methods to alter LiDAR data input in a realistic way to produce specific changes in the robot's actions, aiming to provide interpretable insights into the decision-making process.

## Features
- **Counterfactual Generation**: Tools to explore generating realistic LiDAR-based counterfactuals.
- **Deep Reinforcement Learning Integration**: Scripts to interface with DRL agents, evaluate counterfactuals, and visualize agent responses.
- **Example Use Case**: A script to showcase the potential application of counterfactual generation.

## Repository Structure
```
realistic_lidar_counterfactuals/
│
├── data/                     # Sample input data for training or experiments
├── examples/                 # Example scripts to demonstrate usage
├── models/                   # Pre-trained models or saved checkpoints
├── realistic_lidar_counterfactuals/ # Main package code
├── setup.py                  # For installation as a package
├── README.md                 # Project description, setup, and usage instructions
├── LICENSE                   # License information for your project
├── .gitignore                # Files to ignore in the repo
└── requirements.txt          # Python dependencies
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/sbremman/realistic_lidar_counterfactuals.git
   ```
2. Navigate to the project directory:
   ```sh
   cd realistic_lidar_counterfactuals
   ```
3. Install the core package dependencies:
   ```sh
   pip install .
   ```

### Optional Dependencies
To run the example scripts, you'll also need additional dependencies:
```sh
pip install torch stable-baselines3
```

## Usage
After installation, you can start by running the example scripts to understand how the counterfactuals are generated.

```sh
python examples/basic_usage.py
```

Refer to the `examples/` directory for guidance on using the package. Currently, there is one example script provided to demonstrate the usage. Please note that this is a demonstration project, and the methods may require further adaptation for specific use cases.

## Contributing
Contributions are welcome! If you have suggestions for improvements, please feel free to fork the repository, make changes, and submit a pull request.

## License
This project is licensed under the terms of the MIT license. See the **LICENSE** file for details.

## Contact
If you have any questions, feel free to open an issue on the repository.
