# Realistic LiDAR Counterfactuals

This repository provides an implementation of generating realistic counterfactual explanations for Deep Reinforcement Learning (DRL) agents that control mobile robots using raw LiDAR data as input. The project demonstrates methods to alter LiDAR data input in a realistic way to produce specific changes in the robot's actions, providing interpretable insights into the decision-making process.

## Features
- **Counterfactual Generation**: Methods to generate realistic LiDAR-based counterfactuals.
- **Deep Reinforcement Learning Integration**: Tools to interface with DRL agents, evaluate counterfactuals, and visualize agent responses.
- **Example Use Cases**: Scripts to showcase the application of counterfactual generation in real-world scenarios.

## Repository Structure
```
realistic_lidar_counterfactuals/
│
├── data/                     # Sample input data for training or experiments
├── examples/                 # Example scripts to demonstrate usage
├── models/                   # Pre-trained models or saved checkpoints
├── realistic_lidar_counterfactuals/ # Main package code
├── tests/                    # Unit tests for your package
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
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
After installation, you can start by running the example scripts to understand how the counterfactuals are generated.

```sh
python examples/generate_counterfactual.py
```

Refer to the `examples/` directory for more detailed examples and guidance on using the package.

## Running Tests
To run the unit tests, use the following command:
```sh
pytest tests/
```

## Contributing
Contributions are welcome! If you have suggestions for improvements, please feel free to fork the repository, make changes, and submit a pull request.

## License
This project is licensed under the terms of the MIT license. See the **LICENSE** file for details.

## Contact
If you have any questions, feel free to open an issue on the repository or contact Sindre Benjamin Remman.
