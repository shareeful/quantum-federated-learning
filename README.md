# Federated Hybrid Quantum Machine Learning & Reinforcement Learning

This repository contains the implementation and experimental results for a research project investigating the "Quantum Advantage" in distributed and dynamic learning environments.

The project evaluates a **Hybrid Quantum-Classical Neural Network** architecture across two distinct domains:

1. **Federated Learning** for Computer Vision (CIFAR-10).
2. **Quantum Reinforcement Learning** for Dynamic Control (Safety Gym).

## 🚀 Key Features

* **Hybrid Architecture:** Integrates classical deep learning (CNNs/ResNet) with Variational Quantum Circuits (VQCs) using **PennyLane** and **PyTorch**.
* **Federated Learning Engine:** Custom implementation of the **FedAvg** algorithm to simulate distributed training with Non-IID data partitions.
* **Quantum PPO Agent:** A Proximal Policy Optimization agent where the policy network is approximated by an entangled quantum circuit.
* **Custom RL Environment:** A "Safety Gym" environment built with **PyGame** to test safe navigation and obstacle avoidance.
* **Benchmarking Tools:** Scripts to compare Quantum vs. Classical performance regarding convergence speed, sample efficiency, and parameter usage.

## 🛠️ Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch torchvision numpy matplotlib pandas
pip install pennylane
pip install pygame
pip install seaborn
```

## 🔬 Experiment I: Federated Static Pattern Recognition (CIFAR-10)

This experiment validates the ability of a Federated Hybrid QNN to aggregate static visual features from distributed clients. It compares **Federated Quantum**, **Centralized Quantum**, and **Federated Classical** models.

### Architecture
-   **Classical Backbone:** ResNet18 (frozen/pretrained) or Custom CNN.
-   **Quantum Layer:** 4-Qubit Variational Circuit with `AngleEmbedding` and `StronglyEntanglingLayers`.
-   **Aggregation:** FedAvg across 5 clients.

### Running the Experiment
To train the Federated Hybrid QNN on CIFAR-10:

```bash
run FL_QNN_CIFAR10.ipynb
```
## 🤖 Experiment II: Quantum Reinforcement Learning (Safety Gym)
This experiment isolates the quantum architecture to test its Sample Efficiency in a dynamic control task. It compares a Hybrid Quantum PPO (RL-Q) agent against a Simple Classical PPO (Simple RL) agent.
### Environment
- A custom 2D navigation task where a robot must reach a goal while avoiding randomized hazard zones.
- State: 11-dimensional vector (Position, Goal Vector, Lidar sensors).
- Action: Continuous velocity control ($v_x, v_y$).
### Running the Experiment
To train the Quantum RL agent with real-time visualization:
```bash
run QNN-FL/training.py
```
