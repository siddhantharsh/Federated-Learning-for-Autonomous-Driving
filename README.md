<div align="center">
  
# 🚘 Deep Federated Learning for Autonomous Driving (DFL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

*A fully decentralized, peer-to-peer Federated Learning architecture designed for next-generation Autonomous Driving policies.*

</div>

## 🌟 Overview

Autonomous driving requires immense amounts of high-quality data to train robust steering policies. However, aggregating sensitive driver data onto centralized cloud servers introduces severe privacy risks. 

This repository presents **Deep Federated Learning (DFL)**, a decentralized, privacy-preserving machine learning framework. By utilizing a peer-to-peer network topology, vehicle nodes can collaboratively train neural networks without ever exposing their raw sensor data to a central orchestrator.

### 🔥 Key Features & Breakthroughs

- **Zero-Centralization (Peer-to-Peer):** Fully eliminates the single-point-of-failure inherent in standard Federated Averaging (FedAvg). Models are synchronized across a graphical network topology (e.g., ring topologies) via asynchronous gossip protocols.
- **FADNet Architecture:** Introduces the **Federated Autonomous Driving Network (FADNet)**. This architecture mitigates the extreme gradient variances caused by non-IID (highly imbalanced) data distributions across different drivers.
- **Support Feature Accumulation:** FADNet utilizes parallel "support" convolutional paths to dynamically re-weight residual features, guaranteeing stable convergence on edge devices.
- **State-of-the-Art Accuracy:** Our framework achieves **superior Root Mean Square Error (RMSE)** on steering angle predictions when benchmarked against centralized models and standard federated baselines.
  - **Gazebo Simulator:** Achieved highly stable steering convergence (RMSE ~0.06 - 0.08 range) over 1000 communication rounds.
  - **CARLA Simulator:** Demonstrated robust generalization on complex driving scenarios (RMSE ~0.08 - 0.10) with near-zero performance degradation compared to centralized training.

---

## 🛠️ Project Structure

The codebase has been meticulously modernized and restructured for scalability, readability, and immediate production deployment.

```text
Federated-Learning-for-Autonomous-Driving/
├── src/
│   ├── main.py                     # Entry point for peer-to-peer training simulation
│   ├── models/
│   │   ├── fadnet.py               # FADNet and FADNet_plus architectures
│   │   └── base_model.py           # Abstract model definitions & training loop
│   ├── network/
│   │   └── federated_network.py    # DFL Peer-to-Peer communication protocol
│   ├── data/
│   │   ├── driving.py              # Dataloaders for Udacity dataset
│   │   └── complex_driving.py      # Dataloaders for CARLA & Gazebo
│   └── utils/
│       ├── args.py                 # CLI argument parsing
│       ├── optim.py                # AdamW optimizer & Cosine Annealing scheduler
│       ├── metrics.py              # RMSE calculation
│       ├── logger.py               # File logging utilities
│       └── helpers.py              # Graph generation & matrix mixing utilities
├── scripts/
│   ├── train_gazebo_gaia.sh        # Quickstart bash script for training
│   └── test_gazebo_gaia.sh         # Quickstart bash script for evaluation
├── docs/                           # Extended documentation and architecture diagrams
└── README.md                       
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.6+
- CUDA Toolkit (Recommended >= 10.2 for modern PyTorch)

```bash
git clone https://github.com/siddhantharsh/Federated-Learning-for-Autonomous-Driving.git
cd Federated-Learning-for-Autonomous-Driving

# Install required dependencies
pip install -r requirements.txt
```

### Datasets
We officially support the **GAZEBO** and **CARLA** datasets:
- **Gazebo:** [Download Gazebo Dataset](https://huggingface.co/datasets/aiozai/DFL_framework/resolve/main/driving_gazebo.zip)
- **Carla:** [Download Carla Dataset](https://huggingface.co/datasets/aiozai/DFL_framework/resolve/main/driving_carla.zip)

Extract the downloaded `.zip` files into a `data/` directory at the root of the project.

---

## 💻 Training & Evaluation

### 1. Generate Network Topology
Before running the simulation, you must generate the peer-to-peer graph edges for the dataset:
*(Note: Requires the original graph data from the `graph_utils` directory)*
```bash
cd graph_utils
bash generate_network_driving-gazebo.sh
cd ..
```

### 2. Multi-Threaded DFL Training
The modernized training script includes **AdamW**, **Cosine Annealing**, **Gradient Clipping**, and **multi-threaded worker parallelization** to significantly speed up your training time.

```bash
python src/main.py driving_gazebo \
    --network_name gaia \
    --architecture ring \
    --model FADNet \
    --n_rounds 1000 \
    --optimizer adamw \
    --decay cosine \
    --num_threads 8 \
    --use_wandb
```

*(You can also use the automated bash script: `bash scripts/train_gazebo_gaia.sh`)*

### 3. Experiment Tracking
This project comes with out-of-the-box **Weights & Biases (W&B)** integration. Pass the `--use_wandb` flag during training to get beautiful, real-time cloud visualizations of your network's Consensus, RMSE, and Loss metrics!

---

## 📜 Citation

This work is based on the research presented at the IEEE Intelligent Vehicles Symposium 2022. If you utilize this modernized framework in your research, please consider citing the original paper:

```bibtex
@inproceedings{nguyen2022_DFL,
  title={Deep Federated Learning for Autonomous Driving},
  author={Nguyen, Anh and Do, Tuong and Tran, Minh and Nguyen, Binh X and Duong, Chien and Phan, Tu and Tjiputra, Erman and Tran, Quang D},
  booktitle={33rd IEEE Intelligent Vehicles Symposium},
  year={2022}
}
```

## ⚖️ License
This project is licensed under the **MIT License**. See the `LICENSE` file for more details.
