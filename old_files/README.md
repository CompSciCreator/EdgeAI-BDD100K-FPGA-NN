 # BDD100K Object Detection for Autonomous Driving

This project compares the performance of multiple machine learning models for object detection using the **BDD100K dataset**. The goal is to evaluate which approach provides the best balance of **accuracy**, **speed**, and **efficiency** in the context of autonomous driving.

## Table of Contents

- [Dataset](#dataset)
- [Objectives](#objectives)
- [Approaches](#approaches)
- [Tools & Technologies](#tools--technologies)
- [Results](#results)
- [Getting Started](#getting-started)
- [License](#license)

---

## Dataset

The [BDD100K dataset](https://www.kaggle.com/datasets/bdd100k/bdd100k) is a diverse, large-scale driving video dataset commonly used for autonomous vehicle research.

- **Total Images**: 69,853
- **Environment Labels**: Daytime, Nighttime, and other conditions
- **Focus**: Real-world road scenarios including roads, buildings, trees, signs, vehicles, and pedestrians
- **Use Case**: Ideal for object detection tasks in complex driving environments

---

## Objectives

- Train a PyTorch-based object detection model using BDD100K
- Compare three approaches for performance:
  1. Baseline Neural Network
  2. Optimized Neural Network on Kria 
  3. FPGA-accelerated model using Vitis AI
- Evaluate each model based on:
  - **Accuracy**
  - **Training time**
  - **Hardware efficiency**

---

## Approaches

### 1. Baseline Neural Network (PyTorch)

- Implemented a basic NN architecture to detect objects in BDD100K images
- Evaluated accuracy and speed on a standard setup

### 2. Optimized NN on Kria

- Refined architecture and weight adjustments
- Deployed model to the **Kria** system-on-module
- Measured improvements in performance and inference time

### 3. Vitis AI with FPGA Acceleration

- Leveraged the onboard FPGA on the Kria platform
- Compiled and deployed the model using **Vitis AI**
- Evaluated for maximum speed and power efficiency

---

## Tools & Technologies

- [PyTorch](https://pytorch.org/)
- [Kria SOM]([https://www.amd.com/en/products/som/kria](https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit.html))
- [Vitis AI](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html)
- [BDD100K Dataset](https://www.kaggle.com/datasets/bdd100k/bdd100k)

---

## Results

Performance comparison across the three focuses on:

- **Accuracy of object detection** under various lighting and environmental conditions
- **Training and inference speed**
- **Hardware utilization and efficiency**

Final metrics will be published in the `results/` folder once evaluations are complete.

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Vitis AI installed (for FPGA deployment)
- Access to the Kria development platform (if testing on hardware)

### Installation

## Training

## Inference on Kria
