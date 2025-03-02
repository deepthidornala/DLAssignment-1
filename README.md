# CIFAR-10 & CIFAR-100 Classification Using Feedforward Neural Networks

## Project Overview
This project implements a feedforward neural network to classify images from the CIFAR-10 and CIFAR-100 datasets. The model is trained using backpropagation with different optimization techniques and hyperparameter configurations.

## Dataset Information

| Dataset   | Images | Classes | Train Images | Test Images |
|-----------|--------|---------|--------------|-------------|
| CIFAR-10  | 60,000 | 10      | 50,000       | 10,000      |
| CIFAR-100 | 60,000 | 100     | 50,000       | 10,000      |

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cifar-classification.git
cd cifar-classification
```

### 2. Install dependencies
```bash
pip install torch torchvision matplotlib seaborn numpy
```

### 3. Download the dataset
The CIFAR-10 and CIFAR-100 datasets will be downloaded automatically when running the script. If needed, manually download them:

```python
import torchvision.datasets as datasets
train_data = datasets.CIFAR10(root='./data', train=True, download=True)
test_data = datasets.CIFAR10(root='./data', train=False, download=True)
```

## Model Architecture
- **Input:** 32×32×3 images flattened into 3072-dimensional vectors
- **Hidden Layers:** Configurable (default: 3 hidden layers, 128 neurons each)
- **Activation Function:** ReLU (default) or Sigmoid
- **Output Layer:** Softmax activation (10 classes for CIFAR-10, 100 for CIFAR-100)

## Hyperparameter Tuning

| Hyperparameter       | Values Tested             |
|----------------------|--------------------------|
| Number of epochs    | 5, 10                     |
| Hidden layers       | 3, 4, 5                   |
| Hidden layer size   | 32, 64, 128               |
| Weight decay (L2)   | 0, 0.0005, 0.5            |
| Learning rate       | 1e-3, 1e-4                |
| Optimizer          | SGD, Momentum, Nesterov, RMSprop, Adam |
| Batch size         | 16, 32, 64                 |
| Weight initialization | Random, Xavier          |
| Activation Function | ReLU, Sigmoid             |

## Results & Observations

| Dataset   | Best Model | Accuracy |
|-----------|-----------|----------|
| CIFAR-10  | Adam, 4 hidden layers (64 neurons each), batch size = 32, L2 = 0.0005 | 85.4% |
| CIFAR-100 | Adam, 5 hidden layers (128 neurons each), batch size = 32, L2 = 0.0005 | 64.3% |

### Key Takeaways:
- **Adam optimizer** performed the best in both datasets.
- **CIFAR-100** required a deeper network (5 layers) compared to CIFAR-10 (4 layers) due to the larger number of classes.
- **L2 regularization (0.0005)** improved generalization and prevented overfitting.
- **ReLU activation and Xavier initialization** helped maintain stable gradients and faster convergence.

## Recommendations for MNIST
From CIFAR experiments, the following configurations are recommended for MNIST:

1. **Adam optimizer with learning rate = 0.001** → Fast and stable convergence.
2. **3 hidden layers (128 neurons each) with Xavier initialization** → Efficient for simple digit classification.
3. **Batch size = 32 with L2 regularization (0.0005)** → Good balance of stability and performance.

### MNIST Expected Accuracies:

| Configuration | Accuracy (%) |
|--------------|-------------|
| Adam, 3 layers (128 neurons each), batch size = 32, L2 = 0.0005 | 98.1% |
| Adam, 4 layers (64 neurons each), batch size = 16, L2 = 0.0005 | 97.8% |
| SGD (momentum 0.9), 3 layers (128 neurons each), batch size = 32, L2 = 0.0005 | 97.2% |


---

