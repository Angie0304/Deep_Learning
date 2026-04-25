# CNN with Residual Blocks for Fashion MNIST

This module implements a Convolutional Neural Network (CNN) for Fashion MNIST classification, incorporating residual connections, data augmentation, and advanced training strategies to enhance model performance.

## Module Structure 

``` text 
residual_blocks_fashion_mnist/
├── README.md                 # Documentation and execution guide
├── cnn_fashion.py            # CNN with residual blocks for Fashion MNIST
└── requirments.txt           # Dependencies
```

## How it works

The module follows these steps:

1. Load and normalize the Fashion MNIST dataset
2. Apply data augmentation techniques
3. Build a CNN with residual blocks
4. Compile the model with optimizer and loss function
5. Train the model with callbacks and learning rate scheduling
6. Evaluate model performance on test data
7. Visualize training accuracy over epochs

# Usage 

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run the program 
```bash
python cnn_fashion.py
```

3. Output
- Display training and validation accuracy
- Save the best model during training

# Dataset 

This project uses the Fashion MNIST dataset, which is automatically downloaded using TensorFlow when the program is executed.

# Status 
Complete
