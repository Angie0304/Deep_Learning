# Mutilayer Perceptron 

This module implements a Multilayer Perceptron (MLP) for fault classification using structured industrial data. It includes data preprocessing, feature preparation, and model training to classify different types of faults based on input features.

# Module Structure 

```text
multilayer_perceptron/
├── data/                             # Dataset storage
│   ├── raw/                          # Original data files
│   │   └── Faults.NNA
│   └── processed/                    # Cleaned and prepared data
│       └── faults_processed.csv
├── notebooks/                        # Implementation and preprocessing
│   ├── perceptron_multicapa.ipynb
|   └── preprocesamiento.ipynb
├── requirements.txt                  # Dependencies
└── README.md                         # Documentation and usage guide
```
# How it works 

The module follows these steps:

1. Load the raw dataset containing industrial fault data
2. Preprocess and clean the data for model training
3. Transform and organize features into a structured format
4. Split the dataset into training and testing sets
5. Build a Multilayer Perceptron (MLP) model
6. Train the model using the processed data
7. Evaluate the model on test data
8. Analyze classification performance
