# CosineGatedLSTM: A Comparative Study of RNN Models

This repository is dedicated to comparing novel Cosine Gated LSTM models (CGLSTMv0 and CGLSTMv1) with traditional recurrent neural network (RNN) models (LSTM, GRU, and RAU) across multiple datasets and tasks. The primary goal is to evaluate the performance of CGLSTM models in various contexts, focusing on their ability to capture complex temporal dynamics compared to their traditional counterparts.

## Detailed Task Overview

Each task is designed to test different aspects of RNN capabilities, from handling simple numerical sequences to complex language modeling. The tasks, datasets, training procedures, and result handling are as follows:

### 1. Adding Problem

- **Dataset**: Synthetic dataset generated to test the models' ability to identify and sum two significant numbers in a sequence filled with noise.
- **Approach**: Models are trained to predict the sum of two marked numbers in sequences of varying lengths, emphasizing the ability to capture long-range dependencies.
- **Result Handling**: Performance is quantified using the Mean Absolute Error (MAE) between the predicted and actual sums, stored in a comparative table highlighting differences between models.

### 2. Fashion-MNIST Item Classification

- **Dataset**: Fashion-MNIST, a dataset comprising 60,000 training and 10,000 test images of fashion items across 10 categories.
- **Approach**: Despite being an image dataset, each image is treated as a sequence of pixel rows to test the sequential processing capabilities of each RNN model.
- **Result Handling**: Classification accuracy on the test set serves as the primary performance metric, with results compiled in a table to facilitate model comparison.

### 3. Row-wise MNIST Handwritten Digits Recognition

- **Dataset**: The classic MNIST dataset of handwritten digits, processed in a row-wise sequential manner.
- **Approach**: Similar to Fashion-MNIST, this task challenges the models to recognize digits from sequential row input, testing their effectiveness on a well-established benchmark.
- **Result Handling**: Accuracy metrics are collected and presented in a comparative table, showcasing each model's performance on digit recognition.

### 4. Sentiment Analysis on the IMDB Movie Reviews Dataset

- **Dataset**: The IMDB dataset containing 50,000 reviews split evenly into training and test sets, aimed at binary sentiment classification.
- **Approach**: Models process tokenized review texts to predict the sentiment, testing each model's ability to understand natural language and capture sentiment indicators.
- **Result Handling**: Results are evaluated based on accuracy, precision, recall, and F1 score, with detailed performance comparisons stored for analysis.

### 5. Word-level Language Modeling on the Penn Treebank Corpus

- **Dataset**: The Penn Treebank dataset, a standard benchmark for language modeling, challenging models to predict the next word in a sequence.
- **Approach**: This task assesses the models' capacity for learning and generating coherent text, requiring an understanding of syntax, grammar, and context.
- **Result Handling**: Perplexity is the main metric, offering insight into how well each model predicts word sequences. Comparative results are tabulated, highlighting the strengths and weaknesses of each model.

## Training and Evaluation

Each model undergoes rigorous training using a consistent framework to ensure fair comparison. Hyperparameters are carefully selected to optimize performance across tasks, with results meticulously recorded to facilitate analysis. Training involves:

- Using cross-entropy loss for classification tasks and mean squared error for regression tasks.
- Applying Adam optimizer for adaptive learning rate adjustments.
- Implementing early stopping based on validation loss to prevent overfitting.

## Results and Comparative Analysis

Upon completion of training for each task, results are aggregated into comprehensive tables, illustrating:

- Task-specific metrics (accuracy, MAE, perplexity) for direct performance comparison.
- Statistical analysis, including t-tests, to assess the significance of performance differences.
- Training efficiency metrics (time, number of parameters) to evaluate model complexity and scalability.

These tables serve as the basis for evaluating the proposed CGLSTM models against traditional RNN architectures, shedding light on areas of improvement, applicability to different tasks, and overall effectiveness in sequence modeling challenges.

## Contributing and License

Contributions to enhance model performance, introduce new tasks, or improve result analysis are welcome. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project. This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
