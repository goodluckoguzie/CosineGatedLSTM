
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
from scipy.stats import ttest_ind
def main():

    # Extend the system path to include the parent directory for module imports
    sys.path.append('../')

    # Import the custom recurrent model class
    from model import FM_RecurrentModel

    # Configure the device (use CUDA if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Function to create a balanced validation set from the training dataset
    def create_validation_set(dataset, num_samples_per_class=500):
        """
        Creates indices for a balanced validation set.
        
        Parameters:
        - dataset: The full training dataset
        - num_samples_per_class: Number of samples per class to include in the validation set
        
        Returns:
        - validation_indices: Indices for the validation samples
        """
        targets = dataset.targets.numpy()
        class_indices = [np.where(targets == i)[0] for i in range(10)]
        validation_indices = np.hstack([np.random.choice(indices, num_samples_per_class, replace=False) for indices in class_indices])
        return validation_indices

    # Function to train the model for one epoch
    def train_model(model, train_loader, criterion, optimizer):
        """
        Trains the model for one epoch.

        Parameters:
        - model: The neural network model to train
        - train_loader: DataLoader for the training dataset
        - criterion: Loss function
        - optimizer: Optimization algorithm
        
        Returns:
        - avg_loss: Average loss over the training dataset
        - accuracy: Training accuracy
        """
        model.train()
        total_loss = 0
        total_correct = 0
        for images, labels in train_loader:
            images = images.view(-1, 28, 28).to(device)  
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * total_correct / len(train_loader.dataset)
        return avg_loss, accuracy

    # Function to evaluate the model on a given dataset
    def evaluate_model(model, loader, criterion):
        """
        Evaluates the model's performance on a dataset.

        Parameters:
        - model: The neural network model to evaluate
        - loader: DataLoader for the dataset to evaluate on
        - criterion: Loss function
        
        Returns:
        - avg_loss: Average loss over the dataset
        - accuracy: Accuracy on the dataset
        - duration: Time taken for evaluation
        """
        model.eval()
        total_loss = 0
        total_correct = 0
        start_time = time.time()
        with torch.no_grad():
            for images, labels in loader:
                images = images.view(-1, 28, 28).to(device)  # Reshape for the recurrent model
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
        avg_loss = total_loss / len(loader)
        accuracy = 100 * total_correct / len(loader.dataset)
        duration = time.time() - start_time
        return avg_loss, accuracy, duration

    # Function to count the trainable parameters in the model
    def count_trainable_parameters(model):
        """
        Counts the number of trainable parameters in a model.

        Parameters:
        - model: The neural network model
        
        Returns:
        - The count of trainable parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Main training and evaluation function
    def train_and_evaluate(model_type, learning_rate, seed):
        """
        Trains and evaluates a model.

        Parameters:
        - model_type: The type of recurrent cell to use
        - learning_rate: Learning rate for the optimizer
        - seed: Random seed for reproducibility
        
        Returns:
        - A dictionary containing training and evaluation metrics
        """
        torch.manual_seed(seed)
        # Initialize the model with specified parameters
        model = FM_RecurrentModel(input_size, hidden_size, output_size, model_type).to(device)

        print(f"Training model type: {model_type} with learning rate: {learning_rate}, seed: {seed}")
        print(f"Number of trainable parameters in {model_type}: {count_trainable_parameters(model)}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer)
            val_loss, val_accuracy, val_duration = evaluate_model(model, validation_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch [{epoch+1}/{num_epochs}], {model_type} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

        test_loss, test_accuracy, test_duration = evaluate_model(model, test_loader, criterion)

        # Save model and results
        save_dir = 'results/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_path = os.path.join(save_dir, f'{model_type}_model_seed_{seed}.pth')
        torch.save(model.state_dict(), model_save_path)

        print(f"\n{model_type} Test Accuracy: {test_accuracy:.2f}%")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "num_parameters": count_trainable_parameters(model),
            "training_time": val_duration,
            "testing_time": test_duration
        }


    # Set hyperparameters and load data
    input_size = 28  # Input size for FashionMNIST (28x28 images)
    hidden_size = 128  # Size of the hidden layer in the model
    output_size = 10  # Number of classes in FashionMNIST
    num_epochs = 213  # Number of epochs for training
    batch_size = 128  # Batch size for training and evaluation

    seeds = [10, 20,400]  # Seeds for reproducibility

    learning_rates = {
       
        'CGLSTMv0': 1e-3,'LSTM': 1e-3, 'CGLSTMv1': 1e-3, 'GRU': 1e-3, 'Transformer': 1e-3, 'RAUCell': 1e-3
    }


    # Define transformations for the datasets
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load and prepare datasets
    train_dataset_full = datasets.FashionMNIST(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=test_transform)

    validation_indices = create_validation_set(train_dataset_full)
    validation_dataset = Subset(train_dataset_full, validation_indices)
    train_indices = list(set(range(len(train_dataset_full))) - set(validation_indices))
    train_dataset = Subset(train_dataset_full, train_indices)

    # Data loaders for training, validation, and testing
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Training and evaluation loop for each model type and seed
    all_model_results = {}
    for model_type, lr in learning_rates.items():
        model_results = {}
        for seed in seeds:
            results = train_and_evaluate(model_type, lr, seed)
            model_results[seed] = results
            pd.DataFrame(results).to_csv(f'results/{model_type}_metrics_seed_{seed}.csv', index=False)
        all_model_results[model_type] = model_results

    # Analyze and display final results
    prepare_and_display_final_results(all_model_results, learning_rates, seeds)

# Function to prepare and display the final aggregated results
def prepare_and_display_final_results(all_model_results, learning_rates, seeds):
    """
    Prepares and displays final aggregated results from all model evaluations.
    
    Parameters:
    - all_model_results: Dictionary containing results from all models and seeds
    - learning_rates: Dictionary of learning rates used for each model type
    - seeds: List of seeds used for training
    
    No return value; prints and saves the final results to a CSV file.
    """

    # Calculate means and prepare data
    model_metrics = {model_type: {'train_accuracy': [], 'val_accuracy': [], 'test_accuracy': [], 'training_time': [], 'testing_time': []} for model_type in learning_rates}

    for model_type, results in all_model_results.items():
        for seed, metrics in results.items():
            model_metrics[model_type]['train_accuracy'].append(metrics['train_accuracy'])
            model_metrics[model_type]['val_accuracy'].append(metrics['val_accuracy'])
            model_metrics[model_type]['test_accuracy'].append(metrics['test_accuracy'])
            model_metrics[model_type]['training_time'].append(metrics['training_time'])
            model_metrics[model_type]['testing_time'].append(metrics['testing_time'])

    # T-tests
    t_test_results = {}
    for model_type in learning_rates:
        for metric in ['train_accuracy', 'val_accuracy', 'test_accuracy']:
            _, p_value = ttest_ind(model_metrics[model_type][metric], model_metrics['CGLSTMv0'][metric], nan_policy='omit')
            t_test_results[(model_type, metric)] = p_value

    # Creating the final table
    final_table_data = []
    for model_type in learning_rates:
        final_table_data.append({
            'Model': model_type,
            'Mean Train Accuracy (%)': np.mean(model_metrics[model_type]['train_accuracy']),
            'Mean Val Accuracy (%)': np.mean(model_metrics[model_type]['val_accuracy']),
            'Mean Test Accuracy (%)': np.mean(model_metrics[model_type]['test_accuracy']),
            'Mean Training Time (s)': np.mean(model_metrics[model_type]['training_time']),
            'Mean Testing Time (s)': np.mean(model_metrics[model_type]['testing_time']),
            'T-test p-value (Train Acc)': t_test_results[(model_type, 'train_accuracy')],
            'T-test p-value (Val Acc)': t_test_results[(model_type, 'val_accuracy')],
            'T-test p-value (Test Acc)': t_test_results[(model_type, 'test_accuracy')]
        })

    df = pd.DataFrame(final_table_data)
    df.to_csv('results/model_comparison_final.csv', index=False)
    print(df)

if __name__ == "__main__":
    main()
