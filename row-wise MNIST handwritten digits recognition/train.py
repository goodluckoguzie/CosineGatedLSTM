def main():

    import pandas as pd
    from scipy.stats import ttest_ind
    import numpy as np

    # Function to aggregate results, perform t-tests, and display the final comparison table
    def prepare_and_display_final_results(all_model_results, learning_rates, seeds):
        """
        Prepares and displays final aggregated results, including performing t-tests 
        to statistically compare model performances and saving the final comparison 
        table as a CSV file.
        
        Parameters:
        - all_model_results: A dictionary holding all the models' results keyed by model type.
        - learning_rates: A dictionary of learning rates used for each model type.
        - seeds: A list of seeds used for initializing the training process.
        """
        # Calculate means and prepare data
        model_metrics = {model_type: {'train_accuracy': [], 'val_accuracy': [], 'test_accuracy': [], 'training_time': [], 'testing_time': []} for model_type in learning_rates}

        for model_type, results in all_model_results.items():
            for seed, metrics in results.items():
                model_metrics[model_type]['train_accuracy'].extend(metrics['train_accuracies'])
                model_metrics[model_type]['val_accuracy'].extend(metrics['val_accuracies'])
                model_metrics[model_type]['test_accuracy'].append(metrics['test_accuracy'])
                model_metrics[model_type]['training_time'].append(metrics['training_time'])
                model_metrics[model_type]['testing_time'].append(metrics['testing_time'])

        # T-tests
        t_test_results = {}
        for model_type in learning_rates:
            for metric in ['train_accuracy', 'val_accuracy', 'test_accuracy']:
                _, p_value = ttest_ind(model_metrics[model_type][metric], model_metrics['CGLSTMv1'][metric], nan_policy='omit')
                t_test_results[(model_type, metric)] = p_value

        # Compile the final table data with mean metrics and p-values from t-tests
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
        # Save the final comparison table to a CSV file and print it
        df = pd.DataFrame(final_table_data)
        df.to_csv('results/model_comparison_final.csv', index=False)
        print(df)

    
    """
    Main function to train and evaluate different recurrent neural network models on the MNIST dataset,
    perform statistical tests to compare their performances, and save the results.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    import time
    import sys
    from scipy.stats import ttest_ind
    import pandas as pd

    sys.path.append('../')

    # Import your model
    from model import RowWise_RecurrentModel

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set hyperparameters and dataset transformations
    input_size = 28  # MNIST images are 28x28 pixels
    hidden_size = 128  # Number of features in the hidden state of the RNN
    output_size = 10  # MNIST has 10 classes (0 to 9)
    num_epochs = 213  # Number of times the whole dataset is passed through the network
    batch_size = 128  # Number of samples per batch to load


    Transformer_rate,RAU_learning_rate, CGLSTMCellv0_learning_rate, CGLSTMCellv1_learning_rate, GRU_learning_rate, LSTM_learning_rate = 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3


    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=test_transform)

    train_size = 55000
    validation_size = len(train_dataset) - train_size
    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    def evaluate_model(model, loader, criterion):
        """
        Evaluates the model's performance on a given dataset.
        
        Parameters:
        - model: The neural network model to evaluate
        - loader: DataLoader for the dataset to evaluate on
        - criterion: Loss function
        
        Returns:
        - avg_loss: Average loss over the dataset
        - accuracy: Accuracy percentage on the dataset
        """
        model.eval()
        total_loss = 0
        total_correct = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.view(-1, 28, input_size).to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
        avg_loss = total_loss / len(loader)
        accuracy = 100 * total_correct / len(loader.dataset)
        return avg_loss, accuracy

    def calculate_metrics(model, loader):
        """
        Calculates precision, recall, f1-score, and generates a confusion matrix
        for the model's performance on a given dataset.
        
        Parameters:
        - model: The neural network model to evaluate
        - loader: DataLoader for the dataset
        
        Returns:
        - precision: Weighted precision of the model's predictions
        - recall: Weighted recall of the model's predictions
        - f1_score: Weighted F1 score of the model's predictions
        - conf_matrix: Confusion matrix of the model's predictions
        """
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.view(-1, 28, input_size).to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_preds)
        return precision, recall, f1_score, conf_matrix

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train_and_evaluate(model_type, learning_rate, seed=None):
        """
        Trains and evaluates a model of specified type with given learning rate and seed,
        then aggregates and returns the results including training and validation losses,
        accuracies, precision, recall, f1-score, and timings.
        
        Parameters:
        - model_type: Type of the recurrent cell (e.g., LSTMCell, GRUCell)
        - learning_rate: Learning rate for the optimizer
        - seed: Random seed for reproducibility (optional)
        
        Returns:
        - A dictionary containing aggregated results from training and evaluation
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups, if applicable
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Initialize model
        model = RowWise_RecurrentModel(input_size, hidden_size, output_size, model_type).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Lists to store per-epoch metrics
        epoch_train_losses, epoch_val_losses = [], []
        epoch_train_accuracies, epoch_val_accuracies = [], []

        # Best accuracies initialization
        best_train_acc = 0
        best_val_acc = 0
        # Directory for saving models
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Record the start time of the training
        training_start_time = time.time()

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss, train_accuracy = 0, 0
            for images, labels in train_loader:
                images, labels = images.view(-1, 28, input_size).to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_accuracy += (predicted == labels).sum().item()
            train_loss /= len(train_loader)
            train_accuracy = 100 * train_accuracy / len(train_loader.dataset)
            epoch_train_losses.append(train_loss)
            epoch_train_accuracies.append(train_accuracy)

            # Validation phase
            model.eval()
            val_loss, val_accuracy = 0, 0
            with torch.no_grad():
                for images, labels in validation_loader:
                    images, labels = images.view(-1, 28, input_size).to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_accuracy += (predicted == labels).sum().item()
            val_loss /= len(validation_loader)
            val_accuracy = 100 * val_accuracy / len(validation_loader.dataset)
            epoch_val_losses.append(val_loss)
            epoch_val_accuracies.append(val_accuracy)

            # Check if current epoch's accuracies are the best so far
            if train_accuracy > best_train_acc and val_accuracy > best_val_acc:
                best_train_acc = train_accuracy
                best_val_acc = val_accuracy
                # Save model
                model_save_path = f'models/{model_type}_lr_{learning_rate}_seed_{seed}_epoch_{epoch+1}.pth'
                torch.save(model.state_dict(), model_save_path)
                # print(f'Model saved to {model_save_path} for epoch {epoch+1} with train acc: {train_accuracy}%, val acc: {val_accuracy}%')

            print(f'Epoch [{epoch+1}/{num_epochs}], {model_type} - Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Train Accuracy: {train_accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Calculate total training duration
        training_duration = time.time() - training_start_time

        # Record the start time of the testing
        testing_start_time = time.time()
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
        # Calculate testing duration
        testing_duration = time.time() - testing_start_time
        precision, recall, f1_score, _ = calculate_metrics(model, test_loader)

        # Return aggregated results
        return {
            'epoch_train_losses': epoch_train_losses,
            'epoch_val_losses': epoch_val_losses,
            'train_accuracies': epoch_train_accuracies,
            'val_accuracies': epoch_val_accuracies,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'training_time': training_duration,
            'testing_time': testing_duration
        }


    if not os.path.exists('results'):
        os.makedirs('results')

    learning_rates = {'Transformer': Transformer_rate,'LSTM': LSTM_learning_rate, 'CGLSTMv0': CGLSTMCellv0_learning_rate, 'CGLSTMv1': CGLSTMCellv1_learning_rate, 'RAUCell': RAU_learning_rate, 'GRU': GRU_learning_rate}
    seeds = [42, 6, 456, 789, 112]
    all_model_results = {}

    for model_type, lr in learning_rates.items():
        model_results = {}
        for seed in seeds:
            result = train_and_evaluate(model_type, lr, seed)
            model_results[seed] = result

            # Modify here to store lists as objects
            result_for_csv = {
                'seed': seed,
                'train_loss': result['epoch_train_losses'],  # Stored directly as list object
                'val_loss': result['epoch_val_losses'],  # Stored directly as list object
                'train_accuracy': result['train_accuracies'],  # Stored directly as list object
                'val_accuracy': result['val_accuracies'],  # Stored directly as list object
                'test_accuracy': [result['test_accuracy']],  # Single values wrapped in a list for consistency
                'precision': [result['precision']],
                'recall': [result['recall']],
                'f1_score': [result['f1_score']],
                'training_time': [result['training_time']],
                'testing_time': [result['testing_time']]
            }

            # Convert to DataFrame, ensuring each metric is handled as its appropriate data type
            df = pd.DataFrame([result_for_csv])  # Encapsulate in a list to ensure DataFrame creation

            # File naming and saving
            filename = f'results/{model_type}_lr_{lr:.1e}_seed_{seed}.csv'
            df.to_csv(filename, index=False)
            # print(f"Saved results to {filename}")

        all_model_results[model_type] = model_results
        # Prepare data for t-tests and final table
    prepare_and_display_final_results(all_model_results, learning_rates, seeds)



if __name__ == "__main__":
    main()
