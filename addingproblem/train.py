import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ttest_ind
import time

# Extend the system path to include the parent directory, allowing for module imports from there.
sys.path.append('../')
from model import AP_RecurrentModel

def generate_data(T, batch_size, device):
    """
    Generates synthetic data for training and evaluation.
    
    :param T: Time series length.
    :param batch_size: Number of samples per batch.
    :param device: PyTorch device (cpu or cuda) for tensor allocation.
    :return: A tuple of combined input and labels tensors.
    """
    # Generate random sequences and initialize markers and labels.
    seq = torch.rand(batch_size, T, device=device)  # Random numbers between 0 and 1
    markers = torch.zeros(batch_size, T, device=device)
    labels = torch.zeros(batch_size, 1, device=device)
    
    # Assign random markers and calculate labels for each sequence.
    for i in range(batch_size):
        indices = np.sort(np.random.choice(T, 2, replace=False))
        markers[i, indices] = 1
        labels[i] = seq[i, indices].sum()
    
    # Combine sequence and markers as model input.
    combined_input = torch.stack((seq, markers), dim=-1)  
    return combined_input, labels

def calculate_mae(predictions, targets):
    """
    Calculates the Mean Absolute Error (MAE) between predictions and targets.
    
    :param predictions: Model predictions.
    :param targets: Ground truth labels.
    :return: MAE value as a float.
    """
    return torch.mean(torch.abs(predictions - targets)).item()

def train_model(model_type, T, num_steps, train_set_size, val_set_size, test_set_size, batch_size, learning_rate, seed, hidden_size, device):
    """
    Trains the model with specified parameters and synthetic data.
    
    :param model_type: The type of recurrent cell to use in the model.
    :param T: Time series length.
    :param num_steps: Number of training epochs.
    :param train_set_size: Size of the training dataset.
    :param val_set_size: Size of the validation dataset.
    :param test_set_size: Size of the test dataset.
    :param batch_size: Batch size for training.
    :param learning_rate: Learning rate for optimizer.
    :param seed: Random seed for reproducibility.
    :param hidden_size: Size of the hidden layer in the model.
    :param device: PyTorch device (cpu or cuda).
    :return: A dictionary with individual results for further analysis.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AP_RecurrentModel(2, hidden_size, 1, model_type=model_type).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_time = 0
    testing_time = 0
    val_maes = []
    test_maes = []

    for step in range(1, num_steps + 1):
        start_time = time.time()
        model.train()
        train_loss_sum = 0

        for _ in range(train_set_size // batch_size):
            combined_input, labels = generate_data(T, batch_size, device)
            optimizer.zero_grad()
            output = model(combined_input)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        average_train_loss = train_loss_sum / (train_set_size // batch_size)

        model.eval()
        with torch.no_grad():
            # Validate the model with a separate dataset and calculate MAE.
            combined_input, labels = generate_data(T, val_set_size, device)
            val_output = model(combined_input)
            val_loss = criterion(val_output, labels).item()
            val_mae = calculate_mae(val_output, labels)
            val_maes.append(val_mae)

            # Log training and validation loss for insight.
            if step % 100 == 0:
                print(f"Epoch {step}/{num_steps} - Train Loss: {average_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            
            # Evaluate the model on the test set.
            combined_input, labels = generate_data(T, test_set_size, device)
            start_test_time = time.time()
            test_output = model(combined_input)
            end_test_time = time.time()
            testing_time += end_test_time - start_test_time
            test_mae = calculate_mae(test_output, labels)
            test_maes.append(test_mae)

        training_time += time.time() - start_time

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Save the model state for future use.
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_save_path = f'{models_dir}/{model_type}_T{T}_seed{seed}.pth'
    torch.save(model.state_dict(), model_save_path)
    
    # Organize individual results for detailed analysis.
    individual_results = {
        'seed': seed,
        'val_maes': val_maes,
        'test_maes': test_maes,
        'average_training_time': training_time / num_steps,
        'average_testing_time': testing_time / num_steps,
        'num_parameters': num_parameters
    }

    # Save detailed results for further analysis and graph plotting.
    results_dir = 'results/detailed'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    detailed_results_path = f'{results_dir}/{model_type}_T{T}_seed{seed}_detailed_results.csv'
    pd.DataFrame(individual_results).to_csv(detailed_results_path, index=False)

    return individual_results

def prepare_and_display_final_results(all_model_results, learning_rates, seeds):
    """
    Aggregates and displays final results from all models and configurations, including statistical comparison using t-tests.
    
    :param all_model_results: Dictionary of all model results.
    :param learning_rates: Dictionary of learning rates for each model type.
    :param seeds: List of seeds used for training.
    """
    # Initialize containers for aggregated metrics.
    aggregated_metrics = {model_type: {'val_mae': [], 'test_mae': [], 'training_time': [], 'testing_time': []} for model_type in learning_rates}
    
    # Aggregate results from all configurations.
    for model_type, seeds_results in all_model_results.items():
        for seed, metrics in seeds_results.items():
            aggregated_metrics[model_type]['val_mae'].extend(metrics['val_maes'])
            aggregated_metrics[model_type]['test_mae'].extend(metrics['test_maes'])
            aggregated_metrics[model_type]['training_time'].append(metrics['average_training_time'])
            aggregated_metrics[model_type]['testing_time'].append(metrics['average_testing_time'])
    
    # Prepare DataFrame for final results.
    final_results = []
    for model_type, metrics in aggregated_metrics.items():
        final_results.append({
            'Model Type': model_type,
            'Mean Val MAE': np.mean(metrics['val_mae']),
            'STD Val MAE': np.std(metrics['val_mae']),
            'Mean Test MAE': np.mean(metrics['test_mae']),
            'STD Test MAE': np.std(metrics['test_mae']),
            'Mean Training Time': np.mean(metrics['training_time']),
            'STD Training Time': np.std(metrics['training_time']),
            'Mean Testing Time': np.mean(metrics['testing_time']),
            'STD Testing Time': np.std(metrics['testing_time']),
        })
    final_df = pd.DataFrame(final_results)

    # Perform t-tests between CGLSTMCellv1 and other models for MAE.
    benchmark_model = 'CGLSTMCellv1'
    for metric in ['val_mae', 'test_mae']:
        if benchmark_model not in aggregated_metrics:  # Skip if benchmark model not in results
            continue
        benchmark_data = aggregated_metrics[benchmark_model][metric]
        for model_type in learning_rates:
            if model_type == benchmark_model:
                continue
            if model_type not in aggregated_metrics:  # Ensure comparison model is in results
                continue
            model_data = aggregated_metrics[model_type][metric]
            t_stat, p_value = ttest_ind(benchmark_data, model_data, equal_var=False)
            final_df.loc[final_df['Model Type'] == model_type, f'P-Value {metric} vs. {benchmark_model}'] = p_value

    # Save and display the final aggregated results.
    final_results_path = 'results/final_aggregated_results.csv'
    final_df.to_csv(final_results_path, index=False)
    
    print("Final aggregated results saved to:", final_results_path)
def main():
    """
    Main function to execute the training and evaluation process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 128
    batch_size = 20
    num_steps = 10
    T_values = [100,1000]
    train_set_size = 100_000
    val_set_size = 10_000
    test_set_size = 10_000
    seeds = [42, 56, 30,59,6]
    learning_rates = {'Transformer': 1e-3,'GRUCell': 1e-3, 'LSTMCell': 1e-3, 'CGLSTMCellv0': 1e-3, 'CGLSTMCellv1': 1e-3, 'RAUCell': 1e-3}
    
    all_model_results = {}
    for model_type in learning_rates:
        model_results = {}
        for T in T_values:
            for seed in seeds:
                print(f"Training {model_type} for T={T} and seed={seed}...")
                result = train_model(model_type, T, num_steps, train_set_size, val_set_size, test_set_size, batch_size, learning_rates[model_type], seed, hidden_size, device)
                model_results[seed] = result
        all_model_results[model_type] = model_results

    prepare_and_display_final_results(all_model_results, learning_rates, seeds)

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
