import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ttest_ind
import time
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

def get_batch(time_steps, batch_size):
    """Generate the adding problem dataset"""
    add_values = torch.rand(batch_size, time_steps)  # Swap the dimensions
    add_indices = torch.zeros_like(add_values)
    half = int(time_steps / 2)
    for i in range(batch_size):
        first_half = np.random.randint(half)
        second_half = np.random.randint(half, time_steps)
        add_indices[i, first_half] = 1  # Note the index change to match batch dimension
        add_indices[i, second_half] = 1
    inputs = torch.stack((add_values, add_indices), dim=-1)
    targets = torch.mul(add_values, add_indices).sum(dim=1).unsqueeze(1)  # Ensure correct dimension reduction
    return inputs, targets


def calculate_mae(predictions, targets):
    """Calculates the Mean Absolute Error (MAE) between predictions and targets."""
    return torch.mean(torch.abs(predictions - targets)).item()

# def train_model(model_type, T, num_steps, train_set_size, val_set_size, test_set_size, batch_size, learning_rate, seed, hidden_size, device):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     model = AP_RecurrentModel(2, hidden_size, 1, model_type=model_type).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     training_time = 0
#     testing_time = 0
#     val_maes = []
#     test_maes = []
    
#     for step in range(1, num_steps + 1):
#         start_time = time.time()
#         model.train()
#         train_loss_sum = 0
        
#         for _ in range(train_set_size // batch_size):
#             combined_input, labels = get_batch(T, batch_size)
#             combined_input, labels = combined_input.to(device), labels.to(device)
#             optimizer.zero_grad()
#             output = model(combined_input)
#             loss = criterion(output, labels)
#             loss.backward()
#             optimizer.step()
#             train_loss_sum += loss.item()

def adjust_learning_rate(optimizer, step, initial_lr, step_size=20000, lr_decay=0.5):
    """Adjusts learning rate every `step_size` steps by multiplying it with `lr_decay`."""
    if step % step_size == 0 and step > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print(f"Reduced learning rate to {param_group['lr']} at step {step}")

def train_model(model_type, T, num_steps, train_set_size, val_set_size, test_set_size, batch_size, learning_rate, seed, hidden_size, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AP_RecurrentModel(2, hidden_size, 1, model_type=model_type).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    training_time = 0
    testing_time = 0
    val_maes = []
    test_maes = []
    step = 0  # Initialize step counter
    
    for epoch in range(1, num_steps + 1):
        start_time = time.time()
        model.train()
        train_loss_sum = 0
        
        for _ in range(train_set_size // batch_size):
            step += 1  # Increment step counter
            adjust_learning_rate(optimizer, step, learning_rate)  # Adjust learning rate if needed
            
            combined_input, labels = get_batch(T, batch_size)
            combined_input, labels = combined_input.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(combined_input)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()       
        average_train_loss = train_loss_sum / (train_set_size // batch_size)
        
        model.eval()
        with torch.no_grad():
            combined_input, labels = get_batch(T, val_set_size)
            combined_input, labels = combined_input.to(device), labels.to(device)
            val_output = model(combined_input)
            val_loss = criterion(val_output, labels).item()
            val_mae = calculate_mae(val_output, labels)
            val_maes.append(val_mae)
            
            combined_input, labels = get_batch(T, test_set_size)
            combined_input, labels = combined_input.to(device), labels.to(device)
            start_test_time = time.time()
            test_output = model(combined_input)
            end_test_time = time.time()
            testing_time += end_test_time - start_test_time
            test_mae = calculate_mae(test_output, labels)
            test_maes.append(test_mae)
        
        training_time += time.time() - start_time

    # Save model
    model_save_path = f'models/{model_type}/'
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), f'{model_save_path}model_seed_{seed}.pth')

    # Prepare individual results
    individual_results = {
        'seed': seed,
        'val_mae': np.mean(val_maes),  # Aggregate to a single mean value
        'test_mae': np.mean(test_maes),  # Aggregate to a single mean value
        'average_training_time': training_time / num_steps,
        'average_testing_time': testing_time / num_steps,  # Calculate average testing time across all steps
        'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    # Save individual results
    results_save_path = f'results/{model_type}/'
    os.makedirs(results_save_path, exist_ok=True)
    df_individual_results = pd.DataFrame([individual_results])
    df_individual_results.to_csv(f'{results_save_path}results_seed_{seed}.csv', index=False)
    
    return individual_results

import pandas as pd
from scipy.stats import ttest_ind

def prepare_and_display_final_results(all_model_results, learning_rates):
    all_model_averages = {}
    # Extract the test MAEs for CGGRU to use as a baseline for t-tests
    cglstm_test_maes = np.array([result['test_mae'] for result in all_model_results['CGLSTM'].values()])

    for model_type, seeds_results in all_model_results.items():
        dfs = []
        # Collect all seed results for the current model
        for seed, results in seeds_results.items():
            df = pd.DataFrame([results])
            dfs.append(df)
        
        model_aggregate = pd.concat(dfs, ignore_index=True)
        
        model_mean = model_aggregate.mean().to_dict()
        model_std = model_aggregate.std().to_dict()
        model_mean['Model Type'] = model_type
        model_mean['STD Val MAE'] = model_std['val_mae']
        model_mean['STD Test MAE'] = model_std['test_mae']
        
        # Perform t-test if current model is not CGLSTM
        if model_type != 'CGLSTM':
            current_test_maes = np.array([result['test_mae'] for result in seeds_results.values()])
            t_stat, p_value_test = ttest_ind(cglstm_test_maes, current_test_maes, equal_var=False)
            
            model_mean['T-test Statistic (vs. CGLSTM)'] = t_stat
            model_mean['T-test p-value (vs. CGLSTM)'] = p_value_test
        
        all_model_averages[model_type] = model_mean

    # Convert aggregated means to DataFrame for easy CSV writing and display
    df_model_averages = pd.DataFrame(list(all_model_averages.values()))
    df_model_averages.to_csv('model_averages_across_seeds.csv', index=False)
    print("Model averages across seeds saved to 'model_averages_across_seeds.csv'.")
    print(df_model_averages)


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    hidden_size = 128
    batch_size = 128
    num_steps = 35
    T = 1000
    train_set_size = 80000
    val_set_size = 40000
    test_set_size = 40000
    seeds = [23,443,54]  
    learning_rates = {'Transformer': 1e-3,'CGLSTM': 1e-3,'RAUCell': 1e-3,'LSTM': 1e-3, 'GRU': 1e-3,}

    all_model_results = {}
    for model_type in learning_rates:
        model_results = {}
        for seed in seeds:
            print(f"Training {model_type} for T={T} and seed={seed}...")
            result = train_model(model_type, T, num_steps, train_set_size, val_set_size, test_set_size, batch_size, learning_rates[model_type], seed, hidden_size, device)
            model_results[seed] = result
        all_model_results[model_type] = model_results

    prepare_and_display_final_results(all_model_results, learning_rates)

if __name__ == "__main__":
    main()
