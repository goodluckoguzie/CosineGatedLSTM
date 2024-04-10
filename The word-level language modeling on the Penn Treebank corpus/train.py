


import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchtext.datasets as datasets
from torchtext.data.utils import get_tokenizer
from collections import Counter
import sys
import time
#from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append('../')
from model import LanguageModel  # Make sure this import matches your model's actual location

def build_vocab(max_size=10000):
    tokenizer = get_tokenizer("basic_english")
    train_iter = datasets.PennTreebank(split='train')
    token_counter = Counter()
    for text in train_iter:
        tokens = tokenizer(text)
        token_counter.update(tokens)
    most_common_tokens = token_counter.most_common(max_size)
    vocab = {token: i for i, (token, _) in enumerate(most_common_tokens)}
    vocab["<unk>"] = len(vocab)  # Unknown token
    vocab["<pad>"] = len(vocab)  # Padding token
    return vocab

class PTBDataset(Dataset):
    def __init__(self, split, vocab=None):
        self.data_iter = datasets.PennTreebank(split=split)
        self.vocab = vocab
        self.tokenizer = get_tokenizer("basic_english")
        self.tokens = self._tokenize_data()

    def _tokenize_data(self):
        tokens = []
        for text in self.data_iter:
            tokens.extend([self.vocab.get(token, self.vocab["<unk>"]) for token in self.tokenizer(text)])
        return tokens

    def __len__(self):
        return len(self.tokens) - 1

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx], dtype=torch.long), torch.tensor(self.tokens[idx + 1], dtype=torch.long)

def load_data(BATCH_SIZE=20):
    vocab = build_vocab()
    train_dataset = PTBDataset(split='train', vocab=vocab)
    valid_dataset = PTBDataset(split='valid', vocab=vocab)
    test_dataset = PTBDataset(split='test', vocab=vocab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return vocab, train_loader, valid_loader, test_loader


# def train_and_evaluate(model_type, lr, seed, vocab, train_loader, valid_loader, test_loader):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     EMBEDDING_DIM = 200
#     HIDDEN_DIM = 200
#     NUM_LAYERS = 2
#     DROPOUT = 0
#     N_EPOCHS = 15

#     model = LanguageModel(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, model_type).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()

#     metrics = {'train_loss': [], 'valid_loss': [], 'train_ppl': [], 'valid_ppl': [], 'test_loss': [], 'test_ppl': [], 'train_time': [], 'test_time': []}

#     for epoch in range(N_EPOCHS):
#         start_train_time = time.time()
#         model.train()
#         total_train_loss = 0
#         for text, targets in train_loader:
#             text, targets = text.to(device), targets.to(device)
#             optimizer.zero_grad()
#             output = model(text)
#             loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()
#         metrics['train_time'].append(time.time() - start_train_time)
def train_and_evaluate(model_type, lr, seed, vocab, train_loader, valid_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model and optimizer setup remains unchanged
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 200
    NUM_LAYERS = 2
    DROPOUT = 0
    N_EPOCHS = 13

    model = LanguageModel(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)

    # Training loop with scheduler step
    metrics = {'train_loss': [], 'valid_loss': [], 'train_ppl': [], 'valid_ppl': [], 'test_loss': [], 'test_ppl': [], 'train_time': [], 'test_time': []}

    for epoch in range(N_EPOCHS):
        start_train_time = time.time()
        model.train()
        total_train_loss = 0
        for text, targets in train_loader:
            text, targets = text.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        metrics['train_time'].append(time.time() - start_train_time)
        
        # Step the scheduler
        # scheduler.step()
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for text, targets in valid_loader:
                text, targets = text.to(device), targets.to(device)
                output = model(text)
                loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
                total_val_loss += loss.item()
                
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(valid_loader)

        metrics['train_loss'].append(avg_train_loss)
        metrics['valid_loss'].append(avg_val_loss)
        metrics['train_ppl'].append(np.exp(avg_train_loss))
        metrics['valid_ppl'].append(np.exp(avg_val_loss))
        scheduler.step(avg_val_loss)  

    # After all epochs, calculate test metrics
    start_test_time = time.time()
    total_test_loss = 0
    with torch.no_grad():
        for text, targets in test_loader:
            text, targets = text.to(device), targets.to(device)
            output = model(text)
            loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
            total_test_loss += loss.item()
    metrics['test_time'].append(time.time() - start_test_time)

    avg_test_loss = total_test_loss / len(test_loader)
    metrics['test_loss'].append(avg_test_loss)
    metrics['test_ppl'].append(np.exp(avg_test_loss))

    # Save the model
    model_save_directory = 'models'
    if not os.path.exists(model_save_directory):
        os.makedirs(model_save_directory)
    model_save_path = os.path.join(model_save_directory, f'{model_type}_seed_{seed}.pt')
    torch.save(model.state_dict(), model_save_path)


    # Ensure the models directory exists
    model_save_directory = 'results'
    if not os.path.exists(model_save_directory):
        os.makedirs(model_save_directory)



    # Saving epoch-wise metrics
    epoch_metrics = {
        'train_loss': metrics['train_loss'],
        'valid_loss': metrics['valid_loss'],
        'train_ppl': metrics['train_ppl'],
        'valid_ppl': metrics['valid_ppl']
    }
    epoch_results_path = f'results/{model_type}_seed_{seed}_epoch_metrics.csv'
    pd.DataFrame(epoch_metrics).to_csv(epoch_results_path, index=False)

    # Prepare and save final metrics
    final_metrics = {
        'test_loss': [avg_test_loss],
        'test_ppl': [np.exp(avg_test_loss)],
        'mean_train_time': [np.mean(metrics['train_time'])],
        'mean_test_time': [np.mean(metrics['test_time'])],
        'num_parameters': [sum(p.numel() for p in model.parameters() if p.requires_grad)]
    }
    final_results_path = f'results/{model_type}_seed_{seed}_final_metrics.csv'
    pd.DataFrame(final_metrics).to_csv(final_results_path, index=False)

    return {
        'train_perplexity': np.mean(metrics['train_ppl']),
        'valid_perplexity': np.mean(metrics['valid_ppl']),
        'test_perplexity': np.exp(avg_test_loss),
        'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'mean_train_time': np.mean(metrics['train_time']),
        'mean_test_time': np.mean(metrics['test_time']),
        'model_save_path': model_save_path
    }

# # def prepare_and_display_final_results(all_model_results):
# #     final_summary = {
# #         'Model Type': [], 'Mean Train Perplexity': [], 'Std Train Perplexity': [],
# #         'Mean Valid Perplexity': [], 'Std Valid Perplexity': [], 'Mean Test Perplexity': [],
# #         'Std Test Perplexity': [], 'Mean Train Time': [], 'Mean Test Time': [], 'Mean Num Parameters': [],
# #         'P-Value Train PPL vs. CGGRU': [], 'P-Value Valid PPL vs. CGGRU': [], 'P-Value Test PPL vs. CGGRU': []
# #     }
    
# #     benchmark_model = 'CGGRU'
# #     benchmark_results = {'train_ppl': [], 'valid_ppl': [], 'test_ppl': []}
    
# #     if benchmark_model in all_model_results:
# #         for seed, results in all_model_results[benchmark_model].items():
# #             benchmark_results['train_ppl'].append(results['train_perplexity'])
# #             benchmark_results['valid_ppl'].append(results['valid_perplexity'])
# #             benchmark_results['test_ppl'].append(results['test_perplexity'])

# #     for model_type, seeds_results in all_model_results.items():
# #         train_ppls, valid_ppls, test_ppls, train_times, test_times, num_params = [], [], [], [], [], []

# #         for seed, results in seeds_results.items():
# #             train_ppls.append(results['train_perplexity'])
# #             valid_ppls.append(results['valid_perplexity'])
# #             test_ppls.append(results['test_perplexity'])
# #             train_times.append(results['mean_train_time'])
# #             test_times.append(results['mean_test_time'])
# #             num_params.append(results['num_parameters'])

# #         # Calculating mean and std
# #         final_summary['Model Type'].append(model_type)
# #         final_summary['Mean Train Perplexity'].append(np.mean(train_ppls))
# #         final_summary['Std Train Perplexity'].append(np.std(train_ppls))
# #         final_summary['Mean Valid Perplexity'].append(np.mean(valid_ppls))
# #         final_summary['Std Valid Perplexity'].append(np.std(valid_ppls))
# #         final_summary['Mean Test Perplexity'].append(np.mean(test_ppls))
# #         final_summary['Std Test Perplexity'].append(np.std(test_ppls))
# #         final_summary['Mean Train Time'].append(np.mean(train_times))
# #         final_summary['Mean Test Time'].append(np.mean(test_times))
# #         final_summary['Mean Num Parameters'].append(np.mean(num_params))
        
# #         # Performing t-tests if benchmark model results are available
# #         if benchmark_model in all_model_results and model_type != benchmark_model:
# #             for metric in ['train_ppl', 'valid_ppl', 'test_ppl']:
# #                 model_metric_values = [train_ppls, valid_ppls, test_ppls][['train_ppl', 'valid_ppl', 'test_ppl'].index(metric)]
# #                 t_stat, p_value = ttest_ind(benchmark_results[metric], model_metric_values, equal_var=False)
# #                 # Correctly format the metric name to match the keys in final_summary
# #                 metric_name = metric.replace('_', ' ').capitalize().replace("ppl", "PPL")
# #                 key_name = f'P-Value {metric_name} vs. {benchmark_model}'
# #                 final_summary[key_name].append(p_value)
# #         else:
# #             for metric in ['train_ppl', 'valid_ppl', 'test_ppl']:
# #                 # Correctly format the metric name to match the keys in final_summary
# #                 metric_name = metric.replace('_', ' ').capitalize().replace("ppl", "PPL")
# #                 key_name = f'P-Value {metric_name} vs. {benchmark_model}'
# #                 final_summary[key_name].append(np.nan)

# #     df_summary = pd.DataFrame(final_summary)
# #     summary_path = 'results/final_summary.csv'
# #     df_summary.to_csv(summary_path, index=False)
# #     print(df_summary)

# def prepare_and_display_final_results(all_model_results):
#     final_summary = {
#         'Model Type': [], 'Mean Test Perplexity': [], 'Std Test Perplexity': [], 
#         'Mean Train Time': [], 'Mean Test Time': [], 'Mean Num Parameters': [],
#         'Test PPL T-test Statistic (vs. CGLSTM)': [], 'Test PPL T-test p-value (vs. CGLSTM)': []
#     }
    
#     benchmark_model = 'CGLSTM'
#     benchmark_test_ppl = np.array([r['test_perplexity'] for r in all_model_results[benchmark_model].values()])
#     benchmark_val_ppl = np.array([r['valid_perplexity'] for r in all_model_results[benchmark_model].values()])

#     for model_type, seeds_results in all_model_results.items():
#         val_ppls,test_ppls, train_times, test_times, num_params = [], [], [], [],[]

#         for seed, results in seeds_results.items():
#             test_ppls.append(results['test_perplexity']) 
#             val_ppls.append(results['valid_perplexity']) 
#             train_times.append(results['mean_train_time'])
#             test_times.append(results['mean_test_time'])
#             num_params.append(results['num_parameters'])

#         # Calculate mean and std for test perplexity and other metrics
#         final_summary['Model Type'].append(model_type)
#         final_summary['Mean Test Perplexity'].append(np.mean(test_ppls))
#         final_summary['Std Test Perplexity'].append(np.std(test_ppls))
#         final_summary['Mean Val Perplexity'].append(np.mean(val_ppls))
#         final_summary['Std Val Perplexity'].append(np.std(val_ppls))
#         final_summary['Mean Train Time'].append(np.mean(train_times))
#         final_summary['Mean Test Time'].append(np.mean(test_times))
#         final_summary['Mean Num Parameters'].append(np.mean(num_params))
        
#         # Performing t-tests for test perplexity if benchmark model results are available and current model is not the benchmark
#         if model_type != benchmark_model:
#             t_stat, p_value = ttest_ind(test_ppls, benchmark_test_ppl, equal_var=False)
#             final_summary['Test PPL T-test Statistic (vs. CGLSTM)'].append(t_stat)
#             final_summary['Test PPL T-test p-value (vs. CGLSTM)'].append(p_value)
#             t_stat, p_value = ttest_ind(val_ppls, benchmark_val_ppl, equal_var=False)
#             final_summary['Val PPL T-test Statistic (vs. CGLSTM)'].append(t_stat)
#             final_summary['Val PPL T-test p-value (vs. CGLSTM)'].append(p_value)

#         else:
#             final_summary['Test PPL T-test Statistic (vs. CGLSTM)'].append(None)
#             final_summary['Test PPL T-test p-value (vs. CGLSTM)'].append(None)

#     df_summary = pd.DataFrame(final_summary)
#     summary_path = 'results/final_summary.csv'
#     df_summary.to_csv(summary_path, index=False)
#     print(df_summary)


def prepare_and_display_final_results(all_model_results):
    final_summary = {
        'Model Type': [],
        'Mean Test Perplexity': [],
        'Std Test Perplexity': [],
        'Mean Train Time': [],
        'Mean Test Time': [],
        'Mean Num Parameters': [],
        'Test PPL T-test Statistic (vs. CGLSTM)': [],
        'Test PPL T-test p-value (vs. CGLSTM)': [],
        'Mean Val Perplexity': [],  # Added to fix the KeyError
        'Std Val Perplexity': [],   # Added to fix the KeyError
        'Val PPL T-test Statistic (vs. CGLSTM)': [],  # For validation perplexity comparison
        'Val PPL T-test p-value (vs. CGLSTM)': []    # For validation perplexity comparison
    }
    
    benchmark_model = 'CGLSTM'
    benchmark_test_ppl = [results['test_perplexity'] for results in all_model_results.get(benchmark_model, {}).values()]
    benchmark_val_ppl = [results['valid_perplexity'] for results in all_model_results.get(benchmark_model, {}).values()]

    for model_type, seeds_results in all_model_results.items():
        test_ppls, val_ppls, train_times, test_times, num_params = [], [], [], [], []

        for results in seeds_results.values():
            test_ppls.append(results['test_perplexity'])
            val_ppls.append(results['valid_perplexity'])
            train_times.append(results['mean_train_time'])
            test_times.append(results['mean_test_time'])
            num_params.append(results['num_parameters'])

        final_summary['Model Type'].append(model_type)
        final_summary['Mean Test Perplexity'].append(np.mean(test_ppls))
        final_summary['Std Test Perplexity'].append(np.std(test_ppls))
        final_summary['Mean Val Perplexity'].append(np.mean(val_ppls))  # Correct calculation
        final_summary['Std Val Perplexity'].append(np.std(val_ppls))    # Correct calculation
        final_summary['Mean Train Time'].append(np.mean(train_times))
        final_summary['Mean Test Time'].append(np.mean(test_times))
        final_summary['Mean Num Parameters'].append(np.mean(num_params))

        # Statistical comparisons
        if model_type != benchmark_model:
            t_stat, p_value = ttest_ind(test_ppls, benchmark_test_ppl, equal_var=False)
            final_summary['Test PPL T-test Statistic (vs. CGLSTM)'].append(t_stat)
            final_summary['Test PPL T-test p-value (vs. CGLSTM)'].append(p_value)

            t_stat_val, p_value_val = ttest_ind(val_ppls, benchmark_val_ppl, equal_var=False)
            final_summary['Val PPL T-test Statistic (vs. CGLSTM)'].append(t_stat_val)
            final_summary['Val PPL T-test p-value (vs. CGLSTM)'].append(p_value_val)
        else:
            final_summary['Test PPL T-test Statistic (vs. CGLSTM)'].append(None)
            final_summary['Test PPL T-test p-value (vs. CGLSTM)'].append(None)
            final_summary['Val PPL T-test Statistic (vs. CGLSTM)'].append(None)
            final_summary['Val PPL T-test p-value (vs. CGLSTM)'].append(None)

    df_summary = pd.DataFrame(final_summary)
    summary_path = 'results/final_summary.csv'
    df_summary.to_csv(summary_path, index=False)
    print(df_summary)


def main():
    BATCH_SIZE = 20
    # learning_rates = {'GRU': 1e-3,}
    learning_rates = {'Transformer': 1e-3,'CGLSTM':1e-3,'RAUCell': 1e-3,'LSTM': 1e-3, 'GRU': 1e-3}
    seeds = [68,944]

    vocab, train_loader, valid_loader, test_loader = load_data(BATCH_SIZE)
    all_model_results = {}

    for model_type, lr in learning_rates.items():
        seeds_results = {}
        for seed in seeds:
            print(f"Training model: {model_type} with LR: {lr}, Seed: {seed}")
            results = train_and_evaluate(model_type, lr, seed, vocab, train_loader, valid_loader, test_loader)
            seeds_results[seed] = results
        all_model_results[model_type] = seeds_results

    prepare_and_display_final_results(all_model_results)

if __name__ == "__main__":
    main()
