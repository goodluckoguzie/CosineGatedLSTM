
def main():
    """
    Main function to conduct sentiment analysis on the IMDB dataset using different
    recurrent neural network models. It trains and evaluates the models, compares their
    performance through statistical tests, and visualizes training and validation results.
    """
    # Import necessary libraries
    import os
    import time
    import pandas as pd
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from collections import Counter
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    import re
    import nltk
    from scipy.stats import ttest_rel
    from sklearn.metrics import precision_score, recall_score, f1_score
    import sys
    sys.path.append('../')
    from scipy.stats import ttest_ind

    # Import your model
    from model import SA_RecurrentModel

    # Check if the "results" folder exists, and create it if not
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Function to count trainable parameters in a model
    def count_trainable_parameters(model):
        """
        Counts the number of trainable parameters in the given PyTorch model.
        
        Parameters:
        - model: The PyTorch model.
        
        Returns:
        - The total count of trainable parameters in the model.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train_and_evaluate(model_type, learning_rate, train_loader, val_loader, test_loader, seeds):
        """
        Trains and evaluates a sentiment analysis model specified by model_type with
        the given learning rate, training, validation, and test data loaders, across
        different seeds for reproducibility. Measures and prints performance metrics.
        
        Parameters:
        - model_type: String specifying the type of RNN model to use.
        - learning_rate: Learning rate for the optimizer.
        - train_loader: DataLoader for the training dataset.
        - val_loader: DataLoader for the validation dataset.
        - test_loader: DataLoader for the test dataset.
        - seeds: List of seeds for ensuring reproducibility.
        
        Returns:
        - model: The trained PyTorch model.
        - all_results: A dictionary containing training, validation, and testing results.
        """
        all_results = {}
        for seed in seeds:
            print(f"Running with seed {seed}")
            # Set seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Start time for training
            start_time = time.time()

            model = SA_RecurrentModel(vocab_size + 1, embedding_dim, hidden_dim, output_size, model_type, dropout)
            model.to(device)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            epochs = 100  # Adjust number of epochs as needed
            for epoch in range(epochs):
                model.train()
                train_loss, train_correct, total_train = 0.0, 0, 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    output = model(inputs)

                    loss = criterion(output.squeeze(), labels.float())

                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    train_loss += loss.item()
                    predicted = (output >= 0.5).float()
                    train_correct += predicted.eq(labels.view_as(predicted)).sum().item()
                    total_train += labels.size(0)
                train_accuracy = (train_correct / total_train) * 100

                # Validation loop
                model.eval()
                val_loss, val_correct, total_val = 0.0, 0, 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model(inputs)
                        loss = criterion(output.squeeze(), labels.float())
                        val_loss += loss.item()
                        predicted = (output >= 0.5).float()
                        val_correct += predicted.eq(labels.view_as(predicted)).sum().item()
                        total_val += labels.size(0)
                val_accuracy = (val_correct / total_val) * 100

                print(f'Epoch [{epoch + 1}/{epochs}], {model_type} - Train Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

            # Test loop
            model.eval()
            test_loss = 0.0
            test_correct = 0
            total_test = 0
            test_predictions = []
            test_labels = []
            test_start_time = time.time()  # Start time for testing
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model(inputs)
                    loss = criterion(output.squeeze(), labels.float())
                    test_loss += loss.item()
                    predicted = (output >= 0.5).float()
                    test_correct += predicted.eq(labels.view_as(predicted)).sum().item()
                    total_test += labels.size(0)
                    test_predictions.extend(predicted.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
            test_end_time = time.time()  # End time for testing
            test_duration = test_end_time - test_start_time
            test_accuracy = (test_correct / total_test) * 100
            test_precision = precision_score(test_labels, test_predictions)
            test_recall = recall_score(test_labels, test_predictions)
            test_f1_score = f1_score(test_labels, test_predictions)
            print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%, Test Precision: {test_precision:.2f}, Test Recall: {test_recall:.2f}, Test F1 Score: {test_f1_score:.2f}')

            # End time for training
            end_time = time.time()
            training_duration = end_time - start_time
            print(f"Training Time for {model_type}: {training_duration:.2f} seconds")

            # Save the individual results
            all_results[f'Seed_{seed}'] = {
                "train_losses": train_loss / len(train_loader),
                "val_losses": val_loss / len(val_loader),
                "train_accuracies": train_accuracy,
                "val_accuracies": val_accuracy,
                "test_losses": test_loss / len(test_loader),
                "test_accuracies": test_accuracy,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1_score": test_f1_score,
                "training_time": training_duration,
                "testing_time": test_duration  
            }

        # Return the model and all results
        return model, all_results

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess dataset
    data = pd.read_csv('IMDB Dataset.csv')  

    # Preprocessing functions
    def clean_text(text):
        """
        Cleans the input text by removing HTML tags, URLs, punctuation, and converting to lowercase.
        
        Parameters:
        - text: The input text string to be cleaned.
        
        Returns:
        - Cleaned text string.
        """
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        return text

    data['cleaned_review'] = data['review'].apply(clean_text)

    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def tokenize_and_lemmatize(text):
        words = word_tokenize(text)
        return [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    data['processed_review'] = data['cleaned_review'].apply(tokenize_and_lemmatize)

    data['label'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    # Prepare data for training
    vocab = Counter()
    for review in data['processed_review']:
        vocab.update(review)
    vocab_size = len(vocab)

    word2int = {word: i + 1 for i, (word, _) in enumerate(vocab.most_common())}

    def encode_review(review):
        return [word2int.get(word, 0) for word in review]

    data['encoded_review'] = data['processed_review'].apply(encode_review)

    def calculate_max_length(reviews, percentile=95):
        lengths = [len(review) for review in reviews]
        return int(np.percentile(lengths, percentile))

    # Applying the function to your data
    max_len = calculate_max_length(data['processed_review'])
    print(f"Calculated max length: {max_len}")

    # Then proceed with padding as before
    def pad_features(encoded_reviews, seq_length=max_len):
        features = np.zeros((len(encoded_reviews), seq_length), dtype=int)
        for i, review in enumerate(encoded_reviews):
            len_review = len(review)
            if len_review != 0:
                features[i, -min(len_review, seq_length):] = np.array(review)[:seq_length]
        return features

    encoded_reviews = list(data['processed_review'].apply(encode_review))
    features = pad_features(encoded_reviews)

    # Split data into training, validation, and test sets
    # Using stratify to ensure equal distribution of positive and negative samples
    X_train, X_temp, y_train, y_temp = train_test_split(features, data['label'].values, test_size=0.5, random_state=42,
                                                        stratify=data['label'].values)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42, stratify=y_temp)

    # Create DataLoader for training, validation, and test sets
    batch_size = 128
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    # Model parameters
    output_size = 1
    embedding_dim =  64
    hidden_dim = 128  # 00
    n_layers = 1
    dropout = 0.5

    # Define model types and their respective learning rates
    model_types = ['Transformer','LSTMCell', 'GRUCell', 'RAUCell', 'CGLSTMCellv0', 'CGLSTMCellv1']
    learning_rates = [1e-3,1e-3, 1e-3, 1e-3, 1e-3, 1e-3]  # Example learning rates

    # Set seeds for reproducibility
    seeds = [42, 1337, 2022, 777, 999]

    # Store all model results
    all_model_results = {}

    # Training and evaluating each model
    for model_type, lr in zip(model_types, learning_rates):
        print(f"Training model type: {model_type}")
        model, all_results = train_and_evaluate(model_type, lr, train_loader, val_loader, test_loader, seeds)
        # print(f"Results for {model_type}: {all_results}")

        # Save the model
        model_save_path = os.path.join(results_folder, f'sentiment_model_{model_type}_lr{lr}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # Save results to a file for each seed
        for seed, seed_results in all_results.items():
            seed_results_file_path = os.path.join(results_folder, f'{model_type}_lr{lr}_seed_{seed}_results.txt')
            with open(seed_results_file_path, 'w') as f:
                f.write(f"Model Type: {model_type}\n")
                f.write(f"Learning Rate: {lr}\n")
                for key, value in seed_results.items():
                    f.write(f"{key}: {value}\n")
            print(f"Results for {model_type} with seed {seed} saved to {seed_results_file_path}")

        # Store results for this model
        all_model_results[model_type] = all_results

    # Calculate aggregated metrics for each model
    aggregated_results = {}
    for model_type, results in all_model_results.items():
        aggregated_results[model_type] = {
            'mean_train_accuracy': np.mean([r['train_accuracies'] for r in results.values()]),
            'mean_val_accuracy': np.mean([r['val_accuracies'] for r in results.values()]),
            'mean_test_accuracy': np.mean([r['test_accuracies'] for r in results.values()]),
            'mean_training_time': np.mean([r['training_time'] for r in results.values()]),
            'mean_testing_time': np.mean([r['testing_time'] for r in results.values()]),
        }


    # Perform independent two-sample t-tests between models for statistical significance on validation, training, and test accuracy
    t_test_results = {'val_accuracy': {}, 'train_accuracy': {}, 'test_accuracy': {}}
    baseline_val_accuracies = [r['val_accuracies'] for r in all_model_results['CGLSTMCellv1'].values()]
    baseline_train_accuracies = [r['train_accuracies'] for r in all_model_results['CGLSTMCellv1'].values()]
    baseline_test_accuracies = [r['test_accuracies'] for r in all_model_results['CGLSTMCellv1'].values()]
    for model_type in model_types:
        if model_type == 'CGLSTMCellv1':  # Skip baseline comparison with itself
            continue
        val_accuracies = [r['val_accuracies'] for r in all_model_results[model_type].values()]
        train_accuracies = [r['train_accuracies'] for r in all_model_results[model_type].values()]
        test_accuracies = [r['test_accuracies'] for r in all_model_results[model_type].values()]

        # Perform t-tests for validation accuracy
        _, p_value_val = ttest_ind(val_accuracies, baseline_val_accuracies)
        t_test_results['val_accuracy'][model_type] = p_value_val

        # Perform t-tests for training accuracy
        _, p_value_train = ttest_ind(train_accuracies, baseline_train_accuracies)
        t_test_results['train_accuracy'][model_type] = p_value_train

        # Perform t-tests for test accuracy
        _, p_value_test = ttest_ind(test_accuracies, baseline_test_accuracies)
        t_test_results['test_accuracy'][model_type] = p_value_test

    # Prepare final results table
    final_table_data = []
    for model_type, metrics in aggregated_results.items():
        final_table_data.append({
            'Model': model_type,
            'Mean Train Accuracy (%)': metrics['mean_train_accuracy'],
            'Mean Val Accuracy (%)': metrics['mean_val_accuracy'],
            'Mean Test Accuracy (%)': metrics['mean_test_accuracy'],
            'Mean Training Time (s)': metrics['mean_training_time'],
            'Mean Testing Time (s)': metrics['mean_testing_time'],
            'T-test p-value (vs. CGLSTMCellv1) (Validation Accuracy)': t_test_results['val_accuracy'].get(model_type, 'N/A'),
            'T-test p-value (vs. CGLSTMCellv1) (Training Accuracy)': t_test_results['train_accuracy'].get(model_type, 'N/A'),
            'T-test p-value (vs. CGLSTMCellv1) (Test Accuracy)': t_test_results['test_accuracy'].get(model_type, 'N/A')
        })

    # Convert to DataFrame and save to CSV
    final_results_df = pd.DataFrame(final_table_data)
    final_results_csv_path = os.path.join(results_folder, 'final_model_comparison_results.csv')
    final_results_df.to_csv(final_results_csv_path, index=False)

    print(f"Final results table saved to {final_results_csv_path}")

    # # Plotting
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(12, 6))

    # # # Plot training losses
    # plt.subplot(1, 2, 1)
    # for model_type in model_types:
    #     for seed in seeds:
    #         plt.plot(all_model_results[model_type][f'Seed_{seed}']['train_losses'], label=f'{model_type} Train Loss (Seed {seed})')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.legend()

    # # Plot validation accuracies
    # plt.subplot(1, 2, 2)
    # for model_type in model_types:
    #     for seed in seeds:
    #         plt.plot(all_model_results[model_type][f'Seed_{seed}']['val_accuracies'], label=f'{model_type} Validation Accuracy (Seed {seed})')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Validation Accuracy')
    # plt.legend()

    # plt.tight_layout()
    # plt.savefig('results/training_validation_results.png')
    # plt.show()

if __name__ == "__main__":
    main()
