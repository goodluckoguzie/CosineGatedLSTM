from collections import defaultdict
import numpy as np

# Your all_results structure
all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)), {
    'LSTMCell': defaultdict(lambda: defaultdict(list), {
        'train_ppl': [2.813446670093808, 2.8185831927457907],
        'valid_ppl': [2.7360506986208835, 2.737553923046916],
        'test_ppl': [2.79621247581381, 2.7993419517128904],
        'training_time': [44.43605017662048, 44.96340894699097],
        'testing_time': [2.6385903358459473, 2.628495931625366]
    }),
    'GRUCell': defaultdict(lambda: defaultdict(list), {
        'train_ppl': [2.830261998834628, 2.827708476677966],
        'valid_ppl': [2.7348626419031947, 2.7333390826225954],
        'test_ppl': [2.795751560981358, 2.7953323121821207],
        'training_time': [42.04689383506775, 43.72522187232971],
        'testing_time': [2.42233943939209, 2.4367613792419434]
    })
})

# Initialize an empty dictionary to hold the mean values
mean_values = defaultdict(dict)
print("all_results",all_results)
# Iterate over each model type and metric to calculate the mean values
for model_type, metrics in all_results.items():
    for metric, values in metrics.items():
        # Calculate the mean of the metric values
        mean_value = np.mean(values)
        # Store the mean value in the mean_values dictionary
        mean_values[model_type][metric] = mean_value

# Print the mean values for each model type and metric
for model_type, metrics in mean_values.items():
    print(f"Model Type: {model_type}")
    for metric, mean_value in metrics.items():
        print(f"  {metric}: {mean_value:.4f}")
    print()  # Newline for better readability
