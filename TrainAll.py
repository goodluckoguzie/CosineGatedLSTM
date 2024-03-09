# import subprocess
# import os

# directories = [
#     'sentiment analysis experiment on the IMDB movie reviews dataset',
#     'The word-level language modeling on the Penn Treebank corpus',

#     'addingproblem',
#     'fashion-MNIST_item_classification',
#     'row-wise MNIST handwritten digits recognition'
# ]

# original_dir = os.getcwd()

# for dir in directories:
#     target_dir = os.path.join(original_dir, dir)
#     os.chdir(target_dir)
    
#     print(f"Executing script in {dir} within {os.getcwd()}...")
    
#     try:
#         # result = subprocess.run(["python", "train.py"], capture_output=False, text=True, timeout=3600)
#         result = subprocess.run(["python", "train.py"], capture_output=False, text=True, timeout=3600)
#         print("Output:", result.stdout)
#         if result.stderr:
#             print("Error:", result.stderr)
#     except subprocess.TimeoutExpired:
#         print(f"Script in {dir} exceeded timeout.")
#     except Exception as e:
#         print(f"Failed to execute script in {dir}: {e}")

#     os.chdir(original_dir)

# print("Training completed for all directories.")


# Import necessary libraries
import subprocess
import os

# List of directories to navigate through. Each directory contains a different project or experiment.
directories = [

    'fashion-MNIST_item_classification',  # Directory for fashion-MNIST classification project
    'row-wise MNIST handwritten digits recognition',  # Directory for MNIST digits recognition project,
       'sentiment analysis experiment on the IMDB movie reviews dataset',  # Directory for sentiment analysis project
    'The word-level language modeling on the Penn Treebank corpus',     # Directory for language modeling project

    'addingproblem',  # Directory for an experiment on the adding problem
]

# Save the current working directory to navigate back to it later
original_dir = os.getcwd()

# Loop through each directory specified in the 'directories' list
for dir in directories:
    # Construct the path to the target directory and change the current working directory to it
    target_dir = os.path.join(original_dir, dir)
    os.chdir(target_dir)
    
    # Inform the user about the script execution start in the current directory
    print(f"Executing script in {dir} within {os.getcwd()}...")
    
    try:
        # Execute the training script 'train.py' in the current directory, with a timeout of 1 hour (3600 seconds)
        # 'capture_output=False' allows the output to be displayed directly instead of being captured
        result = subprocess.run(["python", "train.py"], capture_output=False, text=True)
        
        # If there's stdout output, print it (though capture_output is False, so this may not be needed)
        if result.stdout:
            print("Output:", result.stdout)
        
        # If there's stderr output, print it (though capture_output is False, so this may not be needed)
        if result.stderr:
            print("Error:", result.stderr)
    except subprocess.TimeoutExpired:
        # Handle the case where the script execution takes longer than the timeout
        print(f"Script in {dir} exceeded timeout.")
    except Exception as e:
        # Handle any other exceptions that may occur during script execution
        print(f"Failed to execute script in {dir}: {e}")

    # Navigate back to the original directory after executing the script in each target directory
    os.chdir(original_dir)

# Inform the user that the training has been completed for all specified directories
print("Training completed for all directories.")
