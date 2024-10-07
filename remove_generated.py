import os

def delete_files_recursively(start_path, target_suffix='generated_samples.json'):
    # Walk through all directories and files in the starting directory
    for root, dirs, files in os.walk(start_path):
        for file in files:
            # Check if the file ends with the target suffix
            if file.endswith(target_suffix):
                file_path = os.path.join(root, file)
                print(f"Deleting file: {file_path}")  # Print which file is about to be deleted
                os.remove(file_path)  # Uncomment this line to actually delete files

# Specify the starting directory
start_directory = '.'  # Change this to the path you want to start from

# Dry run: Initially run the script with the `os.remove` line commented out to ensure correct files are targeted.
delete_files_recursively(start_directory)

# After confirming the correct files are printed, uncomment the `os.remove(file_path)` line to actually delete the files.
