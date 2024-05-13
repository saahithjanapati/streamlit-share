import streamlit as st
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

print("Streamlit app is running...")

PATH_TO_RESULTS_DIR = Path("results/")  # set this to the results directory

# Options for selection
models = [
    "EleutherAI/pythia-1b", 
    "EleutherAI/pythia-1.4b", 
    "EleutherAI/pythia-2.8b", 
    "EleutherAI/pythia-6.9b", 
    "EleutherAI/pythia-12b",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",    
]


model_name_to_size = {
    "EleutherAI/pythia-1b": 1,
    "EleutherAI/pythia-1.4b": 1.4,
    "EleutherAI/pythia-2.8b": 2.8,
    "EleutherAI/pythia-6.9b": 6.9,
    "EleutherAI/pythia-12b": 12,
    "meta-llama/Llama-2-7b-hf": 7,
    "meta-llama/Llama-2-13b-hf": 13,
    "meta-llama/Llama-2-70b-hf": 70,
}


model_sizes = [1, 1.4, 2.8, 6.9, 12, 2.7, 13, 70]

datasets = ["qnli", "rte", "sentiment", "wnli"]
k_shots = [0, 1, 2, 5, 10]
prompt_modes = ["default", "randomized", "flipped", "same", "same-random"]

# Title of the app
st.title('Analyzing ICL with Intrinsic Dimension')

# Selection fields
model_name = st.multiselect('Select Model(s)', models)
template = "simple"  # Fixed value
dataset = st.selectbox('Select Dataset', datasets)
k_shot = st.multiselect('Select k-shot', k_shots)
prompt_mode = st.multiselect('Select Prompt Mode', prompt_modes)


def fetch_accuracy_data(model_name, dataset, k_shot, prompt_mode, template):
    # Returns a value for accuracy
    path_to_results = PATH_TO_RESULTS_DIR / "accuracy_results" / Path(f"{model_name}") / f"{dataset}-{prompt_mode}-{template}-{k_shot}_results.json"
    with path_to_results.open("r") as file:
        data = json.load(file)
    return data["accuracy"]


def fetch_lid_data(model_name, dataset, k_shot, prompt_mode, template):
    # Return a list of lid values for the specified parameter combination
    path_to_results = PATH_TO_RESULTS_DIR / "lid_results" / Path(f"{model_name}-{dataset}-{k_shot}-{prompt_mode}-{template}_lid.json")
    
    if path_to_results.exists() == False:
        return None

    with path_to_results.open("r") as file:
        data = json.load(file)
        data = {int(k): v for k, v in data.items()}
    return data



def fetch_data(model_name, dataset, k_shot, prompt_mode, template):
    
    if type(model_name) != list:
        model_name = [model_name]

    if type(dataset) != list:
        dataset = [dataset]
    
    if type(k_shot) != list:
        k_shot = [k_shot]
    
    if type(prompt_mode) != list:
        prompt_mode = [prompt_mode]

    accuracy_dict = {}
    lid_dict = {}

    # loop through all the combinations of the parameters and fetch the data
    for model in model_name:
        for ds in dataset:
            for k in k_shot:
                for mode in prompt_mode:
                    model_size = model_name_to_size[model]

                    accuracy_dict[(model_size, k, model, ds, mode, template)] = (fetch_accuracy_data(model, ds, k, mode, template), str((model, ds, k, mode, template)))
                    lid_dict[(model_size, k, model, ds, mode, template)] = fetch_lid_data(model, ds, k, mode, template), str((model, ds, k, mode, template))
    
    return accuracy_dict, lid_dict


def generate_plots(acc_data, lid_data):
    # Convert keys to a list for indexing
    keys_list = list(acc_data.keys())
    
    fig, axs = plt.subplots(2, 1, figsize=(15, 20))  # Creating two subplots, one above the other

    # Variables to hold bar width and the set of all unique models for legend
    bar_width = 0.35
    unique_models = set()

    # Plot each set of model results with a distinct bar
    for i, key in enumerate(sorted(keys_list)):
        # Extract model name from the key for the legend
        # model_name = key.split(',')[0].strip('(').strip('\'')
        model_name = acc_data[key][1]
        unique_models.add(model_name)
        
        # Plotting each model's accuracy as a separate bar
        axs[0].bar(i, acc_data[key][0], width=bar_width, label=model_name)
    
    # axs[0].set_xticklabels([i for key in keys_list], rotation='vertical', fontsize=8)  # Vertical and smaller font size

    # axs[0].set_xticks(indices)  # Set numerical indices as x-ticks
    # axs[0].set_xticklabels([f'{i+1}' for i in range(len(keys_list))], rotation=45)  # Optional: show indices starting from 1
    axs[0].legend(loc='right', title="Models")  # Positioning legend on the right side with title
    axs[0].set_title('Accuracy Plot')
    axs[0].set_xlabel('Experiment Index', labelpad=20)  # Example x-label
    axs[0].set_ylabel('Accuracy', labelpad=20)  # Example y-label


    # Assuming LID data plot remains unchanged
    axs[1].set_title('LID Plot')
    axs[1].set_xlabel('Layer Index', labelpad=20)  # Example x-label
    axs[1].set_ylabel('LID', labelpad=20)  # Example y-label



    # Plotting LID data (assuming lid_data is correctly formatted and available)
    if lid_data != None:
        for key in sorted(lid_data.keys()):
            

        # for label, values_tup in lid_data.items():
            
            values, label = lid_data[key]
            print(label, values)
            

            if values == None:
                st.text(f"No LID data available for {label}")
                continue


            x_values = list(values.keys())
            y_values = list(values.values())

            axs[1].plot(x_values, y_values, label=label)

        axs[1].legend()

    axs[1].set_title('LID Plot')
    axs[1].set_xlabel('Layer Index', labelpad=20)  # Example x-label
    axs[1].set_ylabel('LID', labelpad=20)  # Example y-label


    


    plt.tight_layout()  # Adjust layout to make it neat
    st.pyplot(fig)



if st.button('Fetch Data and Plot'):
    acc_data, lid_data = fetch_data(model_name, dataset, k_shot, prompt_mode, template)
    generate_plots(acc_data, lid_data)


# def generate_plots(acc_data, lid_data):
#     # Convert keys to a list for indexing
#     keys_list = list(acc_data.keys())
#     # Creating numerical indices for x-axis
#     indices = np.arange(len(keys_list))
    
#     fig, axs = plt.subplots(2, 1, figsize=(5, 8))  # Creating two subplots, one above the other
    
#     # Plotting accuracy data with numerical x-axis and setting labels to the right
#     axs[0].bar(indices, acc_data.values(), label='Accuracy')
#     axs[0].set_xticks(indices)  # Set numerical indices as x-ticks
#     axs[0].set_xticklabels([f'{i+1}' for i in range(len(keys_list))], rotation=45)  # Optional: show indices starting from 1
#     axs[0].legend(loc='upper right')  # Positioning legend on the right side
#     axs[0].set_title('Accuracy Plot')
#     axs[0].set_xlabel('Experiment Index', labelpad=20)  # Example x-label
#     axs[0].set_ylabel('Accuracy', labelpad=20)  # Example y-label

#     # Plotting LID data (assuming lid_data is correctly formatted and available)
#     # if lid_data:
#     #     for label, values in lid_data.items():
#     #         axs[1].plot(values, label=label)
#     #     axs[1].legend()
#     axs[1].set_title('LID Plot')
#     axs[1].set_xlabel('Layer Index', labelpad=20)  # Example x-label
#     axs[1].set_ylabel('LID', labelpad=20)  # Example y-label

#     plt.tight_layout()  # Adjust layout to make it neat
#     st.pyplot(fig)

# Displaying the selected options
# st.write('Selected Model(s):', model_name)
# st.write('Template:', template)
# st.write('Selected Dataset:', dataset)
# st.write('Selected k-shot:', k_shot)
# st.write('Selected Prompt Mode:', prompt_mode)

# Fetching data and plotting

