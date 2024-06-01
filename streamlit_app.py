import streamlit as st
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np
import os
import pickle

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

project_gutenberg_models = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
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

free_text_token_lengths = [10, 30, 60, 140, 200]

prompt_modes = ["default", "randomized", "flipped", "same", "same-random"]

finetune_options = ["base", "finetuned"]

# Title of the app
st.title('Analyzing ICL with Intrinsic Dimension')

# Creating tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Original Experiments", 
                                  "Project Gutenberg Free-Text Generation", 
                                  "Fine-Tuning",
                                  "ICL with No Query", "Intrinsic Dimension of Queries"])



def find_directory(base_dir, start_string):
    """
    Finds the name of the first directory in base_dir that starts with start_string.
    
    Parameters:
    base_dir (str): The directory to search within.
    start_string (str): The starting string to match directory names.
    
    Returns:
    str: The name of the first matching directory or None if no match is found.
    """
    try:
        # List all items in the base directory
        items = os.listdir(base_dir)
        
        # Iterate over the items to find directories that start with the given string
        for item in items:
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith(start_string):
                return item
        
        # If no matching directory is found, return None
        return None
    
    except FileNotFoundError:
        print(f"Error: The directory '{base_dir}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



# Content for each tab
with tab1:
    st.header("Original Experiments")

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
        
        if not path_to_results.exists():
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
            model_name = acc_data[key][1]
            unique_models.add(model_name)
            
            # Plotting each model's accuracy as a separate bar
            axs[0].bar(i, acc_data[key][0], width=bar_width, label=model_name)
        
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

with tab2:
    st.header("Project Gutenberg Free-Text Generation")

    # Selection fields
    model_names = st.multiselect('Select Model(s)', project_gutenberg_models)
    k_shots = st.multiselect('Select token-length(s)', free_text_token_lengths)

    def fetch_lid_data_project_gutenberg(model_name, k_shot):
        # Return a list of lid values for the specified parameter combination
        path_to_results = PATH_TO_RESULTS_DIR / "lid_results" / Path(f"{model_name}-project_gutenberg-{k_shot}-free_lid.json")
        print(path_to_results)
        if not path_to_results.exists():
            return None

        with path_to_results.open("r") as file:
            data = json.load(file)
            data = {int(k): v for k, v in data.items()}
        return data


    def fetch_lid_data_multiple(models, token_lengths):
        lid_dict = {}
        for model in models:
            for token_length in token_lengths:
                lid_dict[(model, token_length)] = fetch_lid_data_project_gutenberg(model, token_length)
        return lid_dict


    def generate_lid_plot_multiple(lid_data):
        fig, ax = plt.subplots(figsize=(15, 10))

        for (model_name, k_shot), data in lid_data.items():
            if data is not None:
                x_values = list(data.keys())
                y_values = list(data.values())

                ax.plot(x_values, y_values, label=f"{model_name} - {k_shot}-tokens")
            else:
                st.text(f"No LID data available for {model_name} - {k_shot}-tokens")

        ax.set_title('LID Plot for Project Gutenberg Free-Text Generation')
        ax.set_xlabel('Layer Index', labelpad=20)
        ax.set_ylabel('LID', labelpad=20)
        ax.legend()

        plt.tight_layout()
        st.pyplot(fig)

    if st.button('Fetch LID Data and Plot'):
        lid_data = fetch_lid_data_multiple(model_names, k_shots)
        generate_lid_plot_multiple(lid_data)

with tab3:
    st.header("Fine-Tuning")

    # Fixed values
    model_name = "meta-llama/Llama-2-7b-hf"
    template = "simple"
    dataset = "sst2_val"
    prompt_mode = "default"
    template = "simple"

    k_shots_finetuning = [0, 1, 2, 5, 10]


    # Selection fields
    k_shot = st.multiselect('Select k-shot', k_shots_finetuning, key="fine_tuning_k_shot")
    finetune_option = st.multiselect('Select Option(s)', finetune_options)

    def fetch_accuracy_data_fine_tuning(model_name, dataset, k_shot, finetune_option, template):
        # Returns a value for accuracy
        
        finetune_option_str = ""
        if finetune_option == "finetuned":
            finetune_option_str = f"-lora_sst-{k_shot}-shot"

        path_to_results = PATH_TO_RESULTS_DIR / "accuracy_results" / Path(f"{model_name}") / f"{dataset}-{prompt_mode}-{template}-{k_shot}{finetune_option_str}_results.json"
        print(path_to_results)
        with path_to_results.open("r") as file:
            data = json.load(file)
        print(data)

        return data["accuracy"]

    def fetch_lid_data_fine_tuning(model_name, dataset, k_shot, finetune_option, template):
        # Return a list of lid values for the specified parameter combination

        finetune_option_str = ""
        if finetune_option == "finetuned":
            finetune_option_str = f"-lora-sst-{k_shot}-shot"

        path_to_results = PATH_TO_RESULTS_DIR / "lid_results" / Path(f"{model_name}-{dataset}-{k_shot}-{prompt_mode}-{template}{finetune_option_str}_lid.json")
        print(path_to_results)
        
        if not path_to_results.exists():
            return None

        with path_to_results.open("r") as file:
            data = json.load(file)
            data = {int(k): v for k, v in data.items()}
        return data

    def fetch_data_fine_tuning(model_name, dataset, k_shot, finetune_option, template):
        if type(k_shot) != list:
            k_shot = [k_shot]
        
        if type(finetune_option) != list:
            finetune_option = [finetune_option]

        accuracy_dict = {}
        lid_dict = {}

        for k in k_shot:
            for option in finetune_option:
                accuracy_dict[(k, model_name, dataset, option, template)] = (fetch_accuracy_data_fine_tuning(model_name, dataset, k, option, template), str((model_name, dataset, k, option, template)))
                lid_dict[(k, model_name, dataset, option, template)] = fetch_lid_data_fine_tuning(model_name, dataset, k, option, template), str((model_name, dataset, k, option, template))
        
        return accuracy_dict, lid_dict

    def generate_plots_fine_tuning(acc_data, lid_data):
        # Convert keys to a list for indexing
        keys_list = list(acc_data.keys())
        
        fig, axs = plt.subplots(2, 1, figsize=(15, 20))  # Creating two subplots, one above the other

        # Variables to hold bar width and the set of all unique models for legend
        bar_width = 0.35
        unique_models = set()

        # Plot each set of model results with a distinct bar
        for i, key in enumerate(sorted(keys_list)):
            model_name = acc_data[key][1]
            unique_models.add(model_name)
            
            # Plotting each model's accuracy as a separate bar
            axs[0].bar(i, acc_data[key][0], width=bar_width, label=model_name)
        
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

    if st.button('Fetch Data and Plot', key="fine_tuning"):
        acc_data, lid_data = fetch_data_fine_tuning(model_name, dataset, k_shot, finetune_option, template)
        generate_plots_fine_tuning(acc_data, lid_data)

with tab4:
    st.header("ICL with No Query")
    k_shots_no_query = [1, 2, 5, 10]
    # Selection fields
    model_name = st.multiselect('Select Model(s)', models[-3:], key="icl_no_query-model")
    dataset = st.selectbox('Select Dataset', datasets, key="icl_no_query-dataset")
    k_shot = st.multiselect('Select k-shot', k_shots_no_query, key="icl_no_query-k_shot")
    prompt_mode = st.multiselect('Select Prompt Mode', prompt_modes, key="icl_no_query-prompt_mode")
    query_present = st.multiselect('Query Present', ["True", "False"], key="icl_no_query-query_present")

    def fetch_lid_data_no_query(model_name, dataset, k_shot, prompt_mode, template, query_present):
        # Adjust the path as needed

        if query_present == "True":
            path_to_results = PATH_TO_RESULTS_DIR / "lid_results" / Path(f"{model_name}-{dataset}-{k_shot}-{prompt_mode}-{template}_lid.json")
        else:
            path_to_results = PATH_TO_RESULTS_DIR / "lid_results" / Path(f"{model_name}-{dataset}-{k_shot}-{prompt_mode}-{template}_False_lid.json")
        print(path_to_results)
        if not path_to_results.exists():
            return None

        with path_to_results.open("r") as file:
            data = json.load(file)
            data = {int(k): v for k, v in data.items()}
        return data

    def fetch_data_no_query(model_name, dataset, k_shot, prompt_mode, template, query_present):
        if type(model_name) != list:
            model_name = [model_name]

        if type(dataset) != list:
            dataset = [dataset]
        
        if type(k_shot) != list:
            k_shot = [k_shot]
        
        if type(prompt_mode) != list:
            prompt_mode = [prompt_mode]

        if type(query_present) != list:
            query_present = [query_present]

        lid_dict = {}

        # loop through all the combinations of the parameters and fetch the data
        for model in model_name:
            for ds in dataset:
                for k in k_shot:
                    for mode in prompt_mode:
                        for query_present_val in query_present:
                            model_size = model_name_to_size[model]

                            lid_dict[(model_size, k, model, ds, mode, template, query_present_val)] = fetch_lid_data_no_query(model, ds, k, mode, template, query_present_val), str((model, ds, k, mode, template, "Query Present: " + query_present_val))
        
        return lid_dict

    def generate_lid_plot_no_query(lid_data):
        fig, ax = plt.subplots(figsize=(15, 10))

        for key in sorted(lid_data.keys()):
            values, label = lid_data[key]

            if values is None:
                st.text(f"No LID data available for {label}")
                continue

            x_values = list(values.keys())
            y_values = list(values.values())

            ax.plot(x_values, y_values, label=label)

        ax.set_title('LID Plot for ICL with Query / No Query')
        ax.set_xlabel('Layer Index', labelpad=20)
        ax.set_ylabel('LID', labelpad=20)
        ax.legend()

        plt.tight_layout()
        st.pyplot(fig)

    if st.button('Fetch Data and Plot', key="icl_no_query_button"):
        lid_data = fetch_data_no_query(model_name, dataset, k_shot, prompt_mode, template, query_present)
        generate_lid_plot_no_query(lid_data)

    # Add your content for Experiment 4 here


with tab5:

    
    def populate_main(folder, selected_queries):
        # get graph of the LID's of each query corresponding to the dataset
        num_queries = 50
        num_lid_samples = 80
        num_layers = 41
        num_acc_samples = 80

        print(folder)

        if "13b" in str(folder):
            print("13B")
            num_layers = 41
            num_queries = 100
            num_lid_samples = 80
            num_acc_samples = 80

        elif "70b" in str(folder):
            print("70B")
            num_layers = 81
            num_queries = 60
            num_lid_samples = 100
            num_acc_samples = 100

        st.write(f"### Number of queries: {num_queries}")
        st.write(f"### Number of LID samples: {num_lid_samples}")
        st.write(f"### Number of accuracy samples: {num_acc_samples}")


        accuracies = []

        # load the selected query text and accuracy data
        text_to_display = ""

        for query in range(num_queries):

            query_path = folder / "accuracy_results" / f"query_{query}_results.json"
            query_data = json.load(open(query_path, "r"))
            acc = query_data["accuracy"]
            accuracies.append(acc)

            query_text_path = folder/ "queries.json"
            query_text = json.load(open(query_text_path, "r"))["queries"][query]


            if query not in selected_queries:
                continue

            text_to_display += f"### Query #{query}\n"
            text_to_display += f"Text: {query_text}\n"
            text_to_display += f"Query Accuracy: {acc}\n"

            text_to_display += "\n"
        st.write(f"### Average Accuracy Across All Queries: {np.average(accuracies): .4f}")
        st.write(text_to_display)


        regression_data_path = folder / "regression_results" / "regression_data.pkl"
        regression_data = pickle.load(open(regression_data_path, "rb"))

        # make a plot of the LIDs of each query for each layer



        st.write("### Analyzing Average LID Across Queries")

        layers = list(range(1, num_layers))


        fig1_handles = []

        fig1, ax1 = plt.subplots()
        for query in range(num_queries):
            query_lids = []
            for layer in layers:
                query_lids.append(regression_data[(layer, query)]["avg_lid"])

            if len(selected_queries) == 0 or query in selected_queries:
                handle, = ax1.plot(query_lids, label=f"Query {query} - Acc: {accuracies[query]}", linewidth=2)
                
                if query in selected_queries:
                    fig1_handles.append(handle)
            else:
                ax1.plot(query_lids, alpha=0.1, label=f"Query {query} - Acc: {accuracies[query]}")

            
        # add x-axis label
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Average LID Across Demonstration Prompts")
        
        ax1.set_title("Average LID of Queries for Each Layer")
        ax1.legend(handles=fig1_handles)

        st.pyplot(fig1)

        ###########################################################################################
        regression_results_path = folder / "regression_results" / "regression_results.json"
        regression_results = json.load(open(regression_results_path, "r"))

        # plot the r values across layers for average LID
        r_values = []
        for layer in layers:
            r_values.append(regression_results["avg_lid"][str(layer)]["r_value"]**2)
        

        # find max r value and corresponding layer
        best_layer = None
        for layer, r in zip(layers, r_values):
            if r == max(r_values):
                # st.write(f"Layer with Highest R Value: {layer} - {r}")
                best_layer = layer



        fig3, ax3 = plt.subplots()
        ax3.plot(layers, r_values, label="R^2 Value")
        ax3.plot([best_layer], [max(r_values)], 'ro', label="Max R Value")
        ax3.set_xlabel("Layer")
        ax3.set_ylabel("R^2 Value")
        ax3.set_title("R^2 Correlation Coeffecient for Average LID Across Queries")
        st.pyplot(fig3)



        ###########################################################################################
        # create scatterplot of the average LID and accuracy at the fixed layer for all queries
        accuracies = []
        avg_lids = []
        for query in range(num_queries):
            accuracies.append(regression_data[(best_layer, query)]["accuracy"])
            avg_lids.append(regression_data[(best_layer, query)]["avg_lid"])
        
        fig5, ax5 = plt.subplots()
        ax5.scatter(avg_lids, accuracies)
        ax5.set_xlabel(f"Average LID Across Demonstration Prompts")
        ax5.set_ylabel("Accuracy")
        ax5.set_title(f"Average LID vs Accuracy at Layer {best_layer} (R^2 = {max(r_values): .4f})")
        st.pyplot(fig5)


        ###########################################################################################


        st.write("### Analyzing Variance of LID Measurements Across Queries")

        fig2_handles = []
        # make a plot of the variance of for each layer for each query
        fig2, ax2 = plt.subplots()
        for query in range(num_queries):
            query_vars = []
            for layer in layers:
                query_vars.append(regression_data[(layer, query)]["var_lid"])
            # ax2.plot(query_vars, label=f"Query {query}")

            if len(selected_queries) == 0 or query in selected_queries:
                handle,  = ax2.plot(query_vars, label=f"Query {query} - Acc: {accuracies[query]}", linewidth=2)

                if query in selected_queries:
                    fig2_handles.append(handle)
            else:
                ax2.plot(query_vars, alpha=0.1, label=f"Query {query} - Acc: {accuracies[query]}")

        # add x-axis label
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Variance of LID Across Demonstrations")

        ax2.set_title("Variance of LID of Queries for Each Layer")

        # legends
        ax2.legend(handles=fig2_handles)

        st.pyplot(fig2)


        ###########################################################################################

        # plot the r values across layers for average LID
        r_values = []
        for layer in layers:
            r_values.append(regression_results["var_lid"][str(layer)]["r_value"]**2)


        best_layer = None
        #find max r value and corresponding layer
        for layer, r in zip(layers, r_values):
            if r == max(r_values):
                # st.write(f"Layer with Highest R Value: {layer} - {r}")
                best_layer = layer

        
        fig4, ax4 = plt.subplots()
        ax4.plot(layers, r_values, label="R^2 Value")
        ax4.plot([best_layer], [max(r_values)], 'ro', label="Max R^2 Value")
        ax4.set_xlabel("Layer")
        ax4.set_ylabel("R^2 Value")
        ax4.set_title("R^2 Correlation Coeffecient for Variance of LID Across Queries")
        st.pyplot(fig4)

        ###########################################################################################
        # create scatterplot of the average LID and accuracy at the fixed layer for all queries
        accuracies = []
        avg_lids = []
        for query in range(num_queries):
            accuracies.append(regression_data[(best_layer, query)]["accuracy"])
            avg_lids.append(regression_data[(best_layer, query)]["var_lid"])
        
        fig5, ax5 = plt.subplots()
        ax5.scatter(avg_lids, accuracies)
        ax5.set_xlabel(f"Variance of LID Measurments Across Demonstrations")
        ax5.set_ylabel("Accuracy")
        ax5.set_title(f"Variance of LID vs Accuracy at Layer {best_layer} (R^2 = {max(r_values): .4f})")
        st.pyplot(fig5)

        
        ################################################################################################







    st.header("Intrinsic Dimension of Queries")
    datasets = ["boolq", "cola", "mrpc", "qnli", "qqp"]


    models = ["Llama-2-13b-hf", "Llama-2-70b-hf"]
    query_indices = list(range(50))


    model = st.selectbox('Select Model', models)


    if model == "Llama-2-13b-hf":
        datasets = ["cola", "commonsense_qa", "mrpc", "mrpc", "mnli", "qnli", "qqp", "rte"]
    
    if model == "Llama-2-70b-hf":
        datasets = ["boolq", "cola", "mrpc", "qnli", "qqp"]

    dataset = st.selectbox('Select Dataset', datasets)
    selected_queries = st.multiselect('Select Queries to Highlight', query_indices)

    folder = find_directory(Path("results") / "query_lid_acc" / model, dataset)
    
    folder = Path("results") / "query_lid_acc" / model / folder


    if st.button('Populate', key="populate_button_query_lid"):
        # populate the data
        populate_main(folder, selected_queries)





    





