import streamlit as st
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np
import os

PATH_TO_RESULTS_DIR = Path("concise_results/")  # set this to the results directory


def get_average_accuracy(model, dataset, experiment_type, accuracy_type):
    def load_accuracy(path):
        try:
            with path.open("r") as f:
                data = json.load(f)
                return data.get("accuracy", None)
        except FileNotFoundError:
            print(f"File not found: {path}")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {path}")
            return None

    if "icl" in experiment_type:
        k_shot = int(experiment_type.split("-")[1])
        path_to_accuracy = PATH_TO_RESULTS_DIR / f"icl/accuracy_results/{model}/{dataset}/{k_shot}-shot/"

    elif experiment_type == "finetune 1k":
        path_to_accuracy = PATH_TO_RESULTS_DIR / f"ft/accuracy_results/{model}_final/{dataset}/0-shot/"

    elif experiment_type == "finetune 10":
        path_to_accuracy = PATH_TO_RESULTS_DIR / f"few-sample-ft/accuracy_results/{model}_final/{dataset}/0-shot/"

    else:
        print(f"Unknown experiment type: {experiment_type}")
        return None

    if accuracy_type == "logit-accuracy":
        path_to_accuracy /= "accuracy.json"
    elif accuracy_type == "generation accuracy":
        path_to_accuracy /= "acc-results.json"
    else:
        print(f"Unknown accuracy type: {accuracy_type}")
        return None

    return load_accuracy(path_to_accuracy)



def get_intrinsic_dimensions(model, dataset, experiment_type, estimator):
    path_to_id = PATH_TO_RESULTS_DIR / "id/"

    if "icl" in experiment_type:
        k_shot = int(experiment_type.split("-")[1])
        path_to_id /= f"icl/{model}/{dataset}/{k_shot}-shot/{estimator}.json"

    elif experiment_type == "finetune 1k":
        path_to_id /= f"ft/{model}_final/{dataset}/0-shot/"
        path_to_id /= f"{estimator}.json"

    elif experiment_type == "finetune 10":
        path_to_id /= f"few-sample-ft/{model}_final/{dataset}/10-samples/0/0-shot/"
        path_to_id /= f"{estimator}.json"

    try:
        with path_to_id.open("r") as f:
            data = json.load(f)

            if estimator == "twonn":
                id_vals = [data[str(i)] for i in range(len(data))]
                return id_vals[1:]
            
            else:
                id_vals = [data[str(i)]["estimate"] for i in range(len(data))]
                return id_vals[1:]
            
    except FileNotFoundError:
        print(f"File not found: {path_to_id}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {path_to_id}")
        return None



def get_num_layers(model_name):
    """Returns the number of layers for a given model name."""
    model_layers = {
        "meta-llama/Llama-2-70b-hf": 81,
        "meta-llama/Llama-2-13b-hf": 41,
        "meta-llama/Llama-2-7b-hf": 33,
        "meta-llama/Meta-Llama-3-8B": 33,
        "meta-llama/Meta-Llama-3-70B": 81,
        "EleutherAI/pythia-1b": 17,
        "EleutherAI/pythia-1.4b": 25,
        "EleutherAI/pythia-2.8b": 33,
        "EleutherAI/pythia-6.9b": 33,
        "EleutherAI/pythia-12b": 37,
        "mistralai/Mistral-7B-v0.3": 33,
        "google/gemma-2-27b": 47,
        "google/gemma-2-9b": 43,
    }

    layers = model_layers.get(model_name)
    if layers is None:
        for key in model_layers:
            if key in model_name:
                layers = model_layers[key]
                break
    
    if layers is None:
        print(f"Model name {model_name} not found in the layers dictionary")
    else:
        print(f"Model {model_name} has {layers} layers")
    
    return layers

# Streamlit app
st.title("Experiment Results Visualization")

# Model selection
model = st.selectbox(
    "Select a model:",
    [
        "meta-llama/Meta-Llama-3-8B", 
        "mistralai/Mistral-7B-v0.3", 
        "meta-llama/Llama-2-7b-hf", 
        "meta-llama/Llama-2-13b-hf"
    ]
)

# Dataset selection
dataset = st.selectbox(
    "Select a dataset:",
    ["sst2", "cola", "qnli", "qqp", "mnli", "ag_news", "commonsense_qa", "mmlu"]
)

# Intrinsic dimension estimator selection
estimator = st.selectbox(
    "Select an intrinsic dimension estimator:",
    ["mle_15", "mle_25", "mle_50", "mle_75", "mle_100", "twonn"]
)

# Experiment type selection
experiment_types = st.multiselect(
    "Select experiment types:",
    ["icl-0", "icl-1", "icl-2", "icl-5", "icl-10", "finetune 1k", "finetune 10"]
)

accuracy_type = st.selectbox(
    "Select accuracy type:",
    ["logit-accuracy", "generation accuracy"]
)


# Generate graphs
if st.button("Generate Graphs"):
    # Bar chart for average accuracy
    accuracies = []
    for exp_type in experiment_types:
        accuracy = get_average_accuracy(model, dataset, exp_type, accuracy_type)
        if accuracy is None:
            st.warning(f"Accuracy data not found for {exp_type}.")
        accuracies.append(accuracy)

    plt.figure(figsize=(10, 5))
    plt.bar(experiment_types, [acc if acc is not None else 0 for acc in accuracies])
    plt.title("Average Accuracy by Experiment Type")
    plt.xlabel("Experiment Type")
    plt.ylabel("Average Accuracy")
    st.pyplot(plt)

    # Line chart for intrinsic dimensions
    num_layers = get_num_layers(model)
    plt.figure(figsize=(10, 5))
    for exp_type in experiment_types:
        intrinsic_dims = get_intrinsic_dimensions(model, dataset, exp_type, estimator)
        if intrinsic_dims is None:
            st.warning(f"Intrinsic dimension data not found for {exp_type}.")
            intrinsic_dims = [0] * (num_layers-1)  # Default to zero if data is missing
        plt.plot(range(num_layers-1), intrinsic_dims, label=exp_type)
    plt.title("Intrinsic Dimensions by Layer Index")
    plt.xlabel("Layer Index")
    plt.ylabel("Intrinsic Dimension")
    plt.legend()
    st.pyplot(plt)

# ... existing code ...





