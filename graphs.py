import wandb
import numpy as np
import matplotlib.pyplot as plt

api = wandb.Api()
project_name = "deep_learning"
# filters = {
#         "config.dataset.name": "DTD047"
#     }
filters = None
# filters = {
#         "name": {"$regex": "^dtd_.*no_freeze.*$"}
#     }
runs = api.runs(project_name, filters=filters)
#run_id = "sc9pohpa"

def find_convergence(acc_list, window_width=4):
    convolve = np.convolve(acc_list, np.ones(window_width) / window_width, mode='same')
    index = np.argmax(convolve)
    max_acc = np.max(convolve)
    return index, max_acc

import matplotlib.pyplot as plt
import seaborn as sns

def plot_acc_vs_batches(acc_lists, num_batches_lists, labels, title, file_name, y_lims=None):
    # Set a more appealing style using seaborn
    sns.set_theme(style="whitegrid", context="talk")

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Define a colormap for distinguishing lines
    colors = sns.color_palette("tab10", len(labels))

    # Plot each accuracy vs. batches line
    for i, (acc_list, num_batches_list, label) in enumerate(zip(acc_lists, num_batches_lists, labels)):
        plt.plot(
            num_batches_list,
            acc_list,
            label=label,
            color=colors[i],
            linewidth=2.25,
            marker="o",
            markersize=4,
        )
        # Annotate the last point
        xytext = (0, -12) if label == r"$p = 0.6$" else (0, 7)
        plt.annotate(
            f"{acc_list[-1]:.4f}",
            (num_batches_list[-1], acc_list[-1]),
            textcoords="offset points",
            xytext=xytext,
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=colors[i],
        )

    # Add title and labels
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Number of Batches", fontsize=14)
    plt.ylabel("Validation Accuracy", fontsize=14)

    # Customize y-axis limits if provided
    if y_lims is not None:
        plt.ylim(y_lims[0], y_lims[1])

    # Customize legend
    plt.legend(title="Models", fontsize=12, title_fontsize=14)

    # Save the figure to the specified file
    plt.tight_layout()
    plt.savefig(f"figures/{file_name}", dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    #run = api.run(f"{project_name}/{run_id}")
    for run in runs:
        if "no_freeze" in run.name:
            print(run.name)
            history = run.history()  
            val_accuracies = history['val_accuracy']
            max_index, max_acc = find_convergence(val_accuracies.to_list())
            print(max_index, max_acc, history['total_datapoints'][max_index])

    graphs_info = [
        # {
        #     "file_name" : "dtd_ours_vs_vanilla.png",
        #     "title" : "ADS vs. Vanilla Training on DTD",
        #     "runs_graphed" : {
        #         "dtd_ours_no_freeze_2_5_mixone" : "(2, 5)",
        #         "dtd_ours_no_freeze_3_12_mixone" : "(3, 12)",
        #         "dtd_vanilla_no_freeze" : "Vanilla"
        #     },
        # },
        {
            "file_name" : "dtd_prop_comparison.png",
            "title" : "Varying reweighting proportion on DTD",
            "runs_graphed" : {
                "dtd_ours_no_freeze_2_5_mixone" : r"$p = 0.8$",
                "dtd_ours_no_freeze_2_5_prop1_mixone" : r"$p = 1$",
                "dtd_ours_no_freeze_2_5_prop0.6_mixone" : r"$p = 0.6$"
            },
        },
        # {
        #     "file_name" : "fl_ours_vs_vanilla.png",
        #     "title" : "ADS vs. Vanilla Training on Flowers",
        #     "runs_graphed" : {
        #         "fl_ours_no_freeze_2_25_mixone" : "(2, 25)",
        #         "fl_ours_no_freeze_2_25" : "(2, 25) without mix one",
        #         "fl_ours_no_freeze_4_25_mixone" : "(4, 25)",
        #         "fl_vanilla_no_freeze" : "Vanilla"
        #     },
        # },
        {
            "file_name" : "food_from_scratch.png",
            "title" : "ADS vs. Vanilla Training on Food",
            "runs_graphed" : {
                "fo_ours_no_freeze_scratch_2_25_25_subset" : "ADS",
                "fo_vanilla_no_freeze_scratch_25_subset_again" : "Vanilla"
            },
            "y_lim": False,
            "no_stopping": True,
        },
        {
            "file_name" : "fl_no_freeze_all_freeze.png",
            "title" : "Effects of freezing transformer encoder during fine-tuning on Flowers",
            "runs_graphed" : {
                "fl_vanilla_no_freeze" : "No Freezing",
                "fl_vanilla_all_freeze" : "Freezing"
            },
            "y_lim": False,
        },
        {
            "file_name" : "dtd_no_freeze_all_freeze.png",
            "title" : "Effects of freezing transformer encoder during fine-tuning on DTD",
            "runs_graphed" : {
                "dtd_vanilla_no_freeze" : "No Freezing",
                "dtd_vanilla_all_freeze" : "Freezing"
            },
            "y_lim": False,
        },
        {
            "file_name" : "fl_mixone_nomixone.png",
            "title" : "Alternating phases vs. only targeted epochs",
            "runs_graphed" : {
                "fl_ours_no_freeze_2_25_mixone" : "Alternating",
                "fl_ours_no_freeze_2_25" : "Only Targeted",
            },
        },
    ]

    for graph in graphs_info:
        accuracies = []
        batches = []
        labels = []
        y_values = []
        for run in runs:
            if run.name in graph["runs_graphed"]:
                print(run.name)
                history = run.history()  
                max_index, max_acc = find_convergence(history['val_accuracy'].to_list())
                if graph.get("no_stopping", False):
                    accuracies.append(history['val_accuracy'].to_list())
                    batches.append(history['total_datapoints'].to_list())
                else:
                    accuracies.append(history['val_accuracy'].to_list()[:max_index])
                    batches.append(history['total_datapoints'].to_list()[:max_index])
                labels.append(graph["runs_graphed"][run.name])
                y_values.append(max_acc)

        mean_y_val = np.mean(y_values)
        y_lims = (mean_y_val - 0.1, mean_y_val + 0.03) if graph.get("y_lim", True) else None
        plot_acc_vs_batches(
            accuracies, 
            batches,
            labels,
            graph["title"],
            graph["file_name"],
            y_lims,
        )