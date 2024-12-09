import wandb
import numpy as np

api = wandb.Api()
project_name = "deep_learning"
filters = {
        "config.dataset.name": "DTD047"
    }
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


if __name__ == '__main__':
    #run = api.run(f"{project_name}/{run_id}")
    for run in runs:
        if "no_freeze" in run.name:
            print(run.name)
            history = run.history()  
            val_accuracies = history['val_accuracy']
            max_index, max_acc = find_convergence(val_accuracies.to_list())
            print(max_index, max_acc)
        


