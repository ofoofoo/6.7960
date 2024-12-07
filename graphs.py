import wandb

api = wandb.Api()
project_name = "deep_learning"
run_id = "iykisvn4"

run = api.run(f"{project_name}/{run_id}")
history = run.history()  
val_accuracies = history['val_accuracy']
val_loss = history['val_loss']

print(val_accuracies.to_list())
print(val_loss.to_list())

