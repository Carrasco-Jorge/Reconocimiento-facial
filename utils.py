import os

def create_path(path, PATH = f"./models"):
  return os.path.join(PATH, path)

def save_hyperparams(path, **kwargs):
    file = open(path, "w")
    file.write(f"# {'-'*5} Hyperparameters {'-'*5} #\n\n")
    for key, value in kwargs.items():
        file.write(f"{key:<15} : {value}\n")

# Launch TensorBoard
# python -m tensorboard.main --logdir=models/tensorboard_logs