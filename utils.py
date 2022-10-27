import os

def create_path(path, PATH = f"./models"):
  return os.path.join(PATH, path)

def save_hyperparams(path, **kwargs):
    file = open(path, "w")
    file.write(f"# {'-'*5} Hyperparameters {'-'*5} #\n\n")
    for key, value in kwargs.items():
        file.write(f"{key:<15} : {value}\n")

def get_dataset_partitions_tf(ds, ds_size, batch_size, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    # set batch size

    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)
    
    return train_ds, val_ds, test_ds

# Launch TensorBoard
# cd /d D:\jorge\Documents\9noSemestre\CodigosANN\FacialRec
# activate deeplearning
# python -m tensorboard.main --logdir=models/tensorboard_logs