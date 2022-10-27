# Import general modules
import numpy as np
from time import time
import random
from utils import create_path, save_hyperparams, get_dataset_partitions_tf

# Import DL framework modules
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import TensorBoard, ReduceLROnPlateau

# Import viz modules
import matplotlib.pyplot as plt
from PIL import Image

# Workaround for OMP issue
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ------------ Hyperparameter values ------------ #
image_size = (112,92)
batch_size = 10
epochs = 3
dropout = 0.2
last_activation = "sigmoid"
loss = "binary_crossentropy"
learning_rate = 0.001
optimizers = {"adam":Adam(learning_rate=learning_rate),
              "rmsprop":RMSprop(learning_rate=learning_rate)}
opt = "adam"
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, min_delta=0.01, patience=5, min_lr=1e-4)
seed = 2022

optimizer = optimizers[opt]

NAME = f"my_model_{int(time())}"
path = create_path(NAME)
os.mkdir(path)
tensorboard = TensorBoard(log_dir=create_path(NAME,PATH=create_path("my_tensorboard_logs")))
save_hyperparams(path=create_path("hyperparameters.txt",PATH=path),
                 image_size = image_size,batch_size = batch_size,epochs = epochs,
                 dropout = dropout,last_activation = last_activation,loss = loss,
                 learning_rate = learning_rate,optimizer=opt, seed = seed)

# ------------ Load data ------------ #
my_data_path = os.path.join("./my_data_png")
data_path = os.path.join("./data_png")

files = []
labels = []
for i in range(332):
    files.append(create_path(f"{i+1}.png",PATH=my_data_path))
    labels.append(1)

for d in range(40):
    for f in range(10):
        files.append(create_path(f"s{d+1}/{f+1}.png",PATH=data_path))
        labels.append(0)

lst = list(zip(files,labels))
random.shuffle(lst)
files, labels = zip(*lst)

files = tf.data.Dataset.from_tensor_slices(list(files))
labels = tf.data.Dataset.from_tensor_slices(list(labels))

data = tf.data.Dataset.zip((files, labels))

def process_file(file_name, label):
    image = tf.io.read_file(file_name)
    image = tf.io.decode_png(image, channels=1)
    image = tf.image.resize(image, image_size)
    # image /= 255.0 # Done inside model
    return image, label

dataset = data.map(process_file)

train_ds, val_ds, test_ds = get_dataset_partitions_tf(
    dataset, 332+400, batch_size=batch_size, 
    train_split=0.6,val_split=0.3,test_split=0.1
)

# for image, label in dataset.take(10):
#     plt.imshow(image)
#     plt.title(int(label))
#     plt.show()

# ------------ Load model ------------ #
num_model = 2
models = [
    "models/model_1666634308/model_1666634308",
    "models/model_1666644878/model_1666644878",
    "models/model_1666638061/model_1666638061"
]

pre_trained_model = keras.models.load_model(models[num_model])

pre_trained_model.summary()

model = Sequential()
# Add pre-trained layers
num_layers = [10, 16, 10]
for i in range(num_layers[num_model]):
    l = pre_trained_model.get_layer(index=i)
    l.trainable=False
    model.add(l)

# New classifier
# model.add(layers.Dropout(0.2))

# model.add(layers.Dense(50))
# model.add(layers.LeakyReLU(alpha=0.2))

# model.add(layers.Dense(50))
# model.add(layers.LeakyReLU(alpha=0.2))

# units = 10, 20, 50
model.add(layers.Dense(50,activation="relu"))

model.add(layers.Dense(1,activation=last_activation))

with open(create_path('model_summary.txt',PATH=path),'w') as file:
    model.summary(print_fn=lambda x: file.write(x + '\n'))

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=["accuracy"]
)

# -------- Training -------- #

history = model.fit(train_ds,
                    epochs=epochs, 
                    validation_data=val_ds, 
                    verbose=0,
                    callbacks=[tensorboard,reduce_lr])

# -------- Training Viz -------- #
fig, ax = plt.subplots()
ax.plot(history.history["accuracy"], label="Training")
ax.plot(history.history["val_accuracy"], label="Validation")
ax.legend()
fig.savefig(create_path("accuracy.png", PATH=path))

fig, ax = plt.subplots()
ax.plot(history.history["loss"], label="Training")
ax.plot(history.history["val_loss"], label="Validation")
ax.legend()
fig.savefig(create_path("loss.png", PATH=path))

model.save(create_path(NAME, PATH=path))

print(model.evaluate(test_ds))
