# Import general modules
import numpy as np
from time import time
from utils import create_path, save_hyperparams

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

# -------- Hyperparameter values -------- #
image_size = (112,92)
batch_size = 10
epochs = 500
dropout = 0.2
last_activation = "softmax"
loss = "categorical_crossentropy"
learning_rate = 0.0001
optimizers = {"adam":Adam(learning_rate=learning_rate),
              "rmsprop":RMSprop(learning_rate=learning_rate)}
opt = "adam"
seed = 2022

optimizer = optimizers[opt]

NAME = f"model_{int(time())}"
path = create_path(NAME)
os.mkdir(path)
tensorboard = TensorBoard(log_dir=create_path(NAME,PATH=create_path("tensorboard_logs")))
save_hyperparams(path=create_path("hyperparameters.txt",PATH=path),
                 image_size = image_size,batch_size = batch_size,epochs = epochs,
                 dropout = dropout,last_activation = last_activation,loss = loss,
                 learning_rate = learning_rate,optimizer=opt, seed = seed)

# -------- Image dataset from directory -------- #
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data_png",
    label_mode="categorical",
    color_mode="grayscale",
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data_png",
    label_mode="categorical",
    color_mode="grayscale",
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
)


train_ds = train_ds.prefetch(buffer_size=batch_size)
val_ds = val_ds.prefetch(buffer_size=batch_size)


# -------- Model -------- #
def block(x, filters=32, sep=True):
    if sep:
        l = layers.SeparableConv2D(filters=filters, kernel_size=(4,4))(x)
        l = layers.BatchNormalization()(l)
        l = layers.Activation("relu")(l)
    else:
        l = layers.Conv2D(filters=filters, kernel_size=(4,4))(x)
        l = layers.BatchNormalization()(l)
        l = layers.Activation("relu")(l)
    l = layers.MaxPooling2D(pool_size=(2,2))(l)
    return l

# Utilize sequential model to perform data augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2,0.2),
        layers.RandomTranslation(0.2,0.2)
    ]
)

input_shape = image_size + (1,)
inputs = keras.Input(shape=input_shape)
# Data augmentation
x = data_augmentation(inputs)
# Data normalization
x = layers.Rescaling(1./255)(x)

# Conv layers
x = block(x, filters=64,sep=False)
x = block(x, filters=128,sep=True)
x = block(x, filters=128,sep=True)

# Dense layers
x = layers.Flatten()(x)

x = layers.Dropout(dropout)(x)
x = layers.Dense(units=40, activation="relu")(x)

x = layers.Dense(units=40, activation="relu")(x)

x = layers.Dense(units=40, activation=last_activation)(x)
outputs = x

model = keras.Model(inputs, outputs)

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
                    callbacks=[tensorboard])

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
