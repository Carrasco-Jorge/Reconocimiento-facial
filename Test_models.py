# Import general modules
import numpy as np
from time import time
import random
from glob import glob
from utils import create_path
# from sklearn.metrics import confusion_matrix

# Import DL framework modules
import tensorflow as tf
from tensorflow import keras

# Import viz modules
import matplotlib.pyplot as plt
import seaborn as sns

# Workaround for OMP issue
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

image_size = (112,92)
batch_size = 10

# ------------ Load data ------------ #
# Get new images to test (me, brother, mother, etc.)
my_data_path = os.path.join("./test_data_png/me")
data_path = os.path.join("./test_data_png/not_me")

files = []
labels = []
for i in range(50):
    files.append(create_path(f"{i+1}.png",PATH=my_data_path))
    labels.append(1)

for i in range(191):
    files.append(create_path(f"{i+1}.png",PATH=data_path))
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

# for image, label in dataset.take(10):
#     plt.imshow(image)
#     plt.title(int(label))
#     plt.show()

# Get labels
labels = []
for _ , label in list(dataset.as_numpy_iterator()):
    labels.append(label)

labels = tf.convert_to_tensor(labels)

dataset = dataset.batch(10)

# ------------ Test models ------------ #

def build_model(model_path):
    model_0 = keras.models.load_model(model_path)
    # model = keras.models.Sequential()
    # model.add(keras.layers.Rescaling(1./255, input_shape=(112,92,1)))
    # # Remove data augmentation section
    # for i in range(2,len(model.layers)):
    #     l = model_0.get_layer(index=i)
    #     model.add(l)
    # # Compile model
    # model.compile(
    #     optimizer="adam",
    #     loss="binary_crossentropy",
    #     metrics=["accuracy"]
    # )
    return model_0

def test_model(model_path):
    # Load model
    model = build_model(model_path)

    #Predict
    y_prediction = model.predict(dataset)

    #Create confusion matrix and normalizes it over predicted (columns)
    confusion_matrix = tf.math.confusion_matrix(
        labels=labels,
        predictions=y_prediction,
        num_classes=2
    )

    model_name_path = "./Test_results/"+model_path.split('\\')[-1]
    confusion_matrix = tf.math.divide(confusion_matrix,tf.reduce_sum(confusion_matrix))

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, ax=ax, annot=True)
    ax.set_title(f"Accuracy : {confusion_matrix[0,0]+confusion_matrix[1,1]}")
    ax.set_ylabel("Real labels")
    ax.set_xlabel("Precitions")
    fig.savefig(model_name_path+".png")


models = glob(f"models/my_model_*/my_model_*")
for model in models:
    test_model(model)
