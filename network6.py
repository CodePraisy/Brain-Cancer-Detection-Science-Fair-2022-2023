import os
import pickle
import numpy as np 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras
from keras import layers

from time import perf_counter
from sklearn.model_selection import train_test_split

def train():
    list_of_images = 'list_of_images.pkl'
    list_of_answers = 'list_of_answers.pkl'

    open_images_file = open(list_of_images, "rb")
    images_list = pickle.load(open_images_file)
    open_images_file.close()

    open_answers_file = open(list_of_answers, "rb")
    answers_list = pickle.load(open_answers_file)
    open_answers_file.close()

    images_list = list(images_list)
    answers_list = list(answers_list)

    images_list = np.array(images_list)
    answers_list = np.array(answers_list)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images_list, answers_list, test_size=0.25
    )

    train_images = train_images  / 255.0
    test_images = test_images / 255.0

    print(train_images.shape)
    print(train_labels.shape)

    model = keras.Sequential([
        keras.Input(shape=(256, 256, 1)),
        
        layers.Conv2D(16, (3, 3), activation="relu", kernel_initializer='he_uniform'),
        layers.Conv2D(16, (3, 3), activation="relu", kernel_initializer='he_uniform'),
        layers.Conv2D(16, (3, 3), activation="relu", kernel_initializer='he_uniform'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_uniform'),
        layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_uniform'),
        layers.MaxPooling2D(pool_size=(2, 2)),  
        
        layers.Flatten(),
        
        layers.Dense(32, kernel_initializer='he_uniform'),
        layers.Dropout(0.5),  
        layers.Dense(1)
    ])

    print(model.summary())

    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    optim = keras.optimizers.Adam(learning_rate=0.01)
    metrics = ["accuracy"]

    model.compile(optimizer=optim, loss=loss, metrics=metrics)

    batch_size = 64
    epochs = 15

    start_time = perf_counter()

    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=2)
    model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=2)

    model.save("BrainCancerModelLiteTurboPentarieCell.h5")

    end_time = perf_counter()

    print("\n-----------------------------\n")

    print(f"Operation took {round(end_time - start_time)} second(s) to complete!")

if __name__ == "__main__": train()
