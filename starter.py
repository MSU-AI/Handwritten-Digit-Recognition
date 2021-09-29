import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt


def load_data():
    ##################### Load MNIST dataset ####################################
    # Use MNIST handwriting dataset
    mnist = tf.keras.datasets.mnist
    
    # Prepare data for training
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # If you are curious, you can have a look at a random sample from your data
    # random_index = 0
    # plt.imshow(x_train[random_index])
    # plt.show()
    # print(y_train[random_index])

    ##################### Data pre-processing ####################################
    # Scale x data into a 0-1 range
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Transform y-data into a usable format (one-hot encoding)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    
    # Re-shaping your data (just modifies it such that it's digestable for the CNN)
    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
    )
    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
    )


def create_model(x_train, y_train, x_test, y_test):
    ########################## Create your CNN ####################################
    # 1. Build the structure of your NN
    
        # Add the convolutional input layer
        # Specify the input shape, activation, kernel size, and neurons
        # Add pooling
        # Flatten
        # (Add Hidden Layers??)
        # Add output layer for all 10 digits

    # 2. Compile neural network
    # You can try with "adam" optimizer and "categorical_crossentropy" loss

    # 3. Fit neural network, choose n° of epochs
    # Start only with the number of epochs and do not worry about validation data
    
    # 4. Evaluate neural network performance
    # Call .evaluate method on your model
    
    # 5. Save the model as "model.h5"
    
    # 6. Return your model
    
    return
    

def import_model():
    try:
        model = load_model("model.h5")
    except:
        model = create_model()
    return model