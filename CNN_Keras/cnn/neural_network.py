from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

class CNN:
    @staticmethod
    def build(width, height, depth, total_classes, Saved_Weights_Path=None):
        # Initialize the model
        model = Sequential()

        # First CONV => RELU => POOL Layer
        model.add(Conv2D(
            filters=20,
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
            input_shape=(height, width, depth)  # channels last
        ))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second CONV => RELU => POOL Layer
        model.add(Conv2D(
            filters=50,
            kernel_size=(5, 5),
            padding="same",
            activation="relu"
        ))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Third CONV => RELU => POOL Layer
        model.add(Conv2D(
            filters=100,
            kernel_size=(5, 5),
            padding="same",
            activation="relu"
        ))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Flatten and Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(500, activation="relu"))
        model.add(Dense(total_classes, activation="softmax"))

        # Load pre-trained weights if provided
        if Saved_Weights_Path is not None:
            model.load_weights(Saved_Weights_Path)

        return model
# ---------------------- End of CNN -----------------------
