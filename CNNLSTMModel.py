from keras import layers, metrics, Sequential
from metrics import dice_coefficient
import tensorflow as tf

def create_cnn_lstm_model(image_shape=(128, 128), num_slices=50):
    """
    Create a CNN-LSTM model for slice-by-slice 3D image segmentation using Sequential API.
    
    Parameters:
        image_shape: tuple, the shape of each 2D slice (height, width).
        num_slices: int, number of slices in the 3D image.
    
    Returns:
        model: Compiled Keras model.
    """
    model = Sequential(name="CNN_LSTM_Segmentation")
    
    # Input layer
    model.add(layers.InputLayer(input_shape=(num_slices, image_shape[0], image_shape[1], 1)))
    
    # TimeDistributed CNN layers
    model.add(layers.TimeDistributed(layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=2)))
    model.add(layers.TimeDistributed(layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=2)))
    model.add(layers.TimeDistributed(layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')))
    model.add(layers.TimeDistributed(layers.Flatten()))  # Flatten for LSTM input
    
    # LSTM layers
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(64, return_sequences=True))
    
    # TimeDistributed Dense layer for pixel-wise probabilities
    model.add(layers.TimeDistributed(layers.Dense(image_shape[0] * image_shape[1], activation='sigmoid')))
    
    # Reshape back to spatial dimensions
    model.add(layers.Reshape((num_slices, image_shape[0], image_shape[1], 1)))

    # Compile the model

    model.compile(
        optimizer='adam', 
        loss=tf.keras.losses.BinaryFocalCrossentropy(), # "binary_crossentropy"
        metrics=[ 
            metrics.BinaryIoU(target_class_ids=[1]),
            dice_coefficient
            ]
        )

    return model