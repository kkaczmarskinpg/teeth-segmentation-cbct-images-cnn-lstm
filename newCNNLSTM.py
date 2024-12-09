import tensorflow as tf
from tensorflow.keras import layers, models

def cnn_lstm_teeth_segmentation(input_shape=(None, 128, 128, 1)):
    """
    Creates a CNN-LSTM network for teeth segmentation from CBCT scans.

    Args:
        input_shape: Tuple, shape of the input data (num_slices, height, width, channels).
        num_classes: Number of output classes for segmentation.

    Returns:
        model: A compiled Keras model.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder (CNN layers)
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

    x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

    # Flatten the spatial dimensions for LSTM processing
    x = layers.TimeDistributed(layers.Flatten())(x)

    # LSTM layers
    x = layers.LSTM(256, return_sequences=True, activation='relu')(x)
    x = layers.LSTM(128, return_sequences=False, activation='relu')(x)

    # Decoder (Deconvolutional layers)
    x = layers.RepeatVector(input_shape[0])(x)  # Repeat for each time step
    x = layers.TimeDistributed(layers.Reshape((16, 16, 128)))(x)  # Reshape to 2D spatial format

    x = layers.TimeDistributed(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'))(x)

    outputs = x

    # Create the model
    model = models.Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Example usage
input_shape = (10, 128, 128, 1)  # batch size, number of slices, height, width, channels
model = cnn_lstm_teeth_segmentation(input_shape=input_shape)
model.summary()
