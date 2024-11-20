from keras import layers, models, Input
from metrics import jaccard_index, dice_coefficient

def create_cnn_lstm_model(image_shape=(128, 128), num_slices=652):
    """
    Create a CNN-LSTM model for 3D teeth segmentation with resized input.
    
    Parameters:
        image_shape: tuple, shape of the resized slices (height, width).
        num_slices: int, number of slices in the 3D scan.

    Returns:
        model: Keras model instance.
    """
    input_layer = Input(shape=(num_slices, image_shape[0], image_shape[1], 1))  # Adjust input shape

    # CNN model for feature extraction
    cnn_input = Input(shape=(image_shape[0], image_shape[1], 1))
    x = layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(cnn_input)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, kernel_size=2, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')(x)
    cnn_model = models.Model(cnn_input, x, name="CNN_FeatureExtractor")

    # Apply CNN to each slice
    time_distributed_layer = layers.TimeDistributed(cnn_model)(input_layer)  # Shape: (652, 256, 256, 1)

    # LSTM for sequential processing
    lstm_input = layers.Reshape((num_slices, -1))(time_distributed_layer)  # Flatten spatial dimensions
    lstm_layer = layers.LSTM(128, return_sequences=True)(lstm_input)
    lstm_layer = layers.LSTM(64, return_sequences=True)(lstm_layer)
    lstm_output = layers.Dense(image_shape[0] * image_shape[1], activation='sigmoid')(lstm_layer)  # Flattened mask

    # Reshape back to spatial dimensions
    output_layer = layers.Reshape((num_slices, image_shape[0], image_shape[1], 1))(lstm_output)

    # Define the complete model
    model = models.Model(inputs=input_layer, outputs=output_layer, name="CNN_LSTM_Segmentation")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', jaccard_index, dice_coefficient])

    return model