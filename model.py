import tensorflow as tf
from keras import layers, models, Input

def create_cnn_lstm_model(image_shape=(797, 797), num_slices=652):
    """
    Create a CNN-LSTM model for 3D teeth segmentation.

    Parameters:
        image_shape: tuple, shape of the 2D slices (height, width).
        num_slices: int, number of slices in the 3D scan.

    Returns:
        model: Keras model instance.
    """
    # CNN model for feature extraction
    cnn_input = Input(shape=(image_shape[0], image_shape[1], 1))  # Single-channel (grayscale) 2D image input
    x = layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(cnn_input)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    cnn_model = models.Model(cnn_input, x, name="CNN_FeatureExtractor")

    # LSTM for sequential processing of extracted features
    lstm_input = Input(shape=(num_slices, cnn_model.output_shape[-1]))  # Sequence of feature vectors
    y = layers.LSTM(128, return_sequences=True)(lstm_input)
    y = layers.LSTM(64, return_sequences=True)(y)
    y = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(y)  # Output per slice
    
    # Define the complete model
    model = models.Model(inputs=[cnn_input, lstm_input], outputs=y, name="CNN_LSTM_Segmentation")
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model