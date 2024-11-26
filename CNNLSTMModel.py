from keras import layers, models, Input, metrics, Sequential, losses

def create_cnn_lstm_model(target_size=(128, 128)):
    """
    Create a CNN-LSTM model for slice-by-slice 3D image segmentation.

    Parameters:
        target_size: tuple, the shape of each 2D slice (height, width).
        num_slices: int, number of slices in the 3D image.

    Returns:
        model: Compiled Keras model.
    """
    # Input for the 3D image (slice sequence)
    input_layer = Input(shape=(None, target_size[0], target_size[1], 1))  # (num_slices, H, W, 1)
    
    # CNN for feature extraction (shared across slices)
    cnn_input = Input(shape=(target_size[0], target_size[1], 1))  # Single 2D slice
    x = layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(cnn_input)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)  # Flatten the features for LSTM input
    cnn_model = models.Model(inputs=cnn_input, outputs=x, name="CNN_FeatureExtractor")

    # Apply CNN to each slice using TimeDistributed
    time_distributed = layers.TimeDistributed(cnn_model)(input_layer)  # Shape: (num_slices, features)

    # LSTM for sequence learning
    lstm = layers.LSTM(128, return_sequences=True)(time_distributed)
    lstm = layers.LSTM(64, return_sequences=True)(lstm)

    # Dense layer to predict pixel-wise probabilities for each slice
    dense = layers.TimeDistributed(
        layers.Dense(target_size[0] * target_size[1], activation='sigmoid')
    )(lstm)

    # Reshape back to spatial dimensions for segmentation mask
    output_layer = layers.Reshape((-1, target_size[0], target_size[1], 1))(dense)

    # Define the model
    model = models.Model(inputs=input_layer, outputs=output_layer, name="CNN_LSTM_Segmentation")

    # Compile the model

    model.compile(
        optimizer='adam', 
        loss=losses.Dice(reduction='sum_over_batch_size', name='dice'), 
        metrics=[
            'accuracy', 
            metrics.BinaryIoU(target_class_ids=[1]),
            losses.Dice(reduction='sum_over_batch_size', name='dice')
            ]
        )

    return model