from keras import layers, models, Input, losses
from metrics import dice_coefficient, jaccard_index, dice_loss

def create_cnn_lstm_model(image_shape=(64, 64), num_slices=304):
    """
    Create a CNN-LSTM model for slice-by-slice 3D image segmentation.

    Parameters:
        image_shape: tuple, the shape of each 2D slice (height, width).
        num_slices: int, number of slices in the 3D image.

    Returns:
        model: Compiled Keras model.
    """
    # Input for the 3D image (slice sequence)
    input_layer = Input(shape=(num_slices, image_shape[0], image_shape[1], 1))  # (num_slices, H, W, 1)

    # CNN feature extractor
    cnn_features = layers.TimeDistributed(
        layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')
    )(input_layer)

    cnn_features = layers.TimeDistributed(
        layers.MaxPooling2D(pool_size=2)
    )(cnn_features)

    cnn_features = layers.TimeDistributed(
        layers.Dropout(rate=0.3)
    )(cnn_features)

    cnn_features = layers.TimeDistributed(
        layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')
    )(cnn_features)

    cnn_features = layers.TimeDistributed(
        layers.MaxPooling2D(pool_size=2)
    )(cnn_features)

    cnn_features = layers.TimeDistributed(
        layers.Dropout(rate=0.3)
    )(cnn_features)

    cnn_features = layers.TimeDistributed(
        layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')
    )(cnn_features)

    cnn_features = layers.TimeDistributed(
        layers.Flatten()
    )(cnn_features)  # Shape: (num_slices, features)

    # LSTM for sequence learning
    lstm = layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(cnn_features)
    lstm = layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(lstm)

    # Dense layer dependent on image_shape
    dense_output_size = (image_shape[0] // 4) * (image_shape[1] // 4) * 128
    dense = layers.TimeDistributed(
        layers.Dense(dense_output_size, activation='relu')
    )(lstm)

    # Reshape to match spatial dimensions
    reshaped = layers.Reshape((num_slices, image_shape[0] // 4, image_shape[1] // 4, 128))(dense)

    # Deconvolutional layer to upsample back to original size
    deconv = layers.TimeDistributed(
        layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation='relu')
    )(reshaped)  # Upsampling step 1 -> (H/2, W/2)

    deconv = layers.TimeDistributed(
        layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation='relu')
    )(deconv)  # Upsampling step 2 -> (H, W)

    # Final 1D convolution activation layer
    final_conv = layers.TimeDistributed(
        layers.Conv2D(1, kernel_size=1, activation="sigmoid")
    )(deconv)

    # Define the model
    model = models.Model(inputs=input_layer, outputs=final_conv, name="CNN_LSTM_Deconv_Segmentation")

    # Compile the model
    model.compile(
        optimizer='adam', 
        loss=losses.Dice(), 
        metrics=[
            jaccard_index,
            dice_coefficient,
        ]
    )

    return model