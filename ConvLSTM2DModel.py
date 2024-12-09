from keras import layers, Sequential, losses
from metrics import dice_coefficient, jaccard_index, dice_loss
import tensorflow as tf

def create_cnn_convlstm2d_model(image_shape=(128, 128), num_slices=304):
    model = Sequential()

    
    input_shape = (num_slices, image_shape[0], image_shape[1], 1)


    model.add(layers.ConvLSTM2D(filters=16, 
                                kernel_size=(3, 3),
                                activation="tanh",
                                recurrent_dropout=0.2,
                                padding='same',
                                return_sequences=True,
                                input_shape=input_shape))
    model.add(layers.MaxPool3D(pool_size=(1, 2, 2), padding='same'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))

    model.add(layers.ConvLSTM2D(filters=32, 
                                kernel_size=(3, 3),
                                activation="tanh",
                                recurrent_dropout=0.2,
                                padding='same',
                                return_sequences=True))
    model.add(layers.MaxPool3D(pool_size=(1, 2, 2), padding='same'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))

    model.add(layers.ConvLSTM2D(filters=64, 
                                kernel_size=(3, 3),
                                activation="tanh",
                                recurrent_dropout=0.2,
                                padding='same',
                                return_sequences=True))
    model.add(layers.MaxPool3D(pool_size=(1, 2, 2), padding='same'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))

    
    model.add(layers.TimeDistributed(layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')))
    model.add(layers.TimeDistributed(layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')))
    model.add(layers.TimeDistributed(layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')))
    
   
    model.add(layers.TimeDistributed(layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same')))

    
    model.compile(
        optimizer='adam',
        loss = losses.BinaryCrossentropy(),
        #loss=tf.keras.losses.Dice(
        #    reduction="sum_over_batch_size", name="dice", axis=None, dtype=None),
        metrics=[dice_coefficient, jaccard_index]
    )

    return model