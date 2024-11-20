import os
from utils import load_dicom, load_nifti, cbct_data_generator, plot_image_with_mask
import matplotlib.pyplot as plt
from model import create_cnn_lstm_model

photos_path = "./photos/"
masks_path = "./masks/"

photos_names = os.listdir(photos_path)
masks_names = os.listdir(masks_path)

batch_size = 1
epochs = 20
model = create_cnn_lstm_model()
model.summary()


train_gen = cbct_data_generator(photos_path, masks_path, photos_names, masks_names, batch_size)

model.fit(
    train_gen, 
    steps_per_epoch=len(photos_names)//batch_size,
    epochs=epochs)

"""
scan = load_dicom(photos_path+photos_names[0])
mask = load_nifti(masks_path+masks_names[0])
plot_image_with_mask(scan,mask)
"""