import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import datetime as dt
import json

from io import BytesIO
from zipfile  import ZipFile
from pydicom import dcmread
from natsort import natsorted
from keras import Model
from sklearn.model_selection import train_test_split
from itertools import cycle

def pad_scan_mask(image, target_slices=304):
    num_slices = image.shape[0]
    if num_slices < target_slices:
        pad_width = target_slices - num_slices
        padded_image = np.pad(image, ((0, pad_width), (0, 0), (0, 0)), mode='constant')
        return padded_image
    else:
        cropped_image = image[:target_slices]
        return cropped_image

def load_dicom(zip_file_path, target_size=(32, 32)):
    slices = []
    with ZipFile(zip_file_path, 'r') as zip_ref:
    
        for file in natsorted(zip_ref.namelist())[150:200]: # Only 50 slices for testing
            if file.endswith('.dcm'):
                dcm_file = BytesIO(zip_ref.read(file))
                ds = dcmread(dcm_file)
                resized_slice = cv2.resize(ds.pixel_array, target_size, interpolation=cv2.INTER_LINEAR)
                slices.append(resized_slice.astype("float32")/255.0)
                    
    return np.array(slices)

def load_nifti_cbct_scan(file_path, target_size=(16, 16), target_slices=304):
    photo = sitk.ReadImage(file_path)
    photo_array = sitk.GetArrayFromImage(photo)
    photo_array = (photo_array - np.min(photo_array))/(np.max(photo_array)-np.min(photo_array))
    if  target_size != -1:
        photo_array = np.array([cv2.resize(slice_, target_size, interpolation=cv2.INTER_NEAREST) for slice_ in photo_array]) # only 50 slices for test
    photo_array = pad_scan_mask(photo_array, target_slices)
    return photo_array

def load_nifti_mask(file_path, target_size=(16, 16), target_slices=304):
    mask_image = sitk.ReadImage(file_path) 
    mask_array = sitk.GetArrayFromImage(mask_image)

    if target_size != -1:
        mask_array = np.array([cv2.resize(slice_, target_size, interpolation=cv2.INTER_NEAREST) for slice_ in mask_array]) # only 50 slices for testing
    mask_array = pad_scan_mask(mask_array, target_slices)
    return mask_array
    
def cbct_data_generator(scan_path: str, masks_path: str, scan_names: list):
    for name in cycle(scan_names):  
        cbct_scan_file = name + "_0000.nii.gz"
        mask_file = name + ".nii.gz"

        cbct_scan = load_nifti_cbct_scan(scan_path + cbct_scan_file)
        mask = load_nifti_mask(masks_path + mask_file)

        cbct_scan = cbct_scan[..., np.newaxis]  
        mask = mask[..., np.newaxis]   

        scan_ready = np.expand_dims(cbct_scan, axis=0)
        mask_ready = np.expand_dims(mask, axis=0)

        print(f"{cbct_scan_file}: {scan_ready.shape}, {mask_file}: {mask_ready.shape}")

        yield scan_ready, mask_ready

def plot_image_with_mask_grid(cbct_scan, mask, rows: int = 3, cols: int = 6, start_idx = 150, end_idx: int = 500):
    num_of_slices = rows*cols

    idx_increment = (end_idx-start_idx)//num_of_slices

    idx = start_idx

    plt.figure(figsize=(16,8))
    for i in range(num_of_slices):
        plt.subplot(rows, cols, i+1)
        plt.imshow(cbct_scan[idx,:,:],cmap="gray")

        teeth = np.ma.masked_where(mask[idx,:,:] == 0, mask[idx,:,:])
        plt.imshow(teeth, cmap="Set1")

        plt.title(str(idx))
        plt.axis("off")
        idx += idx_increment

    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.show()
    return

def plot_image_with_mask(cbct_scan, mask, idx_slice):
    plt.figure(figsize=(8,8))
    plt.imshow(cbct_scan[idx_slice,:,:],cmap='gray')
    teeth = np.ma.masked_where(mask[idx_slice,:,:] == 0, mask[idx_slice,:,:])
    plt.imshow(teeth, cmap="Set1")
    plt.title(str(idx_slice))
    plt.axis("off")
    plt.show()
    return

def save_model(model: Model):
    os.makedirs("models", exist_ok=True)
    time = dt.datetime.now().strftime("%m-%d-%Y-%H-%M")
    
    model_path = "models/CNN-LSTM-teeth-segmentation-model-"+time+".keras"
    print(f"Saving model...")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    history_path = "models/history-"+time+".json"
    print("Saving history...")
    with open(history_path, 'w') as f:
        json.dump(model.history.history, f)
    print(f"History saved to {history_path}")
    

def split_train_val_test(scan_names: list, 
                         train_size: float,
                         val_size: float,
                         test_size: float):
    train_photos_names, X_temp = train_test_split(scan_names, train_size=train_size)
    val_photos_names, test_photos_names = train_test_split(X_temp, train_size=(val_size/(val_size+test_size)))

    return train_photos_names, val_photos_names, test_photos_names