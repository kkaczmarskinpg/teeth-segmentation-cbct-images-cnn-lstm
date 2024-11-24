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
from alive_progress import alive_bar
from keras import Model


def load_dicom(zip_file_path, target_size=(128, 128)):
    slices = []
    with ZipFile(zip_file_path, 'r') as zip_ref:
        with alive_bar(50) as bar:
            for file in natsorted(zip_ref.namelist())[150:200]: # Only 50 slices for testing
                if file.endswith('.dcm'):
                    dcm_file = BytesIO(zip_ref.read(file))
                    ds = dcmread(dcm_file)
                    resized_slice = cv2.resize(ds.pixel_array, target_size, interpolation=cv2.INTER_LINEAR)
                    slices.append(resized_slice.astype("float32")/255.0)
                    bar()
    return np.array(slices)


def load_nifti(file_path, target_size=(128, 128)):
    mask_image = sitk.ReadImage(file_path) 
    mask_array = sitk.GetArrayFromImage(mask_image)

    lower_teeth = (mask_array == 4).astype(np.uint8)
    upper_teeth = (mask_array == 3).astype(np.uint8)
    binary_mask = lower_teeth + upper_teeth
    resized_mask = np.array([cv2.resize(slice_, target_size, interpolation=cv2.INTER_NEAREST) for slice_ in binary_mask])[150:200] # only 50 slices for testing
    return resized_mask

def cbct_data_generator(scan_path: str, masks_path: str, scan_files: list, mask_files: list):
    while True:
        for i in range(0,len(scan_files)):
            cbct_scan = load_dicom(scan_path+scan_files[i])
            mask = load_nifti(masks_path+mask_files[i])

            
            cbct_scan = cbct_scan[..., np.newaxis]
            mask = mask[..., np.newaxis]

            scan_ready = np.expand_dims(cbct_scan, axis=0)
            mask_ready = np.expand_dims(mask, axis=0)

            print(scan_ready.shape, mask_ready.shape)

            yield scan_ready, mask_ready

def plot_image_with_mask(cbct_scan, mask, rows: int = 3, cols: int = 6, start_idx = 150, end_idx: int = 500):
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
    