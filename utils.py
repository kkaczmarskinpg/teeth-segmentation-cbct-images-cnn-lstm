import io
import zipfile
import pydicom
import numpy as np
import SimpleITK as sitk
from natsort import natsorted
from alive_progress import alive_bar
import cv2
import matplotlib.pyplot as plt

def load_dicom(zip_file_path, target_size=(256, 256)):
    slices = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        with alive_bar(652) as bar:
            for file in natsorted(zip_ref.namelist()):
                if file.endswith('.dcm'):
                    dcm_file = io.BytesIO(zip_ref.read(file))
                    ds = pydicom.dcmread(dcm_file)
                    #resized_slice = cv2.resize(ds.pixel_array, target_size, interpolation=cv2.INTER_LINEAR)
                    slices.append(ds.pixel_array)
                    bar()
    return np.array(slices)


def load_nifti(file_path, target_size=(256, 256)):
    mask_image = sitk.ReadImage(file_path) 
    mask_array = sitk.GetArrayFromImage(mask_image)

    lower_teeth = (mask_array == 4).astype(np.uint8)
    upper_teeth = (mask_array == 3).astype(np.uint8)
    binary_mask = lower_teeth + upper_teeth
    #resized_mask = np.array([cv2.resize(slice_, target_size, interpolation=cv2.INTER_NEAREST) for slice_ in binary_mask])
    return binary_mask

def cbct_data_generator(scan_path: str, masks_path: str, scan_names: list, mask_names: list, batch_size: int = 1):
    while True:
        for i in range(len(scan_names)):
            cbct_scan = load_dicom(scan_path+scan_names[i])
            mask = load_nifti(masks_path+mask_names[i])

            cbct_scan = cbct_scan.astype('float32')/255.0
            cbct_scan = cbct_scan[..., np.newaxis]

            mask = mask[..., np.newaxis]
            print(cbct_scan.shape, mask.shape)
            yield np.expand_dims(cbct_scan, axis=0), np.expand_dims(mask, axis=0)

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
    
    