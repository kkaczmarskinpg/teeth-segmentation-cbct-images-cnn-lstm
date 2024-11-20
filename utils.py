import io
import zipfile
import pydicom
import numpy as np
import SimpleITK as sitk
from natsort import natsorted
from alive_progress import alive_bar
import cv2

def load_dicom(zip_file_path, target_size=(256, 256)):
    slices = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        with alive_bar(652) as bar:
            for file in natsorted(zip_ref.namelist()):
                if file.endswith('.dcm'):
                    dcm_file = io.BytesIO(zip_ref.read(file))
                    ds = pydicom.dcmread(dcm_file)
                    resized_slice = cv2.resize(ds.pixel_array, target_size, interpolation=cv2.INTER_LINEAR)
                    slices.append(resized_slice)
                    bar()
    return np.array(slices)


def load_nifti(file_path, target_size=(256, 256)):
    mask_image = sitk.ReadImage(file_path) 
    mask_array = sitk.GetArrayFromImage(mask_image)

    binary_mask = (mask_array == 4).astype(np.uint8)
    resized_mask = np.array([cv2.resize(slice_, target_size, interpolation=cv2.INTER_NEAREST) for slice_ in binary_mask])
    return resized_mask

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