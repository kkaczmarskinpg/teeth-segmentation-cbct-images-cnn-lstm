import io
import zipfile
import pydicom
import numpy as np
import SimpleITK as sitk
from natsort import natsorted
from alive_progress import alive_bar

def load_dicom(zip_file_path):
    slices = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        with alive_bar(652) as bar:
            for file in natsorted(zip_ref.namelist()):
                if file.endswith('.dcm'):
                    dcm_file = io.BytesIO(zip_ref.read(file))
                    ds = pydicom.dcmread(dcm_file)
                    slices.append(ds.pixel_array)
                    bar()
    return np.array(slices)


def load_nifti(file_path):
    mask_image = sitk.ReadImage(file_path) 
    mask_array = sitk.GetArrayFromImage(mask_image)

    binary_mask = (mask_array == 4).astype(np.uint8)
    return binary_mask