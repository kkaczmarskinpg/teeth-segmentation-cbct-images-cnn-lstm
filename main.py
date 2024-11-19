import os
from utils import load_dicom, load_nifti

photos_path = "./photos/"
masks_path = "./masks/"

photos_names = os.listdir(photos_path)
masks_names = os.listdir(masks_path)

cbct_scan = load_dicom(photos_path+photos_names[0])
mask= load_nifti(masks_path+masks_names[0])

cbct_scan = cbct_scan.astype('float32')/255.0
