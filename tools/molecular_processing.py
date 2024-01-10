import os
import pandas as pd

from utils.processing_utils import generate_pl_bm

raw_labels = pd.read_csv("data/molecular_label_raw.csv")
patients = raw_labels["TCGA barcode"].to_list()
subtypes = raw_labels["Molecular Subtype"].to_list()
patient_subtype_tb = {k: v for k, v in zip(patients, subtypes)}

slide_subtype_tb = dict()
for root, dirs, files in os.walk("/data1/r10user3/TCGA-WSI/"):
    for file in files:
        patient = file[:12]
        full_pth = os.path.join(root, file)
        if patient in patients:
            slide_subtype_tb[file] = patient_subtype_tb[patient]

# # generate data/raw_data/csv
# generate_pl_bm(
#         WSI_dir="/data1/r10user3/TCGA-WSI/STAD/STAD", 
#         save_dir="/home/r10user13/TOP/data/raw_data/", 
#         base_patch_size=448, 
#         target_mag=20,
#         number="all",
#         WSI_name="STAD"
# )

