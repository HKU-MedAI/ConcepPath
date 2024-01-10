"""_summary_
调整splits csv文件
补充full train实验
"""

# split
from sklearn.model_selection import train_test_split
import os, shutil, random
import pandas as pd

luad_data = list(pd.read_csv("/home/r10user13/Capstone/TOP/experiment/LUAD/csv/bm.csv")["slide_path"])
luad_label = ["luad"]*len(luad_data)

lusc_data = list(pd.read_csv("/home/r10user13/Capstone/TOP/experiment/LUSC/csv/bm.csv")["slide_path"])
lusc_label = ["lusc"]*len(lusc_data)

def random_split(data, label, seed=2023):
    # train, valid, test = 0.8, 0.1, 0.1
    train_data, valntest_data, train_labels, valntest_labels = train_test_split(
    data, label, test_size=0.2, stratify=label, random_state=seed)
    val_data, test_data, val_labels, test_labels = train_test_split(
    valntest_data, valntest_labels, test_size=0.5, stratify=valntest_labels, random_state=seed)
    return {"train": {"data": train_data, "label": train_labels},
            "val": {"data": val_data, "label": val_labels},
            "test": {"data": test_data, "label": test_labels},}
    # return {"train": {"data": train_data, "label": train_labels},
    #     "val": {"data": [], "label": []},
    #     "test": {"data": valntest_data, "label": valntest_labels},}

def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
            
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            
         
out_dir = "/home/r10user13/Capstone/TOP/experiment/LUNG/split"

sample_num = 5

out_root_path = os.path.join(out_dir, f"full")

if os.path.exists(out_root_path):
    clear_directory(out_root_path)

if not os.path.exists(out_root_path):
    os.makedirs(out_root_path)

for i in range(sample_num):
    
    luad_splits = random_split(luad_data, luad_label, seed=2023-i)
    lusc_splits = random_split(lusc_data, lusc_label, seed=2023-i)
    
    train_data = luad_splits["train"]["data"] + lusc_splits["train"]["data"]
    test_data = luad_splits["test"]["data"] + lusc_splits["test"]["data"]
    val_data = luad_splits["val"]["data"] + lusc_splits["val"]["data"]
    print(f"sample #{i}: train {len(train_data)}, test {len(test_data)}, val {len(val_data)}")
    
    out = pd.DataFrame({
        "data_path": train_data + test_data + val_data, 
        "type": ["train"]*len(train_data) + ["test"]*len(test_data) + ["val"]*len(val_data)
    })
    out.to_csv(os.path.join(out_root_path, f"{str(len(os.listdir(out_root_path)))}.csv"))

