import openslide, glob
import pandas as pd
import os

def generate_label_file(directory_paths, dataset_path, file_name, label_map):
    def list_all_files(directory, targe_path, label_map):
        out1 = []
        out2 = []
        out3 = []
        """列出目录及其所有子目录中的所有文件"""
        for root, dirs, files in os.walk(directory):
            for file in files:
                if ".h5" in file:
                    wsi_path = os.path.join(targe_path, file.replace("h5", "svs"))
                    if wsi_path not in label_map.keys():
                        continue
                    out3.append(
                        label_map[wsi_path]
                    )   
                    out1.append(os.path.abspath(os.path.join(root, file)))
                    out2.append(
                        wsi_path
                    )       
        return list(out1), list(out2), list(out3)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    seg_file_list = []
    wsi_file_list = []
    label_list = []
    for directory_path in directory_paths:
        seg_file_list_i, wsi_file_list_i, label_i = list_all_files(directory_path, directory_paths[directory_path], label_map)
        seg_file_list+=seg_file_list_i
        wsi_file_list+=wsi_file_list_i
        label_list+=label_i
    out = pd.DataFrame({"slide_fp": wsi_file_list, "seg_fp": seg_file_list, "label": label_list})
    out.to_csv(os.path.join(dataset_path, file_name))
    
def generate_pl_bm(
        WSI_dir, 
        save_dir, 
        base_patch_size, 
        target_mag,
        number,
        WSI_name
    ):

    process_list = {} 
    base_mag_csv = {
        "slide_fp": [],
        "base_mag": []
    }
    WSI_name = WSI_name

    if not os.path.exists(os.path.join(save_dir, WSI_name, "csv")):
        os.makedirs(os.path.join(save_dir, WSI_name, "csv"))

    for WSI in glob.glob(WSI_dir+"/*"):
        slide = openslide.open_slide(WSI)
        wsi_name = WSI.split("/")[-1]
        if slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER) == None:
            continue

        base_mag = int(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER))
        target_min_patch_size = int(base_patch_size*(base_mag/target_mag))

        # Update for process_list
        if target_min_patch_size not in process_list:
            process_list[target_min_patch_size] = [wsi_name]
        else:
            process_list[target_min_patch_size].append(wsi_name)

        # Update for bm
        base_mag_csv["slide_fp"].append(WSI)
        base_mag_csv["base_mag"].append(base_mag)

    if number=="all":
        cut_len = len(glob.glob(WSI_dir+"/*"))
    else:
        cut_len = number

        # save bm.csv  
    df = pd.DataFrame(base_mag_csv)[:cut_len]
    df.to_csv(os.path.join(save_dir, f"{WSI_name}/csv/bm.csv"))

    # save patch_size_i process_list.csv
    for k in process_list.keys():
        df = pd.DataFrame({
            "slide_id": process_list[k]
        })[:cut_len]
        df.to_csv(os.path.join(save_dir, f"{WSI_name}/csv/pl_mag{target_mag}x_patch{base_patch_size}_{k}.csv"))
        

def metrics_analysis(input_filepath):
    # 读取csv文件
    df = pd.read_csv(input_filepath)

    # 使用正则表达式去除id中的foldi部分，得到group_id
    df['group_id'] = df['id'].str.replace(r'fold\d', '', regex=True)

    # 根据group_id分组并计算均值和方差
    means = df[[col for col in df.columns if col not in ['id']]].groupby('group_id').mean()
    stds = df[[col for col in df.columns if col not in ['id']]].groupby('group_id').std()
    
    # 去掉'id'列，这样我们只针对其他的列
    cols = [col for col in df.columns if col not in ['id', 'group_id']]

    # 创建新的DataFrame，其中的列为group_id和其他的列，每个列的值为"{均值}±{方差}"的形式
    result_df = pd.DataFrame({'group_id': means.index})
    
    for col in cols:
        result_df_col = []
        for index, mean_row in means.iterrows():
            std_row = stds.loc[index]
            result_df_col.append("{:.4f}".format(mean_row[col])+"±"+"{:.4f}".format(std_row[col]))
        result_df[col] = result_df_col
    
    no_test_col = [col for col in result_df if "test" not in col] + ["test_loss"]
    new_cols = ["group_id"]+[col for col in result_df if col not in no_test_col]
    result_df = result_df[new_cols]
    result_df["group_id"] = result_df["group_id"].str[17:]
    
    # 将结果保存到输出文件路径
    return result_df
