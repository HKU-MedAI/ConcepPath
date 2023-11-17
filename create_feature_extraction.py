from torch.utils.data import Dataset, DataLoader
import torch, os, h5py
import torchvision.transforms as transforms
import pandas as pd
import openslide, argparse
import numpy as np
import pickle


parser = argparse.ArgumentParser(description='seg and patch')

parser.add_argument("--vlm_model", type = str, default="quilt1m", choices=["quilt1m", "plip", "clip"],
                    help="vlm_model")
parser.add_argument('--label_fp', type = str, default="/home/r10user13/TOP/data/datasets/LUNG/subtyping_label.csv",
					help='file path for label file')
parser.add_argument('--batch_size', type = int, default=32,
					help='batch_size')
parser.add_argument('--num_workers', type = int, default=32,
					help='num_workers')
parser.add_argument('--save_rp', type = str, default="/data2/r10user13/",
					help='')
parser.add_argument('--base_mag', type = int, default=20,
					help='')
parser.add_argument('--base_patch_size', type = int, default=448,
					help='')


# seg_patches_fp_path = "/home/r10user13/Capstone/TOP/experiment/LUNG/seg_patches_fp.csv"
# label_path = "/home/r10user13/Capstone/TOP/experiment/LUNG/label.csv"


class TOPDataset(Dataset):
    def __init__(self, coords, slide, patch_level, transform, patch_size):

        self.slide = slide
        self.patch_level = patch_level
        self.coords = coords
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):

        coord = self.coords[idx]
        img = self.slide.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img = img.resize((224,224))

        return self.transform(img)
    

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    data = {}
    df_seg_fp = pd.read_csv(args.label_fp)

    # top_transform = transforms.ToTensor()
    # top_transform = transforms.Compose([
	# 	# transforms.RandomHorizontalFlip(),
	# 	transforms.ToTensor(),
	# 	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	# ]),
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vlm_model = args.vlm_model
    
    if vlm_model == "clip":
        from clip import clip
        clip_model, preprocess = clip.load('ViT-B/16', device=device)
        model = clip_model.visual
        
    elif vlm_model == "plip":
        from clip import clip
        from transformers import CLIPModel, CLIPProcessor
        _, preprocess = clip.load('ViT-B/16', device=device)
        clip_model = CLIPModel.from_pretrained("vinid/plip").to(device)
        model = clip_model
        
    elif vlm_model == "quilt1m":
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        clip_model = clip_model.to(device)
        model = clip_model.visual

    
    model.eval()

    # transform = transforms.ToTensor()
    
    l,_ = df_seg_fp.shape
    i = 0
    batch_size = args.batch_size

    for row in df_seg_fp.itertuples(index=True, name='Pandas'):
        # print(row.Index, row.A, row.B)

        file_path, wsi_path = row.seg_fp, row.slide_fp
        patch_size = int(file_path.split("/")[-3].split("_")[-1])
        # print(file_path, patch_size)
        
        wsi_name = wsi_path.split("/")[-1].replace(".svs","")
        dataset_name = args.label_fp.split("/")[-2].lower()
        
        save_rp_ = os.path.join(args.save_rp, f"{dataset_name}_{args.vlm_model}_{args.base_mag}x_{args.base_patch_size}")
        if not os.path.exists(save_rp_):
            os.makedirs(save_rp_)
            
        out_fp = os.path.join(save_rp_, f'{wsi_name}.pkl')
        
        if os.path.exists(out_fp):
            continue
        
        else:
            # print("no jump")
            
            try:
                data = {}
        
                print(f"process: {wsi_path}")
        
                patch_size = int(file_path.split("/")[-3].split("_")[-1])
                slide = openslide.open_slide(wsi_path)
                h5_content = h5py.File(file_path,'r')
                patch_level = h5_content["coords"].attrs['patch_level']
                coords = h5_content["coords"][:]

                dataset = TOPDataset(
                    coords, slide, patch_level, preprocess, patch_size
                )

                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

                output = []
                for image in data_loader:
                    
                    input = image.to(device) 
                    
                    with torch.no_grad():
                        if vlm_model == "plip":
                            out = model.visual_projection(model.vision_model(input).pooler_output)
                        elif vlm_model == "clip":
                            # print("before clip output")
                            out = model(input.type(clip_model.dtype))
                            # print("after clip output")
                        else:
                            # print("before output")
                            out = model(input)
                            # print("after output")
                    
                    output.append(out.cpu().detach().numpy())
                    
                data["data"] = np.concatenate(output, axis=0)
        
                slide.close()

                with open(out_fp, 'wb') as file:
                    pickle.dump(data, file)
                    
            except:
                print(f"{wsi_name} is not normal")
           
         
        i+=1    
        print(f"{vlm_model} complete: {i}/{l}")
        # break