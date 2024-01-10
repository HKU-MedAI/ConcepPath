# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os, h5py, openslide, glob
from PIL import Image
import numpy as np
import time
import argparse
import pdb
import pandas as pd
import pickle
from utils.heatmap_utils import drawHeatmap
from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler
import cv2

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str, default="/data1/r10user3/TCGA-WSI/LUSC/LUSC",
					help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--target_rp', type=str, default=None)

parser.add_argument("--label_fp", type=str)
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--vlm_model', type = str, default="quilt1m", choices=["clip", "plip", "quilt1m"],
					help='directory to for heatmap processing data')

parser.add_argument('--experiment_rp', type = str, 
					help='root path of experiment')
parser.add_argument('--experiment_name', type = str, 
					help='name of experiment')

parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0, 
					help='downsample level at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')
parser.add_argument('--blur',  default=False, action='store_true')
parser.add_argument('--n_flp', type=int, default=0)
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--n_patch_prompt', type=int, default=26)
parser.add_argument('--is_clam', type=bool, default=False)

def generate_thumbnail(slide_path, heatmap_rp_i, img_size):
    
    slide_ = openslide.OpenSlide(slide_path)
    
    # img = Image.open(glob.glob(thumbnail_path+"/*")[0])
    slide_heatmap_size = img_size # (max(list(img_size)), max(list(img_size)))
    # img.close()
    
    # Generate the thumbnail
    thumbnail = slide_.get_thumbnail(slide_heatmap_size)
    
    # Save the thumbnail
    
    # thumbnail.save(os.path.join(thumbnail_path+"_lung adenocarcinoma", "aa_thumbnail.png"), "PNG")
    thumbnail.save(os.path.join(heatmap_rp_i, "aa_thumbnail.png"), "PNG")
    
    print("Thumbnail save")
    slide_.close()
    
    return thumbnail

def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	# Segment	
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.process_contours(**kwargs)


	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed


def seg_and_patch(source=None, 
                  prompt_map=None, 
                  label_df=None, 
                  mask_heatmap_root_path=None, stitch_heatmap_root_path=None, 
                  heatmap_root_path=None, patch_heatmap_root_path=None, attn_score_root_path=None,
				  patch_size = 256, step_size = 256, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None, 
      			  n_classes=2):

	# print("aaa", heatmap_root_path)
 
	slides = sorted(os.listdir(source))
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]

	label_unique = sorted(list(set(list(label_df["label"]))))
	label_map = dict(zip(
		label_unique, range(len(label_unique))
	))
	prompt_per_class = int(len(list(prompt_map.keys()))/n_classes)
  
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	mask = df['process'] == 1
	process_stack = df[mask]
	# print(process_stack)
	# print(process_stack["label"])

	total = len(process_stack)
 
	custom_downsample = 1
	cmap = "jet"
	alpha = 0.4
	

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.
	heatmap_rp = heatmap_root_path
	attn_score_rp = attn_score_root_path
	

	for i in range(total):
		df.to_csv(os.path.join(heatmap_rp, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
  
		label = process_stack.loc[idx, 'label']
		# label_map = {"luad": "LUAD", "lusc": "LUSC"}
		# label = label_map[label]
  
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)

		if auto_skip and os.path.isfile(os.path.join(patch_heatmap_root_path, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Inialize WSI
		full_path = os.path.join(slide)

		WSI_object = WholeSlideImage(full_path)
  
		slide_name = slide.split("/")[-1].replace(".svs","")
  
		attn_score_fp = os.path.join(attn_score_rp, f"{args.vlm_model}", f"{slide_name}_{args.vlm_model}.pkl")

		coord_fp = label_df[label_df["slide_fp"]==full_path]["seg_fp"].to_list()[0]
		# print(coord_fp)
		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}


			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


		seg_time_elapsed = -1
  
		# patch_prompt_df = pd.read
  
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)
			# mask_file = os.path.join(mask_heatmap_root_path, slide_id+'.pkl') 
			# WSI_object.saveSegmentation(mask_file)
			wsi_ref_downsample = WSI_object.level_downsamples[patch_level]
   
			# print(attn_score_fp)
   
			with open(attn_score_fp, "rb") as file:
				data = pickle.load(file)
    
			with h5py.File(coord_fp, 'r') as file:
    		# 读取数据集
				coords = file['coords'][:]
				patch_size = file["coords"].attrs["patch_size"]
				vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * custom_downsample).astype(int))
    
			# print(data)
			attention_scores = data["att_score"]
			
			heatmap_rp_ = os.path.join(heatmap_rp, "attn_map", args.vlm_model, slide_name)
			if not os.path.exists(heatmap_rp_):
				os.makedirs(heatmap_rp_)
			
			cur_label_i = label_map[label]
			min_, max_ = cur_label_i*prompt_per_class, cur_label_i*prompt_per_class+prompt_per_class
			start = 0
			
   
			for i in range(min_, max_):
				if args.is_clam:
					attention_score_ = attention_scores.cpu().numpy()
				else:
					attention_score_ = attention_scores[:,i].cpu().numpy()
				# print(attention_score_)

				print(prompt_map[i])
				heatmap = drawHeatmap(attention_score_, coords, None, wsi_object=WSI_object, cmap=cmap, alpha=alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=True, blur=args.blur,
					thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
				print(prompt_map[i], " done")
				if start==0:
					thumbnail = generate_thumbnail(full_path, heatmap_rp_, heatmap.size)
					start+=1
				
				heatmap_fp = os.path.join(heatmap_rp_, f"{prompt_map[i]}.png")

				alpha = 0.8
				beta = 1 - alpha
				gamma = 20.0

				image_array = np.array(heatmap)
    
				# 计算最小值和最大值
				min_val = np.min(image_array)
				max_val = np.max(image_array)
    			# 线性归一化到 [0, 1]
				normalized_array = (image_array - min_val) / (max_val - min_val)
    
				# 缩放到 [0, 255] 并转换为整数
				normalized_array = (normalized_array * 255).astype(np.uint8)
    
				# 将归一化后的 NumPy 数组转换回 PIL 图像
				normalized_image = Image.fromarray(normalized_array)
    
				heatmap = cv2.addWeighted(cv2.cvtColor(np.array(normalized_image), cv2.COLOR_RGB2BGR), alpha, cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2BGR), beta, gamma)

				cv2.imwrite(heatmap_fp, heatmap)
    
				del heatmap
				
				if args.is_clam:
					break


if __name__ == '__main__':
    
	args = parser.parse_args()
	
	if args.target_rp==None:
		heatmap_root_path = os.path.join(args.experiment_rp, args.experiment_name, 'output/heatmap')
	else:
		heatmap_root_path = args.target_rp
		if not os.path.exists(heatmap_root_path):
			os.makedirs(heatmap_root_path)
	attn_score_root_path = os.path.join(args.experiment_rp, args.experiment_name, 'output/attn_score')
 
	patch_heatmap_root_path = os.path.join(heatmap_root_path, 'patches')
	mask_heatmap_root_path = os.path.join(heatmap_root_path, 'masks')
	stitch_heatmap_root_path = os.path.join(heatmap_root_path, 'stitches')

	if args.process_list:
		process_list = args.process_list

	else:
		process_list = None

	print('source: ', args.source)
	print('patch_heatmap_root_path: ', patch_heatmap_root_path)
	print('mask_heatmap_root_path: ', mask_heatmap_root_path)
	print('stitch_heatmap_root_path: ', stitch_heatmap_root_path)
	
	directories = {'source': args.source, 
				   'heatmap_root_path': heatmap_root_path,
				   'patch_heatmap_root_path': patch_heatmap_root_path, 
				   'mask_heatmap_root_path' : mask_heatmap_root_path, 
				   'stitch_heatmap_root_path': stitch_heatmap_root_path} 

	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

	print(parameters)
 
	import json
	with open(os.path.join(args.experiment_rp, args.experiment_name, "input/prompt/patch_prompts.json"), 'r', encoding='utf-8') as file:
		patch_level_prompts = json.load(file)
        
	patch_level_prompts_ = []
	for i in range(args.n_classes):
		key_i = list(patch_level_prompts.keys())[i]
		if args.n_flp!=0:
			patch_level_prompts_+=[f"{key_i}'s fully learnable prompt {j}" for j in range(args.n_flp)]
		patch_level_prompts_ += [f"{key_i}'s {k}" for k in list(patch_level_prompts[key_i].keys())]
  
	prompt_map = dict(zip(
		range(len(patch_level_prompts_)), patch_level_prompts_
	))
	

	label_df = pd.read_csv(args.label_fp)
	
	print(heatmap_root_path)
	seg_and_patch(
				source=args.source,
                prompt_map=prompt_map, 
                label_df = label_df,
                heatmap_root_path = heatmap_root_path,
                attn_score_root_path = attn_score_root_path,
                patch_heatmap_root_path=patch_heatmap_root_path,
				patch_size = args.patch_size, 
    			step_size=args.step_size, 
				seg = args.seg,  use_default_params=False, save_mask = True, 
				stitch= args.stitch,
				patch_level=args.patch_level, patch = args.patch,
				process_list = process_list, auto_skip=args.no_auto_skip, n_classes=args.n_classes)
