from train import train
from test import inference
import os, torch, warnings, argparse, random
from utils.utils import set_random_seed
import numpy as np

# 忽略所有警告消息
warnings.filterwarnings("ignore")

# devices = os.environ.get('CUDA_VISIBLE_DEVICES', '1')
# print(f'CUDA visible devices: {devices}')
# torch.cuda.set_device(4)

def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_classes', type=int, default=2, 
                        help='number of classes')
    parser.add_argument('--orth_ratio', type=float, default=1)
    
    parser.add_argument("--weighted_type", type=str, default="ori")
    
    parser.add_argument('--n_flp', type=int, default=0, 
                        help='number of classes')
    
    parser.add_argument('--n_sp', type=int, default=0, 
                        help='number of classes')
    
    parser.add_argument('--num_patch_prompt', type=int, default=26)
    
    parser.add_argument('--seed', type=int, default=2023, 
                        help='random seed')
    
    parser.add_argument('--n_ctx', type=int, default=12)
    
    parser.add_argument('--experiment_rp', type=str, default="/home/r10user13/TOP/experiment")
    
    parser.add_argument('--experiment_name', type=str, default="lung_typing_full_train")
    
    parser.add_argument('--task_type', type=str, default="train", 
                        help='number of classes')

    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                        help='shot number or None')
    
    parser.add_argument("--mask_ratio", type=float, default=0.0)
    
    parser.add_argument('--attn_type', type=str, default="",  choices=["SA", "CA", "SC", ""],
                        help='type of attention mechanism')
    
    parser.add_argument("--is_shared", default=False, action='store_true')
    
    parser.add_argument("--only_learn", default=False, action='store_true')
    
    parser.add_argument('--num_epochs', type=int, default=20, 
                        help='number of epochs')
    
    parser.add_argument("--is_adapted", default=False, action='store_true')
    parser.add_argument("--tr_ratio", type=float, default=0)
    
    
    parser.add_argument('--proj_name', type=str, default="top", 
                        help='name of project')
    
    parser.add_argument('--vlm_name', type=str, default="quilt1m", choices=["clip", "plip", "quilt1m", "rop_quilt1m", "top_ori"],
                        help='name of project')
    
    parser.add_argument('--feature_rp', type=str, default='/data1/r10user13/TOP/lung_quilt1m_20x_448/', 
                        help='path of feature file')
    
    parser.add_argument('--fold_name', type=str, default="fold3", 
                        help='fold_name')
    
    parser.add_argument('--model_fp', type=str, default='', 
                        help='path of patch prompt file')
    
    parser.add_argument('--early_stop', default=False, action='store_true',
                        help='path of split file')

    parser.add_argument('--clip_base_model', type=str, default='ViT-B/16', choices= ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16'], 
                        help='path of patch prompt file')
    
    parser.add_argument('--save_pred_detail', type=bool, default=False, 
                        help='save prediction details to ./output/metrics')
    
    args, _ = parser.parse_known_args()
    return args
    
if __name__ == '__main__':
    try:
        args = get_params()
        set_random_seed(args.seed)
        print("========================================Parameters========================================")
        for arg in vars(args):
            print(f'{arg}: {getattr(args, arg)}')
        print("==========================================================================================")
        # setup_seed(args.seed)
        if args.task_type == "train":
            train(args)
        elif args.task_type == "test":
            # print("test")
            inference(args)
    except Exception as exception:
        raise
