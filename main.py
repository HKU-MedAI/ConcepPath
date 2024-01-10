from train import train
from test import inference
import os, torch, warnings, argparse, random
from utils.utils import set_random_seed
import numpy as np

warnings.filterwarnings("ignore")

def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_classes', type=int, default=2, 
                        help='number of classes')
    
    parser.add_argument('--orth_ratio', type=float, default=1,
                        help='ratio of orthogonal loss')
    
    parser.add_argument('--n_ddp', type=int, default=0, 
                        help='number of data-driven concept')
    
    parser.add_argument('--num_patch_prompt', type=int, default=26,
                        help='number of patch prompt')
    
    parser.add_argument('--seed', type=int, default=2023, 
                        help='random seed')
    
    parser.add_argument('--n_ctx', type=int, default=16,
                        help='number of learnable prompt')
    
    parser.add_argument('--experiment_rp', type=str, default="/home/r10user13/TOP/experiment",
                        help='root path of experiment')
    
    parser.add_argument('--experiment_name', type=str, default="lung_subtyping",
                        help='name of experiment')
    
    parser.add_argument('--task_type', type=str, default="train", 
                        help='type of task (train/test)')

    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                        help='learning rate')
    
    parser.add_argument("--is_shared", default=False, action='store_true')
    
    parser.add_argument('--num_epochs', type=int, default=50, 
                        help='number of epochs')
    
    parser.add_argument("--adapted_ratio", type=float, default=0,
                        help="ratio of adapter")
    
    parser.add_argument('--vlm_name', type=str, default="quilt1m", choices=["clip", "plip", "quilt1m"],
                        help='name of VLM')
    
    parser.add_argument('--feature_rp', type=str, default='/data1/r10user13/TOP/lung_quilt1m_20x_448/', 
                        help='path of feature files')
    
    parser.add_argument('--fold_name', type=str, default="fold3", 
                        help='fold name')
    
    parser.add_argument('--model_fp', type=str, default='', 
                        help='path of trained model')
    
    parser.add_argument('--early_stop', default=False, action='store_true',
                        help='use early stop or not')
    
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
            inference(args)
    except Exception as exception:
        raise
