import torch
#import torch.nn as nn
from tqdm import tqdm
import numpy as np

import sys, os
sys.path.append("..")
from transfer_vgg_model import VGG19
from preprocess import load_image_as_tensor, save_tensor_as_image



sys.path.append("baseline-gram-matrix")
sys.path.append("patch-based")
sys.path.append("feedforward-cnn-gram-matrix")
import GramMatrixLossTransfer
import PatchBasedTransfer
from image_transform_net import ImageTransformerRef

RESULTS_DIR = "results"
CONTENT_IM_DIR = "data/content-images"
STYLE_IM_DIR = "data/style-images"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = ImageTransformerRef().to(device)
model.load_state_dict(torch.load("feedforward-cnn-gram-matrix/model.pt"))
model.eval()

for content_im_path in os.listdir(CONTENT_IM_DIR):
    for style_im_path in os.listdir(STYLE_IM_DIR):
        content_split = content_im_path.split(".")
        style_split = style_im_path.split(".")[0].split("-")
        res_name = content_split[0]+"-"+style_split[0]+"-"+style_split[1]+".jpg"
        
        # BASELINE GRAM MATRIX RESULTS
        x_baseline = GramMatrixLossTransfer.transfer(os.path.join(CONTENT_IM_DIR, content_im_path), 
                                              os.path.join(STYLE_IM_DIR, style_im_path), 
                                              num_iter=2000, alpha=1, beta=1e7)
        save_path = os.path.join(RESULTS_DIR, content_split[0], "baseline")
        if not os.path.isdir(save_path):
            print("NO FUNSD")
            os.system("mkdir -p " + save_path)
        save_tensor_as_image(x_baseline, os.path.join(save_path, res_name))
        
        # BASELINE CNN APPROX
        x_baseline_cnn, shape = load_image_as_tensor(os.path.join(CONTENT_IM_DIR, content_im_path), l=256)
        x_baseline_cnn = model(x_baseline_cnn.to(device))
        save_path = os.path.join(RESULTS_DIR, content_split[0], "baseline-cnn")
        if not os.path.isdir(save_path):
            print("NO FUNSD")
            os.system("mkdir -p " + save_path)
        save_tensor_as_image(x_baseline_cnn.cpu(), os.path.join(save_path, res_name))
        
        # PATCH-BASED RESULTS
        x_patch = PatchBasedTransfer.transfer(os.path.join(CONTENT_IM_DIR, content_im_path), 
                                              os.path.join(STYLE_IM_DIR, style_im_path), 
                                              layer_num=11, patch_size=5,stride=3, 
                                              num_iter=1000, _lambda=1e-2)
        save_path = os.path.join(RESULTS_DIR, content_split[0], "patch")
        if not os.path.isdir(save_path):
            print("NO FUNSD")
            os.system("mkdir -p " + save_path)
        save_tensor_as_image(x_patch, os.path.join(save_path, res_name))
        
