import sys
sys.path.append(".")
sys.path.append("..")
import os
import torch
import clip
import json
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import h5py
import random
import streamlit as st
from configs import data_configs
from models.t2s import t2s
from datasets.images_dataset import ImagesDataset
from streamlit_gui.streamlit_utils import text2image, text2NN
    

def main():
    
    st.title("Fast text2StyleGAN")
    net, clip_model, transform, CLIPemb, train_dataset, preprocess = prepare_models()
    
    with st.sidebar:
        mode = st.selectbox("Which mode to run?", ('KNN','Text only','Text+Image'))
        user_input = st.text_area("Text prompt:", "")
        image_file = None
        
        if mode == 'Text+Image':
            image_file = st.file_uploader("Image input", type=["jpg","png",'jpeg'])
        elif mode == 'KNN':
            K = st.number_input("K (# of candidates for KNN):",value=50,step=1, format='%i')
            M = st.number_input("M (# of samples randomly selected from K candidates):",value=10,step=1, format='%i')
            diri = st.number_input("alpha (parameter of Dirichlet distribution):",value=0.2,step=0.1, format='%f')
        clicked = st.button('Generate')
    if image_file is not None:
        st.sidebar.image(image_file, use_column_width=True)
    if clicked:
        if mode == 'Text only':
            results = text2image(net, clip_model, [user_input], VAE_img='random')
        elif mode == 'KNN':
            results = text2NN(clip_model, net, CLIPemb, user_input, num_nn=K,num_choice=M, alpha=diri)
        elif mode == 'Text+Image':
            image = Image.open(image_file)
            image = image.convert('RGB')
            image = transform(image)
            results = text2image(net, clip_model, [user_input], VAE_img=[image])
        for im_i in range(len(results)):
            st.image(results[im_i],width=None, output_format='PNG')
@st.cache
def prepare_models():
    # Change trial and iteration_i to your own model
    trial = "cvae_v7"
    iteration_i = 500000
    truncation = 0.9
    f = open(f'logs/{trial}/opt.json')
    data = json.load(f)
    f.close()

    device = 'cuda'
    opts = argparse.Namespace(**data)
    opts.truncation=truncation
    opts.device = device
    opts.checkpoint_path = f'logs/{trial}/checkpoints/iteration_{iteration_i}.pt'

    net = t2s(opts).to(device)
    net = net.eval()
    clip_model, preprocess = clip.load("ViT-B/16", device='cuda')
    clip_model = clip_model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    CLIPemb_file = h5py.File('data/FFHQ.hdf5', "r")
    CLIPemb = np.array(CLIPemb_file["image"])

    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
								transform=transforms_dict['transform_gt_train'],
								opts=opts)


    return net, clip_model, transform, CLIPemb, train_dataset, preprocess

def find_img_NN(image_file, num_NN, clip_model, clip_emb, clip_preprocess):
    image = clip_preprocess(Image.open(image_file)).unsqueeze(0).to('cuda')
    
    normalized_emb = torch.tensor(clip_emb/np.linalg.norm(clip_emb, axis=1, keepdims=True)).to('cuda')
    with torch.no_grad():
        image_features = clip_model.encode_image(image).float()
        image_features = image_features/image_features.norm(dim=1,keepdim=True)
    NN_mat = image_features @ normalized_emb.T # B x 70000
    distances, indices = torch.topk(NN_mat,num_NN,dim=1)
    indices = indices.cpu().numpy()
    return indices


if __name__ == '__main__':
	main()