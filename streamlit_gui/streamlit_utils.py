import sys
sys.path.append(".")
sys.path.append("..")
import torch
import clip
import numpy as np
from utils import common
from scipy.spatial import distance_matrix

def text2image(net, clip_model, prompts, VAE_img='random'):
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to('cuda')
        text_features = clip_model.encode_text(text_tokens).float().to('cuda')
    
    if VAE_img == 'random':
        x_hat,_,_,_, latent = net.forward(text_features,img='random', resize=False, return_latents=True)      
    else:
        xs = torch.stack(VAE_img,axis = 0).to('cuda')
        x_hat,_,_,_, latent = net.forward(text_features,img=xs, resize=False, return_latents=True)
    
    display_count = x_hat.shape[0]
    results = []
    for i in range(display_count):
        img = np.asarray(common.tensor2im(x_hat[i]))
        results.append(img)
    return results   


def get_NN(text, clip_model, clip_emb, num_nn):
    normalized_emb = torch.tensor(clip_emb/np.linalg.norm(clip_emb, axis=1, keepdims=True)).to('cuda')
    text_tokens = clip.tokenize([text]).to('cuda')
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()
        text_features = text_features/text_features.norm(dim=1,keepdim=True)
    NN_mat = text_features @ normalized_emb.T # B x 70000
    distances, indices = torch.topk(NN_mat,num_nn,dim=1)
    return distances.cpu().numpy(), indices.cpu().numpy()


def get_data(idx_list, CLIPemb):
    es = []
    for idx in idx_list:
        es.append(torch.tensor(CLIPemb[idx]))
    es = torch.stack(es,axis = 0)
    with torch.no_grad():
        es = es.to('cuda')
    return es


def get_furthest(candidates, indices, num_choice):
    normalized_emb = candidates/np.linalg.norm(candidates, axis=1, keepdims=True)
    dist = distance_matrix(normalized_emb, normalized_emb)
    assert dist.shape[0] == candidates.shape[0] and dist.shape[0] == dist.shape[1]
    selected = np.empty(num_choice)
    selected[0] = np.random.choice(np.arange(len(indices)), 1, replace=False)[0]
    for i in range(1,num_choice):
        selected[i] = dist[:,selected[:i].astype(int)].min(1).argmax()
    selected = selected.astype(int)
    return indices[selected]

def text2NN(clip_model, net, clip_emb, prompt, num_nn=50,num_choice=10, num_gen=1, alpha=0.5):
    convex_fs=[]
    with torch.no_grad():
        distances, indices_full = get_NN(prompt, clip_model, clip_emb, num_nn)
        broad_candidates = get_data(list(indices_full.flatten()), clip_emb)
        for i in range(num_gen):
            indices = get_furthest(broad_candidates.cpu().numpy(), indices_full.flatten(), num_choice)
            input_features = get_data(list(indices), clip_emb)
            rand_w = torch.tensor(np.random.dirichlet([alpha]*num_choice,size=1).flatten()).to('cuda')
            convex_comb = (input_features*rand_w.view(num_choice,-1)).sum(0)
            convex_fs.append(convex_comb)
        convex_fs = torch.stack(convex_fs,axis=0)
        x_hat, _,_,_,latent = net.forward(convex_fs.float(), img='random', return_latents=True,resize=False)
        display_count = x_hat.shape[0]
        results=[]
        for i in range(display_count):
            img = np.asarray(common.tensor2im(x_hat[i]))
            results.append(img)
    return results