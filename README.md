# Text-Free Learning of a Natural Language Interface for Pretrained Face Generators

<p align="center">
<img src="docs/GUI.gif" width=500/>

> We propose Fast text2StyleGAN, a natural language interface that adapts pre-trained GANs for text-guided human face synthesis. Leveraging the recent advances in Contrastive Language-Image Pre-training (CLIP), no text data is required during training. Fast text2StyleGAN is formulated as a conditional variational autoencoder (CVAE) that provides extra control and diversity to the generated images at test time. Our model does not require re-training or fine-tuning of the GANs or CLIP when encountering new text prompts. In contrast to prior work, we do not rely on optimization at test time, making our method orders of magnitude faster than prior work. Empirically, on FFHQ dataset, our method offers faster and more accurate generation of images from natural language descriptions with varying levels of detail compared to prior work. 

<p align="center">
<img src="docs/teaser.png"/>
<br>
Examples of text-driven face image synthesis by our proposed method Fast text2StyleGAN. The text prompts are increasingly more detailed. Each image takes about 0.09s to produce.
</p>

## Description   
This repo contains the official implementation of the Fast text2StyleGAN. Besides training code, code for a [Streamlit](https://streamlit.io/) inference GUI is also included. Please refer to `streamlit_gui/README.md` for details.




## Installation
- Clone this repo:
``` 
git clone https://github.com/duxiaodan/Fast_text2StyleGAN.git
cd Fast_text2StyleGAN
```
- Dependencies:  
Create a new [Conda](https://docs.anaconda.com/anaconda/install/) environment using `environment.yml`.
Then install [CLIP](https://github.com/openai/CLIP) with following commands:
```shell script
pip install git+https://github.com/openai/CLIP.git
```

## Usage
### Generating samples using pre-trained model
1. Download pre-trained model from [this link](https://drive.google.com/file/d/1kP2xrY24B0WdLybe-oZKUOVas7kmf0cA/view?usp=sharing) and put the file under `logs/cvae_v7/checkpoints/`. You can change `cvae_v7` to the name you like but make sure variable `trial` in function `prepare_models()` from file `streamlit_gui/app.py` matches the new name.
2. Download config file for the pre-trained model from [this link](https://drive.google.com/file/d/1PvObvkTaepRzEVrKAFLADgARGqt6RYDI/view?usp=sharing) and put the file under `logs/cvae_v7/`.
3. Our KNN method ("Ours" in the paper) also requires CLIP embeddings of FFHQ dataset. We've pre-computed them and you can download the hdf5 file from [this link](https://drive.google.com/file/d/1ES0l0n33nEOJjRFsSPGH4WdEvJVwEQWK/view?usp=sharing). Put it under `data/`. Though not our best method, you can also play with the other two methods ("Text Only" and "Text+Image") if you want.
4. Now you can refer to `streamlit_gui/README.md` for how to launch the Streamlit GUI and start to generate faces!

### Train your own model
1. To train your own model, put both training data (FFHQ) and testing data (CelebAHQ) under `data/`, in separate folders. Folder structure should be like `data/FFHQ/00000.png`. 
2. The dataloader takes both images and CLIP embeddings therefore you'll also need to pre-compute CLIP embeddings for both the training data and the testing data. We pre-computed CLIP embeddings with `ViT-B/16` encoder for FFHQ and CelebAHQ. You can find them at [here](https://drive.google.com/file/d/1ES0l0n33nEOJjRFsSPGH4WdEvJVwEQWK/view?usp=sharing) and [here](https://drive.google.com/file/d/15Xnx9vwI47PvfmdQqMcE9Sl9WI64pFJc/view?usp=sharing). 
3. In `configs/paths_config.py`, change the paths so that they matched with your own data files. 
4. Download stylegan2 model pre-trained on FFHQ from [this link](https://drive.google.com/file/d/1RnE5R_ofbDGeKtKrBSBSkR-1O0HufvES/view?usp=sharing) and put it under `pretrained_models`. In the terminal, run the command below to start training.
    ```
    bash launch_training.sh
    ```
5. [Weights & Biases](https://wandb.ai/site) is also suppported. Just turn on the flag `--use_wandb`
6. If you want to start from some previous checkpoint, specify its path using the flag `--checkpoint_path`.
7. If you want to run SLURM sequential jobs, turn on the flag `--sequential` and set `--checkpoint_path` to `"auto"`.

## Credits
**Our code is adapted from the pSp implementation:**  
https://github.com/eladrich/pixel2style2pixel  
Copyright (c) 2020 Elad Richardson, Yuval Alaluf
License (MIT) https://github.com/eladrich/pixel2style2pixel/blob/master/LICENSE

**StyleGAN2 implementation:**  
https://github.com/rosinality/stylegan2-pytorch  
Copyright (c) 2019 Kim Seonghyeon  
License (MIT) https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE  

**LPIPS implementation:**  
https://github.com/S-aiueo32/lpips-pytorch  
Copyright (c) 2020, Sou Uchida  
License (BSD 2-Clause) https://github.com/S-aiueo32/lpips-pytorch/blob/master/LICENSE  

**VAE implementation:**  
https://github.com/AntixK/PyTorch-VAE  
Copyright (c) 2020, Anand Krishnamoorthy Subramanian
License (Apache License 2.0) https://github.com/AntixK/PyTorch-VAE/blob/master/LICENSE.md


**Please Note**: The CUDA files under the [StyleGAN2 ops directory](https://github.com/duxiaodan/Fast_text2StyleGAN/tree/master/models/stylegan2/op) are made available under the [Nvidia Source Code License-NC](https://nvlabs.github.io/stylegan2/license.html)


## Citation
If you use this code for your research, please cite our paper <a href="">Text-Free Learning of a Natural Language Interface for Pretrained Face Generators</a>:

```
@InProceedings{
}
```
