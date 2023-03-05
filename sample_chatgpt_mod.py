import os
import torch
import random
import imageio
import argparse
import numpy as np

from stable_diffusion.mixture_of_diffusion import StableDiffusion
from stable_diffusion.vis_utils import plot_spatial_maps
from chatGPT import generate_box

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters',      type=int, default=10000,            help="training iters")
    parser.add_argument('--text',       type=str, default="A hamburger",    help="text prompt")
    parser.add_argument('--workspace',  type=str, default="results",        help="workspace to store result")
    parser.add_argument('--min_step',   type=int, default=20,               help="min step, range [0, max_step]")
    parser.add_argument('--max_step',   type=int, default=980,              help="max step, range [min_step, 1000]")
    parser.add_argument('--guidance',   type=float, default=100,            help="guidance scale")
    parser.add_argument('--v2',         action='store_true',                help="SD v2.1, default v1.5")
    ### rich text configs
    parser.add_argument('--foldername', type=str, default="eval",           help="folder name under workspace")
    parser.add_argument('--seed',       type=int, default=0,                help="random seed")
    opt = parser.parse_args()
    seed = opt.seed
    seed_everything(seed)

    # stable diffusion guidance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    guidance = StableDiffusion(device, opt.v2, opt.min_step, opt.max_step)

    os.makedirs(opt.workspace, exist_ok=True)

    
    save_path = os.path.join(opt.workspace, opt.foldername)
    os.makedirs(save_path, exist_ok=True)


    negative_text = ''

    # define the prompts for each box
    # text_prompts = ['an Easter egg, highly detailed',
    #                 'an Easter egg, highly detailed',
    #                 'an Easter egg, highly detailed',
    #                 'an Easter egg, highly detailed',
    #                 'Four Easter eggs in the field']

    text = 'a cat on the left of a dog'
    names, boxes =  generate_box(text)
    # boxes = [(50, 50, 200, 200), (300, 50, 450, 200), (50, 300, 200, 450), (300, 300, 450, 450) ]
    text_prompts = []
    for name in names:
        text_prompts.append('a ' + name)
    text_prompts.append(text)
    print('object', names)
    print('boxes', boxes)
    height=512
    width=512
    
    spatial_map_overall = torch.ones([1, 4,  width//8, height//8]).cuda()
    spatial_maps = []
    for i, o in enumerate(names):

        layout_xy = boxes[i] # [cat, dog]: # 
        latent_layout_xy = [xy//8 for xy in layout_xy]
        spatial_map = torch.zeros([1, 4, width//8, height//8]).cuda()
        spatial_map[:, :, latent_layout_xy[0]:latent_layout_xy[2], latent_layout_xy[1]:latent_layout_xy[3]] = 1.
        spatial_map_overall[:, :, latent_layout_xy[0]:latent_layout_xy[2], latent_layout_xy[1]:latent_layout_xy[3]] = 0.
        spatial_maps.append(spatial_map)

    spatial_maps.append(spatial_map_overall)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    file_name = text.replace(' ', '_')
    plot_spatial_maps([spatial_maps], [file_name], save_path, seed)
    assert (torch.cat(spatial_maps, 0).sum(0) == 1).all()
    # generate baseline image according to only prompt
    seed_everything(seed)
    guidance.masks = torch.ones([1, 4,  width//8, height//8]).cuda()
    img = guidance.prompt_to_img(text_prompts[-1:], [negative_text], 
                                  height=height, width=width, num_inference_steps=41, 
                                  guidance_scale=8.5)
    imageio.imwrite(os.path.join(save_path, file_name + '_seed%d_base.png' % (seed)), img[0])

    # generate the image according to spatial maps and prompts
    guidance.masks = spatial_maps
    seed_everything(seed)
    bg_aug_end=830 # cang lon cang the prompt cuoi (3 qua trung), cang nho thi cang bi mat background
    img = guidance.prompt_to_img(text_prompts, [negative_text], 
                                  height=height, width=width, num_inference_steps=41, 
                                  guidance_scale=8.5, bg_aug_end=bg_aug_end)
    # img = guidance.prompt_to_img(text_prompts, [negative_text], 
    #                               height=height, width=width, num_inference_steps=41, 
    #                               guidance_scale=10, bg_aug_end=bg_aug_end)
    
    imageio.imwrite(os.path.join(save_path, file_name + '_seed%d_style.png' % (seed)), img[0])


# 840/ 61