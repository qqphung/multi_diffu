import os
import torch
import random
import imageio
import argparse
import numpy as np

# from stable_diffusion.mixture_of_diffusion import StableDiffusion
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
    parser.add_argument('--foldername', type=str, default="test_map",           help="folder name under workspace")
    parser.add_argument('--seed',       type=int, default=0,                help="random seed")
    opt = parser.parse_args()
    seed = opt.seed
    seed_everything(seed)

    # stable diffusion guidance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   

    os.makedirs(opt.workspace, exist_ok=True)

    
    save_path = os.path.join(opt.workspace, opt.foldername)
    os.makedirs(save_path, exist_ok=True)


    negative_text = ''

   
    text_prompts = ['a dog',
                    
                    'a chair',
                    'a dog sitting on the chair']
                    
    list_objects = ['dog', 'chair']               
    objects = generate_box(text_prompts[-1], list_objects)
   
    print(objects)
    height=512
    width=512
    
   
    
    spatial_map_overall = torch.ones([1, 4,  width//8, height//8]).cuda()
    spatial_maps = []
    for o in list_objects:

        for layout_xy in objects[o]: # [cat, dog]: # 
            latent_layout_xy = [xy//8 for xy in layout_xy]
            spatial_map = torch.zeros([1, 4, width//8, height//8]).cuda()
            spatial_map[:, :, latent_layout_xy[0]:latent_layout_xy[2], latent_layout_xy[1]:latent_layout_xy[3]] = 1.
            spatial_map_overall[:, :, latent_layout_xy[0]:latent_layout_xy[2], latent_layout_xy[1]:latent_layout_xy[3]] = 0.
            spatial_maps.append(spatial_map)

    spatial_maps.append(spatial_map_overall)
    plot_spatial_maps([spatial_maps], ['newdog_table_gpt_api'], save_path, seed)
    assert (torch.cat(spatial_maps, 0).sum(0) == 1).all()


   ''' # objects = {'cat': [(88, 372, 283, 480)], 'dog': [(309, 149, 487, 339)], 'ball': [(119, 220, 291, 329)]}
    # objects = {'cat': [(40, 110, 204, 252), (220, 50, 357, 204), (321, 240, 501, 462), (175, 259, 303, 423)]}
    # objects = {'cat': [(241,100, 402, 296)], 'dog':[(217, 317, 410, 485)]}
    # objects = {'ball': [(74, 280, 193, 412), (42, 69, 210, 202), (294, 52, 461,184), (310, 218, 456, 336), (320, 375, 448, 482)]}
    # objects = {'cat': [(192, 153, 300, 295)], 'car':[(300,50 , 453, 464)]}
    # objects = {'elephant': [(90, 50, 313, 201)], 'cat':[( 66,278, 304, 446 )]}
    print(objects)
    height=512
    width=512
    
    # define the coordinates of the four egg boxes
    # egg1 = (45, 67, 120, 188)
    # egg2 = (133, 214, 250, 311)
    # egg3 = (320, 119, 390, 204)
    # egg4 = (395, 209, 480, 305)


    # egg1 = (100, 150, 200, 250)
    # egg2 = (250, 100, 350, 200)
    # egg3 = (400, 150, 500, 250)
    # egg4 = (175, 300, 275, 400)

    # egg1 = (220, 100, 300, 190)
    # egg2 = (220, 220, 300, 300)
    # egg3 = (200, 320, 300, 440)
    # egg4 = (300, 80, 450, 450)
    # cat = (100, 150, 220, 270)
    # dog = (300, 100, 440, 240)
    # egg1 = (60, 120, 180, 240)
    # egg2 = (260, 220, 380, 340)
    # egg3 = (410, 80, 530, 200)
    # egg4 = (340, 350, 460, 470)
    # create and visualize the spatial maps'''