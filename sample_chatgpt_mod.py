import os
import torch
import random
import imageio
import argparse
import numpy as np
import csv
from stable_diffusion.mixture_of_diffusion import StableDiffusion
from stable_diffusion.vis_utils import plot_spatial_maps
from chatGPT import generate_box, save_img, load_gt, load_box, draw_box
from PIL import Image
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def refine_boxes(boxes):
    result = []
    for box in boxes:
        result.append((box[1], box[0], box[3], box[2]))
    return result

def format_box(names, boxes):
    result_name = []
    
    for i, name in enumerate(names):
        result_name.append('a ' + name.replace('_',' '))
       
    return result_name, np.array(boxes)
def read_csv(path_file, t):
    list_prompts = []
    with open(path_file,'r') as f:
        reader = csv.reader(f)
        # import pdb; pdb.set_trace()
        for i, row in enumerate(reader):
            if i >0:
                
                if  row[1] ==t:# 'Positional': #row[1] == 'Positional' or
                    list_prompts.append(row[0])
    return list_prompts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters',      type=int, default=10000,            help="training iters")
    parser.add_argument('--text',       type=str, default="A hamburger",    help="text prompt")
    parser.add_argument('--workspace',  type=str, default="HRS_evaluate",        help="workspace to store result")
    parser.add_argument('--min_step',   type=int, default=20,               help="min step, range [0, max_step]")
    parser.add_argument('--max_step',   type=int, default=980,              help="max step, range [min_step, 1000]")
    parser.add_argument('--guidance',   type=float, default=100,            help="guidance scale")
    parser.add_argument('--v2',         action='store_true',                help="SD v2.1, default v1.5")
    ### rich text configs
    parser.add_argument('--foldername', type=str, default="counting",           help="folder name under workspace")
    parser.add_argument('--seed',       type=int, default=0,                help="random seed")
    parser.add_argument('--type', type=str, default="counting",           help="folder name under workspace")
    parser.add_argument('--data', type=str, default="counting",           help="folder name under workspace")
    opt = parser.parse_args()
    seed = opt.seed
    # seed_everything(seed)

    # stable diffusion guidance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    guidance = StableDiffusion(device, opt.v2, opt.min_step, opt.max_step)

    os.makedirs(opt.workspace, exist_ok=True)

    
    save_path = os.path.join(opt.workspace, opt.foldername)
    os.makedirs(save_path, exist_ok=True)


    negative_text = ''
    if opt.data == 'HRS':
        if opt.type=='spatial':
            list_data_box = ['/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/spatial.p',
            '/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/spatial_2.p']
            prompts, _ = load_gt('/vulcanscratch/chuonghm/data_evaluate_LLM/HRS/spatial_compositions_prompts.csv')
        elif opt.type=='counting':
            list_data_box = ['/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/counting.p',
            '/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/counting_500_1499.p',
            '/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/counting_1500_2499.p',
            '/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/counting_5.p']
            _, prompts = load_gt('/vulcanscratch/chuonghm/data_evaluate_LLM/HRS/counting_prompts.csv')
        elif opt.type=='size':
            list_data_box = ['/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/size.p']
            prompts,_ = load_gt('/vulcanscratch/chuonghm/data_evaluate_LLM/HRS/size_compositions_prompts.csv')
        elif opt.type=='color':
            prompts, _ = load_gt("/vulcanscratch/chuonghm/data_evaluate_LLM/HRS/colors_composition_prompts.csv")   
            list_data_box = ['/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/color.p']
    else:
        if opt.type == 'counting':
            prompts = read_csv('/vulcanscratch/chuonghm/data_evaluate_LLM/drawbench/drawbench.csv','Counting')
            list_data_box = ['/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box_drawbench/counting.p']
        else:
            prompts = read_csv('/vulcanscratch/chuonghm/data_evaluate_LLM/drawbench/drawbench.csv','Positional')
            list_data_box = ['/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box_drawbench/spatial.p']
    for file_box in list_data_box:
        data_boxes = load_box(file_box)
        for id_img, prompt in enumerate(prompts):
            # if id_img == 132 or id_img ==2927 or id_img== 2535 or id_img==131 or id_img == 2880 or id_img ==1423: continue
            # if id_img < 1624: continue
            # if imd_id == 
            print(id_img)
            text = prompt
            if not prompt in data_boxes.keys(): continue
            names, boxes = data_boxes[prompt]
            boxes = refine_boxes(boxes)
            text_prompts = []
            for name in names:
                text_prompts.append('a ' + name.replace("_"," "))
            new_text = 'on the background'
            text_copy = text_prompts.copy()
            text_prompts.append(new_text)
            text_copy.append(text)
            # text_prompts= ['a chair','a person','in the room']
            # text_copy = ['a chair','a person','A chair below a person in the room']

            print('object', text_prompts)
            print('boxes', boxes)
            height=512
            width=512
            boxes = np.array(boxes)
            
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
            spatial_maps = [spatial_map / torch.cat(spatial_maps, 0).sum(0, keepdim=True) for spatial_map in spatial_maps]
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            # save_path_base = save_path+'_base'
            # if not os.path.isdir(save_path_base):
            #     os.mkdir(save_path_base)
                # print(save_path+'_box')
            file_name = text.replace(' ', '_')
            # plot_spatial_maps([spatial_maps], [file_name], save_path+'_box', seed)
            assert (torch.cat(spatial_maps, 0).sum(0) == 1).all()
            
            # generate baseline image according to only prompt
           
            # guidance.masks = torch.ones([1, 4,  width//8, height//8]).cuda()
            # img = guidance.produce_attn_maps(text_prompts[-1:], [negative_text], 
            #                             height=height, width=width, num_inference_steps=41, 
            #                             guidance_scale=8.5)
            # imageio.imwrite(os.path.join(save_path, file_name + '_seed%d_base.png' % (seed)), img[0])
            # save_img(save_path_base, Image.fromarray(img[0]), prompt, 0, id_img)
            # generate the image according to spatial maps and prompts
            guidance.masks = spatial_maps
            for i in range(1):
                
                # seed_everything(seed)
                bg_aug_end=830 # cang lon cang the prompt cuoi (3 qua trung), cang nho thi cang bi mat background
                img = guidance.prompt_to_img(text_prompts,text_copy, [negative_text]*len(text_prompts), 
                                            height=height, width=width, num_inference_steps=41, 
                                            guidance_scale=8.5, bg_aug_end=bg_aug_end)
                # img = guidance.prompt_to_img(text_prompts, [negative_text], 
                #                               height=height, width=width, num_inference_steps=41, 
                #                               guidance_scale=10, bg_aug_end=bg_aug_end)
                save_img(save_path, Image.fromarray(img[0]), prompt, i, id_img)
            # draw_box(save_path, Image.fromarray(img[0]), prompt, 0, id_img,names, boxes )
        

# 840/ 61
