import cv2
import os
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from chatGPT import read_csv, generate_box

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle


feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def detect(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
    objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # let's only keep detections with score > 0.9

        if score > 0.9:
            draw = ImageDraw.Draw(image)
            draw.rectangle(box, outline=(255, 0, 0))
            # font = ImageFont.truetype("sans-serif.ttf", 16)
            # draw.text((x, y),"Sample Text",(r,g,b))
            # import pdb; pdb.set_trace()
            t = model.config.id2label[label.item()]
            objects.append(t)
            fnt = ImageFont.truetype("/vulcanscratch/chuonghm/detect/arial.ttf", 20)
            draw.text((box[0]+5, box[1]+5),t, font=fnt, fill=(0,255,0, 255))
            # print(
            #     f"Detected {model.config.id2label[label.item()]} with confidence "
            #     f"{round(score.item(), 3)} at location {box}"
            # )
    return objects, image
def rearrange_files(path_folder, csv_prompts):
    
    files = os.listdir(path_folder)
    texts = []
    for f in files:
        
        stt = f.split(".")[1]
        stt = int(stt)
        stt = int(stt /10)
        texts.append(csv_prompts[stt])
    return files, texts

def delete(p, l):
    result = []
    found = False
    for i in l:
        if (i != p) or (found==True):
            result.append(i)
        else:
            found = True
    return result

def lower(st):
    result = []
    for s in st:
        result.append(s.lower())
    return result

if __name__ == "__main__":
    
    path_folder = 'save_to_detect/generated_img_base_true_base'
    csv_prompts = read_csv('drawbench.csv')
    files, prompts = rearrange_files(path_folder, csv_prompts)
    num_true = 0
    num_false = 0
    total_predict = 0
    total_gt = 0
    num_lack = 0
    binary_true = 0
    save_folder = 'save_to_detect/detect_img_base_true_base'
    os.makedirs(save_folder, exist_ok=True)
    with open('../create_label_ldm/counting_label.p','rb') as f:
        GT_phrases = pickle.load(f)
    for i, f in enumerate(files):
        prompt = f.split('.')[0]
        prompt = prompt.replace('_',' ')
        # phrases, _ = generate_box(prompt)
        phrases = GT_phrases[prompt +'.']
        phrases = lower(phrases)
        phrases_o = phrases.copy()
        # import pdb; pdb.set_trace()
        path_img = os.path.join(path_folder, f)
        img = Image.open(path_img)
        object_names, img_draw = detect(img)
        object_names = lower(object_names)
        img_draw.save(os.path.join(save_folder, f))
        total_gt += len(phrases)
        # import pdb; pdb.set_trace()
        slg_true = 0
        slg_false = 0
        for o in object_names:
            finded = 0
            
            for p in phrases:
                if o.lower() == p.lower():
                    slg_true += 1
                    num_true += 1
                    phrases = delete(p, phrases)
                    finded = 1
                    total_predict += 1
                    break
                
            if finded == 0 and (o in phrases_o):
                slg_false += 1
                num_false +=1
                total_predict += 1
        print(f)
        print('GT: ',phrases_o )
        print('predict: ', object_names)
        print('slg true:', slg_true)
        print('slg_false: ', slg_false)
        print('lack:',len(phrases) )    
            
        if (slg_true == len(phrases_o)):
            binary_true += 1
        num_lack += len(phrases)
        
        # import pdb; pdb.set_trace()
        
            
    print('true:', num_true)
    print('num_false: ', num_false)
    print('num lack: ',num_lack)
    print('total_predict:', total_predict)
    print('total_gt', total_gt)
    print('true: predict:',num_true/total_predict )
    print('true: gt: ', num_true / total_gt)  
    print('number true images: ', binary_true)
    print('number of file:', len(files))
        
        




