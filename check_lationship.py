import cv2
import os
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from chatGPT import read_csv, generate_box, read_txt_label

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
    boxes = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # let's only keep detections with score > 0.9

        if score > 0.85:
            boxes.append(box)
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
    return objects, boxes, image

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

def check_location(box1, box2, label):
    mean_1 = (box1[0] + box1[2]) / 2 , (box1[1] + box1[3]) /2
    mean_2 = (box2[0] + box2[2]) / 2 , (box2[1] + box2[3]) /2
    if label == 'top':
        if mean_1[1] < mean_2[1]: return True
    if label == 'underneath':
        if mean_1[1] > mean_2[1]: return True
    if label == 'left':
        if mean_1[0] < mean_2[0]: return True
    if label == 'right':
        if mean_1[0] > mean_2[0]: return True
    return False

if __name__ == "__main__":
    
    path_folder = 'save_to_detect/spatial_relationship_base'
    csv_prompts = read_csv('drawbench.csv')
    files, prompts = rearrange_files(path_folder, csv_prompts)
    num_true = 0
    num_false = 0
    total_predict = 0
    total_gt = 0
    num_lack = 0
    total = 0
    save_folder = 'save_to_detect/spatial_detected_2'
    # os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    dict_label = read_txt_label('/vulcanscratch/chuonghm/GLIGEN/label_position.txt')
    
    with open('../create_label_ldm/position_label.p','rb') as f:
        GT_phrases = pickle.load(f)

    for i, f in enumerate(files):
        prompt = f.split('.')[0]
        prompt = prompt.replace('_',' ')
        phrases = GT_phrases[prompt +'.']
        # phrases, _ = generate_box(prompt)
        
        phrases = lower(phrases)
        phrases_o = phrases.copy()
        # import pdb; pdb.set_trace()
        path_img = os.path.join(path_folder, f)
        img = Image.open(path_img)
        object_names, predicted_boxes, img_draw = detect(img)
        object_names = lower(object_names)
        img_draw.save(os.path.join(save_folder, f))
        total_gt += len(phrases)
        # import pdb; pdb.set_trace()
        predict_boxes_true = []
        result_true = []

        for i, o in enumerate(object_names):
            finded = 0
            for p in phrases:
                if o.lower() == p.lower():
                    phrases = delete(p, phrases)
                    finded = 1
                    total_predict += 1
                    predict_boxes_true.append(predicted_boxes[i])
                    result_true.append(o)
                    break
        label = dict_label[prompt + '.']
        # import pdb; pdb.set_trace()
        if len(predict_boxes_true) == 2:
            if result_true[0] == phrases_o[0]:
                aligned  = check_location(predict_boxes_true[0],predict_boxes_true[1], label )
            else:
                aligned  = check_location(predict_boxes_true[1],predict_boxes_true[0], label )
        else:
            aligned = False
        if aligned: num_true += 1
        total += 1 
        print(f)
        print('GT',phrases_o)
        print('predict: ', object_names)
        print('aligned', aligned)   
            
    print('true:', num_true)
    print('total: ', total)
    print('ratio', num_true/total)
    
        
        




