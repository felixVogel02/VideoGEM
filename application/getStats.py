import torch
import videogem
import requests
from PIL import Image
import numpy as np
from utilities.data import vhico_dataset
from torch.utils.data import DataLoader
from utilities.analysis import spatial_eval, visualize_model
from utilities.visualization import visualize
import cv2 as cv2

def testGEM_basic(preprocess, gem_model, save_path="GEM_cat", url="http://images.cocodataset.org/val2017/000000039769.jpg", device="cuda"):
    """For trying out without a dataloader. With a dataloader the preprocess would already has been done.
    """
    # load image and text
    image = preprocess(
        Image.open(requests.get(url, stream=True).raw)
                ).unsqueeze(0).to(device)
    text = ['cat']  # , 'remote control']

    with torch.no_grad():
        logits = gem_model(image, text)  # [B, num_prompt, W, H]
        res_coord = np.unravel_index(logits.argmax().to("cpu"), logits.shape)
        visualize(image, text, logits, save_path=save_path)  # (optional visualization)

def applyHeuristic(model, img, label, tokenizer, template="{}", device="cuda"):
    """When using with a dataloader, the image is already preprocessed.
    """
    
    text = [template.format(label[0])]
    img = img.to(device)
    # print(img.shape)
    x = img.shape[-2]
    y = img.shape[-1]
    
    return (int(x/2), int(y/2)), None, text


def main():
    # result with only < and >: Total: 1658, correct: 986, acc: 0.594692400482509
    # result with <= and >=: Total: 1658, correct: 991, acc: 0.597708082026538
    # print(gem.available_models())
    # return
    model_name = 'ViT-B/16'  # 'ViT-L/14-336'  #  'ViT-B/32'  # 'ViT-L/14'  # 'ViT-B/16'  # 'ViT-B-16-quickgelu'
    pretrained = 'openai'  # 'laion2b_s32b_b82k' # 'openai'  # 'metaclip_400m'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    template="A person {}"  #"A photo of {}."
    # init model and image transform
    preprocess = videogem.get_gem_img_transform()
    
    
    
    test_data = vhico_dataset(transform=preprocess)
    # spatial_eval(None, applyHeuristic, test_data, None, template, device, withStats=True, test_ioU=False, mode=1)
    # visualize_model(gem_model, applyGEM, test_data, None, template, device, save_path="testing/images/gem/")
    

if __name__ == "__main__":
    main()