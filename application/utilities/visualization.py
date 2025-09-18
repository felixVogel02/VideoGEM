import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv2
import open_clip
import clip
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from torchvision import models
from torchsummary import summary
import torch.nn as nn
import numpy as np

def show_model():
    """
    Print a model to see f.e. if CLIP and Open_CLIP have the same structure.
    """

    device = "cuda:0"
    # CLIP
    # model, preprocess = clip.load("ViT-B/16", device=device)
    # model.to(device)
    # visualModel = model.visual
    # clipModel = [module for module in visualModel.modules() if not isinstance(module, nn.Sequential)]
    # f = open("/data1/felix/master_thesis_plots/models/clip_VisionTransformer.txt", "a")
    # f.write(str(clipModel))
    # f.close()
    
    # Open-CLIP
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    model.to(device)
    visualModel = model.visual
    clipModel = [module for module in visualModel.modules() if not isinstance(module, nn.Sequential)]
    f = open("/data1/felix/master_thesis_plots/models/openClip_VisionTransformer.txt", "a")
    f.write(str(clipModel))
    f.close()

    return

def visualize_simple(image, caption, bbox, save_path, resize=True):
    """
    Visualize an image with a bounding box.
    bbox: (xtl, ytl, ybr, ybr)
    """

    if type(image) == str:
        image = Image.open(image)
    
    width, height = image.size
    
    image = np.array(image)
    
    if resize:
        if height > 360 and width > 640:
            image = cv2.resize(image, (640, 360))
    
    if bbox:
        image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0 , 0), 2)
    
    plt.imshow(image)
    plt.title(caption)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)

def visualize(image, text, logits, max_loc, bboxes, bbox=None, alpha=0.6, save_path=None, add_text=True, model_name="clip", otherLocations=[], inRGB=False):
    """

    Args:
        image (_type_): _description_
        text (_type_): _description_
        logits (_type_): _description_
        bboxes (_type_): _description_
        bbox (_type_, optional): Big bounding box. If None, it is not shown. Defaults to None.
        alpha (float, optional): _description_. Defaults to 0.6.
        save_path (_type_, optional): _description_. Defaults to None.

    Raises:
        f: _description_
    """
    W, H = logits.shape[-2:]
    if isinstance(image, Image.Image):
        image = image.resize((W, H))
    elif isinstance(image, np.ndarray):
        image = np.resize(image, (W, H))
        # image1 = image.shape
        # # image = torch.from_numpy(image)
        # image2 = image.shape
        
        image = Image.fromarray(image.astype('uint8'))  # convert to PIL

    elif isinstance(image, torch.Tensor):
        if model_name in ["clip", "openclip"]:
            if image.ndim > 3:
                image = image.squeeze(0)
            image_unormed = (image.detach().cpu() * torch.Tensor(OPENAI_DATASET_STD)[:, None, None]) \
                            + torch.Tensor(OPENAI_DATASET_MEAN)[:, None, None]  # undo the normalization
            image = Image.fromarray((image_unormed.permute(1, 2, 0).numpy() * 255).astype('uint8'))  # convert to PIL
        elif model_name in ["viclip"]:
            v_mean = np.array([0.406, 0.456, 0.485])
            v_std = np.array([0.225, 0.224, 0.229])
            
            if image.ndim > 3:
                image = image.squeeze(0)
            image_unormed = (image.detach().cpu() * torch.Tensor(v_std)[:, None, None]) \
                            + torch.Tensor(v_mean)[:, None, None]  # undo the normalization
            image = Image.fromarray((image_unormed.permute(1, 2, 0).numpy() * 255).astype('uint8'))  # convert to PIL


    else:
        raise f'image should be either of type PIL.Image.Image or torch.Tensor but found {type(image)}'

    # plot image
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    b = logits.shape
    c = logits.ndim
    mini = torch.min(logits)
    maxi = torch.max(logits)

    if logits.ndim > 3:
        logits = logits.squeeze(0)
    logits = logits.detach().cpu().numpy()


    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    logits_old = logits.copy().squeeze()
    
    logits = (logits * 255).astype('uint8')
    a = logits.shape
    heat_maps = [cv2.applyColorMap(logit, cv2.COLORMAP_JET) for logit in logits]
    if inRGB:
        heat_maps_final = []
        for map in heat_maps:
            heat_maps_final.append(cv2.cvtColor(map.astype('uint8'), cv2.COLOR_BGR2RGB))
        heat_maps = heat_maps_final
        
    # For heatmpas use a grayscale image as background.
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)  # To get the correct number of channels back agaiin.
    vizs = [(1 - alpha) * img_cv + alpha * heat_map for heat_map in heat_maps]
    # vizs = [img_cv]

    # Add circle around the highest activation.
    # min_val, max_val, min_loc, max_loc = max_pos  # cv2.minMaxLoc(logits_old)  # Otherwise problematic with binary mask.
    if type(bboxes)!=bool:
        for box in bboxes:
            # print((box[0].item(), box[1].item()))
            # print((box[2].item(), box[3].item()))
            # print("box: ", box)
            continue
            vizs = [cv2.rectangle(viz, (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())),(0,255,0),3) for viz in vizs]
    
    if bbox:
        # pass
        vizs = [cv2.rectangle(viz, (int(bbox[0].item()), int(bbox[1].item())), (int(bbox[2].item()), int(bbox[3].item())),(0,0,139),1) for viz in vizs]

    
    
    if max_loc:
        pass
        
        # vizs = [cv2.drawMarker(viz, (max_loc[-2], max_loc[-1]),  (255, 47, 2214), markerType=cv2.MARKER_TILTED_CROSS, 
        # markerSize=30, thickness=4, line_type=cv2.LINE_AA) for viz in vizs]  # Final prediction has the same color as bounding box.

        
        # vizs = [cv2.drawMarker(viz, (max_loc[-2], max_loc[-1]), (0,255,0), markerType=cv2.MARKER_TILTED_CROSS, 
        # markerSize=30, thickness=4, line_type=cv2.LINE_AA) for viz in vizs]  # Final prediction has the same color as bounding box.

        
        # vizs = [cv2.drawMarker(viz, (max_loc[-2], max_loc[-1]),(0,255,0), markerType=cv2.MARKER_TILTED_CROSS, 
        # markerSize=10, thickness=2, line_type=cv2.LINE_AA) for viz in vizs]  # Final prediction has the same color as bounding box.
        # vizs = [cv2.circle(viz, (max_loc[-2], max_loc[-1]), 10, (0,255,0), 1) for viz in vizs]  # Final prediction has the same color as bounding box.
        # vizs = [cv2.circle(viz, (max_loc[-2], max_loc[-1]), 10, (255, 47, 2214), 1) for viz in vizs]  # Final prediction has the same color as bounding box.
    
    colors = [  # BGR
        (0, 255, 245),  # yellow (verb)
        (255, 0, 10),  # blue (object)
        (214, 47, 255),  # pink (total)
    ]
    if inRGB:  # RGB
        colors = [
        (245, 255, 0),  # yellow (verb)
        (10, 0, 255),  # blue (object)
        (255, 47, 2214),  # pink (total)
    ]
    for i, pos in enumerate(otherLocations):
        continue
        vizs = [cv2.drawMarker(viz, (pos[-2], pos[-1]),colors[i], markerType=cv2.MARKER_TILTED_CROSS, 
        markerSize=20, thickness=4, line_type=cv2.LINE_AA) for viz in vizs]
        # vizs = [cv2.circle(viz, (pos[-2], pos[-1]), 8-i, colors[i], 1) for viz in vizs]
        

    # ALso save the original image.
    # if not inRGB:
    #     img_cv = cv2.cvtColor(img_cv.astype('uint8'), cv2.COLOR_BGR2RGB)
    # else:
    #     img_cv = img_cv.astype('uint8')
    
    # if add_text:
    #     plt.savefig(save_path+"-"+"original.png")
    # else:
    #     plt.savefig(save_path+"original.png")
    
    
    for viz, cls_name in zip(vizs, text):

        if not inRGB:
            viz = cv2.cvtColor(viz.astype('uint8'), cv2.COLOR_BGR2RGB)
        else:
            viz = viz.astype('uint8')
        plt.imshow(viz)
        plt.title(cls_name)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        if save_path is not None:
            if add_text:
                plt.savefig(save_path+"-"+cls_name.replace(" ", "_",)+".png")
            else:
                plt.savefig(save_path+".png")
                


def visualize_backup(image, text, logits, alpha=0.6, save_path=None):
    W, H = logits.shape[-2:]
    if isinstance(image, Image.Image):
        image = image.resize((W, H))
    elif isinstance(image, torch.Tensor):
        if image.ndim > 3:
            image = image.squeeze(0)
        image_unormed = (image.detach().cpu() * torch.Tensor(OPENAI_DATASET_STD)[:, None, None]) \
                        + torch.Tensor(OPENAI_DATASET_MEAN)[:, None, None]  # undo the normalization
        image = Image.fromarray((image_unormed.permute(1, 2, 0).numpy() * 255).astype('uint8'))  # convert to PIL
    else:
        raise f'image should be either of type PIL.Image.Image or torch.Tensor but found {type(image)}'

    # plot image
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if logits.ndim > 3:
        logits = logits.squeeze(0)
    logits = logits.detach().cpu().numpy()


    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    logits_old = logits.copy().squeeze()
    
    logits = (logits * 255).astype('uint8')
    heat_maps = [cv2.applyColorMap(logit, cv2.COLORMAP_JET) for logit in logits]
    print("heat maps1: ", heat_maps[0][72][168])
    
    heat_maps[0][72][168] = np.array([0, 0, 0])
    print("heat maps2: ", heat_maps[0][72][168])
    print("heat maps3: ", len(heat_maps))
    
    vizs_1 = [(1 - alpha) * img_cv + alpha * heat_map for heat_map in heat_maps]

    # Add circle around the highest activation.
    print(f"logits.argmax() {logits.argmax()}")
    print(f"logits.shape {logits.shape}")
    # print(logits_old)
    print(logits_old.shape)
    print("Max: ", np.max(logits_old))
    print("Max: ", np.argmax(logits_old))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(logits_old)
    print(f"minMaxLoc ", min_val, max_val, min_loc, max_loc)
    res_coord = np.unravel_index(logits.argmax(), logits.shape)
    print(res_coord)
    print("max found: ", logits_old[max_loc[1]][max_loc[0]])
    
    
    vizs = [cv2.circle(viz, (res_coord[-1], res_coord[-2]), 10, (0, 0, 0), 2) for viz in vizs_1]
    
    for viz, cls_name in zip(vizs, text):

        viz = cv2.cvtColor(viz.astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.imshow(viz)
        plt.title(cls_name)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        if save_path is not None:
            plt.savefig(f'heatmap_{cls_name}.png')


if __name__ == "__main__":
    show_model()