import logging
from typing import Any, Union, List, Optional, Tuple, Dict
import open_clip
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv2
import torch.nn as nn

import sys
sys.path.append("/data1/felix/InternVideo/Data/InternVid")

import viclip

from .gem_wrapper import GEMWrapper


_MODELS = {
    # B/32
    "ViT-B/32": [
        "openai",
        "laion400m_e31",
        "laion400m_e32",
        "laion2b_e16",
        "laion2b_s34b_b79k",
    ],

    "ViT-B/32-quickgelu": [
        "metaclip_400m",
        "metaclip_fullcc"
    ],
    # B/16
    "ViT-B/16": [
        "openai",
        "laion400m_e31",
        "laion400m_e32",
        "laion2b_s34b_b88k",
    ],
    "ViT-B/16-quickgelu": [
        "metaclip_400m",
        "metaclip_fullcc",
    ],
    "ViT-B/16-plus-240": [
        "laion400m_e31",
        "laion400m_e32"
    ],
    # L/14
    "ViT-L/14": [
        "openai",
        "laion400m_e31",
        "laion400m_e32",
        "laion2b_s32b_b82k",
    ],
    "ViT-L/14-quickgelu": [
        "metaclip_400m",
    "metaclip_fullcc"
    ],
    "ViT-L/14-336": [
        "openai",
    ]
}


class Brian_proj_img(nn.Module):
    def __init__(self, vis_dim=512, num_classes=512):
        super(Brian_proj_img, self).__init__()
        self.vis_proj_mat_g = nn.Linear(vis_dim, num_classes)
        
    def forward(self, x):
        x = self.vis_proj_mat_g(x)
        return x

class Brian_proj_txt(nn.Module):
    def __init__(self, text_dim=512, num_classes=512):
        super(Brian_proj_txt, self).__init__()
        self.lang_proj_mat_g = nn.Linear(text_dim, num_classes)
        
    def forward(self, x):
        x = self.lang_proj_mat_g(x)
        return x

def available_models() -> List[str]:
    """Returns the names of available GEM-VL models"""
    # _str = "".join([": ".join([key, value]) + "\n" for key, values in _MODELS2.items() for value in values])
    _str = "".join([": ".join([key + " "*(20 - len(key)), value]) + "\n" for key, values in _MODELS.items() for value in values])
    return _str

def get_tokenizer(
        model_name: str = '',
        context_length: Optional[int] = None,
        **kwargs,
):
    """ Wrapper around openclip get_tokenizer function """
    return open_clip.get_tokenizer(model_name=model_name, context_length=context_length, **kwargs)


def get_gem_img_transform(
        img_size:  Union[int, Tuple[int, int]] = (448, 448),
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    std = std or OPENAI_DATASET_STD
    transform = transforms.Compose([
        transforms.Resize(size=img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform


def create_gem_model(
        model_name: str,
        pretrained: Optional[str] = None,
        gem_depth: int = 7,
        ss_attn_iter: int = 1,
        ss_attn_temp: Optional[float] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        tokenizerName=False,
        addBef=False,
        img_attn_only=0,  # 0 is normal GEM. 1 -> only attend to tokens of the same image, ... (see gem_wrapper.py)
        avg_prompts=1,  # The text embedding of groups of avg_prompts is averaged.
        get_sims=False,  # If True it outputs also the similarities between the prompts and the images.
        useStack=False,  # The outputs of all self-self attention blocks are saved and it is not averaged over them, but on their logits after calculating the similarity to the text.
        stackWeights=False,  # Give individual weights to the initial output and the outputs of the self-self attention.
        stackWeightsAuto=False,  # The weights of the stack are automatically inferred by calculating similarities.
        brian_projection=False,  # Should be true when brians model is loaded. Then, the projection layer is also loaded that is needed for Brians model to work correctly.
        **model_kwargs,
):
    img_proj_final = False
    txt_proj_final = False
    if False:#brian_projection:  # Load his model
        print("use Brians projection heads.")
        checkpoint = torch.load("/data2/walid/epoch0008.pth.tar")
        state_dict = checkpoint["state_dict"]
        # print(state_dict.keys())
        img_proj_final = Brian_proj_img().to(device)
        with torch.no_grad():
            img_proj_final.vis_proj_mat_g.weight.copy_(state_dict['module.vis_proj_mat_g.0.weight'])
            # img_proj_final.vis_proj_mat_g.weight.copy_(state_dict['module.vis_proj_mat.0.weight'])
            img_proj_final.vis_proj_mat_g.bias.copy_(state_dict['module.vis_proj_mat_g.0.bias'])
            # img_proj_final.vis_proj_mat_g.bias.copy_(state_dict['module.vis_proj_mat.0.bias'])
        
        txt_proj_final = Brian_proj_txt().to(device)
        with torch.no_grad():
            txt_proj_final.lang_proj_mat_g.weight.copy_(state_dict['module.lang_proj_mat_g.0.weight'])
            # txt_proj_final.lang_proj_mat_g.weight.copy_(state_dict['module.lang_proj_mat.0.weight'])
            txt_proj_final.lang_proj_mat_g.bias.copy_(state_dict['module.lang_proj_mat_g.0.bias'])
            # txt_proj_final.lang_proj_mat_g.bias.copy_(state_dict['module.lang_proj_mat.0.bias'])
            
    if False:#type(model_name) == str and ("howto" in pretrained):
        model_type = "clip"
        size = None
        model_name = model_name.replace("/", "-")
        open_clip_model = open_clip.create_model("ViT-B/32", device=device)
        
        # For Brians model
        checkpoint = torch.load(pretrained)
        state_dict = checkpoint["state_dict"]
        
        # For openai model
        # state_dict = torch.load(pretrained)
        
        
        new_dict = {}
        for k,v in state_dict.items():
            #k = k.replace('encoder_q.encoder.', 'backbone.')
            #print('bu k',k)
            #if 'text_module.fc1' in k:
            #    print('text_module.fc1 bu val',v[0])
                
            k = k.replace('module.', '')
            
            new_dict[k] = v
        checkpoint = new_dict
        pretrained_dict = {}
        for k in open_clip_model.state_dict():
            #k = k.replace('module.', '')
            # if 'DAVEnet.bn1' in k:
            #     print('DAVEnet.bn1 before val',model.state_dict()[0])
            # if 'lang_proj_mat_g' in k:
            #     print('lang_proj_mat_g before val',model.state_dict()[0])
            if k in checkpoint:
                pretrained_dict[k] = checkpoint[k]
            else:
                #print('not bu k',k)
                pretrained_dict[k] = model.state_dict()[k]
        # print('loaded self-trained CLIP: ', args.pretrain_clip)
        open_clip_model.load_state_dict(pretrained_dict)
        
        if tokenizerName:  # If a path to a checkpoint is used, the tokenizer needs to be selected manually.
            tokenizer = open_clip.get_tokenizer(model_name=tokenizerName)
        else:
            # print("model_name: ", model_name)
            tokenizer = open_clip.get_tokenizer(model_name=model_name)

    elif type(model_name) == str:  # Standard
        model_type = "clip"
        model_name = model_name.replace("/", "-")
        size = None
        logging.info(f'Loading pretrained {model_name} from pretrained weights {pretrained}...')
        # print("model_name: ", model_name)
        # print("pretrained: ", pretrained)
        # print("precision: ", precision)
        # print("device: ", device)
        # open_clip_model = open_clip.create_model(model_name, pretrained, precision, device)
        open_clip_model = open_clip.create_model(model_name, pretrained, precision, device, jit, force_quick_gelu, force_custom_text,
                                    force_patch_dropout, force_image_size, force_preprocess_cfg, pretrained_image,
                                    pretrained_hf, cache_dir, output_dict, require_pretrained, **model_kwargs)
        if tokenizerName:  # If a path to a checkpoint is used, the tokenizer needs to be selected manually.
            tokenizer = open_clip.get_tokenizer(model_name=tokenizerName)
        else:
            tokenizer = open_clip.get_tokenizer(model_name=model_name)
            # print("Correct laoded")
    else:  # For adapting to f.e. viCLIP
        model_type = model_name[0]
        size = model_name[1]
        if model_type == "viclip":
            model_sum = viclip.get_viclip(size, pretrained)
            open_clip_model = model_sum["viclip"].to(device)
            tokenizer = model_sum["tokenizer"]

    gem_model = GEMWrapper(model=open_clip_model, tokenizer=tokenizer, depth=gem_depth,
                           ss_attn_iter=ss_attn_iter, ss_attn_temp=ss_attn_temp, model_type=model_type, size=size,
                           addBef=addBef, img_attn_only=img_attn_only, avg_prompts=avg_prompts, get_sims=get_sims,
                           useStack=useStack, stackWeights=stackWeights, stackWeightsAuto=stackWeightsAuto,
                           img_proj_final=img_proj_final, txt_proj_final=txt_proj_final)
    logging.info(f'Loaded GEM-{model_name} from pretrained weights {pretrained}!')
    return gem_model

def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        gem_depth: int = 7,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        **model_kwargs,
):
    gem_model = create_gem_model(model_name, pretrained, gem_depth, precision, device, jit, force_quick_gelu, force_custom_text,
                                 force_patch_dropout, force_image_size, force_preprocess_cfg, pretrained_image,
                                 pretrained_hf, cache_dir, output_dict, require_pretrained, **model_kwargs)

    transform = get_gem_img_transform(**model_kwargs)
    return gem_model, transform

def visualize(image, text, logits, alpha=0.6, save_path=None):
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
    logits = (logits * 255).astype('uint8')
    heat_maps = [cv2.applyColorMap(logit, cv2.COLORMAP_JET) for logit in logits]

    vizs = [(1 - alpha) * img_cv + alpha * heat_map for heat_map in heat_maps]
    for viz, cls_name in zip(vizs, text):

        viz = cv2.cvtColor(viz.astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.imshow(viz)
        plt.title(cls_name)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        if save_path is not None:
            plt.savefig(f'heatmap_{cls_name}.png')
