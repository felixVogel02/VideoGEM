import torch
import videogem
#import requests
from PIL import Image
import numpy as np
from utilities.data import vhico_dataset, vhico_dataset_video, vidSTG_dataset, youCookInteractions_dataset, groundingYoutube_dataset, daly_dataset, daly_dataset_Video, youCookInteractions_dataset_Video, groundingYoutube_dataset_Video
from torch.utils.data import DataLoader
from utilities.analysis import spatial_eval, visualize_model, temporal_eval
from utilities.visualization import visualize
import cv2 as cv2
from torchvision import transforms
import sys
sys.path.append("/data1/felix/InternVideo/Data/InternVid")
from viclip import get_viclip, retrieve_text, _frame_from_video, frames2tensor
import os
from utilities.stats_helper import min_max, parse_sentence
import json
from pprint import pprint
import spacy
from spacy import displacy
from utilities.stats_helper import get_max_pos


nlp = spacy.load("en_core_web_sm")  # Globally, such that it does not have to be loaded again every time.
with open("/data1/felix/youCookInteractions/prompts_destructed/mistral_prompts_dict.json") as f:
    youCook_deconstructed_sentences = json.load(f)


# from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
# print(OPENAI_DATASET_MEAN)
# print(OPENAI_DATASET_STD)



def frame2tensor_adapted(img, device=torch.device('cuda'), onlyOne=True):
    """
    onlyOne=True if one frame should be repeated 8 times, otherwise img is a list of 8 images.
    """
    if onlyOne:
        img = img.permute(1, 2, 0)  # Transform back from (3, H, W) for the dataloader to (W, H, 3).
        img = img.numpy()
        img = img[:,:,::-1]
        
        vid_tube = 8*[np.expand_dims(img, axis=(0, 1))]
    else:
        vid_tube = []
        for frame in img:
            frame = frame.permute(1, 2, 0)  # Transform back from (3, H, W) for the dataloader to (W, H, 3).
            frame = frame.numpy()
            frame = frame[:,:,::-1]
            frame = np.expand_dims(frame, axis=(0, 1))
            vid_tube.append(frame)

    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube

def transform(image):
    v_mean = np.array([0.406, 0.456, 0.485])
    v_std = np.array([0.225, 0.224, 0.229])
    
    # PIL to cv2
    img_raw = image.convert('RGB')
    open_cv_image = np.array(img_raw)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    
    image = cv2.resize(open_cv_image[:,:,::-1], (224, 224))

    # cv2 to PIL
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    transformIt = transforms.Compose([
        # transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(v_mean, v_std),
    ])
    image = transformIt(image)
    
    return image

def applyGEMPipeline(gem_model, img, label, tokenizer=None, template="{}", device="cuda", normalize=True, mainWeight=3, takeImg=4, singleFrameInput=False, mode=1, prompt_mode=1, stackWeightsAutoOne=True):
    """When using with a dataloader, the image is already preprocessed. Used for spatial inference.
    nlp: if the input is a real text, not just verb object, than the verbs and objects need to be extracted differently.
    stackWeightsAutoOne: If True, for the verb prompts (the first prompt), static and dynamic weights are applied. Not for case 4(since there is no verb prompt used).

    Mode:
    1: Average over the 3 points (Final VideoGem).
    2: Average over the heat maps and take the max point of the averaged heatmap.
    3. Multiply the heatmaps, minmax normalize them and then take the max position.
    prompt_mode:
    1: For VHICO, groundingYoutube
    2: For Daly
    5: For youCookInteractions.
    """
    
    b = img[0].shape
    if not singleFrameInput:
        images = [frame[0] for frame in img]
    else:
        images = img[0]
    bc = images[0].shape
    img = frame2tensor_adapted(images, device=device, onlyOne=singleFrameInput)
    c = img.shape
    
    # For comparability the same prompts are used for every dataset:
    verb_template = "A photo of a person {} something."
    verb_template_empty = "A photo of a person doing something."  # Rarely the case, since every action has a verb. But sometimes for youCook spacy does not find the verb. Than this sentence is needed.
    object_template = "A photo of a person using {}."
    object_template_empty = "A photo of a person."  # If no object is available, only used for Daly. For the others it is assumed, that there is always a verb and an object present.
    full_template = "A photo of a person {}."
    
    if prompt_mode == 1:
        # verb_template = "A photo of {} something."
        # object_template = "A photo of the object {}."
        # full_template = "A photo of {}."
        verb = label[0].split(" ")[0]
        object = " ".join(label[0].split(" ")[1:])
        
        text = [
            verb_template.format(verb),
            object_template.format(object),
            full_template.format(label[0]),
            ]
        verb_cnt = 1
        noun_cnt = 1
        full_cnt = 1

    elif prompt_mode == 2:
        new_prompt = ""
        for i, letter in enumerate(label[0]):  # Add whitespaces before capital letters.
            if i > 0 and letter.isupper():
                new_prompt += " "
            new_prompt += letter
        
        if len(new_prompt.split(" ")) == 1:  # Only a verb is present in the prompt, no need to seperate anything.
            text = [
                verb_template.format(new_prompt),
                object_template_empty,  # Basic part of the object prompt, only used if no object is ava
                full_template.format(new_prompt)
                ]
            verb_cnt = 1
            noun_cnt = 1
            full_cnt = 1
        else:
            verb = new_prompt.split(" ")[0]
            object = " ".join(new_prompt.split(" ")[1:])

            text = [
                verb_template.format(verb),
                object_template.format(object),
                full_template.format(new_prompt),
                ]
            verb_cnt = 1
            noun_cnt = 1
            full_cnt = 1

    elif prompt_mode == 5:  # Later check which verbs or nouns fit the image the best. Then take the best 2 for reference.    

        verbs, nouns = parse_sentence(nlp, label[0], mode=2)

        verb_text = []
        for verb in verbs:
            verb_text.append(verb_template.format(verb))
        if len(verb_text) == 0:
            verb_text.append(verb_template_empty)  # To ensure the presence of a verb.

        noun_text = []
        for noun in nouns:
            noun_text.append(object_template.format(noun))
        if len(noun_text) == 0:
            noun_text.append(object_template_empty)  # To ensure the presence of an object.
        
        text = []
        
        text.extend(verb_text)
        text.extend(noun_text)
        text.extend([full_template.format(label[0])])
        # print("Text: ", text)
        # print("verb_text: ", verb_text)
        # print("noun_text: ", noun_text)
        
        verb_length = len(verb_text)  # How many verb prompts are in text. The best one out of them is chosen.
        obj_length = len(noun_text)  # How many object prompts are in text. The best one out of them is chosen.
        full_cnt = 1
        # Bools in order to control the setting and select different combinations of verbs, objects and full prompts.
        use_verb = True
        use_obj = True
        use_full_prompt = True  # full_cnt -> False is set later on.
        
        # stackWeightsAutoOne = False

    with torch.no_grad():
        if stackWeightsAutoOne:
            if prompt_mode >= 4:  # Apply the weights to all verbs.
                # print("verb_length: ", verb_length)
                logits, sims = gem_model(img, text, normalize=normalize, stackWeightsAutoOne=len(text))  # [B, num_prompt, W, H]
            else:
                logits, sims = gem_model(img, text, normalize=normalize, stackWeightsAutoOne=len(text))  # [B, num_prompt, W, H]
        else:
            logits, sims = gem_model(img, text, normalize=normalize, stackWeightsAutoOne=stackWeightsAutoOne)  # [B, num_prompt, W, H]

        B, N, I, W, H = logits.shape
        a = sims.shape
            
        # logits_initial = logits[0, 0, takeImg, :, :]  # First prompt and first image.
        # # Out of the max locations from all three prompts, get a final maximum location.
        # logits_verb = logits[0, 0, takeImg, :, :]
        # logits_obj = logits[0, 1, takeImg, :, :]
        # logits_total = logits[0, 2, takeImg, :, :]
        # _, _, _, max_loc_verb = cv2.minMaxLoc(logits_verb.detach().cpu().numpy())
        # _, _, _, max_loc_obj = cv2.minMaxLoc(logits_obj.detach().cpu().numpy())
        # _, _, _, max_loc_total = cv2.minMaxLoc(logits_total.detach().cpu().numpy())
        
        verb_pos = []
        obj_pos = []
        total_pos = []
        logits_verb_tot = torch.zeros(W, H).to(device)  # The average of all logits_verb.
        logits_obj_tot = torch.zeros(W, H).to(device)
        logits_total_tot = torch.zeros(W, H).to(device)
        cnt = -1
        if prompt_mode < 4:  # Normal case
            for i in range(verb_cnt):  # The verbs are at the first positions of the text list, and therefore also in the logits.
                cnt += 1
                logits_verb = logits[0, cnt, takeImg, :, :]
                _, _, _, max_loc_verb = cv2.minMaxLoc(logits_verb.detach().cpu().numpy())
                verb_pos.append(max_loc_verb)
                logits_verb_tot += logits_verb / verb_cnt

            for i in range(noun_cnt):  # Then the objects.
                cnt += 1
                logits_obj = logits[0, cnt, takeImg, :, :]
                _, _, _, max_loc_obj = cv2.minMaxLoc(logits_obj.detach().cpu().numpy())
                obj_pos.append(max_loc_obj)
                logits_obj_tot += logits_obj / noun_cnt
                
                
            for i in range(full_cnt):  # Then the full prompts (normally just 1).
                cnt += 1
                logits_total = logits[0, cnt, takeImg, :, :]
                _, _, _, max_loc_total = cv2.minMaxLoc(logits_total.detach().cpu().numpy())
                total_pos.append(max_loc_total)
                logits_total_tot += logits_total / full_cnt
        
        else:  # Always take the main prompt, but for reference only take take_max_number prompts (the most similar ones).
            verb_cnt = 0
            noun_cnt = 0
            if sims.ndim > 1:
                if use_verb and verb_length >= 1:
                    relevant_sims = sims[0][:verb_length]
                    dimis = relevant_sims.shape
                    dimis = dimis[0]
                    val, ind = torch.topk(relevant_sims, 1)
                    # used_text1 = text[ind[0]]
                    print("Used verb index: ", ind[0])
                    
                    verb_cnt = 1
                    logits_verb = logits[0, ind[0], takeImg, :, :]
                    _, _, _, max_loc_verb = cv2.minMaxLoc(logits_verb.detach().cpu().numpy())
                    verb_pos.append(max_loc_verb)
                    logits_verb_tot += logits_verb

                if use_obj and obj_length >= 1:
                    relevant_sims = sims[0][verb_length:verb_length+obj_length]
                    dimis = relevant_sims.shape
                    dimis = dimis[0]
                    val, ind = torch.topk(relevant_sims, 1)
                    # used_text1 = text[ind[0]]
                    # used_text2 = text[ind[1]]
                    print("Used object index (without added verb length): ", ind[0])
                    

                    noun_cnt = 1
                    
                    logits_obj = logits[0, ind[0]+verb_length, takeImg, :, :]
                    _, _, _, max_loc_obj = cv2.minMaxLoc(logits_obj.detach().cpu().numpy())
                    obj_pos.append(max_loc_obj)
                    logits_obj_tot += logits_obj
            
            if use_full_prompt:
                for i in range(full_cnt):  # Then the full prompts (normally just 1).
                    logits_total = logits[0, -i-1, takeImg, :, :]
                    _, _, _, max_loc_total = cv2.minMaxLoc(logits_total.detach().cpu().numpy())
                    total_pos.append(max_loc_total)
                    logits_total_tot += logits_total / full_cnt
            else:
                full_cnt = 0

        # Average over the verbs, objects, full sentences seperately.
        x_verb = 0
        y_verb = 0
        for i in range(verb_cnt):
            x_verb += verb_pos[i][0]
            y_verb += verb_pos[i][1]
        if verb_cnt > 0:
            max_loc_verb = (int(x_verb/verb_cnt), int(y_verb/verb_cnt))
        else:
            max_loc_verb = (0, 0)  # Dummy value, not seriously used.
        
        x_obj = 0
        y_obj = 0
        for i in range(noun_cnt):
            x_obj += obj_pos[i][0]
            y_obj += obj_pos[i][1]
        if noun_cnt > 0:
            max_loc_obj = (int(x_obj/noun_cnt), int(y_obj/noun_cnt))
        else:
            max_loc_obj = (0, 0)  # Dummy value, not seriously used.
        
        x_total = 0
        y_total = 0
        for i in range(full_cnt):
            x_total += total_pos[i][0]
            y_total += total_pos[i][1]
        if full_cnt > 0:
            max_loc_total = (int(x_total/full_cnt), int(y_total/full_cnt))
        else:
            max_loc_total = (0, 0)  # Dummy value, not seriously used.
        
        # Now the normal computation continues.
        
        avg_x = 0
        x_cnt = 0
        avg_y = 0
        y_cnt = 0
        if full_cnt >= 1:
            avg_x += mainWeight*max_loc_total[0]
            x_cnt += mainWeight
            avg_y += mainWeight*max_loc_total[1]
            y_cnt += mainWeight
        if verb_cnt >= 1:  # Otherwise no verb was found.  #abs(max_loc_verb[0] - max_loc_total[0]) < W/2 and abs(max_loc_verb[1] - max_loc_total[1]) < H/2:
            avg_x += max_loc_verb[0]
            x_cnt += 1
            avg_y += max_loc_verb[1]
            y_cnt += 1
        if noun_cnt >= 1:#abs(max_loc_obj[0] - max_loc_total[0]) < W/2 and abs(max_loc_obj[1] - max_loc_total[1]) < H/2:
            avg_x += max_loc_obj[0]
            x_cnt += 1
            avg_y += max_loc_obj[1]
            y_cnt += 1
        # logitsFinal = logits[0, :, :, :]
        # a = logitsFinal.shape

        if mode == 1:
            max_loc = (int(avg_x / x_cnt), int(avg_y / y_cnt))
            logitsFinal_prev = (logits_verb_tot + logits_obj_tot + mainWeight * logits_total_tot) / (mainWeight+2)  # Not correct if no verb or noun is found.
            logitsFinal = logits_total_tot.unsqueeze(0).unsqueeze(0)  # logitsFinal_prev.unsqueeze(0).unsqueeze(0)
            
            # return max_loc_obj, logits_total_tot.unsqueeze(0).unsqueeze(0), [text[-1]], [], logits_total_tot.unsqueeze(0).unsqueeze(0)
            

        elif mode == 2:
            logitsFinal_prev = (logits_verb_tot + logits_obj_tot + mainWeight * logits_total_tot) / (mainWeight+2)
            logitsFinal = logitsFinal_prev.unsqueeze(0).unsqueeze(0)
            _, _, _, max_loc = cv2.minMaxLoc(logitsFinal_prev.detach().cpu().numpy())
        
        elif mode == 3:
            logitsFinal_prev = (logits_verb_tot * logits_obj_tot * logits_total_tot ** (mainWeight))
            # mini = torch.min(logitsFinal_prev)
            # maxi = torch.max(logitsFinal_prev)
            # logitsFinal_prev = (logitsFinal_prev - mini) / (maxi - mini)
            logitsFinal = logitsFinal_prev.unsqueeze(0).unsqueeze(0)
            _, _, _, max_loc = cv2.minMaxLoc(logitsFinal_prev.detach().cpu().numpy())
            
            
        all_logits = logits[0, -1, :]
        return max_loc, logitsFinal, [text[-1]], [max_loc_verb, max_loc_obj, max_loc_total], all_logits



def applyGEM_plots(gem_model, img, label, tokenizer=None, template="{}", device="cuda", takeImg=4, singleFrameInput=False, prompt_mode=5, normalize=True):
    """When using with a dataloader, the image is already preprocessed. Used for spatial inference.
    """
    
    if not singleFrameInput:
        images = [frame[0] for frame in img]
    else:
        images = img[0]
    # bc = images[0].shape
    img = frame2tensor_adapted(images, device=device, onlyOne=singleFrameInput)

    
    verb_template = "A photo of a person {} something."
    verb_template_empty = "A photo of a person doing something."  # Rarely the case, since every action has a verb. But sometimes for youCook spacy does not find the verb. Than this sentence is needed.
    object_template = "A photo of a person using {}."
    object_template_empty = "A photo of a person."  # If no object is available, only used for Daly. For the others it is assumed, that there is always a verb and an object present.
    full_template = "A photo of a person {}."
    
    if prompt_mode == 1:
        verb = label[0].split(" ")[0]
        object = " ".join(label[0].split(" ")[1:])
        
        text = [
                verb_template.format(verb),
                object_template.format(object),
                full_template.format(label[0]),
                ]
        verb_cnt = 1
        noun_cnt = 1
        full_cnt = 1

    elif prompt_mode == 2:
        new_prompt = ""
        for i, letter in enumerate(label[0]):  # Add whitespaces before capital letters.
            if i > 0 and letter.isupper():
                new_prompt += " "
            new_prompt += letter

        
        if len(new_prompt.split(" ")) == 1:  # Only a verb is present in the prompt, no need to seperate anything.
            text = [
                verb_template.format(new_prompt),
                object_template_empty,  # Basic part of the object prompt, only used if no object is available.
                full_template.format(new_prompt)
                ]
            verb_cnt = 1
            noun_cnt = 1
            full_cnt = 1
        else:
            verb = new_prompt.split(" ")[0]
            object = " ".join(new_prompt.split(" ")[1:])

            text = [
                verb_template.format(verb),
                object_template.format(object),
                full_template.format(new_prompt),
                ]
            verb_cnt = 1
            noun_cnt = 1
            full_cnt = 1

    elif prompt_mode == 5:  # Later check which verbs or nouns fit the image the best. Then take the best 2 for reference.

        verbs, nouns = parse_sentence(nlp, label[0], mode=2)
        
        verb_text = []
        for verb in verbs:
            verb_text.append(verb_template.format(verb))
        if len(verb_text) == 0:
            verb_text.append(verb_template_empty)  # To ensure the presence of a verb.

        noun_text = []
        for noun in nouns:
            noun_text.append(object_template.format(noun))
        if len(noun_text) == 0:
            noun_text.append(object_template_empty)  # To ensure the presence of an object.
        
        text = []
        
        text.extend(verb_text)
        text.extend(noun_text)
        text.extend([full_template.format(label[0])])
        
        verb_length = len(verb_text)  # How many verb prompts are in text. The best one out of them is chosen.
        obj_length = len(noun_text)  # How many object prompts are in text. The best one out of them is chosen.
        full_cnt = 1
        
        
    with torch.no_grad():
        
        normal_stack_final, importance_stack_final, sims = gem_model(img, text, normalize=normalize, stackWeightsAutoOne=False, get_full_analysis=True)  # [B, num_prompt, W, H]

        # a = len(normal_stack_final)
        # b = len(importance_stack_final)
        # c = normal_stack_final[0].shape
        # d = importance_stack_final[0].shape
        # Out of the max locations from all three prompts, get a final maximum location.
        if prompt_mode < 4:  # Normal case
            
            # Get the results for normal_stack_final
            normal_pos = []  # List of lists with the predicted position for verb, object, full prompt.
            for idx, elm in enumerate(normal_stack_final):
                inner_res = []
                for i in range(3):  # There should be exactly 3 prompts.
                    logits = elm[0, i, takeImg, :, :]
                    _, _, _, max_loc = cv2.minMaxLoc(logits.detach().cpu().numpy())
                    inner_res.append(max_loc)
                normal_pos.append(inner_res)
            # normal_pos.append([0, 0, 0])  # In order to have the same length as importance_pos.
            
            # Get the results for importance_stack_final
            importance_pos = []  # List of lists with the predicted position for verb, object, full prompt.
            for idx, elm in enumerate(importance_stack_final):
                inner_res = []
                for i in range(3):  # There should be exactly 3 prompts.
                    logits = elm[0, i, takeImg, :, :]
                    _, _, _, max_loc = cv2.minMaxLoc(logits.detach().cpu().numpy())
                    inner_res.append(max_loc)
                importance_pos.append(inner_res)
        
        else:
            # Always take the main prompt, but for reference take the most similar verb and object prompt.
            
            # Best fitting verb:
            relevant_sims = sims[0][:verb_length]
            dimis = relevant_sims.shape
            dimis = dimis[0]
            val, ind = torch.topk(relevant_sims, 1)
            verb_pos = ind[0]
            
            # Best fitting object:
            relevant_sims = sims[0][verb_length:verb_length+obj_length]
            dimis = relevant_sims.shape
            dimis = dimis[0]
            val, ind = torch.topk(relevant_sims, 1)
            obj_pos = ind[0]+verb_length
            
            # Get the results for normal_stack_final
            normal_pos = []  # List of lists with the predicted position for verb, object, full prompt.
            for idx, elm in enumerate(normal_stack_final):
                inner_res = []
                
                # The best fitting verb:
                logits_verb = elm[0, verb_pos, takeImg, :, :]
                _, _, _, max_loc_verb = cv2.minMaxLoc(logits_verb.detach().cpu().numpy())
                inner_res.append(max_loc_verb)
                
                # The best fitting object:
                logits_obj = elm[0, obj_pos, takeImg, :, :]  # Because index 0 is at index verb_length of the initial logits.
                _, _, _, max_loc_obj = cv2.minMaxLoc(logits_obj.detach().cpu().numpy())
                inner_res.append(max_loc_obj)
                
                # The (only full prompt):
                logits_total = elm[0, -1, takeImg, :, :]
                _, _, _, max_loc_total = cv2.minMaxLoc(logits_total.detach().cpu().numpy())
                inner_res.append(max_loc_total)
                
                # Append the inner result to the final result.
                normal_pos.append(inner_res)
            
            importance_pos = []  # List of lists with the predicted position for verb, object, full prompt.
            for idx, elm in enumerate(importance_stack_final):
                inner_res = []
                
                # The best fitting verb:
                logits_verb = elm[0, verb_pos, takeImg, :, :]
                _, _, _, max_loc_verb = cv2.minMaxLoc(logits_verb.detach().cpu().numpy())
                inner_res.append(max_loc_verb)
                
                # The best fitting object:
                logits_obj = elm[0, obj_pos, takeImg, :, :]  # Because index 0 is at index verb_length of the initial logits.
                _, _, _, max_loc_obj = cv2.minMaxLoc(logits_obj.detach().cpu().numpy())
                inner_res.append(max_loc_obj)
                
                # The (only full prompt):
                logits_total = elm[0, -1, takeImg, :, :]
                _, _, _, max_loc_total = cv2.minMaxLoc(logits_total.detach().cpu().numpy())
                inner_res.append(max_loc_total)
                
                # Append the inner result to the final result.
                importance_pos.append(inner_res)
        
        return [normal_pos, importance_pos], False, False, False, False


def spatial_test(model=False, dataset= "VHICO", just_visualize=False, masked=True, mask_thresh=0.4,
                 save_path="/data1/felix/master_thesis_plots/images/youcook2/myCode/fromDataloader_",
                 alpha=0.6, mode="normal", filtered=True, template="{}", get_sims=False,
                 mainFramePosition=4, frameStep=1, device="cuda:1", viz_all_imgs=False, viz_number=5,
                 useStack=False, img_attn_only=0, stackWeights=False, stackWeightsAuto=False, get_full_analysis=False):
    """Evaluates GEM for spatial grounding (accuracy and mAp) on the VHICO dataset. It can also visualize the data.
    pipelined: If True, the pipelined version is used. Also the similarities between the prompts and the image are returned.
    get_full_analysis: True if graphs should be created. In order to save computation time, different GEM depths and importances of features are calculated all together.
    """

    # init model and image transform
    preprocess = transform  # gem.get_gem_img_transform()
    # pretrained="/data1/felix/pretrainedModels/ViCLIP/ViCLIP-L_InternVid-200M.pth"
    # gem_model = gem.create_gem_model(model_name=["viclip", "l"],
    #                             pretrained=pretrained,
    #                             device=device)
    pretrained = "/data1/felix/pretrainedModels/ViCLIP/ViCLIP-B_InternVid-FLT-10M.pth"  # "/data2/felix/viCLIP_pretrained/ViCLIP-B_InternVid-200M.pth"  # "/data1/felix/pretrainedModels/ViCLIP/ViCLIP-L_InternVid-10M.pth"
    gem_model = videogem.create_gem_model(model_name=["viclip", "b"],
                                pretrained=pretrained,
                                addBef=False, img_attn_only=img_attn_only, gem_depth=7, ss_attn_iter=1,
                                get_sims=True,
                                device=device, useStack=useStack, stackWeights=stackWeights, stackWeightsAuto=stackWeightsAuto)

    if dataset == "VHICO":
        test_data = vhico_dataset(transform=preprocess, annotations_file="/data2/felix/VHICO/gt_bbox_test.json",
                                  img_dir="/data2/felix/VHICO/data1/alexander/VHICO-DATA/VHICO-Keyframes/",
                                  mode=mode)
        simpleStructure=0
    elif dataset == "YouCookInteractions":
        test_data = youCookInteractions_dataset(transform=preprocess, ignore_outside_boxes=False, filtered=filtered)
        simpleStructure=1
    elif dataset == "GroundingYoutube":
        test_data = groundingYoutube_dataset(transform=preprocess, mode=mode)
        simpleStructure=1
    elif dataset == "Daly":  # The main frame is repeated 8 times in order to get the video.
        test_data = daly_dataset(transform=preprocess)
        simpleStructure=1
    elif dataset == "VHICOVideo":
        test_data = vhico_dataset_video(transform=preprocess, annotations_file="/data2/felix/VHICO/gt_bbox_test.json",
                                  img_dir="/data2/felix/VHICO/data1/alexander/VHICO-DATA/VHICO-Keyframes/",
                                  mode=mode)
        simpleStructure=2
    elif dataset == "DalyVideo":  # 7 surrounding frames of the main frame are used to get the video.
        test_data = daly_dataset_Video(transform=preprocess, mainFramePosition=mainFramePosition, frameStep=frameStep)
        simpleStructure=2
    elif dataset == "YouCookVideo":  # 7 surrounding frames of the main frame are used to get the video.
        test_data = youCookInteractions_dataset_Video(transform=preprocess, mainFramePosition=mainFramePosition, filtered=filtered)
        simpleStructure=2
    elif dataset == "GroundingYoutubeVideo":  # 7 surrounding frames of the main frame are used to get the video.
        test_data = groundingYoutube_dataset_Video(transform=preprocess, mainFramePosition=mainFramePosition)
        simpleStructure=2
    

    use_fn = applyGEMPipeline
    
    if get_full_analysis:
        use_fn = applyGEM_plots
        
    
    if not just_visualize:
        spatial_eval(gem_model, use_fn, test_data, None, template, device, mode=0, test_ioU=False, mask_thresh=mask_thresh, ioU_thresh=0.3, simpleStructure=simpleStructure)
    else:
        visualize_model(gem_model, use_fn, test_data, None, template, device, save_path=save_path, masked=masked, mask_thresh=mask_thresh, simpleStructure=simpleStructure,
                        model_name="viclip", alpha=alpha, viz_all_imgs=viz_all_imgs, viz_number=viz_number, inRGB=True)
    # visualize_model(gem_model, use_fn, test_data, None, template, device, save_path="testing/images/gem/heatmap/", masked=False, simpleStructure=simpleStructure)
    print(f"GEM, dataset: {dataset}, template: {template}, pretrained: {pretrained}, mode: {mode}, filtered: {filtered}, mainFramePosition: {mainFramePosition}, frameStep: {frameStep}")



if __name__ == "__main__":

    # 1. Pip install "videogem"
    # 2. Adapt the paths to the data etc. in "application"
    # 3. Set the default value for "prompt_mode" in applyGEMPipeline(...) according to the specified dataset
    # spatial_test(dataset="YouCookVideo", just_visualize=False, mainFramePosition=4, pipelined=True, device="cuda:0", filtered=False, img_attn_only=4)
    # spatial_test(dataset="GroundingYoutubeVideo", just_visualize=False, mainFramePosition=4, device="cuda:0", pipelined=True, img_attn_only=4)   
    # spatial_test(dataset="VHICOVideo", just_visualize=False, mode="normal", pipelined=True, device="cuda:3", img_attn_only=4)
    # spatial_test(dataset="DalyVideo", just_visualize=False, mainFramePosition=4, frameStep=32, pipelined=True, device="cuda:3", img_attn_only=4)
    pass