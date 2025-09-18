import torch
import math
import copy
import cohere
from json import loads
import ast
import time
import torchvision.transforms as transforms
import cv2
import torch



def get_max_pos(input, blurr=False, multidim=False):
    """
    Get the location of the maximum heat point. If requested, the logits are blurred first.
    """

    # print(input.shape)
    if multidim:
        cols = input.shape[-1]
        max_loc = torch.argmax(torch.flatten(input, start_dim=-2, end_dim=-1), dim=-1)
        # max_loc = torch.stack([max_loc // cols, max_loc % cols], -1)
        max_loc = torch.stack([max_loc % cols, max_loc // cols], -1)
        
    else:
        if blurr:
            transform = transforms.GaussianBlur(kernel_size=(31, 31), sigma=(0.5, 0.5))
            input = transform(torch.unsqueeze(input, 0))
        _, _, _, max_loc = cv2.minMaxLoc(input.squeeze().detach().cpu().numpy())
    
    return max_loc


def apply_thresh(logits, thresh=0.5):
    """Takes in a heatmap and returns a bianry mask by applying a threshold.
    """
    
    mask = torch.zeros_like(logits)
    mask[logits > thresh] = 1
    
    return mask
    


def calc_iOu(mask, bboxs):
    """Calculate the intersection over union (iOu) by comparing a binary mask with a bounding box.

    Args:
        mask (_type_): predicted binary mask
        bboxs (_type_): gt bounding box
    """

    gt_mask = torch.zeros_like(mask)
    for box in bboxs:
        x_min, y_min, x_max, y_max = box
        gt_mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 1

    intersection = mask * gt_mask
    union = torch.logical_or(mask, gt_mask)
    
    ioU = intersection.sum() / union.sum()
    
    return ioU, intersection, union
    
    


def bbox_area(allbboxes, image):
    """Calculates the area of allbboxes as well as the image, and the % of the area of the image
    that the bounding boxes take.
    Takes approximately 1.4s for one sample.
    
    bbox: Big bounding box including all other bounding boxes.
    allbboxes: All bounding boxes which area should be considered.
    """
    
    # Iterate over every pixel in bbox and check, if it is covered by a boudningbox.
    # print("allbboxes ", allbboxes)
    coveredPixels = 0
    # Go columnwise through the image and for each column check which part is covered by a bounding box.
    for x in range(image.shape[-1]):
        # The overall start and end value 
        y_vals = []
        for box in allbboxes:
            x_min, y_min, x_max, y_max = box
            if ( x >= x_min and x <= x_max):
                y_vals.append((y_min, y_max))
        y_vals.sort(key=lambda x: x[0])  # Sorted after y_min.
        # if len(y_vals) > 1:
        #     print("y_vals: ", y_vals)
        excluded = 0
        last_max = 0
        for y_min, y_max in y_vals:
            if last_max >= y_max:  # Next bounding box is completely included in the last one.
                continue
            if y_min > last_max:  # Bbox does not overlap with last one.
                excluded += y_max - y_min + 1  # Count both borders.
            elif y_min == last_max:  # Bbox directly go into each other.
                excluded += y_max - y_min  # First border was already counted.
            elif y_min < last_max:  # Bboxes overlap.
                excluded += y_max - last_max
            last_max = y_max
        coveredPixels += excluded

    image_area = image.shape[-1] * image.shape[-2]
    
    rel_area = coveredPixels / image_area
    
    return coveredPixels, image_area, rel_area

def pred_in_bbox(bboxs, pred):
    """Checks if the predicted x, y coordinates are in the bounding boxes given.
    """

    if type(pred) is list:  # For the case get_full_analysis, where a lot of predictions for different settings are made at the same time.
        if len(bboxs) > 1:  # Then this doesn't work correctly.
            print("Only one bbox can be used when get_full_analysis=True")
            raise "error"
            
    for bbox in bboxs:
        
        x_min = bbox[0].item()
        y_min = bbox[1].item()
        x_max = bbox[2].item()
        y_max = bbox[3].item()
        if type(pred) is torch.Tensor:  # For using several weighting techniques for several prompts in the same run.
            x_inside = (pred[:, :, 0] >= x_min) & (pred[:, :, 0] <= x_max)
            y_inside = (pred[:, :, 1] >= y_min) & (pred[:, :, 1] <= y_max)
            res = x_inside & y_inside
            return res

        elif type(pred) is list:  # For creating the big graphs with several depths.
            res = []
            for elm in pred:  # 2 elements for normal_pos, importance_pos.
                sub_res = []
                for sub_elm in elm:  # 7 or 8 elements.
                    sub_sub_res = []
                    
                    for sub_sub_elm in sub_elm:  # 3 elements for the 3 prompts.
                        x_pred = sub_sub_elm[0]
                        y_pred = sub_sub_elm[1]
                        if (x_pred <= x_max and x_pred >= x_min and y_pred <= y_max and y_pred >= y_min):
                            sub_sub_res.append(True)
                        else:
                            sub_sub_res.append(False)
                    sub_res.append(sub_sub_res)
                res.append(sub_res)
            return res
                        
        else:
            # print("correct")
            x_pred = pred[0]
            y_pred = pred[1]
            if (x_pred <= x_max and x_pred >= x_min and y_pred <= y_max and y_pred >= y_min):
                return True
    return False

def dict_to_mAp(input):
    """Calculates the mean average precision (mAp) over classes for a given dict.

    Args:
        input (dict): {label: {"correct": y, "total": z}}
    """

    prec = []
    for key in input.keys():
        prec.append(input[key]["correct"] / input[key]["total"])
    
    mAp = sum(prec) / len(prec)
    return mAp

def predicted_frame_logits(logits, thresh=0.5, topk=1):
    """For temporal grounding it returns if an image is predicted.
    An image is predicted, if the average of it's topk maximum logits is higher than the given threshold.
    This function works batched.
    """

    std = torch.std(logits, dim=(1, 2, 3))
    
    logits = torch.flatten(logits, start_dim=1)  # Keep the batch dimension (first dimension).
    values, _ = torch.topk(logits, topk, dim=1)
    maxi = torch.mean(values, dim=1)
    res = maxi > thresh

    return res, maxi, std


def predicted_frame_similarity(similarity, thresh=0.5):
    """For temporal grounding it returns if an image is predicted.
    An image is predicted, if the similarity between an image and it's captions is high enough.
    """

    predicted = similarity > thresh
    # print("Thresh: ", thresh)
    # print(predicted)

    return predicted

def get_temporal_ioU_ioD(stats):
    """For temporal evaluation calculate the ioD and IoU where for each video ioD and IoU is first calculated over the classes (captions),
    then averaged over the captions and then averaged over the videos.
    """

    ioD_total = 0
    ioU_total = 0
    total_cnt_ioD = 0
    total_cnt_ioU = 0
    for vid_key in stats.keys():
        vid_res = stats[vid_key]
        ioD = 0
        ioU = 0
        cnt_ioD = 0
        cnt_ioU = 0
        for caption in vid_res.keys():
            elm = vid_res[caption]
            if (elm["tp"]+elm["fp"] != 0):  # If only "tn" or "fn" occurred until now, ignore it.
                ioD += elm["tp"]/(elm["tp"]+elm["fp"])
                cnt_ioD += 1
            if (elm["tp"]+elm["fp"]+elm["fn"] != 0):  # If only "tn" occurred until now.
                ioU += elm["tp"]/(elm["tp"]+elm["fp"]+elm["fn"])
                cnt_ioU += 1
        if cnt_ioD != 0:
            ioD_total += ioD / cnt_ioD
            total_cnt_ioD += 1
        if cnt_ioU != 0:
            ioU_total += ioU / cnt_ioU
            total_cnt_ioU += 1

    ioD_total = ioD_total / total_cnt_ioD
    ioU_total = ioU_total / total_cnt_ioU

    return ioD_total, ioU_total

def min_max(logits):
    if logits.ndim > 2:
        B, num_prompt = logits.shape[:2]
        logits_min = logits.reshape(B, num_prompt, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        logits_max = logits.reshape(B, num_prompt, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        logits = (logits - logits_min) / (logits_max - logits_min)
    else:  # Already only the final 2 dimensions of the logits.
        logits_min = logits.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        logits_max = logits.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        logits = (logits - logits_min) / (logits_max - logits_min)
    return logits


def parse_sentence(nlp, text, mode=2):
    """
    Extract the nouns and verbs of a sentence with nlp.
    mode: What exactl is extracted.
    1: All nouns and verbs
    2: All chunks for nouns (noun + adj), and verbs
    3: Cohere
    4: All objects (rest when verb is removed), verbs
    """

    doc = nlp(text)
    nouns = set()
    verbs = set()
    if mode == 1:
        for token in doc:
            if token.pos_ == "VERB":
                verbs.add(token.text)
                # verbs.add(token.lemma_)
            elif token.pos_ == "NOUN":
                nouns.add(token.text)
                # nouns.add(token.lemma_)
    elif mode == 2:
        for token in doc:
            if token.pos_ == "VERB":
                verbs.add(token.text)

        # For the nouns also use their describing adjectives.
        for chunk in doc.noun_chunks:
            root = chunk.root
            noun = root.text
            adj = ""
            if root.pos_ == "NOUN":
                for tok in chunk:
                    if tok != root and tok.pos_ == "ADJ":
                        adj = tok.text + " "
            if True:#len(adj) > 0:
                nouns.add(adj + noun)
    elif mode == 3:  # Using an AI model to get the information.
        # Outputs several pairs.
        # template = "What is the main physical object with its attributes and the action conducted in the following sentence: '{}'\
        #             Only output a list in the format of [<action>, <object>]. If there is no action replace <action> by ''.\
        #             If there is no object replace <object> by ''."
        
        template = "What is the main physical object with its attributes and the action conducted in the following sentence: '{}'\
                    Only output one element in the format of ['<action>', '<object>']. If there is no action replace <action> by ''.\
                    If there is no object replace <object> by ''."
        
        co = cohere.Client("5mhSgXFcWe4rZBDtNK0W0cMcbkUDWG7WPlh1zzLd")
        response = co.chat(
            message=template.format(text)
        )  # Should output a list.
        # lst = "[0, 2, 9, 4, 8]"
        try:
            res = ast.literal_eval(response.text)
            if len(res) == 2:
                verbs = [res[0]]
                nouns = [res[1]]
            else:
                verbs = []
                nouns = []
            time.sleep(1.5)  # Only 40 calls per minute are allowed.
        except:
            verbs = []
            nouns = []
    elif mode == 4:
        for token in doc:
            if token.pos_ == "VERB":
                verbs.add(token.text)
        noun = text
        for verb in verbs:
            noun = noun.replace(verb, "")
        noun = noun.replace("  ", " ")  # After removing a word in the middle its preceding and aftergoing spaces are still there.
        noun = noun.strip()  # Remove Leading and aftergoing spaces.
        nouns = [noun]  # Only one big object description will be added.
    verbs = list(verbs)
    nouns = list(nouns)
    # verb_res = verbs[0] if len(verbs) > 0 else []
    # noun_res = nouns[0] if len(nouns) > 0 else []
    return verbs, nouns
    # return [], []

def extract_subsentences(sent, client, model="mistral-large-latest"):
    """
    For a given sentence extract its subsentences using Mistral. Expecially used for the captions of youCookInteractions.
    """

    chat_response = client.chat.complete(
        model = model,
        messages = [
            {
                "role": "user",
                "content": f"Deconstruct the following sentence into its subsentences: '{sent}. Only return a list of the form: ['subsentence1', 'subsentence2', ...].",
            },
        ]
    )
    try:
        res = ast.literal_eval(chat_response.choices[0].message.content)
    except:  # Could not extract subsentences
        res = [sent]

    return res