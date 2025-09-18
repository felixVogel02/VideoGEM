import torch
from torch.utils.data import DataLoader
from utilities.visualization import visualize
from utilities.stats_helper import bbox_area, pred_in_bbox, apply_thresh, calc_iOu, dict_to_mAp, predicted_frame_logits, predicted_frame_similarity, get_temporal_ioU_ioD
import math
import time
import psutil
import os
import json




def spatial_eval(model, applyModel, dataset, tokenizer, template="{}", device="cuda", withStats=False, mode=1, test_ioU=True, mask_thresh=0.5, ioU_thresh=0.3, simpleStructure=0):
    """Evaluate a model on a dataset for the pointing game (spatial grounding).
    mode:
    0 -> Use smallest bounding box that includes every human and object bounding box of the image.
    1 -> Use union of all bounding boxes (without the bigbox that includes every bounding box.)
    2 -> Use union of all object bounding boxes
    3 -> Use union of all human bounding boxes.
    
    simpleStructure for f.e. YouCook-Interactions with only one bounding box type.
    """
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    total = 0
    correct_acc = 0
    correct_acc_tensor = False
    # For ioU of the heatmap and the bounding boxes selected by mode.
    correct_ioU = 0
    ioU_sum = 0
    ioU_sum_dict = dict()  # For mAp one should average over the classes. {label: {"correct": y, "total": z}}
    # For ioU of the heatmap and the union of object bounding boxes.
    correct_ioU_objBBox = 0
    ioU_objBBox_sum = 0
    
    bbox_size = 0
    if simpleStructure==0:
        for worked, image, label, bigBBox, objectBoxes, humanBoxes in test_dataloader:
            if not worked:  # Image was probably not found.
                continue
            
            max_loc, logits, _, _, _ = applyModel(model, image, label, tokenizer, template, device=device)
            
            # Get the correct ground truth bounding box depending on the evaluation mode.
            allbboxes = torch.cat((objectBoxes[0], humanBoxes[0]))
            if mode == 0:
                bboxs = bigBBox
            elif mode == 1:
                bboxs = allbboxes
            elif mode == 2:
                bboxs = objectBoxes[0]
            elif mode == 3:
                bboxs = humanBoxes[0]

            correct_predicted = pred_in_bbox(bboxs, max_loc)
            # print("correct_predicted: ", type(correct_predicted))
            # print("max_loc: ", type(max_loc))
            
            if type(correct_predicted) is list:
                correct_predicted = torch.Tensor(correct_predicted).to(device)
                if type(correct_acc_tensor) is bool:  # Initialize it.
                    correct_acc_tensor = torch.zeros_like(correct_predicted).to(device)
                correct_acc_tensor += correct_predicted

            elif type(correct_predicted) is torch.Tensor:
                correct_predicted = correct_predicted.long()
                if type(correct_acc_tensor) is bool:  # Initialize it.
                    correct_acc_tensor = torch.zeros_like(correct_predicted).to(device)
                correct_acc_tensor += correct_predicted
            
            else:
                if correct_predicted:
                    correct_acc += 1
            
            if test_ioU:
                mask = apply_thresh(logits, thresh=mask_thresh)
                mask = mask.squeeze()
                ioU, _, _ = calc_iOu(mask, bboxs)
                ioU_obj, _, _ = calc_iOu(mask, objectBoxes[0])
                ioU_objBBox_sum += ioU_obj
                ioU_sum += ioU
                if label in ioU_sum_dict:
                    ioU_sum_dict[label] = {"correct": ioU_sum_dict[label]["correct"], "total": ioU_sum_dict[label]["total"] + 1}
                else:
                    ioU_sum_dict[label] = {"correct": 0, "total": 1}

                if ioU_obj > ioU_thresh:
                    correct_ioU_objBBox += 1
                if ioU > ioU_thresh:
                    correct_ioU += 1
                    ioU_sum_dict[label] = {"correct": ioU_sum_dict[label]["correct"] + 1, "total": ioU_sum_dict[label]["total"]}
                    

            if withStats:
                _, _, rel = bbox_area(bboxs, image)
                bbox_size += rel
            total += 1
            if total % 100 == 0:
                print(f"Memory taken in MiB: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}")
                if test_ioU:
                    print("mAp: ", dict_to_mAp(ioU_sum_dict))
                if type(correct_acc_tensor) is torch.Tensor:
                    print(f"Acc: {correct_acc_tensor/total}")
                else:
                    print(f"Acc: {correct_acc/total}, ioU_acc: {correct_ioU/total}, ioU_avg: {ioU_sum/total}, ioU_obj_acc: {correct_ioU_objBBox/total}, ioU_obj_avg: {ioU_objBBox_sum/total}, bbox area: {bbox_size/total}")

    elif simpleStructure==1:
        for worked, image, label, bboxs, vid, frame in test_dataloader:
            if not worked:  # Image was probably not found.
                continue
            
            max_loc, logits, _, _, _ = applyModel(model, image, label, tokenizer, template, device=device)

            correct_predicted = pred_in_bbox(bboxs, max_loc)
            
            if type(correct_predicted) is list:
                correct_predicted = torch.Tensor(correct_predicted).to(device)
                if type(correct_acc_tensor) is bool:  # Initialize it.
                    correct_acc_tensor = torch.zeros_like(correct_predicted).to(device)
                correct_acc_tensor += correct_predicted
                
            elif type(correct_predicted) is torch.Tensor:
                correct_predicted = correct_predicted.long()
                if type(correct_acc_tensor) is bool:  # Initialize it.
                    correct_acc_tensor = torch.zeros_like(correct_predicted).to(device)
                correct_acc_tensor += correct_predicted
                
            else:
                if correct_predicted:
                    correct_acc += 1
                    
            if test_ioU:
                mask = apply_thresh(logits, thresh=mask_thresh)
                mask = mask.squeeze()
                ioU, _, _ = calc_iOu(mask, bboxs)
                ioU_sum += ioU
                ioU, _, _ = calc_iOu(mask, bboxs)
                if label in ioU_sum_dict:
                    ioU_sum_dict[label] = {"correct": ioU_sum_dict[label]["correct"], "total": ioU_sum_dict[label]["total"] + 1}
                else:
                    ioU_sum_dict[label] = {"correct": 0, "total": 1}      
                    
                if ioU > ioU_thresh:
                    correct_ioU += 1
                    ioU_sum_dict[label] = {"correct": ioU_sum_dict[label]["correct"] + 1, "total": ioU_sum_dict[label]["total"]}
              

            if withStats:
                _, _, rel = bbox_area(bboxs, image)
                bbox_size += rel
            total += 1
            if total % 100 == 0:
                if test_ioU:
                    print("mAp: ", dict_to_mAp(ioU_sum_dict))
                if type(correct_acc_tensor) is torch.Tensor:
                    print(f"Acc: {correct_acc_tensor/total}")
                else:
                    print(f"Acc: {correct_acc/total}, ioU_acc: {correct_ioU/total}, ioU_avg: {ioU_sum/total}, ioU_obj_acc: {correct_ioU_objBBox/total}, ioU_obj_avg: {ioU_objBBox_sum/total}, bbox area: {bbox_size/total}")

    elif simpleStructure==2:
        for worked, images, framePosition, label, bboxs, vid, frame in test_dataloader:
            if not worked:  # Image was probably not found.
                continue
            
            max_loc, logits, _, _, _ = applyModel(model, images, label, tokenizer, template, device=device, singleFrameInput=False, takeImg=framePosition.item())

            correct_predicted = pred_in_bbox(bboxs, max_loc)
            
            if type(correct_predicted) is list:
                correct_predicted = torch.Tensor(correct_predicted).to(device)
                if type(correct_acc_tensor) is bool:  # Initialize it.
                    correct_acc_tensor = torch.zeros_like(correct_predicted).to(device)
                correct_acc_tensor += correct_predicted
            
            elif type(correct_predicted) is torch.Tensor:
                correct_predicted = correct_predicted.long()
                if type(correct_acc_tensor) is bool:  # Initialize it.
                    correct_acc_tensor = torch.zeros_like(correct_predicted).to(device)
                correct_acc_tensor += correct_predicted
            
            else:
                if correct_predicted:
                    correct_acc += 1
                    
            if test_ioU:
                mask = apply_thresh(logits, thresh=mask_thresh)
                mask = mask.squeeze()
                ioU, _, _ = calc_iOu(mask, bboxs)
                ioU_sum += ioU
                ioU, _, _ = calc_iOu(mask, bboxs)
                if label in ioU_sum_dict:
                    ioU_sum_dict[label] = {"correct": ioU_sum_dict[label]["correct"], "total": ioU_sum_dict[label]["total"] + 1}
                else:
                    ioU_sum_dict[label] = {"correct": 0, "total": 1}      
                    
                if ioU > ioU_thresh:
                    correct_ioU += 1
                    ioU_sum_dict[label] = {"correct": ioU_sum_dict[label]["correct"] + 1, "total": ioU_sum_dict[label]["total"]}
              

            if withStats:
                _, _, rel = bbox_area(bboxs, images[framePosition])
                bbox_size += rel
            total += 1
            if total % 100 == 0:
                if test_ioU:
                    print("mAp: ", dict_to_mAp(ioU_sum_dict))
                if type(correct_acc_tensor) is torch.Tensor:
                    print(f"Acc: {correct_acc_tensor/total}")
                else:
                    print(f"Acc: {correct_acc/total}, ioU_acc: {correct_ioU/total}, ioU_avg: {ioU_sum/total}, ioU_obj_acc: {correct_ioU_objBBox/total}, ioU_obj_avg: {ioU_objBBox_sum/total}, bbox area: {bbox_size/total}")


    print("-------------Final--------------")
    if type(correct_acc_tensor) is torch.Tensor:
        print(f"Total: {total}, Acc: {correct_acc_tensor/total}")
    else:
        print(f"Total: {total}, correct_acc: {correct_acc}, acc: {correct_acc/total}, bbox area: {bbox_size/total}")
    if test_ioU:
        print(f"ioU_acc: {correct_ioU/total}, ioU_avg: {ioU_sum/total}")
        print(f"mAp: {dict_to_mAp(ioU_sum_dict)}")
    if not simpleStructure:
        print(f"ioU_obj_acc: {correct_ioU_objBBox/total}, ioU_obj_avg: {ioU_objBBox_sum/total}")
    


def visualize_model(model, applyModel, dataset, tokenizer, template="{}", device="cuda", save_path="model", masked=False,
                    mask_thresh=0.5, simpleStructure=0, model_name="clip", alpha=0.6, viz_all_imgs=False, viz_number=5, inRGB=False):
    """Visualize a model on a dataset for the pointing game.
    masked: if the heatmap is visualized as a mask (according to mask_thresh)
    """
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    cnt = 0
    # if masked:
    #     save_path = save_path + str(mask_thresh).replace(".", "")
    if simpleStructure==0:
        for worked, image, label, bigBBox, objectBoxes, humanBoxes in test_dataloader:
            if not worked:  # Image was probably not found.
                continue
            predicted, logits, text, otherLocations, _ = applyModel(model, image, label, tokenizer, template, device=device)
            if masked:
                logits = apply_thresh(logits, mask_thresh)
            bboxes = torch.cat((objectBoxes[0], humanBoxes[0]))
            print(logits.shape)
            visualize(image, text, logits, predicted, bboxes, None, save_path=save_path+str(cnt), model_name=model_name,
                      alpha=alpha, otherLocations=otherLocations, inRGB=inRGB)
            cnt += 1
            if cnt == viz_number:
                break
    elif simpleStructure==1:
        for worked, image, label, bigBBox, vid, frame in test_dataloader:
            if not worked:  # Image was probably not found.
                continue
            # if vid[0] != "c00gy-NVzaw":
            #     continue
            predicted, logits, text, otherLocations, _ = applyModel(model, image, label, tokenizer, template, device=device)
            if masked:
                logits = apply_thresh(logits, mask_thresh)
            print(logits.shape)
            visualize(image, text, logits, predicted, bigBBox, None, save_path=save_path + vid[0]+"_"+str(frame[0].item()), model_name=model_name,
                      alpha=alpha, otherLocations=otherLocations, inRGB=inRGB)
            cnt += 1
            if cnt == viz_number:
                break
    elif simpleStructure==2:
        for worked, images, framePosition, label, bigBBox, vid, frame in test_dataloader:
            if not worked:  # Image was probably not found.
                continue
            # if vid[0] != "c00gy-NVzaw":
            #     continue
            predicted, logits, text, otherLocations, all_logits = applyModel(model, images, label, tokenizer, template, device=device, singleFrameInput=False, takeImg=framePosition.item())
            a = all_logits.shape
            if masked:
                logits = apply_thresh(logits, mask_thresh)
            print(logits.shape)
            if viz_all_imgs:  # Visualize all 8 images of the video.
                for i in range(8):
                    visualize(images[i], text, all_logits[i].unsqueeze(0), predicted, bigBBox, None, save_path=save_path + vid[0]+"_"+str(frame[0].item()) + "_pos" + str(i+1) , model_name=model_name,
                              alpha=alpha, otherLocations=otherLocations, inRGB=inRGB)
            else:    
                visualize(images[framePosition.item()], text, logits, predicted, bigBBox, None, save_path=save_path + vid[0]+"_"+str(frame[0].item()), model_name=model_name, alpha=alpha,
                          otherLocations=otherLocations, inRGB=inRGB)  # otherLocations
            cnt += 1
            if cnt == viz_number:
                break

    
    