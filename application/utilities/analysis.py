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

def temporal_eval(model, applyModel, dataset, thresh=0.5, batch_size=15, tokenizer=False, device="cuda",
                  shuffle=False, visData=False, visualize_num=10, save_path="testing/images/gem/temporal",
                  alpha=0.3, log_batches_interval=50, multiply_logits=1, topk=1, save_path_stats=False):
    """Evaluates a model like GEM or LeGrad on temporal grounding. It saves the average of the topk logits
        per video and frame and caption in a dict in order to not have to calculate it every time when just
        the normalization or some postprocessing is adapted.
    """

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    total = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    max_logit_True = 0  # Sum of all max logits where the label of the image is True.
    max_logit_False = 0  # Sum of all max logits where the label of the image is False.
    
    std_logit_True = 0  # Sum of all standard deviations of logits where the label of the image is True.
    std_logit_False = 0  # Sum of all standard deviations of logits where the label of the image is False.
    start = time.time()
    cnt = 0
    batch_count = 0
    stats = dict()  # For ioU and ioD where first for each video is averaged over the classes (captions) and then ovr the videos.
    # Saves the predictions and labels per video and caption and frame. {video: {caption: {preds: [], labels: [], frame: []}}}
    # preds: [], labels: [], frame: [] are in the same order (but not sorted after frames.)
    save_preds = dict()

    for image, caption, label, vid, frame_number in dataloader:

        image = image.to(device)
        label = label.to(device)
        
        logits = applyModel(model, image, caption, device=device, tokenizer=tokenizer)
        logits = logits * multiply_logits
        # print("logits.shape: ", logits.shape)

        # print(torch.amax(logits, dim=(0, 1, 2, 3)), torch.amin(logits, dim=(0, 1, 2, 3)), torch.mean(logits))
        
        
        predictions, max_logits, std_logits = predicted_frame_logits(logits, thresh=thresh, topk=topk)
        # print("predictions", predictions)
        # print("labels", label)
        # print(std_logits)
        # break
        
        total += predictions.shape[0]
        tp1 = torch.logical_and(predictions, label).detach().cpu()
        tn1 = torch.logical_and(predictions==False, label==False).detach().cpu()
        fp1 = torch.logical_and(predictions==True, label==False).detach().cpu()
        fn1 = torch.logical_and(predictions==False, label==True).detach().cpu()
        
        tp += tp1.sum()
        tn += tn1.sum()
        fp += fp1.sum()
        fn += fn1.sum()
        
        max_logit_True += (max_logits*label).sum().detach().cpu()
        max_logit_False += (max_logits*(label==False)).sum().detach().cpu()
        std_logit_True += (std_logits*label).sum().detach().cpu()
        std_logit_False += (std_logits*(label==False)).sum().detach().cpu()
        
        predictions = predictions.detach()
        label = label.detach()
        frame_number = frame_number.detach()

        for idx in range(len(vid)):  # Update stats.
            vid_key = vid[idx]
            caption_key = caption[idx]
            if vid_key in stats.keys():
                found_vid = stats[vid_key]
            else:
                found_vid = dict()
            if caption_key in found_vid.keys():
                res = found_vid[caption_key]
            else:
                res = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            res["tp"] = res["tp"] + tp1[idx]
            res["fp"] = res["fp"] + fp1[idx]
            res["tn"] = res["tn"] + tn1[idx]
            res["fn"] = res["fn"] + fn1[idx]
            found_vid[caption_key] = res
            stats[vid_key] = found_vid

        if save_path_stats:
            for idx in range(len(vid)):  # Update save_preds.
                vid_key = vid[idx]
                caption_key = caption[idx]
                if vid_key in save_preds.keys():
                    found_vid = save_preds[vid_key]
                else:
                    found_vid = dict()
                if caption_key in found_vid.keys():
                    res = found_vid[caption_key]
                else:
                    res = {"preds": [], "labels": [], "frame": []}
                res["preds"].append(predictions[idx])
                res["labels"].append(label[idx])
                res["frame"].append(frame_number[idx])
                found_vid[caption_key] = res
                save_preds[vid_key] = found_vid
        
        if visData:
            cnt += 1
            adder = "label_" + str(label[0].item()) + "_" + str(cnt)
            print(f"Filename: {adder}")
            visualize(image, caption, logits, False, [], None, save_path=save_path+adder, alpha=alpha, add_text=False)
            if cnt >= visualize_num:
                break

        batch_count += 1
        if total % (log_batches_interval*batch_size) == 0:
            end = time.time()
            takenTime = (end-start)/3600
            ioD_correct, ioU_correct = get_temporal_ioU_ioD(stats)  # Averaged as Brian did, first over captions, then over videos.

            
            # print(f"Memory taken in MiB: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}")
            print("Batch count: ", batch_count)
            print(f"Taken time: {takenTime}h, expected left time: {takenTime * 204941 / total - takenTime}, threshold: {thresh}")
            print(f"IoD (precision): {tp/(tp+fp)}, IoU: {tp/(tp+fp+fn)}, accuracy: {(tp+tn)/(tp+tn+fp+fn)}, recall: {tp/(tp+fn)}, positive prediction rate: {(tp+fp)/(tp+tn+fp+fn)}")
            print(f"Average max logit for samples with positive gt: {max_logit_True/(tp+fn)}, with negative gt: {max_logit_False/(tn+fp)}")
            print(f"Average standard deviation of logits for samples with positive gt: {std_logit_True/(tp+fn)}, with negative gt: {std_logit_False/(tn+fp)}")
            print(f"IoD (Brian): {ioD_correct}, ioU (Brian): {ioU_correct}")

        # torch.cuda.empty_cache()  # Otherwise LeGrad runs into memory allocation issues.
        del image, caption, label, logits, predictions, max_logits, std_logits


    print("--------------------Final Stats---------------------")
    end = time.time()
    takenTime = (end-start)/3600
    ioD_correct, ioU_correct = get_temporal_ioU_ioD(stats)  # Averaged as Brian did, first over captions, then over videos.

    print(f"Taken time: {takenTime}h, expected left time: {takenTime * 204941 / total - takenTime}, threshold: {thresh}")
    print(f"Total: {total}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"IoD (precision): {tp/(tp+fp)}, IoU: {tp/(tp+fp+fn)}, accuracy: {(tp+tn)/(tp+tn+fp+fn)}, recall: {tp/(tp+fn)}, positive prediction rate: {(tp+fp)/(tp+tn+fp+fn)}")
    print(f"Average max logit for samples with positive gt: {max_logit_True/(tp+fn)}, with negative gt: {max_logit_False/(tn+fp)}")
    print(f"Average standard deviation of logits for samples with positive gt: {std_logit_True/(tp+fn)}, with negative gt: {std_logit_False/(tn+fp)}")
    print(f"IoD (Brian): {ioD_correct}, ioU (Brian): {ioU_correct}")
    if save_path_stats:
        with open(save_path_stats, "w") as f:
            json.dump(save_preds, f)
        print(f"Stats saved under: {save_path_stats}")
    
    
def temporal_eval_directly(model, applyModel, dataset, thresh=0.5, batch_size=15, device="cuda",
                  shuffle=False, log_batches_interval=50):
    """Evaluates a model like CLIp or LongCLip on temporal grounding.
    """

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    total = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    start = time.time()
    avg_pos_similarity = 0  # Average similarity of images and captions that fit.
    avg_neg_similarity = 0  # Average similarity of images and captions that do not fit.
    batch_count = 0
    stats = dict()  # For ioU and ioD where first for each video is averaged over the classes (captions) and then ovr the videos.
    for image, caption, label, vid, frame_number in dataloader:

        image = image.to(device)
        label = label.to(device)
        
        similarity = applyModel(model, image, caption, device=device)
        
        
        predictions = predicted_frame_similarity(similarity, thresh=thresh)
        
        total += predictions.shape[0]
        tp1 = torch.logical_and(predictions, label).detach().cpu()
        tn1 = torch.logical_and(predictions==False, label==False).detach().cpu()
        fp1 = torch.logical_and(predictions==True, label==False).detach().cpu()
        fn1 = torch.logical_and(predictions==False, label==True).detach().cpu()
        
        tp += tp1.sum()
        tn += tn1.sum()
        fp += fp1.sum()
        fn += fn1.sum()
        for idx in range(len(vid)):
            vid_key = vid[idx]
            caption_key = caption[idx]
            if vid_key in stats.keys():
                found_vid = stats[vid_key]
            else:
                found_vid = dict()
            if caption_key in found_vid.keys():
                res = found_vid[caption_key]
            else:
                res = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            res["tp"] = res["tp"] + tp1[idx]
            res["fp"] = res["fp"] + fp1[idx]
            res["tn"] = res["tn"] + tn1[idx]
            res["fn"] = res["fn"] + fn1[idx]
            found_vid[caption_key] = res
            stats[vid_key] = found_vid
        
        
        avg_pos_similarity += (similarity*label).sum().detach().cpu()
        avg_neg_similarity += (similarity*(label==False)).sum().detach().cpu()
        # print("aaa: ", (similarity*label))
        # print("bbb: ", similarity*(label==False))

        batch_count += 1
        if total % (log_batches_interval*batch_size) == 0:
            end = time.time()
            takenTime = (end-start)/3600
            ioD_correct, ioU_correct = get_temporal_ioU_ioD(stats)  # Averaged as Brian did, first over captions, then over videos.
            
            # print(f"Memory taken in MiB: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}")
            print("Batch count: ", batch_count)
            print(f"Average pos similarity: {avg_pos_similarity/(tp+fn)}, average neg similarity: {avg_neg_similarity/(tn+fp)}")
            print(f"Summed pos similarity: {avg_pos_similarity}, Summed neg similarity: {avg_neg_similarity}")
            print(f"Taken time: {takenTime}h, expected left time: {takenTime * 204941 / total - takenTime}, threshold: {thresh}")
            print(f"IoD (precision): {tp/(tp+fp)}, IoU: {tp/(tp+fp+fn)}, accuracy: {(tp+tn)/(tp+tn+fp+fn)}, recall: {tp/(tp+fn)}, positive prediction rate: {(tp+fp)/(tp+tn+fp+fn)}")
            print(f"IoD (Brian): {ioD_correct}, ioU (Brian): {ioU_correct}")
        # torch.cuda.empty_cache()  # Otherwise LeGrad runs into memory allocation issues.


    print("--------------------Final Stats---------------------")
    end = time.time()
    takenTime = (end-start)/3600
    ioD_correct, ioU_correct = get_temporal_ioU_ioD(stats)  # Averaged as Brian did, first over captions, then over videos.

    print(f"Average pos similarity: {avg_pos_similarity/(tp+fn)}, average neg similarity: {avg_neg_similarity/(tn+fp)}")
    print(f"Summed pos similarity: {avg_pos_similarity}, Summed neg similarity: {avg_neg_similarity}")
    print(f"Taken time: {takenTime}h, expected left time: {takenTime * 204941 / total - takenTime}, threshold: {thresh}")
    print(f"Total: {total}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"IoD (precision): {tp/(tp+fp)}, IoU: {tp/(tp+fp+fn)}, accuracy: {(tp+tn)/(tp+tn+fp+fn)}, recall: {tp/(tp+fn)}, positive prediction rate: {(tp+fp)/(tp+tn+fp+fn)}") 
    print(f"IoD (Brian): {ioD_correct}, ioU (Brian): {ioU_correct}")

    
    
    