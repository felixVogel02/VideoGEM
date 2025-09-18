import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image, ImageFilter
import json
import pickle
import math
from torch.utils.data import DataLoader
import copy

class daly_dataset_Video(Dataset):
    def __init__(self, transform, mainFramePosition=4, frameStep=1):
        """
        mainFramePosition goes from 0 to 7, if 0 the main frame is the first frame out of the 8 frames used for the video.
        frameStep refers to the list framesBefore and framesAfter. The actual frame step is multiplied by 2, since only every second frame was stored.
        If frameStep=32, a seperate annotation file needs to be laoded. This can only be used with mainFramePosition being 3 or 4.
        """
        if frameStep < 32:
            self.frameStep = frameStep
            with open("/data2/felix/Daly/download_videos/frame_annotation.json") as f:
                annos = json.load(f)
        elif frameStep == 32:
            self.frameStep = 1  # Because in the annotation list, the distance between two elements is already 32 frames.
            with open("/data2/felix/Daly/download_videos/frame_annotation.json") as f:
                annos = json.load(f)
        else:
            return "Error, impossible frameStep value"
    
        
        self.transform = transform
        self.mainFramePosition = mainFramePosition
        
        self.data = []
        # vid_num = set()
        for key in annos.keys():
            self.data.append(annos[key])
            # vid_num.add(annos[key]["vid"])
        # print("Anzahl der Videos: ", len(vid_num))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        video = self.data[index]
        vid = video["vid"]
        frameNumber = video["frameNumber"]
        bbox = video["bbox"][0]  # [0] because I used .toList() which added one 0-dimension.
        # bbox are values between 0 and 1.
        caption = video["label"]
        img_path = video["img_path"]
        framesBefore = video["framesBefore"][::-1]  # Invert the ordering of the frames.
        framesAfter = video["framesAfter"]
        img_base_path = video["img_base_path"]
        
        try:
            images = []
            
            # Images before the main image.
            for i in range(0, self.mainFramePosition*self.frameStep, self.frameStep):
                path = img_base_path + str(framesBefore[i]) + ".jpg"
                new_img = Image.open(path)
                images.append(new_img)
            images = images[::-1]  # Invert the order again, such that the frames are ordered as in the video.
            
            # The main image.
            images.append(Image.open(img_path))
            
            # Images after the main image.
            for i in range(0, (7-self.mainFramePosition)*self.frameStep, self.frameStep):
                path = img_base_path + str(framesAfter[i]) + ".jpg"
                new_img = Image.open(path)
                images.append(new_img)
        except:
            return False, 0, 0, 0, 0, 0, 0
        
        if self.transform:
            new_imgs = []
            for img in images:
                new_imgs.append(self.transform(img))
        else:
            new_imgs = images

        # Correct the bounding box position for the main frame (only frame with bounding boxes of interest).
        image = new_imgs[self.mainFramePosition]
        img_shape = image.shape
        img_height = img_shape[-2]
        img_width = img_shape[-1]
        
        bbox_x_min = int(round(bbox[0] * (img_width)))
        bbox_y_min = int(round(bbox[1] * (img_height)))
        bbox_x_max = int(round(bbox[2] * (img_width)))
        bbox_y_max = int(round(bbox[3] * (img_height)))
        new_bbox = torch.Tensor([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max])
    
        return True, new_imgs, self.mainFramePosition, caption, new_bbox, vid, frameNumber

class daly_dataset(Dataset):
    def __init__(self, transform):
        with open("/data2/felix/Daly/download_videos/frame_annotation.json") as f:
            annos = json.load(f)
        
        self.transform = transform
        
        self.data = []
        for key in annos.keys():
            self.data.append(annos[key])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        # index = index + 120
        video = self.data[index]
        vid = video["vid"]
        frameNumber = video["frameNumber"]
        bbox = video["bbox"][0]  # [0] because I used .toList() which added one 0-dimension.
        # bbox are values between 0 and 1.
        caption = video["label"]
        img_path = video["img_path"]
        
        try:
            image = Image.open(img_path)
        except:
            return False, 0, 0, 0, 0, 0
        
        
        if self.transform:
            image = self.transform(image)

        img_shape = image.shape
        img_height = img_shape[-2]
        img_width = img_shape[-1]
        
        # bbox are values between 0 and 1.
        bbox_x_min = int(round(bbox[0] * (img_width)))
        bbox_y_min = int(round(bbox[1] * (img_height)))
        bbox_x_max = int(round(bbox[2] * (img_width)))
        bbox_y_max = int(round(bbox[3] * (img_height)))
        new_bbox = torch.Tensor([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max])
    
        return True, image, caption, new_bbox, vid, frameNumber

class groundingYoutube_dataset_Video(Dataset):
    def __init__(self, transform, mode="normal", mainFramePosition=4):
        with open("/data1/felix/groundingYoutube/annotations/box_anno.json") as f:
            annos = json.load(f)
        
        self.transform = transform
        self.mainFramePosition = mainFramePosition
        
        vid_base_path = "/data1/felix/groundingYoutube/mining_resized_video/"
        img_base_path = "/data1/felix/groundingYoutube/extracted_frames/"
        self.img_surrounding_base_path = "/data2/felix/groundingYoutube/extracted_frames_surrounding/"
        
        self.data = []
        self.captions = set()
        for key in annos.keys():
            for frame in annos[key]:
                vid_path = vid_base_path +  key + ".mp4"
                img_name = img_base_path + key + "/" + str(frame["second"] - 1) + ".jpg"
                if mode == "normal":
                    caption = frame["step_name"]
                elif mode == "verb":
                    caption = frame["step_name"].split(" ")[0]
                elif mode == "object":
                    splitted = frame["step_name"].split(" ")[1:]
                    caption = " ".join(splitted)

                new_res = {
                    "vid": key,
                    "second": frame["second"],
                    "box": frame["box"],
                    "caption": caption,
                    "image_path": img_name,
                    "video_path": vid_path
                }
                self.captions.add(caption)
                self.data.append(new_res)
        
        res = set()
        for i in self.data:
            res.add(i["vid"])
        print("Anzahl der Videos: ", len(res))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        
        # "-2TGG1WFF3A/284"  # Note the saved image has +1 for the frame number!!!
        # "-2TGG1WFF3A/448"
        # "-5m_16X3IMo/79"
        
        # Also good images:
        # /data1/felix/master_thesis_plots/final_images/viclip_vid_grounding_youtube/-_9E7vmxi4o_897-A_photo_of_a_person_flip_pancake..png
        # /data1/felix/master_thesis_plots/final_images/viclip_vid_grounding_youtube/-q5MfvBP9RE_95-A_photo_of_a_person_crack_egg..png
        # /data1/felix/master_thesis_plots/final_images/viclip_vid_grounding_youtube/-q5MfvBP9RE_194-A_photo_of_a_person_flip_egg..png
        # /data1/felix/master_thesis_plots/final_images/viclip_vid_grounding_youtube/-q5MfvBP9RE_266-A_photo_of_a_person_assemble_sandwich..png
        # /data1/felix/master_thesis_plots/final_images/viclip_vid_grounding_youtube/01FLJjr-vJM_51-A_photo_of_a_person_mash_banana..png
        # /data1/felix/master_thesis_plots/final_images/viclip_vid_grounding_youtube/01FLJjr-vJM_96-A_photo_of_a_person_spread_butter..png
        # /data1/felix/master_thesis_plots/final_images/viclip_vid_grounding_youtube/01FLJjr-vJM_92-A_photo_of_a_person_spread_butter..png
        # /data1/felix/master_thesis_plots/final_images/viclip_vid_grounding_youtube/01FLJjr-vJM_103-A_photo_of_a_person_fry_pancake..png
        # /data1/felix/master_thesis_plots/final_images/viclip_vid_grounding_youtube/01FLJjr-vJM_100-A_photo_of_a_person_add_mixture..png
        # /data1/felix/master_thesis_plots/final_images/viclip_vid_grounding_youtube/04sDcaLeEjE_151-A_photo_of_a_person_roll_up..png
        # /data1/felix/master_thesis_plots/final_images/viclip_vid_grounding_youtube/0GGOV4f7mKQ_51-A_photo_of_a_person_heat_pan..png
        
        # Only for visualization.
        # index = 0
        # while True:
        #     elm = self.data[index]
        #     if elm["vid"] == "PzLb9Pc6q1o":
        #         if elm["second"] == 289:
        #             break
        #     index += 1
        # while True:
        #     if "-5m_16X3IMo/79" in self.data[index]["image_path"]:
        #         break
            
            
            # if "01FLJjr-vJM/102" in self.data[index]["image_path"]:
            #     break
            # if "-2TGG1WFF3A/448" in self.data[index]["image_path"]:
            #     break
            # if "-5m_16X3IMo/79" in self.data[index]["image_path"]:
            #     break
            
            
            
            # if "-2TGG1WFF3A/284" in self.data[index]["image_path"]:
            #     break
            # if "-2TGG1WFF3A/448" in self.data[index]["image_path"]:
            #     break
            # if "-5m_16X3IMo/79" in self.data[index]["image_path"]:
            #     break
            # index += 1
        # index = index+950

        video = self.data[index]
        vid = video["vid"]
        second = video["second"]
        bbox = video["box"]
        caption = video["caption"]
        img_path = video["image_path"]
        vid_path = video["video_path"]
        
        try:
            images = []
            dst_name = self.img_surrounding_base_path + vid + "/"
            # Images before the main image.
            for i in range(self.mainFramePosition, 0, -1):
                path = dst_name + str(second - i*0.25) + ".jpg"
                new_img = Image.open(path)
                images.append(new_img)
            
            # The main image.
            images.append(Image.open(img_path))
            
            # Images after the main image.
            for i in range(1, 7-self.mainFramePosition+1):
                path = dst_name + str(second + i*0.25) + ".jpg"
                new_img = Image.open(path)
                images.append(new_img)
        except:
            return False, 0, 0, 0, 0, 0, 0
    
        width, height = images[self.mainFramePosition].size
        if self.transform:
            new_imgs = []
            for img in images:
                new_imgs.append(self.transform(img))
        else:
            new_imgs = images

        img_shape = new_imgs[self.mainFramePosition].shape
        img_height = img_shape[-2]
        img_width = img_shape[-1]
        bbox_x_min = int(round(bbox[0] * (img_width / width)))
        bbox_y_min = int(round(bbox[1] * (img_height / height)))
        bbox_x_max = int(round(bbox[2] * (img_width / width)))
        bbox_y_max = int(round(bbox[3] * (img_height / height)))
        new_bbox = torch.Tensor([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max])
    
        return True, new_imgs, self.mainFramePosition, caption, new_bbox, vid, second

class groundingYoutube_dataset(Dataset):
    def __init__(self, transform, mode="normal"):
        with open("/data1/felix/groundingYoutube/annotations/box_anno.json") as f:
            annos = json.load(f)
        
        self.transform = transform
        
        vid_base_path = "/data1/felix/groundingYoutube/mining_resized_video/"
        img_base_path = "/data1/felix/groundingYoutube/extracted_frames/"
        
        self.data = []
        self.captions = set()
        for key in annos.keys():
            for frame in annos[key]:
                vid_path = vid_base_path +  key + ".mp4"
                img_name = img_base_path + key + "/" + str(frame["second"] - 1) + ".jpg"
                if mode == "normal":
                    caption = frame["step_name"]
                elif mode == "verb":
                    caption = frame["step_name"].split(" ")[0]
                elif mode == "object":
                    splitted = frame["step_name"].split(" ")[1:]
                    caption = " ".join(splitted)

                new_res = {
                    "vid": key,
                    "second": frame["second"],
                    "box": frame["box"],
                    "caption": caption,
                    "image_path": img_name,
                    "video_path": vid_path
                }
                self.captions.add(caption)
                self.data.append(new_res)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        # index = index + 750
        
        video = self.data[index]
        vid = video["vid"]
        second = video["second"]
        bbox = video["box"]
        caption = video["caption"]
        img_path = video["image_path"]
        vid_path = video["video_path"]
        
        try:
            image = Image.open(img_path)  # read_image(img_path)
            # image = image.filter(ImageFilter.BLUR)
            # image = image.filter(ImageFilter.BLUR)
        except:
            return False, 0, 0, 0, 0, 0
    
        width, height = image.size
        if self.transform:
            image = self.transform(image)

        img_shape = image.shape
        img_height = img_shape[-2]
        img_width = img_shape[-1]
        bbox_x_min = int(round(bbox[0] * (img_width / width)))
        bbox_y_min = int(round(bbox[1] * (img_height / height)))
        bbox_x_max = int(round(bbox[2] * (img_width / width)))
        bbox_y_max = int(round(bbox[3] * (img_height / height)))
        new_bbox = torch.Tensor([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max])
    
        return True, image, caption, new_bbox, vid, second


class youCookInteractions_dataset_Video(Dataset):
    def __init__(self, transform, ignore_outside_boxes=False, filtered=True, mainFramePosition=4):
        with open("/data1/felix/youCookInteractions/annotations/frame_annotation.json") as f:
            data = json.load(f)
        
        self.data = []
        self.filtered = filtered
        self.mainFramePosition = mainFramePosition
        self.img_surrounding_base_path = "/data2/felix/youCookInteractions/extracted_frames_surrounding/"

        ignore_vids = [
            "2HsWZdKKBGg",
            "OpURFOTdycE",
            "btikV_DUoCM",
            "9ekEjxd-A_Y",
            "DrXVuj1Qowo",
            "HF49t8uVJOE",
            "FliMoBfG72Y"
        ]
        self.no_rescale_vids = [
            "WYAFPvlDB_A",
            "vU2lND4YQjM",
            "oJZUxU9szWA",
            "1vJp-jaIaeE",
            "4Y8vVGsv4JE",
            "FzhJGCaaYVs",
            "30Q8k57Kbz4"
        ]
        self.rescale_vids = [
            "Re46osq_NkI"
        ]
        
        for key in data.keys():
            if filtered and data[key]["vid"] in ignore_vids:
                continue
            else:
                self.data.append(data[key])
        
        self.transform = transform
        self.ignore_outside_boxes = ignore_outside_boxes
        
        res = set()
        for i in self.data:
            res.add(i["vid"])
        print("Anzahl der Videos: ", len(res))
        print("Anzahl der Bilder: ", len(self.data))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        # index = index +200
        # /data1/felix/master_thesis_plots/final_images/viclip_vid_grounding_youtube/T_fPNAK5Ecg_472-A_photo_of_a_person_add_butter_and_milk_to_the_mashed_potatoes_and_mix..png
        # while True:
        #     if "T_fPNAK5Ecg/473" in self.data[index]["image_path"]:
        #         break
        #     index += 1
        
        video = self.data[index]
        vid = video["vid"]
        frame = video["frame"]
        bbox = video["bbox"]
        caption = video["caption"]
        img_path = video["image_path"]
        
        try:
            second = frame - 1
            images = []
            dst_name = self.img_surrounding_base_path + vid + "/"
            # Images before the main image.
            for i in range(self.mainFramePosition, 0, -1):
                path = dst_name + str(second - i*0.25) + ".jpg"
                new_img = Image.open(path)
                images.append(new_img)
                        
            # The main image.
            images.append(Image.open(img_path))
            
            # Images after the main image.
            for i in range(1, 7-self.mainFramePosition+1):
                path = dst_name + str(second + i*0.25) + ".jpg"
                new_img = Image.open(path)
                images.append(new_img)
        except:
            return False, 0, 0, 0, 0, 0, 0
    
        width, height = images[self.mainFramePosition].size
        
        if self.filtered:
            if vid not in self.no_rescale_vids:
                if height > 360 and width > 640:  # Bounding boxes were created for rescaled images.
                    height = 360
                    width = 640
            if vid in self.rescale_vids:
                height = 360
                width = 640
        else:
            if height > 360 and width > 640:  # Bounding boxes were created for rescaled images.
                height = 360
                width = 640
        
        if self.transform:
            new_imgs = []
            for img in images:
                new_imgs.append(self.transform(img))
        else:
            new_imgs = images
        img_shape = new_imgs[self.mainFramePosition].shape
        # Bounding box needs to be transformed as well to fit the transformed image.
        img_height = img_shape[-2]
        img_width = img_shape[-1]
        bbox_x_min = int(round(bbox[0] * (img_width / width)))
        bbox_y_min = int(round(bbox[1] * (img_height / height)))
        bbox_x_max = int(round(bbox[2] * (img_width / width)))
        bbox_y_max = int(round(bbox[3] * (img_height / height)))
        new_bbox = torch.Tensor([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max])
        
        if self.ignore_outside_boxes:
            if (bbox_x_min < 0 or bbox_x_min > img_width) or \
                    (bbox_x_max < 0 or bbox_x_max > img_width) or \
                    (bbox_y_min < 0 or bbox_y_min > img_height) or \
                    (bbox_y_max < 0 or bbox_y_max > img_height):

                return False, 0, 0, 0, 0, 0, 0
        
        return True, new_imgs, self.mainFramePosition, caption, new_bbox, vid, second



class youCookInteractions_dataset(Dataset):
    def __init__(self, transform, ignore_outside_boxes=False, filtered=True):
        with open("/data1/felix/youCookInteractions/annotations/frame_annotation.json") as f:
            data = json.load(f)
        
        self.data = []
        self.filtered = filtered
        ignore_vids = [
            "2HsWZdKKBGg",
            "OpURFOTdycE",
            "btikV_DUoCM",
            "9ekEjxd-A_Y",
            "DrXVuj1Qowo",
            "HF49t8uVJOE",
            "FliMoBfG72Y"
        ]
        self.no_rescale_vids = [
            "WYAFPvlDB_A",
            "vU2lND4YQjM",
            "oJZUxU9szWA",
            "1vJp-jaIaeE",
            "4Y8vVGsv4JE",
            "FzhJGCaaYVs",
            "30Q8k57Kbz4"
        ]
        self.rescale_vids = [
            "Re46osq_NkI"
        ]
        
        for key in data.keys():
            if filtered and data[key]["vid"] in ignore_vids:
                continue
            else:
                self.data.append(data[key])
        
        self.transform = transform
        # a = len(self.data)
        self.ignore_outside_boxes = ignore_outside_boxes
        print("Anzahl dr Bilder: ", len(self.data))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # index = index + 640
        
        video = self.data[index]
        vid = video["vid"]
        frame = video["frame"]
        bbox = video["bbox"]
        caption = video["caption"]
        img_path = video["image_path"]
        
        try:
            image = Image.open(img_path)  # read_image(img_path)
        except:
            return False, 0, 0, 0, 0, 0
    
        width, height = image.size
        
        if self.filtered:
            if vid not in self.no_rescale_vids:
                if height > 360 and width > 640:  # Bounding boxes were created for rescaled images.
                    height = 360
                    width = 640
            if vid in self.rescale_vids:
                height = 360
                width = 640
        else:
            if height > 360 and width > 640:  # Bounding boxes were created for rescaled images.
                height = 360
                width = 640
        
        if self.transform:
            image = self.transform(image)
        img_shape = image.shape
        # Bounding box needs to be transformed as well to fit the transformed image.
        img_height = img_shape[-2]
        img_width = img_shape[-1]
        bbox_x_min = int(round(bbox[0] * (img_width / width)))
        bbox_y_min = int(round(bbox[1] * (img_height / height)))
        bbox_x_max = int(round(bbox[2] * (img_width / width)))
        bbox_y_max = int(round(bbox[3] * (img_height / height)))
        new_bbox = torch.Tensor([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max])
        
        if self.ignore_outside_boxes:
            if (bbox_x_min < 0 or bbox_x_min > img_width) or \
                    (bbox_x_max < 0 or bbox_x_max > img_width) or \
                    (bbox_y_min < 0 or bbox_y_min > img_height) or \
                    (bbox_y_max < 0 or bbox_y_max > img_height):

                return False, 0, 0, 0, 0, 0
        
        return True, image, caption, new_bbox, vid, frame
    

class vidSTG_dataset(Dataset):
    def __init__(self, transform):
        with open("/data1/felix/VidSTG/annotation/frames/test_frame_annotation.json") as f:
            self.data = json.load(f)
        
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        img_path = data["path"]
        label = data["positive"]
        caption = data["captions"][0]["description"]  # [0] because every sample only has one caption.
        vid = data["vid"]
        frame_number = data["frame_number"]

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        
        return image, caption, label, vid, frame_number


class vhico_dataset_video(Dataset):
    def __init__(self, annotations_file="/data1/alexander/VHICO-DATA/data/gt_bbox_test.json",
                 img_dir="/data1/alexander/VHICO-DATA/VHICO-Keyframes/", transform=None, target_transform=None,
                 mode="normal", mainFramePosition=4, img_surrounding_base_path="/data1/felix/VHICO/extracted_frames/test/"):

        self.mainFramePosition = mainFramePosition
        self.img_surrounding_base_path = img_surrounding_base_path
        
        anno_data = pickle.load( open(annotations_file, 'rb') )
        self.img_annotations = anno_data["annos"]
        self.img_labels = []  # Transform the annotations into an ordered list.
        for elm in self.img_annotations.keys():
            first = self.img_annotations[elm]
            for elm1 in first.keys():
                snd = first[elm1]
                for trd in snd:
                    # Combined boundingbox (including object as well as human)
                    x_min = math.inf
                    y_min = math.inf
                    x_max = 0
                    y_max = 0
                    allbbox = copy.deepcopy(trd["object"])
                    allbbox.extend(trd["human"])
                    if mode == "normal":
                        caption = trd["label"]
                    elif mode == "verb":
                        caption = trd["label"].split(" ")[0]
                    elif mode == "object":
                        splitted = trd["label"].split(" ")[1:]
                        caption = " ".join(splitted)
                    for obj in allbbox:
                        if obj[0] < x_min:
                            x_min = obj[0]
                        if obj[1] < y_min:
                            y_min = obj[1]
                        if obj[2] > x_max:
                            x_max = obj[2]
                        if obj[3] > y_max:
                            y_max = obj[3]
                    path_splitted = trd["path"].split("/")
                    frame = int(path_splitted[-1].split(".")[0])  # Remove the ".jpg" at the end.
                    self.img_labels.append({"class": elm, "vid": elm1, "frame": frame, "path": trd["path"], "object": trd["object"],"human": trd["human"], "bbox":torch.tensor([x_min, y_min, x_max, y_max]), "label": caption, "height": trd["height"], "width": trd["width"]})            
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        # a = len(self.img_labels)
        # b = 3
        vid_num = set()
        for elm in self.img_labels:
            vid_num.add(elm["vid"])
        print("Anzahl der Videos: ", len(vid_num))
        print("Anzahl der Bilder: ", len(self.img_labels))
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # For visualization
        # getty-man-using-board-and-box-method-to-catch-gobies-and-fish-on-mudflat-video-id504531797_14.mp4/67
        # yt-lIucFFQivsE_9
        # yt-sSutdGZ1vEs_33
        
        #'test/catching/getty-man-using-board-and-box-method-to-catch-gobies-and-fish-on-mudflat-video-id504531797_14.mp4/47.jpg'
        # 'test/catching/getty-man-using-board-and-box-method-to-catch-gobies-and-fish-on-mudflat-video-id504531797_14.mp4/67.jpg'
        # while True:
        #     # print(self.img_labels[idx]["path"])
        #     # a = self.img_labels[idx]["path"]
        #     # if "getty-man-using-board-and-box-method-to-catch-gobies-and-fish-on-mudflat-video-id504531797_14.mp4/67" in self.img_labels[idx]["path"]:
        #     #     break
        #     # if "yt-sSutdGZ1vEs_33" in self.img_labels[idx]["path"]:
        #     #     break
        #     if "yt-lIucFFQivsE_9" in self.img_labels[idx]["path"]:
        #         break
        #     idx += 1
        
        # idx = idx+10
            
            
        img_path = os.path.join(self.img_dir, self.img_labels[idx]["path"])
        img_class = self.img_labels[idx]["class"]
        vid = self.img_labels[idx]["vid"]
        frame = self.img_labels[idx]["frame"]
        # print(f"Image path: {img_path}")
        # print(f"Image label: {self.img_labels[idx]['label']}")
        try:
            main_img = Image.open(img_path)
        except:
            return False, 0, 0, 0, 0, 0, 0
        try:
            images = []
            dst_name = self.img_surrounding_base_path + img_class + "/" + vid + "/"
            # Images before the main image.
            for i in range(self.mainFramePosition, 0, -1):
                path = dst_name + str(frame -i*4) + ".jpg"
                new_img = Image.open(path)
                images.append(new_img)
                        
            # The main image.
            images.append(main_img)
            
            # Images after the main image.
            for i in range(1, 7-self.mainFramePosition+1):
                path = dst_name + str(frame +i*4) + ".jpg"
                new_img = Image.open(path)
                images.append(new_img)
        except:
            images = 8*[main_img]  # The main frame is there, just not the surrounding frames.
        
        if self.transform:
            new_imgs = []
            for img in images:
                new_imgs.append(self.transform(img))
        elif self.target_transform:
            new_imgs = []
            for img in images:
                new_imgs.append(self.target_transform(img))
        else:
            new_imgs = images
        
        # a = self.img_labels[idx]
        bbox = self.img_labels[idx]["bbox"]
        label = self.img_labels[idx]["label"]
        width = self.img_labels[idx]["width"]
        height = self.img_labels[idx]["height"]
        dims = torch.tensor([width, height])
        
        img_shape = new_imgs[self.mainFramePosition].shape
        # Bounding box needs to be transformed as well to fit the transformed image.
        img_height = img_shape[-2]
        img_width = img_shape[-1]
        bbox_x_min = int((bbox[0] * (img_width / width)).round())
        bbox_y_min = int((bbox[1] * (img_height / height)).round())
        bbox_x_max = int((bbox[2] * (img_width / width)).round())
        bbox_y_max = int((bbox[3] * (img_height / height)).round())
        new_bbox = torch.Tensor([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max])
        
        # humanBoxes = []
        # objectBoxes = []
        # for obj in self.img_labels[idx]["object"]:
        #     bbox_x_min = int((obj[0] * (img_width / width)).round())
        #     bbox_y_min = int((obj[1] * (img_height / height)).round())
        #     bbox_x_max = int((obj[2] * (img_width / width)).round())
        #     bbox_y_max = int((obj[3] * (img_height / height)).round())
        #     objectBoxes.append(torch.Tensor([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max]))
        # for obj in self.img_labels[idx]["human"]:
        #     bbox_x_min = int((obj[0] * (img_width / width)).round())
        #     bbox_y_min = int((obj[1] * (img_height / height)).round())
        #     bbox_x_max = int((obj[2] * (img_width / width)).round())
        #     bbox_y_max = int((obj[3] * (img_height / height)).round())
        #     humanBoxes.append(torch.Tensor([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max]))
        
        # return True, image, label, new_bbox, objectBoxes, humanBoxes
        return True, new_imgs, self.mainFramePosition, label, new_bbox, vid, frame




class vhico_dataset(Dataset):
    def __init__(self, annotations_file="/data1/alexander/VHICO-DATA/data/gt_bbox_test.json",
                 img_dir="/data1/alexander/VHICO-DATA/VHICO-Keyframes/", transform=None, target_transform=None,
                 mode="normal"):

        anno_data = pickle.load( open(annotations_file, 'rb') )
        self.img_annotations = anno_data["annos"]
        self.img_labels = []  # Transform the annotations into an ordered list.
        for elm in self.img_annotations.keys():
            first = self.img_annotations[elm]
            for elm1 in first.keys():
                snd = first[elm1]
                for trd in snd:
                    # Combined boundingbox (including object as well as human)
                    x_min = math.inf
                    y_min = math.inf
                    x_max = 0
                    y_max = 0
                    allbbox = copy.deepcopy(trd["object"])
                    allbbox.extend(trd["human"])
                    if mode == "normal":
                        caption = trd["label"]
                    elif mode == "verb":
                        caption = trd["label"].split(" ")[0]
                    elif mode == "object":
                        splitted = trd["label"].split(" ")[1:]
                        caption = " ".join(splitted)
                    for obj in allbbox:
                        if obj[0] < x_min:
                            x_min = obj[0]
                        if obj[1] < y_min:
                            y_min = obj[1]
                        if obj[2] > x_max:
                            x_max = obj[2]
                        if obj[3] > y_max:
                            y_max = obj[3]
                    self.img_labels.append({"path": trd["path"], "object": trd["object"],"human": trd["human"], "bbox":torch.tensor([x_min, y_min, x_max, y_max]), "label": caption, "height": trd["height"], "width": trd["width"]})            
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        # a = len(self.img_labels)
        # b = 3
        # # vid_num = set()
        # for elm in self.img_labels:
        #     self.data.append(annos[key])
        #     # vid_num.add(annos[key]["vid"])
        # # print("Anzahl der Videos: ", len(vid_num))
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx]["path"])
        # print(f"Image path: {img_path}")
        # print(f"Image label: {self.img_labels[idx]['label']}")
        try:
            image = Image.open(img_path)  # read_image(img_path)
        except:
            return False, 0, 0, 0, 0, 0
        # a = self.img_labels[idx]
        bbox = self.img_labels[idx]["bbox"]
        label = self.img_labels[idx]["label"]
        width = self.img_labels[idx]["width"]
        height = self.img_labels[idx]["height"]
        dims = torch.tensor([width, height])
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        img_shape = image.shape
        # Bounding box needs to be transformed as well to fit the transformed image.
        img_height = img_shape[-2]
        img_width = img_shape[-1]
        bbox_x_min = int((bbox[0] * (img_width / width)).round())
        bbox_y_min = int((bbox[1] * (img_height / height)).round())
        bbox_x_max = int((bbox[2] * (img_width / width)).round())
        bbox_y_max = int((bbox[3] * (img_height / height)).round())
        new_bbox = torch.Tensor([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max])
        
        humanBoxes = []
        objectBoxes = []
        for obj in self.img_labels[idx]["object"]:
            bbox_x_min = int((obj[0] * (img_width / width)).round())
            bbox_y_min = int((obj[1] * (img_height / height)).round())
            bbox_x_max = int((obj[2] * (img_width / width)).round())
            bbox_y_max = int((obj[3] * (img_height / height)).round())
            objectBoxes.append(torch.Tensor([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max]))
        for obj in self.img_labels[idx]["human"]:
            bbox_x_min = int((obj[0] * (img_width / width)).round())
            bbox_y_min = int((obj[1] * (img_height / height)).round())
            bbox_x_max = int((obj[2] * (img_width / width)).round())
            bbox_y_max = int((obj[3] * (img_height / height)).round())
            humanBoxes.append(torch.Tensor([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max]))
        
        return True, image, label, new_bbox, objectBoxes, humanBoxes