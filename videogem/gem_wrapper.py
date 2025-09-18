import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from open_clip.transformer import VisionTransformer

from .gem_utils import SelfSelfAttention, GEMResidualBlock, GEMResidualBlockViCLIP, GEMResidualBlockLast, modified_vit_forward


class GEMWrapper(nn.Module):
    def __init__(self, model, tokenizer, depth=7, ss_attn_iter=1, ss_attn_temp=None, model_type="clip", size=None, addBef=False, img_attn_only=0, avg_prompts=1, get_sims=False,
                 useStack=False, stackWeights=False, stackWeightsAuto=False, img_proj_final=False, txt_proj_final=False):
        """
        img_proj_final: When Brians model is used, the final vision projection is separate. It needs to be applied afterwards manually.
        txt_proj_final: When Brians model is used, the final language projection is separate. It needs to be applied afterwards manually.
        addBef: The output of the selfself attention and the output of the normal attention are added before.
        get_sims: If True, also the similarities between the image and the prompts are returned.
        stackWeights: Works normally as if no stack was used, but weights for the different outputs of the initial and the following self-self attention outputs can be given.
                    List of weights of same length as are elements in the stack.
        img_attn_only:
            0 -> Normal GEM
            1 -> Attention only to tokens of the same image
            2 -> Attention only to tokens of the other images
            3 -> Attention to tokens of the same image and tokens of the other images separately.
            4 -> Attention to all tokens, tokens of the same image and tokens of the other images separately.
        
        """
        super(GEMWrapper, self).__init__()
        self.img_proj_final = img_proj_final
        self.txt_proj_final = txt_proj_final
        self.model_type = model_type
        self.avg_prompts = avg_prompts
        self.get_sims = get_sims
        self.stackWeightsAuto = stackWeightsAuto
        self.useStack = useStack
        if model_type in ["clip", "openclip"]:
            self.model = model
            self.visual = model.visual
            self.visual.model_type = model_type
            self.visual.stackWeights = stackWeights
            self.tokenizer = tokenizer
            self.depth = depth
            self.ss_attn_iter = ss_attn_iter
            self.ss_attn_temp = ss_attn_temp
            self.patch_size = self.visual.patch_size[0]
            self.addBef=addBef
            self.img_attn_only = img_attn_only
            self.apply_gem()
            
        elif model_type == "viclip":
            self.model = model
            self.visual = model.vision_encoder
            self.visual.model_type = model_type
            self.visual.stackWeights = stackWeights
            
            
            self.tokenizer = tokenizer
            self.depth = depth
            self.ss_attn_iter = ss_attn_iter
            self.ss_attn_temp = ss_attn_temp
            if size in ["l", "L"]:  # 14 for clip_joint_l14, 16 for clip_joint_b16.
                self.patch_size = 14
            elif size in ["b", "B"]:
                self.patch_size = 16
            
            # Imagesize is fixed two 224 * 224 => Depending on the patch_size of 14 or 16, 16 or 14 tokens will be created.
            
            self.addBef=addBef
            self.img_attn_only = img_attn_only
            self.apply_gem()
            

    def apply_gem(self):
        for i in range(1, self.depth+1):
            # Extract info from the original ViT
            num_heads = self.visual.transformer.resblocks[-i].attn.num_heads
            dim = int(self.visual.transformer.resblocks[-i].attn.head_dim * num_heads)
            qkv_bias = True
            # Init the self-self attention layer
            ss_attn = SelfSelfAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                        ss_attn_iter=self.ss_attn_iter, ss_attn_temp=self.ss_attn_temp,
                                        img_attn_only=self.img_attn_only, token_cnt=224//self.patch_size)  # Imagesize is fixed two 224 * 224 => Depending on the patch_size of 14 or 16, 16 or 14 tokens will be created, only used for ViCLIP.
            # Copy necessary weights
            ss_attn.qkv.weight.data = self.visual.transformer.resblocks[-i].attn.in_proj_weight.clone()
            ss_attn.qkv.bias.data = self.visual.transformer.resblocks[-i].attn.in_proj_bias.clone()
            ss_attn.proj.weight.data = self.visual.transformer.resblocks[-i].attn.out_proj.weight.clone()
            ss_attn.proj.bias.data = self.visual.transformer.resblocks[-i].attn.out_proj.bias.clone()
            # Swap the original Attention with our SelfSelfAttention
            self.visual.transformer.resblocks[-i].attn = ss_attn
            # Wrap Residual block to handle SelfSelfAttention outputs
            # if i == 1:
            #     self.visual.transformer.resblocks[-i] = GEMResidualBlockLast(self.visual.transformer.resblocks[-i])
            if self.model_type in ["clip", "openclip"]:
                self.visual.transformer.resblocks[-i] = GEMResidualBlock(self.visual.transformer.resblocks[-i])
            elif self.model_type in ["viclip"]:
                self.visual.transformer.resblocks[-i] = GEMResidualBlockViCLIP(self.visual.transformer.resblocks[-i], addBef=self.addBef)
                
        # Modify ViT's forward function
        self.visual.forward = modified_vit_forward.__get__(self.visual, VisionTransformer)
        return

    def encode_text(self, text: list):
        prompts = [f'{cls}' for cls in text]
        #prompts = [f'a photo of a {cls}.' for cls in text]
        tokenized_prompts = self.tokenizer(prompts).to(self.visual.proj.device)
        text_embedding = self.model.encode_text(tokenized_prompts)
        text_embedding = F.normalize(text_embedding, dim=-1)
        return text_embedding.unsqueeze(0)

    def min_max(self, logits, frames=1):
        """For spatial grounding, minmax normalize over the frames (one 0 and one 1 for each frame).
        """
        B, num_prompt = logits.shape[:2]
        if frames == 1:  # clip or openclip
            logits_min = logits.reshape(B, num_prompt, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
            logits_max = logits.reshape(B, num_prompt, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
            logits = (logits - logits_min) / (logits_max - logits_min)
        else:
            logits_min = logits.reshape(B, num_prompt, frames, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
            logits_max = logits.reshape(B, num_prompt, frames, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
            logits = (logits - logits_min) / (logits_max - logits_min)
        return logits

    def min_max_new(self, logits, frames=1):
        """For temporal grounding, minmax normalize over the whole video (one 0 and one 1 value in the whole video).
        """
        B, num_prompt = logits.shape[:2]
        logits_min = logits.reshape(B, num_prompt, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        logits_max = logits.reshape(B, num_prompt, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        logits = (logits - logits_min) / (logits_max - logits_min)
        return logits

    def avg_text_prompts(self, text, avg_prompts):
        """
        Often it boosts performance, if the embedding of several text prompts is averaged.
        This function averages "avg_prompts" following prompts to one.
        """
        _, num_prompt, dim = text.shape
        text = text.reshape(1, -1, avg_prompts, dim)
        text = torch.mean(text, dim=2)
        return text
    
    
    def get_sim_list(self, stack_init, text_embeddings, mode=1):
        """Getting the similarities based on the similarity of the cls tokens.
        
        mode: 1 -> only static+dynamic weights
        mode: 2 -> output results for dynamic, static, static+dynamic weights
        """

        text_embeddings = F.normalize(text_embeddings, dim=-1)  # [1, N, dim]
        text_embeddings = text_embeddings.squeeze().T

        sims = []
        for elm in stack_init[-3:]:
            if self.img_proj_final:
                elm = self.img_proj_final(elm)
            new_sim = F.normalize(elm, dim=-1)
            new_sim = new_sim[:, 0] @ text_embeddings
            sims.append(new_sim)

        sims_tensor = torch.Tensor(sims)
        sims_tensor = -sims_tensor*20  # When the similarity without the element was high, the element is not very important.
        sims_softmax = sims_tensor.softmax(dim=-1).squeeze()
        # print("Sims: ", sims_softmax)
        # sims_softmax += torch.Tensor([0.5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).to(sims_tensor.device)
        # sims_softmax_final = torch.Tensor([0.3, 0.4, 0.5, 0.6, 0.7, 2/3 - 0.1, 2/3 - 0.1, 2/3 - 0.1]).to(sims_tensor.device)
        # # sims_softmax_final = torch.Tensor([1, 1, 1, 1, 1, 2/3, 2/3, 2/3]).to(sims_tensor.device)
        # # sims_softmax_final = torch.Tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.45, 0.45]).to(sims_tensor.device)
        # sims_softmax_final[-3:] += sims_softmax
        # print("sims_softmax_final: ", sims_softmax_final)
        # print("")
        # return torch.Tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.9, 0.9]).to(sims_tensor.device)
        # dynamic_weights = torch.Tensor([1, 1, 1, 1, 1, 2/3, 2/3, 2/3]).to(sims_tensor.device)
        # dynamic_weights[-3:] += sims_softmax
        # return dynamic_weights
        if mode == 2:
            static_weights = torch.Tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.9, 0.9]).to(sims_tensor.device)
            
            dynamic_weights = torch.Tensor([1, 1, 1, 1, 1, 2/3, 2/3, 2/3]).to(sims_tensor.device)
            dynamic_weights[-3:] += sims_softmax
            
            total_weights = torch.Tensor([0.3, 0.4, 0.5, 0.6, 0.7, 2/3 - 0.1, 2/3 - 0.1, 2/3 - 0.1]).to(sims_tensor.device)
            total_weights[-3:] += sims_softmax
            
            return dynamic_weights, static_weights, total_weights
        
        elif mode == 1:
            sims_softmax_final = torch.Tensor([0.3, 0.4, 0.5, 0.6, 0.7, 2/3 - 0.1, 2/3 - 0.1, 2/3 - 0.1]).to(sims_tensor.device)
            sims_softmax_final[-3:] += sims_softmax
            return sims_softmax_final
        

    def get_sim_list_new(self, sims):
        """Getting the similarities based on the similarity of the patch token that has the maximum similarity."""

        sims_tensor = torch.Tensor(sims)
        sims_tensor = sims_tensor*50  # When the similarity without the element was high, the element is not very important.
        sims_softmax = sims_tensor.softmax(dim=-1).squeeze()
        # print("Sims: ", sims_softmax)
        # sims_softmax += torch.Tensor([0.5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).to(sims_tensor.device)
        sims_softmax_final = torch.Tensor([0.3, 0.4, 0.5, 0.6, 0.7, 2/3 - 0.1, 2/3 - 0.1, 2/3 - 0.1]).to(sims_tensor.device)
        sims_softmax_final[-3:] += sims_softmax
        # print("sims_softmax_final: ", sims_softmax_final)
        # print("")

        return sims_softmax_final



    def get_sim_list_old(self, stack_init, text_embeddings):
        """
        In order to determine the importance of the layer for the current text input and for the heatmaps, the similarity
        between the text embedding and the self-attention outputs is calculated. The more similar they are, the more likely, that
        this layer encodes important information for the heatmaps. Accordingly the corresponding self-self attention output should get more weight.
        Only works with one text.
        """

        sims = []
        # for elm in stack_init[1:-1]:
        # base = torch.zeros_like(stack_init[0])
        for elm in stack_init[1:-1]:
            # new_base = torch.stack(stack_init, dim=-1)
            # new_base = new_base * 0.5
            # new_base = new_base.sum(dim=-1)
            new_sim = F.normalize(elm, dim=-1)
            new_sim = new_sim[:, 0] @ text_embeddings.squeeze().T  # text_embeddings not necessarily normalized. But not needed, because softmax ignores that every element is multiplied by the same factor.
            sims.append(new_sim)
            # base += elm
        
        # Scale the sims:
        sims_tensor = torch.stack(sims, dim=-1)
        maxi = torch.max(sims_tensor)
        mini = torch.min(sims_tensor)
        sims_softmax = sims_tensor.softmax(dim=-1)
        maxi = torch.max(sims_softmax)
        mini = torch.min(sims_softmax)
        sims_softmax = ((sims_softmax - mini/2) / (maxi - mini/2)) + mini
        # sims_softmax = ((sims_tensor - mini) / (maxi - mini)) + 0.3
        
        # "normalize" or adapt the values.
        # print("mini: ", mini, "maxi: ", maxi, "sims_softmax: ", sims_softmax)
        
        # sims_softmax = (sims_softmax / mini) * 0.3
        
        init_weight = torch.Tensor([0.5]).to(sims_tensor.device)
        last_weight = torch.Tensor([1.2]).to(sims_tensor.device)
        sims_new = sims_softmax.squeeze(0)
        sims_new = torch.cat((init_weight, sims_new, last_weight))
        
        print("sims_tensor: ", sims_tensor)
        print("sims_new: ", sims_new)
        print("--------------------------------------")
        # standard_sims = torch.ones_like(sims_softmax)
        return sims_new

    def get_new_result(self, stack, sims, further_process, model_type):
        """Based on the identified weights, the average vector is newly created. After that the last part of the forward method of
        modified_vit_forward needs to be applied again.
        """

        if model_type in ["viclip"]:
            x_final = torch.zeros_like(stack[0])
            for i, elm in enumerate(stack):
                # print("sims[i]: ", sims[i].shape)
                # print("sims: ", sims.shape)
                # print("elm: ", elm.shape)
                # print("x_final: ", x_final.shape)
                x_final = x_final + sims[i] * elm
            
            x_final = further_process["ln_post"](x_final)
            if further_process["proj"] is not None:
                x_final = further_process["dropout"](x_final) @ further_process["proj"]
            x_final = x_final.permute(1, 0, 2)

        else:
            x_final = torch.zeros_like(stack[0].permute(1, 0, 2))
            for i, elm in enumerate(stack):
                elm = elm.permute(1, 0, 2)
                # print("elm.shape: ", elm.shape)
                x_final = x_final + sims[i] * elm
            
            # print("x_final.shape: ", x_final.shape)
            
            x_final = further_process["ln_post"](x_final)
            # print("x_final1.shape: ", x_final.shape)
            
            if further_process["proj"] is not None:
                x_final = x_final @ further_process["proj"]
            # print("x_final2.shape: ", x_final.shape)
            if self.img_proj_final:
                x_final = self.img_proj_final(x_final)
            
        
        return x_final
    
    
    def get_full_result(self, stack, stack_init, further_process, model_type):
        """based on the stack, get the output for every depth of GEM, and the output for leaving out every part of GEM.
        """

        if model_type in ["viclip"]:
            importance_stack = []  # Always exactly one layer is left out (first the self attn. input, then the last self self attn. input, ...).
            base = torch.stack(stack).sum(dim=0)
            for idx, elm in enumerate(stack):  # or stack_init
                current_elm = base - elm  # Full pass without the current element. The more the similarity will drop, the more important the element was.

                current_elm = further_process["ln_post"](current_elm)
                if further_process["proj"] is not None:
                    current_elm = further_process["dropout"](current_elm) @ further_process["proj"]
                current_elm = current_elm.permute(1, 0, 2)
                importance_stack.append(current_elm)
                
            normal_stack = []  # What GEM would output for different numbers of layers.
            base = torch.stack(stack).sum(dim=0)
            for idx, elm in enumerate(stack):  # or stack_init
                if idx == 0:  # The self attn input is never removed. This is GEM for self.depth layers.
                    current_elm = base
                else:  # if idx == self.depth - 1, then it is just the normal clip self attention output (with 0 layers from GEM)
                    base = base - elm + stack_init[idx]  # + stack_init because the self attn input that is used, uses one more layer.
                    current_elm = base  # Full pass without the current element. The more the similarity will drop, the more important the element was.

                current_elm = further_process["ln_post"](current_elm)
                if further_process["proj"] is not None:
                    current_elm = further_process["dropout"](current_elm) @ further_process["proj"]
                current_elm = current_elm.permute(1, 0, 2)
                normal_stack.append(current_elm)
        
            return normal_stack, importance_stack

        else:
            importance_stack = []  # Always exactly one layer is left out (first the self attn. input, then the last self self attn. input, ...).
            base = torch.stack(stack).sum(dim=0)
            for idx, elm in enumerate(stack):  # or stack_init
                current_elm = base - elm  # Full pass without the current element. The more the similarity will drop, the more important the element was.

                current_elm = current_elm.permute(1, 0, 2)
                current_elm = further_process["ln_post"](current_elm)
                if further_process["proj"] is not None:
                    current_elm = current_elm @ further_process["proj"]
                importance_stack.append(current_elm)
            
            normal_stack = []  # What GEM would output for different numbers of layers.
            base = torch.stack(stack).sum(dim=0)
            # print("stack[0].shape: ", stack[0].shape)
            # print("base.shape: ", base.shape)
            for idx, elm in enumerate(stack):  # or stack_init
                # print("stack_init[idx].shape: ", stack_init[idx].shape)
                if idx == 0:  # The self attn input is never removed. This is GEM for self.depth layers.
                    current_elm = base
                # elif idx == self.depth - 1:  # Then there would be no self attn. layer left.
                #     continue
                else:  # if idx == self.depth - 1, then it is just the normal clip self attention output (with 0 layers from GEM)
                    base = base - elm + stack_init[idx]  # + stack_init because the self attn input that is used, uses one more layer.
                    current_elm = base  # Full pass without the current element. The more the similarity will drop, the more important the element was.

                current_elm = current_elm.permute(1, 0, 2)
                current_elm = further_process["ln_post"](current_elm)
                if further_process["proj"] is not None:
                    current_elm = current_elm @ further_process["proj"]
                if self.img_proj_final:
                    current_elm = self.img_proj_final(current_elm)
                normal_stack.append(current_elm)
        
        return normal_stack, importance_stack
            

    def forward(self, image: torch.Tensor, text: list, stackWeightsAutoOne=False, normalize: bool = True, return_ori: bool = False, get_full_analysis: bool = False, single_stats: bool = False):
        """
        :param stackWeightsAutoOne: True when the first prompt is a verb prompt for which static and dynamic weights should be applied.
        :param get_full_analysis: In order to create graphs, the output for leaving out different layers, using different numbers of layers with verb object and full prompt is calculated.
        :param single_stats: If True, verb, obj, full prompt are all tested in one run for dynamic weights, static weights, dynamic + static weights (doing 6 runs in 1).
        :param image: torch.Tensor [1, 3, H, W]
        :param text: list[]
        :param normalize: bool - if True performs min-max normalization
        :param return_ori: bool - if True uses the features from the original visual encoder
        """

        # Image
        #if self.model_type in ["clip", "openclip"]:
        # print("Image shape: ", image.shape)
        W, H = image.shape[-2:]
        feat_gem, feat_ori, stack, stack_init_basic, stack_init, further_process = self.visual(image)
        # print("feat_gem.shape0000000000: ", feat_gem.shape)
        
        if self.img_proj_final:
            feat_ori = self.img_proj_final(feat_ori)
            feat_gem = self.img_proj_final(feat_gem)

        image_feat = feat_ori if return_ori else feat_gem
        # image_feat = feat_ori
        image_feat = F.normalize(image_feat, dim=-1)  # [1, N, dim]
        # print("stack.shape0000000000: ", stack[0].shape)
        

                

        # Text
        if self.model_type in ["clip", "openclip"]:
            text_embeddings = self.encode_text(text)  # [1, num_prompt, dim]
            if self.txt_proj_final:
                text_embeddings = self.txt_proj_final(text_embeddings)
            text_embeddings = self.avg_text_prompts(text_embeddings, self.avg_prompts)
            text_embeddings = F.normalize(text_embeddings, dim=-1)  # Makes no difference in the results.
            
            if single_stats:
                feat_ori = F.normalize(feat_ori, dim=-1)  # [1, N, dim]
                label_probs = (100.0 * feat_ori[:, 0] @ text_embeddings.squeeze().T).softmax(dim=-1)
                _, prompt_cnt, _ = text_embeddings.shape
                
                # The main part. normal, dynamic, static, full prompt acts as a batchsize=3
                for i in range(prompt_cnt):
                    text_embedding = text_embeddings[:, i, :]
                    dynamic_weights, static_weights, total_weights = self.get_sim_list(stack_init, text_embedding, mode=2)
                    feat_gem_dynamic = self.get_new_result(stack, dynamic_weights, further_process, self.model_type)
                    feat_gem_static = self.get_new_result(stack, static_weights, further_process, self.model_type)
                    feat_gem_total = self.get_new_result(stack, total_weights, further_process, self.model_type)
                    
                    feat_gem_dynamic = F.normalize(feat_gem_dynamic, dim=-1)
                    feat_gem_static = F.normalize(feat_gem_static, dim=-1)
                    feat_gem_total = F.normalize(feat_gem_total, dim=-1)
                    
                    feat_gem_new = torch.stack([image_feat, feat_gem_dynamic, feat_gem_static, feat_gem_total], dim=0)
                    # print("feat_gem.shape: ", feat_gem.shape)
                    if i == 0:
                        feat_res = feat_gem_new
                    else:
                        feat_res = torch.cat((feat_res, feat_gem_new), dim=1)  # [settings (4), num_prompt, N, dim]
                    # print("feat_res.shape: ", feat_res.shape)
            
                # Image-Text matching
                img_txt_matching = torch.sum(feat_res[:, :, 1:] * text_embeddings.unsqueeze(2), dim=-1).transpose(1, -1)  # [1, N, num_prompt]
                
                img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h',
                                        w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]
                
                # Interpolate
                img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [1, num_prompt, W, H]
                
                if normalize:
                    img_txt_matching = self.min_max(img_txt_matching, frames=1)
                
                return img_txt_matching, label_probs

            
            elif get_full_analysis:  # Just for creating importance graphs.
                feat_ori = F.normalize(feat_ori, dim=-1)  # [1, N, dim]
                label_probs = (100.0 * feat_ori[:, 0] @ text_embeddings.squeeze().T).softmax(dim=-1)

                # normal_stack is a list of self.depth GEM results with depth self.depth to depth = 1
                # importance_stack is a list of self.depth + 1 GEM results when the first input (self attn input) to the last input (final self self attn layer) is removed. Always exactly one input is missing.
                normal_stack, importance_stack = self.get_full_result(stack, stack_init_basic, further_process, self.model_type)
                
                # For each of the elements of normal_stack get the GEM output (for all prompts).
                normal_stack_final = []
                for normal_feat in normal_stack:
                    normal_feat = F.normalize(normal_feat, dim=-1)
                    img_txt_matching = normal_feat[:, 1:] @ text_embeddings.transpose(-1, -2)  # [1, N, num_prompt]
                
                    img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h',
                                            w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]
                    # Interpolate
                    img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [1, num_prompt, W, H]
                    
                    if normalize:
                        img_txt_matching = self.min_max(img_txt_matching, frames=1)
                    normal_stack_final.append(img_txt_matching)
                
                # For each of the elements of importance_stack get the GEM output (for all prompts)
                importance_stack_final = []
                for importance_feat in importance_stack:
                    importance_feat = F.normalize(importance_feat, dim=-1)
                    img_txt_matching = importance_feat[:, 1:] @ text_embeddings.transpose(-1, -2)  # [1, N, num_prompt]
                
                    img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h',
                                            w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]
                    # Interpolate
                    img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [1, num_prompt, W, H]
                    
                    if normalize:
                        img_txt_matching = self.min_max(img_txt_matching, frames=1)
                    importance_stack_final.append(img_txt_matching)
                
                return normal_stack_final, importance_stack_final, label_probs
                    
            
            else:  # For normal evaluation
                
                if self.get_sims:
                    feat_ori = F.normalize(feat_ori, dim=-1)  # [1, N, dim]
                    label_probs = (100.0 * feat_ori[:, 0] @ text_embeddings.squeeze().T).softmax(dim=-1)
                
                if self.stackWeightsAuto and not stackWeightsAutoOne:
                    
                    sims = self.get_sim_list(stack_init, text_embeddings)
                    feat_gem = self.get_new_result(stack, sims, further_process, self.model_type)
                    image_feat = F.normalize(feat_gem, dim=-1)  # [1, N, dim], just the image feature for the first text prompt (the verb)

                if not stackWeightsAutoOne:
                    
                    # print("ja")
                    # Image-Text matching
                    img_txt_matching = image_feat[:, 1:] @ text_embeddings.transpose(-1, -2)  # [1, N, num_prompt]
                    
                    img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h',
                                            w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]
                    # Interpolate
                    img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [1, num_prompt, W, H]
                
                
                if stackWeightsAutoOne:  # Based on cls token:
                    
                    
                    for i in range(stackWeightsAutoOne):
                        text_embedding = text_embeddings[:, i, :]
                        
                        sims = self.get_sim_list(stack_init, text_embedding)
                        feat_gem = self.get_new_result(stack, sims, further_process, self.model_type)
                        image_feat_one = F.normalize(feat_gem, dim=-1)  # [1, N, dim], just the image feature for the first text prompt (the verb)
                    
                        # Image-Text matching
                        
                        text_embedding = text_embedding.unsqueeze(1)  # To get the 1 dimension back in for the num_prompt.
                        
                        img_txt_matching_one = image_feat_one[:, 1:] @ text_embedding.transpose(-1, -2)  # [1, N, num_prompt]
                        
                        img_txt_matching_one = rearrange(img_txt_matching_one, 'b (w h) c -> b c w h',
                                                w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]
                        
                        # Interpolate
                        img_txt_matching_one = F.interpolate(img_txt_matching_one, size=(W, H), mode='bilinear')  # [1, num_prompt, W, H]

                        if i == 0:
                            img_txt_matching_one_cat = img_txt_matching_one
                        else:
                            img_txt_matching_one_cat = torch.cat((img_txt_matching_one_cat, img_txt_matching_one), dim=1)

                    # print("img_txt_matching_one.shape: ", img_txt_matching_one.shape)
                    # print("img_txt_matching_one_cat.shape: ", img_txt_matching_one_cat.shape)
                    ### Now for the other prompts:
                    # Image-Text matching
                    A, B, C = text_embeddings.shape
                    if B > stackWeightsAutoOne:  # Prevent an error, when only verb prompts are there.
                    
                        img_txt_matching = image_feat[:, 1:] @ text_embeddings[:, stackWeightsAutoOne:, :].transpose(-1, -2)  # [1, N, num_prompt]
                        
                        
                        img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h',
                                                w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]
                        # Interpolate
                        
                        img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [1, num_prompt, W, H]
                        
                        img_txt_matching = torch.cat((img_txt_matching_one_cat, img_txt_matching), dim=1)
                    else:
                        img_txt_matching = img_txt_matching_one_cat
                    
                    
                # Heat Maps
                if normalize:
                    img_txt_matching = self.min_max(img_txt_matching, frames=1)
                
                if self.get_sims:
                    return img_txt_matching, label_probs
                else:
                    return img_txt_matching
            
            
        elif self.model_type in ["viclip"]:
            text_embeddings = self.model.encode_text(text)  # [1, num_prompt, dim]
            text_embeddings = F.normalize(text_embeddings, dim=-1)
            text_embeddings =  text_embeddings.unsqueeze(0)
            text_embeddings = self.avg_text_prompts(text_embeddings, self.avg_prompts)
            
            if single_stats:
                feat_ori = F.normalize(feat_ori, dim=-1)  # [1, N, dim]
                label_probs = (100.0 * feat_ori[:, 0] @ text_embeddings.squeeze().T).softmax(dim=-1)
                _, prompt_cnt, _ = text_embeddings.shape
                
                # The main part. normal, dynamic, static, full prompt acts as a batchsize=3
                for i in range(prompt_cnt):
                    text_embedding = text_embeddings[:, i, :]
                    dynamic_weights, static_weights, total_weights = self.get_sim_list(stack_init, text_embedding, mode=2)
                    feat_gem_dynamic = self.get_new_result(stack, dynamic_weights, further_process, self.model_type)
                    feat_gem_static = self.get_new_result(stack, static_weights, further_process, self.model_type)
                    feat_gem_total = self.get_new_result(stack, total_weights, further_process, self.model_type)
                    
                    feat_gem_dynamic = F.normalize(feat_gem_dynamic, dim=-1)
                    feat_gem_static = F.normalize(feat_gem_static, dim=-1)
                    feat_gem_total = F.normalize(feat_gem_total, dim=-1)
                    
                    feat_gem_new = torch.stack([image_feat, feat_gem_dynamic, feat_gem_static, feat_gem_total], dim=0)
                    # print("feat_gem.shape: ", feat_gem.shape)
                    if i == 0:
                        feat_res = feat_gem_new
                    else:
                        feat_res = torch.cat((feat_res, feat_gem_new), dim=1)  # [settings (4), num_prompt, N, dim]
                # print("feat_res.shape: ", feat_res.shape)
            
                # Image-Text matching
                img_txt_matching = torch.sum(feat_res[:, :, 1:] * text_embeddings.unsqueeze(2), dim=-1).transpose(1, -1)  # [1, N, num_prompt]
                
                img_txt_matching = rearrange(img_txt_matching, 'b (w h t) c -> b c t w h',
                                        w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]
                # print("img_txt_matching1.shape: ", img_txt_matching.shape)
                
                # Interpolate
                img_txt_matching = F.interpolate(img_txt_matching, size=(8, W, H), mode='trilinear')  # [num_prompt, 8, W, H] (8 frames).
                # print("img_txt_matching.shape: ", img_txt_matching.shape)
                
                if normalize:
                    img_txt_matching = self.min_max(img_txt_matching, frames=8)
                
                return img_txt_matching, label_probs
            
            if get_full_analysis:  # Just for creating importance graphs.
                feat_ori = F.normalize(feat_ori, dim=-1)  # [1, N, dim]
                label_probs = (100.0 * feat_ori[:, 0] @ text_embeddings.squeeze().T).softmax(dim=-1)

                # normal_stack is a list of self.depth GEM results with depth self.depth to depth = 1
                # importance_stack is a list of self.depth + 1 GEM results when the first input (self attn input) to the last input (final self self attn layer) is removed. Always exactly one input is missing.
                normal_stack, importance_stack = self.get_full_result(stack, stack_init_basic, further_process, self.model_type)
                
                # For each of the elements of normal_stack get the GEM output (for all prompts).
                normal_stack_final = []
                for normal_feat in normal_stack:
                    normal_feat = F.normalize(normal_feat, dim=-1)
                    img_txt_matching = normal_feat[:, 1:] @ text_embeddings.transpose(-1, -2)  # [1, N, num_prompt]
                
                    img_txt_matching = rearrange(img_txt_matching, 'b (w h t) c -> b c t w h',
                                            t=8, w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]
                    # Interpolate
                    img_txt_matching = torch.squeeze(img_txt_matching, dim=0)
                    img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [num_prompt, 8, W, H] (8 frames).
                    img_txt_matching = torch.unsqueeze(img_txt_matching, dim=0)  # Add [1, num_prompt, 8, W, H]
                    
                    if normalize:
                        img_txt_matching = self.min_max(img_txt_matching, frames=8)
                    normal_stack_final.append(img_txt_matching)
                
                # For each of the elements of importance_stack get the GEM output (for all prompts)
                importance_stack_final = []
                for importance_feat in importance_stack:
                    importance_feat = F.normalize(importance_feat, dim=-1)
                    img_txt_matching = importance_feat[:, 1:] @ text_embeddings.transpose(-1, -2)  # [1, N, num_prompt]
                
                    img_txt_matching = rearrange(img_txt_matching, 'b (w h t) c -> b c t w h',
                                            t=8, w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]
                    # Interpolate
                    img_txt_matching = torch.squeeze(img_txt_matching, dim=0)
                    img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [num_prompt, 8, W, H] (8 frames).
                    img_txt_matching = torch.unsqueeze(img_txt_matching, dim=0)  # Add [1, num_prompt, 8, W, H]
                    
                    if normalize:
                        img_txt_matching = self.min_max(img_txt_matching, frames=8)
                    importance_stack_final.append(img_txt_matching)
                
                return normal_stack_final, importance_stack_final, label_probs

            else:
                if self.get_sims:
                    feat_ori = F.normalize(feat_ori, dim=-1)  # [1, N, dim]
                    label_probs = (100.0 * feat_ori[:, 0] @ text_embeddings.squeeze().T).softmax(dim=-1)
                    
                if self.stackWeightsAuto and not stackWeightsAutoOne:
                    # Based on cls token:
                    sims = self.get_sim_list(stack_init, text_embeddings)
                    feat_gem = self.get_new_result(stack, sims, further_process, self.model_type)
                    image_feat = F.normalize(feat_gem, dim=-1)  # [1, N, dim]
                
                if not stackWeightsAutoOne:
                    # Image-Text matching
                    img_txt_matching = image_feat[:, 1:] @ text_embeddings.transpose(-1, -2)  # [1, N, num_prompt]
                    
                    img_txt_matching = rearrange(img_txt_matching, 'b (w h t) c -> b c t w h',
                                            t=8, w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]

                    # Interpolate (only works for 4D input, therefore remove 1 dimension at the beginning.)
                    img_txt_matching = torch.squeeze(img_txt_matching, dim=0)
                    img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [num_prompt, 8, W, H] (8 frames).
                    img_txt_matching = torch.unsqueeze(img_txt_matching, dim=0)  # Add [1, num_prompt, 8, W, H]
                
                
                
                if stackWeightsAutoOne:  # Based on cls token:
                    
                    for i in range(stackWeightsAutoOne):
                        text_embedding = text_embeddings[:, i, :]
                        
                        sims = self.get_sim_list(stack_init, text_embedding)
                        feat_gem = self.get_new_result(stack, sims, further_process, self.model_type)
                        image_feat_one = F.normalize(feat_gem, dim=-1)  # [1, N, dim], just the image feature for the first text prompt (the verb)
                    
                        # Image-Text matching
                        text_embedding = text_embedding.unsqueeze(1)  # To get the 1 dimension back in for the num_prompt.
                        
                        img_txt_matching_one = image_feat_one[:, 1:] @ text_embedding.transpose(-1, -2)  # [1, N, num_prompt]
                        
                        img_txt_matching_one = rearrange(img_txt_matching_one, 'b (w h t) c -> b c t w h',
                                                t=8, w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]

                        # Interpolate (only works for 4D input, therefore remove 1 dimension at the beginning.)
                        img_txt_matching_one = torch.squeeze(img_txt_matching_one, dim=0)
                        img_txt_matching_one = F.interpolate(img_txt_matching_one, size=(W, H), mode='bilinear')  # [num_prompt, 8, W, H] (8 frames).
                        img_txt_matching_one = torch.unsqueeze(img_txt_matching_one, dim=0)  # Add [1, num_prompt, 8, W, H]
                    
                        if i == 0:
                            img_txt_matching_one_cat = img_txt_matching_one
                        else:
                            img_txt_matching_one_cat = torch.cat((img_txt_matching_one_cat, img_txt_matching_one), dim=1)
                    
                    
                    A, B, C = text_embeddings.shape
                    if B > stackWeightsAutoOne:  # Prevent an error, when only verb prompts are there.
                        # Image-Text matching for the other prompts.
                        img_txt_matching = image_feat[:, 1:] @ text_embeddings[:, stackWeightsAutoOne:, :].transpose(-1, -2)  # [1, N, num_prompt]
                        
                        img_txt_matching = rearrange(img_txt_matching, 'b (w h t) c -> b c t w h',
                                                t=8, w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]

                        # Interpolate (only works for 4D input, therefore remove 1 dimension at the beginning.)
                        img_txt_matching = torch.squeeze(img_txt_matching, dim=0)
                        img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [num_prompt, 8, W, H] (8 frames).
                        img_txt_matching = torch.unsqueeze(img_txt_matching, dim=0)  # Add [1, num_prompt, 8, W, H]
                        
                        img_txt_matching = torch.cat((img_txt_matching_one_cat, img_txt_matching), dim=1)
                    else:
                        img_txt_matching = img_txt_matching_one_cat

    
    

                # Heat Maps
                if normalize:
                    img_txt_matching = self.min_max(img_txt_matching, frames=8)
                
            
            if self.get_sims:
                return img_txt_matching, label_probs
            else:
                return img_txt_matching

    def batched_forward(self, image: torch.Tensor, text: list, normalize: bool = True, return_ori: bool =False):
        """
        :param image: torch.Tensor [B, 3, H, W]
        :param text: list[list[]]
        :param normalize: bool - if True performs min-max normalization
        :param return_ori: bool - if True uses the features from the original visual encoder
        """
        L = len(text)
        cumm_idx = np.cumsum([len(t) for t in text]).tolist()
        B, _, W, H = image.shape
        assert B == L, f'Number of prompts L: {L} should be the same as number of images B: {B}.'

        # Image
        feat_gem, feat_ori = self.visual(image)
        image_feat = feat_ori if return_ori else feat_gem
        image_feat = F.normalize(image_feat, dim=-1)  # [B, N, dim]

        # Text
        flatten_text = [t for sub_text in text for t in sub_text]
        text_embeddings = self.encode_text(flatten_text)  # [B, num_prompt, dim]

        # Image-Text matching
        img_txt_matching = 100 * image_feat[:, 1:] @ text_embeddings.transpose(-1, -2)  # [B, N, num_prompt]
        img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h',
                                     w=W // self.patch_size, h=H // self.patch_size)  # [B, num_prompt, w, h]

        # Interpolate
        img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [B,num_prompt, W, H]

        # Heat Maps
        if normalize:
            img_txt_matching = self.min_max(img_txt_matching)  # [B,num_prompt, W, H]

        # unflatten
        img_txt_matching = torch.tensor_split(img_txt_matching, cumm_idx[:-1], dim=1)
        img_txt_matching = [itm[i] for i, itm in enumerate(img_txt_matching)]
        return img_txt_matching
