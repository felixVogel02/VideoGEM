from typing import Optional, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from open_clip.transformer import _expand_token, to_2tuple



def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True
):
    # sort out sizes, assume square if old size not provided
    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]
    if not old_size:
        old_size = int(math.sqrt(posemb.shape[1] - num_prefix_tokens))
    old_size = to_2tuple(old_size)
    if new_size == old_size:  # might not both be same container type
        return posemb

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)


    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)
    return posemb

class SelfSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ss_attn_iter=1,
                 ss_attn_temp=None, img_attn_only=0, token_cnt=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.ss_attn_iter = ss_attn_iter
        self.ss_attn_temp = ss_attn_temp

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.img_attn_only = img_attn_only
        self.token_cnt = token_cnt

    def forward(self, x, attn_bias=None, prev_attn=None):
        # print(f"x.shape: {x.shape}")
        wasList = False
        if isinstance(x, list) or isinstance(x, tuple):  # Only the case if add_bef is true, which is normally not the case.
            wasList = True
            x_gem, x, stack = x
            x_gem = (x_gem + x ) / 2
            x_gem = x_gem.transpose(0, 1)
            # print("wasList: ", wasList)
        x = x.transpose(0, 1)
        B, N, C = x.shape
        # print(f"B, N, c: {B} {N} {C}")
        # print("wasList elfSelfAttn: ", wasList)
        # print(f"x.shape1: {x.shape}")
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(f"qkv shape: {qkv.shape}")
        q, k, v = qkv[0], qkv[1], qkv[2]
        self.v_values = v
        # original self-attention for the original path
        attn_ori_return = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori_return.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)
        x_ori = attn_ori @ v
        # print(f"x_ori shapeshapeshape: {x_ori.shape}")
        
        x_ori_cls_token = x_ori[:, :, 0, :].unsqueeze(2)  # Because gem does not attend with the class token.

        x_ori = x_ori.transpose(1, 2).reshape(B, N, C)
        x_ori = self.proj_drop(self.proj(x_ori))

        # GEM
        if wasList:
            qkv = self.qkv(x_gem).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.img_attn_only == 0:  # Attend between all tokens over all images
            xs1 = v
            xs2 = k
            xs3 = q

            if self.ss_attn_temp is None:
                pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
                inv_temp = pre_norm * self.scale
            else:
                inv_temp = self.ss_attn_temp

            for it in range(self.ss_attn_iter):
                xs1 = F.normalize(xs1, dim=-1)
                xs2 = F.normalize(xs2, dim=-1)
                xs3 = F.normalize(xs3, dim=-1)

                attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
                attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
                attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

                attn1 = (attn_return1).softmax(dim=-1)
                attn2 = (attn_return2).softmax(dim=-1)
                attn3 = (attn_return3).softmax(dim=-1)

                xs1 = attn1 @ xs1
                xs2 = attn2 @ xs2
                xs3 = attn3 @ xs3

            # Assignment to V
            xs1 = F.normalize(xs1, dim=-1)
            xs2 = F.normalize(xs2, dim=-1)
            xs3 = F.normalize(xs3, dim=-1)

            attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
            attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
            attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

            attn1 = (attn_return1).softmax(dim=-1)
            attn2 = (attn_return2).softmax(dim=-1)
            attn3 = (attn_return3).softmax(dim=-1)

            xs1 = attn1 @ v
            xs2 = attn2 @ v
            xs3 = attn3 @ v
            xs = (xs1 + xs2 + xs3) / 3
            # xs = xs3

            x = xs.transpose(1, 2).reshape(B, N, C)
            x = self.proj_drop(self.proj(x))
        
        elif self.img_attn_only == 3:  # Combine attenting to only tokens of the same image, and attending only to tokens of other images (version 1 and 2).
            
            
            ################## Only attend to tokens of the same image.
            qkv_new = qkv[:, :, :, 1:, :]  # Exclude the class token.
            qkv_new = rearrange(qkv_new, 'q b a (n t) c -> q n b a t c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)  # 196=14*14
            res_sameImg = []
            for i in range(8):
                xs1 = qkv_new[0][i]
                xs2 = qkv_new[1][i]
                xs3 = qkv_new[2][i]
                v = qkv_new[2][i]

                if self.ss_attn_temp is None:
                    pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
                    inv_temp = pre_norm * self.scale
                else:
                    inv_temp = self.ss_attn_temp

                for it in range(self.ss_attn_iter):
                    xs1 = F.normalize(xs1, dim=-1)
                    xs2 = F.normalize(xs2, dim=-1)
                    xs3 = F.normalize(xs3, dim=-1)

                    attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
                    attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
                    attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

                    attn1 = (attn_return1).softmax(dim=-1)
                    attn2 = (attn_return2).softmax(dim=-1)
                    attn3 = (attn_return3).softmax(dim=-1)

                    xs1 = attn1 @ xs1
                    xs2 = attn2 @ xs2
                    xs3 = attn3 @ xs3

                # Assignment to V
                xs1 = F.normalize(xs1, dim=-1)
                xs2 = F.normalize(xs2, dim=-1)
                xs3 = F.normalize(xs3, dim=-1)

                attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
                attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
                attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

                attn1 = (attn_return1).softmax(dim=-1)
                attn2 = (attn_return2).softmax(dim=-1)
                attn3 = (attn_return3).softmax(dim=-1)

                xs1 = attn1 @ v
                xs2 = attn2 @ v
                xs3 = attn3 @ v
                xs = (xs1 + xs2 + xs3) / 3
                # xs = xs3
                res_sameImg.append(xs)
            res_sameImg = torch.cat(res_sameImg, dim=2)
            
            ######################## Only attend to tokens of other images.
            
            qkv = qkv[:, :, :, 1:, :]  # Exclude the class token.
            # print(f"qkv shapeNew: {qkv.shape}")
            v = qkv[2].detach().clone()
            res_otherImg = torch.zeros_like(v)
            qkv = rearrange(qkv, 'q b a (n t) c -> q n b a t c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)  # 196=14*14
            for i in range(8):
                # print("qkv.shape: ", qkv.shape)
                template = torch.zeros_like(qkv)
                template[:, i] = 1
                template = rearrange(template, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
                template = template[0]
                
                template_attend = torch.ones_like(qkv)
                template_attend[:, i] = 0
                template_attend = rearrange(template_attend, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
                template_attend = template_attend[0]
                
                qkv_base = qkv.detach().clone()
                remove_dims = [k for k in range(8)]
                remove_dims.remove(i)
                qkv_base[:, [remove_dims]] = 0  # Only keep the values of the i'th entry.
                qkv_base = rearrange(qkv_base, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
                
                qkv_attend = qkv.detach().clone()
                qkv_attend[:, i] = 0
                qkv_attend = rearrange(qkv_attend, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
            
                xs1 = qkv_base[0]
                xs2 = qkv_base[1]
                xs3 = qkv_base[2]
                
                xs1_attend = qkv_attend[0]
                xs2_attend = qkv_attend[1]
                xs3_attend = qkv_attend[2]

                if self.ss_attn_temp is None:
                    pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
                    inv_temp = pre_norm * self.scale
                else:
                    inv_temp = self.ss_attn_temp

                for it in range(self.ss_attn_iter):
                    xs1 = F.normalize(xs1, dim=-1)
                    xs2 = F.normalize(xs2, dim=-1)
                    xs3 = F.normalize(xs3, dim=-1)
                    # print("xs1.shape: ", xs1.shape)
                    # print("xs1_attend.shape: ", xs1_attend.shape)

                    attn_return1 = (xs1 @ xs1_attend.transpose(-2, -1)) * inv_temp
                    attn_return2 = (xs2 @ xs2_attend.transpose(-2, -1)) * inv_temp
                    attn_return3 = (xs3 @ xs3_attend.transpose(-2, -1)) * inv_temp
                    template_return = (template @ template_attend.transpose(-2, -1))
                    

                    attn_return1[template_return<1e-8] = -float("inf")
                    attn1 = (attn_return1).softmax(dim=-1)
                    attn1[template_return<1e-8] = 0
                    
                    attn_return2[template_return<1e-8] = -float("inf")
                    attn2 = (attn_return2).softmax(dim=-1)
                    attn2[template_return<1e-8] = 0
                    
                    attn_return3[template_return<1e-8] = -float("inf")
                    attn3 = (attn_return3).softmax(dim=-1)
                    attn3[template_return<1e-8] = 0

                    xs1 = attn1 @ xs1
                    xs2 = attn2 @ xs2
                    xs3 = attn3 @ xs3

                # Assignment to V
                xs1 = F.normalize(xs1, dim=-1)
                xs2 = F.normalize(xs2, dim=-1)
                xs3 = F.normalize(xs3, dim=-1)

                attn_return1 = (xs1 @ xs1_attend.transpose(-2, -1)) * inv_temp
                attn_return2 = (xs2 @ xs2_attend.transpose(-2, -1)) * inv_temp
                attn_return3 = (xs3 @ xs3_attend.transpose(-2, -1)) * inv_temp
                template_return = (template @ template_attend.transpose(-2, -1))

                attn_return1[template_return<1e-8] = -float("inf")
                attn1 = (attn_return1).softmax(dim=-1)
                attn1[template_return<1e-8] = 0
                
                attn_return2[template_return<1e-8] = -float("inf")
                attn2 = (attn_return2).softmax(dim=-1)
                attn2[template_return<1e-8] = 0
                
                attn_return3[template_return<1e-8] = -float("inf")
                attn3 = (attn_return3).softmax(dim=-1)
                attn3[template_return<1e-8] = 0
                # print("xs1.shape: ", xs1.shape)
                # print("v.shape: ", v.shape)

                xs1 = attn1 @ v
                xs2 = attn2 @ v
                xs3 = attn3 @ v
                xs = (xs1 + xs2 + xs3) / 3
                res_otherImg += xs
            
            # Combine attention to only tokens of the same image with attention only to tokens of other images.
            # Use a weighted average.
            res = (res_sameImg + res_otherImg) / 2
            
            res = torch.cat([x_ori_cls_token, res], dim=2)
            x = res.transpose(1, 2).reshape(B, N, C)
            x = self.proj_drop(self.proj(x))

        elif self.img_attn_only == 4:  # Combine attenting to only tokens of the same image, attending only to tokens of other images as well as attending to all token (version 1 and 2 and 0).
            
            ################## Attend to all tokens.
            xs1 = v.detach().clone()
            xs2 = k.detach().clone()
            xs3 = q.detach().clone()

            if self.ss_attn_temp is None:
                pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
                inv_temp = pre_norm * self.scale
            else:
                inv_temp = self.ss_attn_temp

            for it in range(self.ss_attn_iter):
                xs1 = F.normalize(xs1, dim=-1)
                xs2 = F.normalize(xs2, dim=-1)
                xs3 = F.normalize(xs3, dim=-1)

                attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
                attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
                attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

                attn1 = (attn_return1).softmax(dim=-1)
                attn2 = (attn_return2).softmax(dim=-1)
                attn3 = (attn_return3).softmax(dim=-1)

                xs1 = attn1 @ xs1
                xs2 = attn2 @ xs2
                xs3 = attn3 @ xs3

            # Assignment to V
            xs1 = F.normalize(xs1, dim=-1)
            xs2 = F.normalize(xs2, dim=-1)
            xs3 = F.normalize(xs3, dim=-1)

            attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
            attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
            attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

            attn1 = (attn_return1).softmax(dim=-1)
            attn2 = (attn_return2).softmax(dim=-1)
            attn3 = (attn_return3).softmax(dim=-1)

            xs1 = attn1 @ v
            xs2 = attn2 @ v
            xs3 = attn3 @ v
            res_original = (xs1 + xs2 + xs3) / 3
            
            ################## Only attend to tokens of the same image.
            qkv_new = qkv[:, :, :, 1:, :].detach().clone()  # Exclude the class token.
            qkv_new = rearrange(qkv_new, 'q b a (n t) c -> q n b a t c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)  # 196=14*14
            res_sameImg = []
            for i in range(8):
                xs1 = qkv_new[0][i]
                xs2 = qkv_new[1][i]
                xs3 = qkv_new[2][i]
                v = qkv_new[2][i]

                if self.ss_attn_temp is None:
                    pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
                    inv_temp = pre_norm * self.scale
                else:
                    inv_temp = self.ss_attn_temp

                for it in range(self.ss_attn_iter):
                    xs1 = F.normalize(xs1, dim=-1)
                    xs2 = F.normalize(xs2, dim=-1)
                    xs3 = F.normalize(xs3, dim=-1)

                    attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
                    attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
                    attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

                    attn1 = (attn_return1).softmax(dim=-1)
                    attn2 = (attn_return2).softmax(dim=-1)
                    attn3 = (attn_return3).softmax(dim=-1)

                    xs1 = attn1 @ xs1
                    xs2 = attn2 @ xs2
                    xs3 = attn3 @ xs3

                # Assignment to V
                xs1 = F.normalize(xs1, dim=-1)
                xs2 = F.normalize(xs2, dim=-1)
                xs3 = F.normalize(xs3, dim=-1)

                attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
                attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
                attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

                attn1 = (attn_return1).softmax(dim=-1)
                attn2 = (attn_return2).softmax(dim=-1)
                attn3 = (attn_return3).softmax(dim=-1)

                xs1 = attn1 @ v
                xs2 = attn2 @ v
                xs3 = attn3 @ v
                xs = (xs1 + xs2 + xs3) / 3
                # xs = xs3
                res_sameImg.append(xs)
            res_sameImg = torch.cat(res_sameImg, dim=2)
            
            ######################## Only attend to tokens of other images.
            
            qkv = qkv[:, :, :, 1:, :]  # Exclude the class token.
            # print(f"qkv shapeNew: {qkv.shape}")
            v = qkv[2].detach().clone()
            res_otherImg = torch.zeros_like(v)
            qkv = rearrange(qkv, 'q b a (n t) c -> q n b a t c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)  # 196=14*14
            for i in range(8):
                # print("qkv.shape: ", qkv.shape)
                template = torch.zeros_like(qkv)
                template[:, i] = 1
                template = rearrange(template, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
                template = template[0]
                
                template_attend = torch.ones_like(qkv)
                template_attend[:, i] = 0
                template_attend = rearrange(template_attend, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
                template_attend = template_attend[0]
                
                qkv_base = qkv.detach().clone()
                remove_dims = [k for k in range(8)]
                remove_dims.remove(i)
                qkv_base[:, [remove_dims]] = 0  # Only keep the values of the i'th entry.
                qkv_base = rearrange(qkv_base, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
                
                qkv_attend = qkv.detach().clone()
                qkv_attend[:, i] = 0
                qkv_attend = rearrange(qkv_attend, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
            
                xs1 = qkv_base[0]
                xs2 = qkv_base[1]
                xs3 = qkv_base[2]
                
                xs1_attend = qkv_attend[0]
                xs2_attend = qkv_attend[1]
                xs3_attend = qkv_attend[2]

                if self.ss_attn_temp is None:
                    pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
                    inv_temp = pre_norm * self.scale
                else:
                    inv_temp = self.ss_attn_temp

                for it in range(self.ss_attn_iter):
                    xs1 = F.normalize(xs1, dim=-1)
                    xs2 = F.normalize(xs2, dim=-1)
                    xs3 = F.normalize(xs3, dim=-1)
                    # print("xs1.shape: ", xs1.shape)
                    # print("xs1_attend.shape: ", xs1_attend.shape)

                    attn_return1 = (xs1 @ xs1_attend.transpose(-2, -1)) * inv_temp
                    attn_return2 = (xs2 @ xs2_attend.transpose(-2, -1)) * inv_temp
                    attn_return3 = (xs3 @ xs3_attend.transpose(-2, -1)) * inv_temp
                    template_return = (template @ template_attend.transpose(-2, -1))
                    

                    attn_return1[template_return<1e-8] = -float("inf")
                    attn1 = (attn_return1).softmax(dim=-1)
                    attn1[template_return<1e-8] = 0
                    
                    attn_return2[template_return<1e-8] = -float("inf")
                    attn2 = (attn_return2).softmax(dim=-1)
                    attn2[template_return<1e-8] = 0
                    
                    attn_return3[template_return<1e-8] = -float("inf")
                    attn3 = (attn_return3).softmax(dim=-1)
                    attn3[template_return<1e-8] = 0

                    xs1 = attn1 @ xs1
                    xs2 = attn2 @ xs2
                    xs3 = attn3 @ xs3

                # Assignment to V
                xs1 = F.normalize(xs1, dim=-1)
                xs2 = F.normalize(xs2, dim=-1)
                xs3 = F.normalize(xs3, dim=-1)

                attn_return1 = (xs1 @ xs1_attend.transpose(-2, -1)) * inv_temp
                attn_return2 = (xs2 @ xs2_attend.transpose(-2, -1)) * inv_temp
                attn_return3 = (xs3 @ xs3_attend.transpose(-2, -1)) * inv_temp
                template_return = (template @ template_attend.transpose(-2, -1))

                attn_return1[template_return<1e-8] = -float("inf")
                attn1 = (attn_return1).softmax(dim=-1)
                attn1[template_return<1e-8] = 0
                
                attn_return2[template_return<1e-8] = -float("inf")
                attn2 = (attn_return2).softmax(dim=-1)
                attn2[template_return<1e-8] = 0
                
                attn_return3[template_return<1e-8] = -float("inf")
                attn3 = (attn_return3).softmax(dim=-1)
                attn3[template_return<1e-8] = 0
                # print("xs1.shape: ", xs1.shape)
                # print("v.shape: ", v.shape)

                xs1 = attn1 @ v
                xs2 = attn2 @ v
                xs3 = attn3 @ v
                xs = (xs1 + xs2 + xs3) / 3
                res_otherImg += xs
            
            # Combine attention to only tokens of the same image with attention only to tokens of other images.
            # Use a weighted average.
            # res = (3*res_original[:, :, 1:] + res_sameImg + res_otherImg) / 5
            # res = (res_original[:, :, 1:] + res_otherImg) / 2
            # res = (res_original[:, :, 1:] + res_sameImg) / 2
            res = (res_sameImg + 3*res_otherImg) / 4
            
            
            
            res = torch.cat([x_ori_cls_token, res], dim=2)
            x = res.transpose(1, 2).reshape(B, N, C)
            x = self.proj_drop(self.proj(x))
                
        elif self.img_attn_only == 2:  # Attend only to tokens of the other images
            cls_token = qkv[:, :, :, 0, :]
            # print(f"cls_token shape: {cls_token.shape}")
            
            qkv = qkv[:, :, :, 1:, :]  # Exclude the class token.
            # print(f"qkv shapeNew: {qkv.shape}")
            v = qkv[2].detach().clone()
            res = torch.zeros_like(v)
            
            
            qkv= rearrange(qkv, 'q b a (n t) c -> q n b a t c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)  # 196=14*14
            for i in range(8):
                # print("qkv.shape: ", qkv.shape)
                template = torch.zeros_like(qkv)
                template[:, i] = 1
                template = rearrange(template, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
                template = template[0]
                
                template_attend = torch.ones_like(qkv)
                template_attend[:, i] = 0
                template_attend = rearrange(template_attend, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
                template_attend = template_attend[0]
                
                qkv_base = qkv.detach().clone()
                remove_dims = [k for k in range(8)]
                remove_dims.remove(i)
                qkv_base[:, [remove_dims]] = 0  # Only keep the values of the i'th entry.
                qkv_base = rearrange(qkv_base, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
                
                qkv_attend = qkv.detach().clone()
                qkv_attend[:, i] = 0
                qkv_attend = rearrange(qkv_attend, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
            
                xs1 = qkv_base[0]
                xs2 = qkv_base[1]
                xs3 = qkv_base[2]
                
                xs1_attend = qkv_attend[0]
                xs2_attend = qkv_attend[1]
                xs3_attend = qkv_attend[2]

                if self.ss_attn_temp is None:
                    pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
                    inv_temp = pre_norm * self.scale
                else:
                    inv_temp = self.ss_attn_temp

                for it in range(self.ss_attn_iter):
                    xs1 = F.normalize(xs1, dim=-1)
                    xs2 = F.normalize(xs2, dim=-1)
                    xs3 = F.normalize(xs3, dim=-1)
                    # print("xs1.shape: ", xs1.shape)
                    # print("xs1_attend.shape: ", xs1_attend.shape)

                    attn_return1 = (xs1 @ xs1_attend.transpose(-2, -1)) * inv_temp
                    attn_return2 = (xs2 @ xs2_attend.transpose(-2, -1)) * inv_temp
                    attn_return3 = (xs3 @ xs3_attend.transpose(-2, -1)) * inv_temp
                    template_return = (template @ template_attend.transpose(-2, -1))
                    

                    attn_return1[template_return<1e-8] = -float("inf")
                    attn1 = (attn_return1).softmax(dim=-1)
                    attn1[template_return<1e-8] = 0
                    
                    attn_return2[template_return<1e-8] = -float("inf")
                    attn2 = (attn_return2).softmax(dim=-1)
                    attn2[template_return<1e-8] = 0
                    
                    attn_return3[template_return<1e-8] = -float("inf")
                    attn3 = (attn_return3).softmax(dim=-1)
                    attn3[template_return<1e-8] = 0

                    xs1 = attn1 @ xs1
                    xs2 = attn2 @ xs2
                    xs3 = attn3 @ xs3

                # Assignment to V
                xs1 = F.normalize(xs1, dim=-1)
                xs2 = F.normalize(xs2, dim=-1)
                xs3 = F.normalize(xs3, dim=-1)

                attn_return1 = (xs1 @ xs1_attend.transpose(-2, -1)) * inv_temp
                attn_return2 = (xs2 @ xs2_attend.transpose(-2, -1)) * inv_temp
                attn_return3 = (xs3 @ xs3_attend.transpose(-2, -1)) * inv_temp
                template_return = (template @ template_attend.transpose(-2, -1))

                attn_return1[template_return<1e-8] = -float("inf")
                attn1 = (attn_return1).softmax(dim=-1)
                attn1[template_return<1e-8] = 0
                
                attn_return2[template_return<1e-8] = -float("inf")
                attn2 = (attn_return2).softmax(dim=-1)
                attn2[template_return<1e-8] = 0
                
                attn_return3[template_return<1e-8] = -float("inf")
                attn3 = (attn_return3).softmax(dim=-1)
                attn3[template_return<1e-8] = 0
                # print("xs1.shape: ", xs1.shape)
                # print("v.shape: ", v.shape)

                xs1 = attn1 @ v
                xs2 = attn2 @ v
                xs3 = attn3 @ v
                xs = (xs1 + xs2 + xs3) / 3
                # print("attn1.shape: ", attn1.shape)
                # print("v.shape: ", v.shape)
                
                # print("xs1.shape: ", xs1.shape)
                # a = torch.count_nonzero(xs1, dim=2)
                # print("a: ", a)
                # print("xs1: ", xs1)
                # return
                res += xs
            # xs = xs3
            # print("x_ori_cls_token.shape: ", x_ori_cls_token.shape)
            # print("res.shape: ", res.shape)
            res = torch.cat([x_ori_cls_token, res], dim=2)
            x = res.transpose(1, 2).reshape(B, N, C)
            x = self.proj_drop(self.proj(x))

        elif self.img_attn_only == 11:  # Same as version 1 but more generalized to adapt it to version 2. Slower, but same results as 1.
            cls_token = qkv[:, :, :, 0, :]
            # print(f"cls_token shape: {cls_token.shape}")
            
            qkv = qkv[:, :, :, 1:, :]  # Exclude the class token.
            # print(f"qkv shapeNew: {qkv.shape}")
            v = qkv[2].detach().clone()
            res = torch.zeros_like(v)
            
            
            qkv= rearrange(qkv, 'q b a (n t) c -> q n b a t c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)  # 196=14*14
            # attn1_sum = torch.zeros(1, 12, 1568, 1568)  # Shape of the attn_return.
            for i in range(8):
                
                # print("qkv.shape: ", qkv.shape)
                template = torch.zeros_like(qkv)
                template[:, i] = 1
                template = rearrange(template, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
                template = template[0]
                
                qkv_base = qkv.detach().clone()
                remove_dims = [k for k in range(8)]
                remove_dims.remove(i)
                qkv_base[:, [remove_dims]] = 0  # Only keep the values of the i'th entry.
                qkv_base = rearrange(qkv_base, 'q n b a t c -> q b a (n t) c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)
                
                xs1 = qkv_base[0]
                xs2 = qkv_base[1]
                xs3 = qkv_base[2]

                if self.ss_attn_temp is None:
                    pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
                    inv_temp = pre_norm * self.scale
                else:
                    inv_temp = self.ss_attn_temp

                for it in range(self.ss_attn_iter):
                    xs1 = F.normalize(xs1, dim=-1)
                    xs2 = F.normalize(xs2, dim=-1)
                    xs3 = F.normalize(xs3, dim=-1)
                    # print("xs1.shape: ", xs1.shape)
                    # print("xs1_attend.shape: ", xs1_attend.shape)

                    attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
                    attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
                    attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp
                    template_return = (template @ template.transpose(-2, -1))
                    # print(i, "attn_return1.shape: ", attn_return1.shape)
                    
                    # attn1 = torch.zeros_like(attn_return1)
                    # b = attn_return1[template_return > 0]
                    # a = (b).softmax(dim=-1)
                    attn_return1[template_return<1e-8] = -float("inf")
                    attn1 = (attn_return1).softmax(dim=-1)
                    attn1[template_return<1e-8] = 0
                    
                    attn_return2[template_return<1e-8] = -float("inf")
                    attn2 = (attn_return2).softmax(dim=-1)
                    attn2[template_return<1e-8] = 0
                    
                    attn_return3[template_return<1e-8] = -float("inf")
                    attn3 = (attn_return3).softmax(dim=-1)
                    attn3[template_return<1e-8] = 0
                    

                    xs1 = attn1 @ xs1
                    xs2 = attn2 @ xs2
                    xs3 = attn3 @ xs3

                # Assignment to V
                xs1 = F.normalize(xs1, dim=-1)
                xs2 = F.normalize(xs2, dim=-1)
                xs3 = F.normalize(xs3, dim=-1)

                attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
                attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
                attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp
                template_return = (template @ template.transpose(-2, -1))  # Was already done, no need to execute it again.

                attn_return1[template_return<1e-8] = -float("inf")
                attn1 = (attn_return1).softmax(dim=-1)
                attn1[template_return<1e-8] = 0
                
                attn_return2[template_return<1e-8] = -float("inf")
                attn2 = (attn_return2).softmax(dim=-1)
                attn2[template_return<1e-8] = 0
                
                attn_return3[template_return<1e-8] = -float("inf")
                attn3 = (attn_return3).softmax(dim=-1)
                attn3[template_return<1e-8] = 0
                # print("xs1.shape: ", xs1.shape)
                # print("v.shape: ", v.shape)

                xs1 = attn1 @ v
                xs2 = attn2 @ v
                xs3 = attn3 @ v
                xs = (xs1 + xs2 + xs3) / 3
                res += xs
            # xs = xs3
            # print("x_ori_cls_token.shape: ", x_ori_cls_token.shape)
            # print("res.shape: ", res.shape)
            res = torch.cat([x_ori_cls_token, res], dim=2)
            x = res.transpose(1, 2).reshape(B, N, C)
            x = self.proj_drop(self.proj(x))

        
        elif self.img_attn_only == 1:  # Attend only to tokens of the same image.
            if self.ss_attn_iter == 1:
                cls_token = qkv[:, :, :, 0, :]
                # print(f"cls_token shape: {cls_token.shape}")
                
                qkv = qkv[:, :, :, 1:, :]  # Exclude the class token.
                # print(f"qkv shapeNew: {qkv.shape}")
                
                qkv = rearrange(qkv, 'q b a (n t) c -> q n b a t c', q=3, b=B, a=self.num_heads, n=8, t=self.token_cnt**2, c=C // self.num_heads)  # 196=14*14
                # print(f"qkv shapeNewshapeNew: {qkv.shape}")
                
                res = []
                
                for i in range(8):
                    xs1 = qkv[0][i]
                    # print(f"xs1 shape: {xs1.shape}")

                    xs2 = qkv[1][i]
                    xs3 = qkv[2][i]
                    v = qkv[2][i]

                    if self.ss_attn_temp is None:
                        pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
                        inv_temp = pre_norm * self.scale
                    else:
                        inv_temp = self.ss_attn_temp

                    for it in range(self.ss_attn_iter):
                        xs1 = F.normalize(xs1, dim=-1)
                        xs2 = F.normalize(xs2, dim=-1)
                        xs3 = F.normalize(xs3, dim=-1)

                        attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
                        attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
                        attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

                        attn1 = (attn_return1).softmax(dim=-1)
                        attn2 = (attn_return2).softmax(dim=-1)
                        attn3 = (attn_return3).softmax(dim=-1)

                        xs1 = attn1 @ xs1
                        xs2 = attn2 @ xs2
                        xs3 = attn3 @ xs3

                    # Assignment to V
                    xs1 = F.normalize(xs1, dim=-1)
                    xs2 = F.normalize(xs2, dim=-1)
                    xs3 = F.normalize(xs3, dim=-1)

                    attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
                    attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
                    attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

                    attn1 = (attn_return1).softmax(dim=-1)
                    attn2 = (attn_return2).softmax(dim=-1)
                    attn3 = (attn_return3).softmax(dim=-1)

                    xs1 = attn1 @ v
                    xs2 = attn2 @ v
                    xs3 = attn3 @ v
                    xs = (xs1 + xs2 + xs3) / 3
                    # xs = xs3
                    res.append(xs)
                    # print("xs shape final: ", xs.shape)
                    # print("x_ori_cls_token shape final: ", x_ori_cls_token.shape)
                    
            else:  # First attend with all tokens and then only within the tokens of one image.
                # First attention between all tokens:
                xs1 = v
                xs2 = k
                xs3 = q

                if self.ss_attn_temp is None:
                    pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
                    inv_temp = pre_norm * self.scale
                else:
                    inv_temp = self.ss_attn_temp

                for it in range(1):
                    xs1 = F.normalize(xs1, dim=-1)
                    xs2 = F.normalize(xs2, dim=-1)
                    xs3 = F.normalize(xs3, dim=-1)

                    attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
                    attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
                    attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

                    attn1 = (attn_return1).softmax(dim=-1)
                    attn2 = (attn_return2).softmax(dim=-1)
                    attn3 = (attn_return3).softmax(dim=-1)

                    xs1 = attn1 @ xs1
                    xs2 = attn2 @ xs2
                    xs3 = attn3 @ xs3
                
                ###############
                # Now the attention only among the tokens of the same image.                
                # print("xs1.shape", xs1.shape)
                xs1 = xs1[:, :, 1:, :]  # Exclude the class token.
                xs2 = xs2[:, :, 1:, :]  # Exclude the class token.
                xs3 = xs3[:, :, 1:, :]  # Exclude the class token.
                
                qkv = qkv[:, :, :, 1:, :]  # Exclude the class token.
                # print(f"qkv shapeNew: {qkv.shape}")
                
                qkv = rearrange(qkv, 'q b a (n t) c -> q n b a t c', q=3, b=B, a=self.num_heads, n=8, t=196, c=C // self.num_heads)  # 196=14*14

                
                xs1_start = rearrange(xs1, 'b a (n t) c -> n b a t c', b=B, a=self.num_heads, n=8, t=196, c=C // self.num_heads)  # 196=14*14
                xs2_start = rearrange(xs2, 'b a (n t) c -> n b a t c', b=B, a=self.num_heads, n=8, t=196, c=C // self.num_heads)  # 196=14*14
                xs3_start = rearrange(xs3, 'b a (n t) c -> n b a t c', b=B, a=self.num_heads, n=8, t=196, c=C // self.num_heads)  # 196=14*14
                                
                res = []

                for i in range(8):
                    xs1 = xs1_start[i]
                    xs2 = xs2_start[i]
                    xs3 = xs3_start[i]
                    # v = qkv[2][i]
                    v = xs1_start[i]

                    for it in range(self.ss_attn_iter - 1):
                        xs1 = F.normalize(xs1, dim=-1)
                        xs2 = F.normalize(xs2, dim=-1)
                        xs3 = F.normalize(xs3, dim=-1)

                        attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
                        attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
                        attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

                        attn1 = (attn_return1).softmax(dim=-1)
                        attn2 = (attn_return2).softmax(dim=-1)
                        attn3 = (attn_return3).softmax(dim=-1)

                        xs1 = attn1 @ xs1
                        xs2 = attn2 @ xs2
                        xs3 = attn3 @ xs3

                    # Assigment to V
                    xs1 = F.normalize(xs1, dim=-1)
                    xs2 = F.normalize(xs2, dim=-1)
                    xs3 = F.normalize(xs3, dim=-1)

                    attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
                    attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
                    attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

                    attn1 = (attn_return1).softmax(dim=-1)
                    attn2 = (attn_return2).softmax(dim=-1)
                    attn3 = (attn_return3).softmax(dim=-1)

                    xs1 = attn1 @ v
                    xs2 = attn2 @ v
                    xs3 = attn3 @ v
                    xs = (xs1 + xs2 + xs3) / 3
                    # xs = xs3
                    res.append(xs)
                    # print("xs shape final: ", xs.shape)
                    # print("x_ori_cls_token shape final: ", x_ori_cls_token.shape)
                
            res = torch.cat(res, dim=2)
            # print("res shape final: ", res.shape)
            
            # res = rearrange(res, 'b a (t n) c -> b a (n t) c', b=B, a=self.num_heads, n=8, t=196, c=C // self.num_heads)  # 196=14*14
            # print("res22 shape final: ", res.shape)
            res = torch.cat([x_ori_cls_token, res], dim=2)
            # print("res33 shape final: ", res.shape)
            
            x = res.transpose(1, 2).reshape(B, N, C)
            
            x = self.proj_drop(self.proj(x))

        if wasList:
            return [x.transpose(0, 1), x_ori.transpose(0, 1), stack]
        else:
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
            

class GEMResidualBlockLast(nn.Module):
    """
    Skip the last residual block with its selfself attention. Without selfself attention the performance is much worse, therefore
    maybe the late "normal" inputs should be reduced.
    """
    def __init__(self, res_block):
        super(GEMResidualBlockLast, self).__init__()
        self.res_block = res_block

    def forward(self,
                q_x: torch.Tensor,
                k_x: Optional[torch.Tensor] = None,
                v_x: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                ):
        return q_x

class GEMResidualBlock(nn.Module):
    def __init__(self, res_block):
        super(GEMResidualBlock, self).__init__()
        self.res_block = res_block

    def forward(self,
                q_x: torch.Tensor,
                k_x: Optional[torch.Tensor] = None,
                v_x: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                ):
        if isinstance(q_x, list):
            x_gem, q_x, stack, stack_init = q_x
        else:
            x_gem = q_x
            stack = [q_x]
            stack_init = [q_x]  # Similar to stack, but saves the outputs of the self-attention blocks (at the same indices as in stack).
            

        x_gem_res, x_ori_res = self.res_block.attn(x=self.res_block.ln_1(q_x))
        x_gem_res, x_ori_res = self.res_block.ls_1(x_gem_res), self.res_block.ls_1(x_ori_res)
        # Original
        x_ori = q_x + x_ori_res
        x_ori = x_ori + self.res_block.ls_2(self.res_block.mlp(self.res_block.ln_2(x_ori)))
        # GEM
        x_gem = x_gem + x_gem_res
        stack.append(x_gem_res)
        stack_init.append(x_ori - sum(stack_init))  # What was added in that layer.
        # print(stack_init)
        return [x_gem, x_ori, stack, stack_init]

class GEMResidualBlockViCLIP(nn.Module):
    def __init__(self, res_block, addBef=False):
        super(GEMResidualBlockViCLIP, self).__init__()
        self.res_block = res_block
        self.addBef = addBef

    def forward(self,
                q_x: torch.Tensor,
                k_x: Optional[torch.Tensor] = None,
                v_x: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                ):
        wasList = False
        if isinstance(q_x, list):
            wasList = True
            x_gem, q_x, stack, stack_init = q_x  # Stack saves the outputs from the self self attention in a list.
        else:
            x_gem = q_x
            
            # x_gem = torch.zeros_like(q_x)  # Dont add the self-attention input.
            stack = [q_x]  # Basic input, before the residuals come in.
            stack_init = [q_x]   # Similar to stack, but saves the outputs of the self-attention blocks (at the same indices as in stack).

        ### For testing with using initial unchanged res_blocks.
        #x_gem_res, x_ori_res = self.res_block.attn(x=self.res_block.ln_1(q_x))
        #x_gem_res, x_ori_res = self.res_block.drop_path1(x_gem_res), self.res_block.drop_path1(x_ori_res)
        # abc = self.res_block.ln_1(q_x)
        # print("attnattnattn: ", abc.shape)
        
        # x_ori_res = self.res_block.attn(abc, abc, abc, need_weights=False, attn_mask=None)[0]
        # x_ori_res = self.res_block.drop_path1(x_ori_res)# Original
        # x_gem_res = self.res_block.drop_path1(x_ori_res)# Original
        # x_ori = q_x + x_ori_res
        # x_ori = x_ori + self.res_block.drop_path2(self.res_block.mlp(self.res_block.ln_2(x_ori)))
        # # GEM
        # x_gem = x_gem + x_gem_res
        # return [x_gem, x_ori]
        # print("wasList resBlock: ", wasList)
        
        if self.addBef and wasList:
            # print("inside")
            x_gem_res, x_ori_res = self.res_block.attn(x=(self.res_block.ln_1(x_gem), self.res_block.ln_1(q_x)))
        else:
            x_gem_res, x_ori_res = self.res_block.attn(x=self.res_block.ln_1(q_x))
        x_gem_res, x_ori_res = self.res_block.drop_path1(x_gem_res), self.res_block.drop_path1(x_ori_res)# Original
        x_ori = q_x + x_ori_res
        x_ori = x_ori + self.res_block.drop_path2(self.res_block.mlp(self.res_block.ln_2(x_ori)))
        # GEM
        stack.append(x_gem_res)  # The residual from GEM.
        # stack_init.append(x_ori_res)  # The residual from self attn. Initially stack_init.
        stack_init.append(x_ori - sum(stack_init))  # What was added in that layer.
        # print(stack_init)
        if self.addBef:
            x_gem = x_gem + x_gem_res
        else:
            x_gem = x_gem + x_gem_res
        # stack.append(x_gem)
        return [x_gem, x_ori, stack, stack_init]

class GEMViT(nn.Module):
    def __init__(self, vit):
        self.vit = vit

def calculate_weights(stackWeights, stack, sim_stack, x_l1):
    """
    Get the weights for the self-self attention outputs based on similarities or use predefined weights. Return the weighted average of the tensors.
    sim_stack and x_l1 are used to calculate the weights if "stackWeights" is set to "dynamic":
    """
    
    if stackWeights == "dynamic":  # Dynamically set the weights based on similarities from x^(l-1) (x_l1) and the features of the sim_stack.
        # print("sim_stack[0].shape: ", sim_stack[0].shape)
        # print("sim_stack[1].shape: ", sim_stack[1].shape)
        # print("sim_stack[2].shape: ", sim_stack[2].shape)
        # print("sim_stack[3].shape: ", sim_stack[3].shape)
        # print("sim_stack[4].shape: ", sim_stack[4].shape)
        # print("x_l1.shape: ", x_l1.shape)
        comparer = torch.zeros_like(x_l1)
        all_sims = torch.zeros(len(sim_stack)-1)
        for i, elm in enumerate(sim_stack):
            if i == 0:
                continue
            comparer = comparer + elm
            comparer_normed = F.normalize(x_l1, dim=-1)
            elm = F.normalize(elm, dim=-1)
            
            sims = comparer_normed * elm
            # print("sims.shape: ", sims.shape)
            sims = sims.sum(dim=-1)  # Now we have the pairwise cosine similarities.
            sims = sims + 1  # To avoid negative and positive similarities to cancel out.
            sims = sims.mean(dim=-1) * (-1)  # Lower cosine similarity means it differs more. That should lead to a bigger weight.
            # print("sims1.shape: ", sims.shape)
            # print("sims: ", i, sims)
            all_sims[i-1] = sims
        weights = all_sims.softmax(dim=-1)*7
        # print("weights.shape: ", weights.shape)
        
        weights = torch.cat((torch.Tensor([1]), weights), dim=0)
        # print("weights: ", i, weights)
    

    elif type(stackWeights) is list or type(stackWeights) is tuple:
        weights = stackWeights
    
    # Now apply the weights.
    x_gem = torch.zeros_like(stack[0])
    for i, elm in enumerate(stack):
        x_gem = x_gem + weights[i] * elm
    
    return x_gem




def modified_vit_forward(self, x: torch.Tensor):
    if self.model_type in ["clip", "openclip"]:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid_h, grid_w = x.shape[2:]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]

        if x.shape[1] != self.positional_embedding.shape[1]:
            pos_emb = resample_abs_pos_embed(self.positional_embedding.unsqueeze(0),
                                             new_size=[grid_h, grid_w],
                                             # old_size=list(self.grid_size),
                                             num_prefix_tokens=1,
                                             interpolation='bicubic',
                                             antialias=True)

        else:
            pos_emb = self.positional_embedding

        x = x + pos_emb.to(x.dtype)
        # x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # print(f"x shape before transformer: {x.shape}")
        
        x_gem, x, stack, stack_init = self.transformer(x)
        # Stack saves all the outputs from the self-self attention blocks in a list.
        # stack_init saves all the outputs from the self attention blocks in a list.

        x = x.permute(1, 0, 2)  # LND -> NLD
        x_gem = x_gem.permute(1, 0, 2)  # LND -> NLD
        
        if self.stackWeights:
            stack_new = []
            for i, elm in enumerate(stack):
                elm = elm.permute(1, 0, 2)
                stack_new.append(elm)
            stack_init_new = []
            for i, elm in enumerate(stack_init):
                elm = elm.permute(1, 0, 2)
                stack_init_new.append(elm)

            x_l1 = stack_new[0]
            x_gem = calculate_weights(self.stackWeights, stack_new, stack_init_new, x_l1)

        # Apply proj
        x = self.ln_post(x)
        # print("x_gem.shape: ", x_gem.shape)
        x_gem = self.ln_post(x_gem)
        if self.proj is not None:
            x = x @ self.proj
            x_gem = x_gem @ self.proj



        final_stack = []
        # base = torch.zeros_like(stack_init[0])
        # base = stack_init[0]
        # print("base.shape: ", base.shape)
        base = torch.stack(stack_init).sum(dim=0)
        for idx, elm in enumerate(stack_init):  # or stack_init
            current_elm = base - elm  # Full pass without the current element. The more the similarity will drop, the more important the element was.
            # print("current_elm.shape: ", current_elm.shape)
            # base += elms
            
            current_elm = current_elm.permute(1, 0, 2)
            current_elm = self.ln_post(current_elm)
            if self.proj is not None:
                current_elm = current_elm @ self.proj
            final_stack.append(current_elm)

        # With the help of the stack_init -> final_stack the importance of the different self-self attention layers is determined.
        # Then based on the weighted average the last steps of this function need to be repeated. Therefore, these functions need to be returned as well.
        further_process = {"ln_post": self.ln_post,
                           "proj": self.proj
                           }

        return [x_gem, x, stack, stack_init, final_stack, further_process]

    elif self.model_type in ["viclip"]:  # For viCLIP.
        if x.ndim == 5:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        else:
            x = x.unsqueeze(2)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if x.shape[1] != self.positional_embedding.shape[1]:
            # print("Pos embed: ")
            pos_emb = resample_abs_pos_embed(self.positional_embedding.unsqueeze(0),
                                             new_size=[H, W],
                                             # old_size=list(self.grid_size),
                                             num_prefix_tokens=1,
                                             interpolation='bicubic',
                                             antialias=True)

        else:
            pos_emb = self.positional_embedding
        x = x + pos_emb.to(x.dtype)

        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        # print(f"x initial: {x.shape}")
        # print(f"cls_tokens initial: {cls_tokens.shape}")
        
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if hasattr(self, 'temporal_positional_embedding'):
            if x.size(1) == 1:
                # This is a workaround for unused parameter issue
                x = x + self.temporal_positional_embedding.mean(1)
            else:
                x = x + self.temporal_positional_embedding
        # print(f"x shape before: {x.shape}")
        
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  #BND -> NBD
        # print(f"x shape before transformer: {x.shape}")
        ###x_gem, x = self.transformer(x)
        x_gem, x, stack, stack_init = self.transformer(x)
        # Stack saves all the outputs from the self-self attention blocks in a list.
        # stack_init saves all the outputs from the self attention blocks in a list.
        
        if self.stackWeights:
            x_l1 = stack[0]
            x_gem = calculate_weights(self.stackWeights, stack, stack, x_l1)

        x = self.ln_post(x)
        x_gem = self.ln_post(x_gem)

        if self.proj is not None:
            x = self.dropout(x) @ self.proj
            x_gem = self.dropout(x_gem) @ self.proj 
        x = x.permute(1, 0, 2)
        x_gem = x_gem.permute(1, 0, 2)
        
        final_stack = []
        for elm in stack:  # or stack_init
            elm = self.ln_post(elm)
            if self.proj is not None:
                elm = self.dropout(elm) @ self.proj
            elm = elm.permute(1, 0, 2)
            final_stack.append(elm)
            

        final_stack = []
        base = torch.stack(stack_init).sum(dim=0)
        for idx, elm in enumerate(stack_init):
            current_elm = base - elm  # Full pass without the current element. The more the similarity will drop, the more important the element was.
            
            current_elm = self.ln_post(current_elm)
            if self.proj is not None:
                current_elm = current_elm @ self.proj
            current_elm = current_elm.permute(1, 0, 2)
            final_stack.append(current_elm)
        
        # With the help of the stack_init -> final_stack the importance of the different self-self attention layers is determined.
        # Then based on the weighted average the last steps of this function need to be repeated. Therefore, these functions need to be returned as well.
        further_process = {"ln_post": self.ln_post,
                           "proj": self.proj,
                           "dropout": self.dropout}

        return [x_gem, x, stack, stack_init, final_stack, further_process]