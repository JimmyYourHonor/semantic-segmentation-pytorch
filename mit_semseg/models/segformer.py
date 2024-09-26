# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math

__all__ = ['MixVisionTransformer', 'mit_b0', 'mit_b1', 'mit_b2']

model_paths = {
    'mit_b0': '/content/drive/MyDrive/pretrained_mit/mit_b0.pth',
    'mit_b1': '/content/drive/MyDrive/pretrained_mit/mit_b1.pth',
    'mit_b2': '/content/drive/MyDrive/pretrained_mit/mit_b2.pth',
}

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

def img_size_to_emb(height, width, channel, freq):
    pos_x = torch.arange(height, device=freq.device, dtype=freq.dtype)
    pos_y = torch.arange(width, device=freq.device, dtype=freq.dtype)
    sin_inp_x = torch.outer(pos_x, freq)
    sin_inp_y = torch.outer(pos_y, freq)
    emb_x = get_emb(sin_inp_x).unsqueeze(1)
    emb_y = get_emb(sin_inp_y)
    emb = torch.zeros(
        (height, width, channel * 2),
        device=freq.device,
        dtype=freq.dtype,
    )
    emb[:, :, : channel] = emb_x
    emb[:, :, channel : 2 * channel] = emb_y

    emb = emb.flatten(0,1).unsqueeze(0).unsqueeze(0)
    return emb

def apply_rotatory_emb(x, pos_emb):
    x_out = torch.cat((pos_emb[:,:,:,1::2]*x[:,:,:,1::2] - \
                       pos_emb[:,:,:,::2]*x[:,:,:,::2], 
                       pos_emb[:,:,:,1::2]*x[:,:,:,1::2] + \
                       pos_emb[:,:,:,::2]*x[:,:,:,::2]), dim=-1)
    return x_out

def unfold_sliding_window(x, kernel, x_shape, num_heads):
    B,C,H,W = x_shape
    stride = kernel//2
    x = x.transpose(2,3).reshape(B,C,H,W)
    x = F.unfold(x, kernel_size=(kernel,kernel), stride=stride, padding=stride)
    x = x.reshape(B,num_heads, C//num_heads, kernel*kernel,-1)\
    .permute(0,4,1,3,2).reshape(-1,num_heads,kernel*kernel,C // num_heads)
    return x

def fold_sliding_window(x, kernel, x_shape):
    B,C,H,W = x_shape
    stride = kernel//2
    # divisor = F.fold(F.unfold(torch.ones(B,C,H,W).to(x.device), 
    #             kernel_size=(kernel,kernel), stride=stride, padding=stride), 
    #             output_size=(H,W), kernel_size=kernel, 
    #             stride=stride, padding=stride)
    x = x.reshape(B, -1, kernel*kernel, C).permute(0,3,2,1)
    x = F.fold(x.reshape(B, C*kernel*kernel, -1), output_size=(H,W), 
               kernel_size=kernel,stride=stride,padding=stride) #/ divisor
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, sliding=False, kernel=128):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sliding = sliding
        self.kernel = kernel

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, pos_emb=None, pos_emb_sr=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if self.sliding:
            kernel = self.kernel
            kernel_sr, H_sr, W_sr = kernel // self.sr_ratio, H // self.sr_ratio, W // self.sr_ratio
            q = unfold_sliding_window(q, kernel, (B,C,H,W), self.num_heads)
            k = unfold_sliding_window(k, kernel_sr, (B,C,H_sr,W_sr), self.num_heads)
            v = unfold_sliding_window(v, kernel_sr, (B,C,H_sr,W_sr), self.num_heads)

        if (pos_emb is not None):
            # Only apply position embedding to q and k
            # q = q + pos_emb
            # k = k + pos_emb_sr
            q = apply_rotatory_emb(q, pos_emb)
            k = apply_rotatory_emb(k, pos_emb_sr)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        if self.sliding:
            x = x.reshape(-1, self.kernel*self.kernel, C)
            x = fold_sliding_window(x, kernel, (B,C,H,W)).reshape(B,C,N)\
            .transpose(1,2)
        else:
            x = x.reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, sliding=False, kernel=128):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, sliding=sliding, kernel=kernel)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, pos_emb=None, pos_emb_sr=None):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, pos_emb, pos_emb_sr))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], use_pos_emb=False, sliding=False, kernels=[64, 32, 16, 8]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.sr_ratios = sr_ratios

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # Positional embedding
        self.use_pos_emb = use_pos_emb
        self.sliding = sliding
        if self.use_pos_emb:
            emb_ch_0 = math.ceil((embed_dims[0]/num_heads[0])/4) * 2
            emb_ch_1 = math.ceil((embed_dims[1]/num_heads[1])/4) * 2
            emb_ch_2 = math.ceil((embed_dims[2]/num_heads[2])/4) * 2
            emb_ch_3 = math.ceil((embed_dims[3]/num_heads[3])/4) * 2
            inv_freq_0 = 1.0 / (10000 ** (torch.arange(0, emb_ch_0, 2).float() / emb_ch_0))
            inv_freq_1 = 1.0 / (10000 ** (torch.arange(0, emb_ch_1, 2).float() / emb_ch_1))
            inv_freq_2 = 1.0 / (10000 ** (torch.arange(0, emb_ch_2, 2).float() / emb_ch_2))
            inv_freq_3 = 1.0 / (10000 ** (torch.arange(0, emb_ch_3, 2).float() / emb_ch_3))
            if self.sliding:
                self.emb_0 = img_size_to_emb(self.kernel_0, self.kernel_0, emb_ch_0, inv_freq_0)
                self.emb_1 = img_size_to_emb(self.kernel_1, self.kernel_1, emb_ch_1, inv_freq_1)
                self.emb_2 = img_size_to_emb(self.kernel_2, self.kernel_2, emb_ch_2, inv_freq_2)
                self.emb_3 = img_size_to_emb(self.kernel_3, self.kernel_3, emb_ch_3, inv_freq_3)
            else:
                self.emb_0 = img_size_to_emb(128, 128, emb_ch_0, inv_freq_0)
                self.emb_1 = img_size_to_emb(64, 64, emb_ch_1, inv_freq_1)
                self.emb_2 = img_size_to_emb(32, 32, emb_ch_2, inv_freq_2)
                self.emb_3 = img_size_to_emb(16, 16, emb_ch_3, inv_freq_3)

        
        if self.sliding:
            self.kernel_0 = kernels[0]
            self.kernel_1 = kernels[1]
            self.kernel_2 = kernels[2]
            self.kernel_3 = kernels[3]
        else:
            self.kernel_0 = None
            self.kernel_1 = None
            self.kernel_2 = None
            self.kernel_3 = None

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], sliding=sliding, kernel=self.kernel_0)
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], sliding=sliding, kernel=self.kernel_1)
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], sliding=sliding, kernel=self.kernel_2)
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3], sliding=sliding, kernel=self.kernel_3)
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        if(self.use_pos_emb):
            if(self.sliding):
                emb = self.emb_0.to(x.device)
                kernel = self.kernel_0
                emb_sr = F.interpolate(emb.reshape(1,kernel,kernel,-1).permute(0,3,1,2), 
                                    size=(kernel//self.sr_ratios[0],kernel//self.sr_ratios[0]), 
                                    mode='bilinear')
                emb_sr = torch.flatten(emb_sr, -2, -1).transpose(2,1).unsqueeze(0)
            else:
                self.emb_0 = self.emb_0.to(x.device)
                emb = F.interpolate(self.emb_0.reshape(1,128,128,-1).permute(0,3,1,2), 
                                    size=(H,W), mode='bilinear')
                emb = torch.flatten(emb, -2, -1).transpose(2,1).unsqueeze(0)
                emb_sr = F.interpolate(self.emb_0.reshape(1,128,128,-1).permute(0,3,1,2), 
                                    size=(H//self.sr_ratios[0],W//self.sr_ratios[0]), 
                                    mode='bilinear')
                emb_sr = torch.flatten(emb_sr, -2, -1).transpose(2,1).unsqueeze(0)
        else:
            emb = None
            emb_sr = None
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W, emb, emb_sr)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        if(self.use_pos_emb):
            if(self.sliding):
                emb = self.emb_1.to(x.device)
                kernel = self.kernel_1
                emb_sr = F.interpolate(emb.reshape(1,kernel,kernel,-1).permute(0,3,1,2), 
                                    size=(kernel//self.sr_ratios[1],kernel//self.sr_ratios[1]), 
                                    mode='bilinear')
                emb_sr = torch.flatten(emb_sr, -2, -1).transpose(2,1).unsqueeze(0)
            else:
                self.emb_1 = self.emb_1.to(x.device)
                emb = F.interpolate(self.emb_1.reshape(1,64,64,-1).permute(0,3,1,2), 
                                      size=(H,W), mode='bilinear')
                emb = torch.flatten(emb, -2, -1).transpose(2,1).unsqueeze(0)
                emb_sr = F.interpolate(self.emb_1.reshape(1,64,64,-1).permute(0,3,1,2), 
                                      size=(H//self.sr_ratios[1],W//self.sr_ratios[1]), 
                                      mode='bilinear')
                emb_sr = torch.flatten(emb_sr, -2, -1).transpose(2,1).unsqueeze(0)
        else:
            emb = None
            emb_sr = None
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W, emb, emb_sr)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        if(self.use_pos_emb):
            if(self.sliding):
                emb = self.emb_2.to(x.device)
                kernel = self.kernel_2
                emb_sr = F.interpolate(emb.reshape(1,kernel,kernel,-1).permute(0,3,1,2), 
                                    size=(kernel//self.sr_ratios[2],kernel//self.sr_ratios[2]), 
                                    mode='bilinear')
                emb_sr = torch.flatten(emb_sr, -2, -1).transpose(2,1).unsqueeze(0)
            else:
                self.emb_2 = self.emb_2.to(x.device)
                emb = F.interpolate(self.emb_2.reshape(1,32,32,-1).permute(0,3,1,2), 
                                      size=(H,W), mode='bilinear')
                emb = torch.flatten(emb, -2, -1).transpose(2,1).unsqueeze(0)
                emb_sr = F.interpolate(self.emb_2.reshape(1,32,32,-1).permute(0,3,1,2), 
                                      size=(H//self.sr_ratios[2],W//self.sr_ratios[2]), 
                                      mode='bilinear')
                emb_sr = torch.flatten(emb_sr, -2, -1).transpose(2,1).unsqueeze(0)
        else:
            emb = None
            emb_sr = None
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W, emb, emb_sr)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        if(self.use_pos_emb):
            if(self.sliding):
                emb = self.emb_3.to(x.device)
                kernel = self.kernel_3
                emb_sr = F.interpolate(emb.reshape(1,kernel,kernel,-1).permute(0,3,1,2), 
                                    size=(kernel//self.sr_ratios[3],kernel//self.sr_ratios[3]), 
                                    mode='bilinear')
                emb_sr = torch.flatten(emb_sr, -2, -1).transpose(2,1).unsqueeze(0)
            else:
                self.emb_3 = self.emb_3.to(x.device)
                emb = F.interpolate(self.emb_3.reshape(1,16,16,-1).permute(0,3,1,2), 
                                      size=(H,W), mode='bilinear')
                emb = torch.flatten(emb, -2, -1).transpose(2,1).unsqueeze(0)
                emb_sr = F.interpolate(self.emb_3.reshape(1,16,16,-1).permute(0,3,1,2), 
                                      size=(H//self.sr_ratios[3],W//self.sr_ratios[3]), 
                                      mode='bilinear')
                emb_sr = torch.flatten(emb_sr, -2, -1).transpose(2,1).unsqueeze(0)
        else:
            emb = None
            emb = None
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W, emb, emb_sr)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x, return_feature_maps=False):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

        
def mit_b0(pretrained=False, use_pos_emb=False, sliding=False, kernels=[64, 32, 16, 8]):
    model = MixVisionTransformer(patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, use_pos_emb=use_pos_emb, sliding=sliding, kernels=kernels)
    if pretrained:
        model.load_state_dict(torch.load(model_paths['mit_b0']), strict=False)
    return model

def mit_b1(pretrained=False):
    model = MixVisionTransformer(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
    if pretrained:
        model.load_state_dict(torch.load(model_paths['mit_b1']), strict=False)
    return model

def mit_b2(pretrained=False):
    model = MixVisionTransformer(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
    if pretrained:
        model.load_state_dict(torch.load(model_paths['mit_b2']), strict=False)
    return model