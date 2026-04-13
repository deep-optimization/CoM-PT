import argparse
import numpy as np
import torch
import timm
import os

# Old Weight Initialization
def uniform_element_selection(wt, s_shape):
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    for dim in range(wt.dim()):
        assert wt.shape[dim] >= s_shape[dim], "Teacher's dimension should not be smaller than student's dimension"  # determine whether teacher is larger than student on this dimension
        if wt.shape[dim] % s_shape[dim] == 0:
            step = wt.shape[dim] // s_shape[dim]
            indices = torch.arange(s_shape[dim]) * step
        else:
            indices = torch.round(torch.linspace(0, wt.shape[dim], s_shape[dim]))
        ws = torch.index_select(ws, dim, indices)
    assert ws.shape == s_shape
    return ws


# Our Weight Initialization for ViT
def uniform_element_duplication(wt, scale=1):
    assert scale >= 1
    H, W = wt.shape
    weight_list = []

    for i in range(scale**2):
        weight_list.append(wt.clone())

    ws = torch.stack(weight_list, dim=0).contiguous()
    ws = ws.view(scale, scale, H, W).permute(0,2,1,3).view(scale*H, scale*W)

    return ws



if __name__ == "__main__":
    ckpt_s = torch.load('/home/disk02/anbang/fjw/CLIP-KD/src/logs/2024_08_27-15_34_27-model_ViT-N-16-lr_0.001-b_128-epochs_32-tag_cc3m-baseline/checkpoints/epoch_32.pt')
    ckpt_t = torch.load()
    for key in ckpt_s['state_dict'].keys():
        print('The shape of ' + key + ': Student:' + str(ckpts['state_dict'][key].shape + '; Teacher: '))
    # print(ckpt['state_dict']['module.positional_embedding'].shape)