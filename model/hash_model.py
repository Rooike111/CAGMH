import os
import torch
import logging
import torch.nn as nn
import numpy as np
from typing import Union
from utils.model import ImageMlp,TextMlp,MambaEncoder
from model.model import build_model
from utils import get_logger, get_summary_writer
import timm
from torch.cuda.amp import GradScaler, autocast
import torchvision
import math


def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False

class CAGMH(nn.Module):
    def __init__(self, outputDim=64, clipPath="./ViT-B-32.pt", writer=None, saveDir="./result/log", logger: logging.Logger=None, is_train=True,device = 1):
        super(CAGMH, self).__init__()
        print()
        os.makedirs(saveDir, exist_ok=True)
        # 保存训练或测试日志
        self.logger = logger if logger is not None else get_logger(os.path.join(saveDir, "train.log" if is_train else "test.log"))
        #写入 tensorboard
        self.writer = writer if writer is not None and is_train else get_summary_writer(os.path.join(saveDir, "tensorboard"))
        self.device = device
        embedDim, self.clip = self.load_clip(clipPath)

        self.FuseTrans = MambaEncoder()
        self.image_hash = ImageMlp(embedDim, outputDim)
        self.text_hash = TextMlp(embedDim, outputDim)
        
        #self.freezen()

    def freezen(self):
        for name, param in self.clip.named_parameters():
            # print(name)
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                                        or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                # print("1")
                continue
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= 12:
                    # print("2")
                    continue
            if name.find("conv2.") == 0:
                # print("3")
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    def load_clip(self, clipPath: str) -> tuple:
        try:
            model = torch.jit.load(clipPath, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(clipPath, map_location="cpu")
        
        return state_dict["text_projection"].shape[1], build_model(state_dict)


    
    def eval(self):
        self.image_hash.eval()
        self.text_hash.eval()
        # self.clip.eval()

    def train(self):
        self.image_hash.train()
        self.text_hash.train()

    def encoding(self,image,text):
        image_embed = self.clip.encode_image(image) 
        text_embed = self.clip.encode_text(text)
        image_embed,text_embed = self.FuseTrans(image_embed=image_embed,text_embed=text_embed)
        image_embed = self.image_hash(image_embed)
        text_embed = self.text_hash(text_embed)
        return image_embed,text_embed


    def forward(self, image, text):
        image_embed, text_embed=self.encoding(image=image,text=text)
        return image_embed, text_embed

