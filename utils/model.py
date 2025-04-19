import torch
import torch.nn as nn
import sys
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm, Dropout, Linear
from torch.nn.functional import normalize
from torch.nn import functional as F
import math
from mamba_ssm import Mamba

    
class MambaEncoder(nn.Module):
    def __init__(self,input_dim=1024, hidden_dim=4096, output_dim=1024):
        super(MambaEncoder, self).__init__()
        
        self.mamba = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=1024, # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
        )
    def feature_enhance(self,image_embed,text_embed):
        i1 = torch.sum(image_embed,dim=1)
        t1 = torch.sum(text_embed, dim=1)
        mi = i1.unsqueeze(1) @ i1.unsqueeze(0)
        mt = t1.unsqueeze(1) @ t1.unsqueeze(0)
        similar_matrix = mi - mt
        similar_matrix = (1-torch.tanh(similar_matrix)**2)*torch.sigmoid(similar_matrix)*(1-torch.sigmoid(similar_matrix))
        feature_a = similar_matrix @ image_embed
        feature_b = similar_matrix @ text_embed
        feature_c = torch.cat((feature_a, feature_b), dim=1)
        return 0.1*feature_c
    
    
    def forward(self, image_embed , text_embed):
        #tokens = self.sattention(image_embed=image_embed,text_embed=text_embed)
        tokens = torch.concat((image_embed, text_embed), dim = 1)
        tokens = tokens.unsqueeze(0)
        result = self.mamba(tokens).squeeze()
        result =result+ self.feature_enhance(image_embed,text_embed)  #+ self.mlp_enhance(image_embed=image_embed,text_embed=text_embed)
        #result = self.mlp_fusion(result)
        result = normalize(result, p =2 ,dim =1)
        return result[:,:512],result[:,512:]


class MLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=4096, output_dim=1024):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim) 
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)  
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

    def forward(self, image_embed,text_embed):
        x = torch.concat((image_embed, text_embed), dim = 1)
        x = self.tanh(self.layer1(x))  
        x = self.tanh(self.dp(self.layer2(x)))  
        x = self.output_layer(x)       
        return x[:,:512],x[:,512:]

class FusionMlp(nn.Module):
    def __init__(self, input_dim, out_put, dim_feedforward=[1024,128,1024], dropout=0.3):#  dropout=0.3
        super(FusionMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(dropout)
        self.tohash = nn.Linear(4096, out_put)
        self.alpha = 1.0
        self.tanh = nn.Tanh()
        #self.kan = KAN([input_dim,hash_lens])

    def _ff_block(self, x):
        x = normalize(x, p =2 ,dim =1)
        # torch.Size([128, 512])
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid*self.alpha)
        return out
    
    #def set_alpha(self,epoch):
    #    self.alpha = math.pow((1.0*epoch+1.0),0.5)

    def forward(self, X):  
        #mlp_output = self.kan(X)
        mlp_output = self._ff_block(X)
        mlp_output = normalize(mlp_output, p =2 ,dim =1)
        return mlp_output


class ImageMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024,128,1024], dropout=0.3):#  dropout=0.3
        super(ImageMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(dropout)
        self.tohash = nn.Linear(4096, hash_lens)
        self.alpha = 1.0
        self.tanh = nn.Tanh()
        #self.kan = KAN([input_dim,hash_lens])

    def _ff_block(self, x):
        x = normalize(x, p =2 ,dim =1)
        # torch.Size([128, 512])
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid*self.alpha)
        return out
    
    #def set_alpha(self,epoch):
    #    self.alpha = math.pow((1.0*epoch+1.0),0.5)

    def forward(self, X):  
        #mlp_output = self.kan(X)
        mlp_output = self._ff_block(X)
        mlp_output = normalize(mlp_output, p =2 ,dim =1)
        return mlp_output



class TextMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024,128,1024], dropout=0.1): # dropout=0.1
        super(TextMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(dropout)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()
        self.alpha = 1.0

    
    def _ff_block(self, x):
        x = normalize(x, p =2 ,dim =1)
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid*self.alpha)
        return out
    
    def forward(self, X):  
        #mlp_output = self.kan(X)
        mlp_output=  self._ff_block(X)
        mlp_output = normalize(mlp_output, p =2 ,dim =1)
        return mlp_output
    

