import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from models.ginet import GINet
from models.gcn import GCN
from collections import OrderedDict

# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""

#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)

# #--------------------------------#
# # QuickGELU激活函数的定义
# # 在transformer结构中的MLP层中被使用
# #--------------------------------#
# class QuickGELU(nn.Module):
#     def forward(self, x: torch.Tensor):
#         return x * torch.sigmoid(1.702 * x)
 
# #-------------------------------------------------#
# # transformer模块的定义,将会在transformer结构中被使用
# # 1.多头注意力层
# # 2.LayerNorm层
# # 3.MLP层
# #-------------------------------------------------#
# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
#         super(ResidualAttentionBlock,self).__init__()
#         #----------------------#
#         # 多头注意力机制
#         #----------------------#
#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         #-------------------------------------------------------------------#
#         # 在MLP层中首先是进行一次全连接,之后是过QuickGELU激活函数,最后是通过投影进行映射
#         #-------------------------------------------------------------------#
#         self.mlp  = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2      = LayerNorm(d_model)
#         self.attn_mask = attn_mask
#     #-------------------------------------#
#     # 该函数的作用是对输入的张量使用多头注意力机制
#     #-------------------------------------#
#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
#     #---------------------------------------------------------------#
#     # 在这个前向传播函数中,对于transformer模块进行了定义以及说明
#     #---------------------------------------------------------------#
#     def forward(self, x: torch.Tensor):
#         x = x + self.attention(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
 
#         return x
 
# #-------------------------------------#
# # transformer结构的定义
# # 即为多个transformer模块按照顺序进行堆叠
# #-------------------------------------#
# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
#         super(Transformer,self).__init__()
#         self.width     = width
#         self.layers    = layers
#         self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
 
#     def forward(self, x: torch.Tensor):
#         return self.resblocks(x)

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim))

    def forward(self, src):
        # batch_size, input_size
        out = self.fc(src)
        # batch_size, output_size
        return out



class GraphCLIP(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 transformer_hid=256,
                 transformer_heads=4,
                 transformer_layers=2
                 ):
        super(GraphCLIP, self).__init__()

        self.gin = GINet(
            num_layer=5, 
            hid_dim=256, 
            feat_dim=2*embed_dim, 
            drop_ratio=0, 
            pool='mean')

        self.mlp = MLP(
            in_dim=768, 
            hid_dim=256, 
            out_dim=embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def encode_graph(self, graph):
        return self.gin(graph)

    def encode_text(self, text_embedding):
        #x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = text_embedding
        x = self.mlp(x)

        return x

    def forward(self, graph, text):
        graph_features = self.encode_graph(graph)
        text_features = self.encode_text(text)


        # 此两个tensor的维度shape = [batch_size, embed_dim]
        return graph_features, text_features, self.logit_scale.exp()