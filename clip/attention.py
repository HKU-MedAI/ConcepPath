from torch import nn
import torch
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.2,
                #  ori_drop_ratio=0.2,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.qkv_bias = qkv_bias
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        # self.ori_drop = nn.Dropout(ori_drop_ratio)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        
        # self.ln = torch.nn.LayerNorm(dim)

    def forward(self, class1_features, class2_features, attn_type = "SA"):
        
        class1_features_ori, class2_features_ori = class1_features, class2_features
        # ["SA", "CA", "SC"]
        self.qkv.weight.data = self.qkv.weight.data.to(class1_features.dtype)
        self.proj.weight.data = self.proj.weight.data.to(class1_features.dtype)
        self.proj.bias.data = self.proj.bias.data.to(class1_features.dtype)
        if self.qkv_bias:
            self.qkv.bias.data = self.qkv.bias.data.to(class1_features.dtype)
            
        class1_features = class1_features.unsqueeze(0)
        class2_features = class2_features.unsqueeze(0)
        
        if attn_type == "SC":
            class_1_len = class1_features.shape[0]
            class_2_len = class2_features.shape[0]
            
            x = torch.cat((class1_features, class2_features), dim=1)
            
            B, N, C = x.shape
        
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            # x = self.proj(x)
            x = self.proj_drop(x)

            class1_features, class2_features = x.squeeze(0)[:class_1_len,:], x.squeeze(0)[-class_2_len:,:]
        
        elif attn_type == "SA":
            
            B, N, C = class1_features.shape
        
            qkv = self.qkv(class1_features).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            class1_features = (attn @ v).transpose(1, 2).reshape(B, N, C)
            # class1_features = self.proj(class1_features)
            class1_features = self.proj_drop(class1_features).squeeze(0)

            B, N, C = class2_features.shape
        
            qkv = self.qkv(class2_features).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            class2_features = (attn @ v).transpose(1, 2).reshape(B, N, C)
            # class2_features = self.proj(class2_features)
            class2_features = self.proj_drop(class2_features).squeeze(0)
        
        elif attn_type == "CA":
            
            B1, N1, C1 = class1_features.shape
            B2, N2, C2 = class2_features.shape
            C = C1
        
            qkv1 = self.qkv(class1_features).reshape(B1, N1, 3, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
            q1, k1, v1= qkv1[0], qkv1[1], qkv1[2]  # make torchscript happy (cannot use tensor as tuple)
            
            qkv2 = self.qkv(class2_features).reshape(B2, N2, 3, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)
            q2, k2, v2= qkv2[0], qkv2[1], qkv2[2]  # make torchscript happy (cannot use tensor as tuple)
            
            attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)

            class1_features = (attn1 @ v2).transpose(1, 2).reshape(B1, N1, C1)
            # class1_features = self.proj(class1_features)
            class1_features = self.proj_drop(class1_features).squeeze(0)
            
            attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)

            class2_features = (attn2 @ v1).transpose(1, 2).reshape(B2, N2, C2)
            # class2_features = self.proj(class2_features)
            class2_features = self.proj_drop(class2_features).squeeze(0)
            
        return class1_features_ori+class1_features, class2_features_ori+class2_features
        
        return class1_features_ori+F.layer_norm(class1_features, (C, )), class2_features_ori+F.layer_norm(class2_features, (C, )) 
        
# class CrossAttention(torch.nn.Module):
#     def __init__(self, d, self_attn=False):
#         super(CrossAttention, self).__init__()
        
#         # 定义线性层为 Q, K, V
#         self.query = nn.Linear(d, d)
#         self.key = nn.Linear(d, d)
#         self.value = nn.Linear(d, d)
        
#         self.d = d

#     def forward(self, class1_features, class2_features, self_attn=False):
        
#         class_1_len = class1_features.shape[0]
#         class_2_len = class2_features.shape[0]
        
#         self.query.weight.data = self.query.weight.data.to(class1_features.dtype)
#         self.query.bias.data = self.query.bias.data.to(class1_features.dtype)
#         self.key.weight.data = self.key.weight.data.to(class1_features.dtype)
#         self.key.bias.data = self.key.bias.data.to(class1_features.dtype)
#         self.value.weight.data = self.value.weight.data.to(class1_features.dtype)
#         self.value.bias.data = self.value.bias.data.to(class1_features.dtype)
        
#         # class1 为 Query, class2 为 Key-Value
#         class_feature = torch.cat((class1_features, class2_features), dim=0)
        

#         q, k, v = self.query(class_feature), self.key(class_feature), self.value(class_feature)
#         # print(q.shape, k.shape, v.shape)
        
#         # 计算注意力分数
#         scores = torch.matmul(q, k.transpose(0, 1)) / (self.d ** 0.5)
#         attention_weights = F.softmax(scores, dim=-1)

#         # 获得输出
#         output = torch.matmul(attention_weights, v)
#         # print(output.shape)
#         return output[:class_1_len,:], output[-class_2_len:,:]
