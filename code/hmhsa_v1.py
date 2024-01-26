import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import _cfg

dims = [64, 128, 320, 512]
head = 64
kernel_sizes = [5, 3, 5, 3]
expansions = [8, 8, 4, 4]
grid_sizes = [2, 2, 2, 1]
ds_ratios = [8, 4, 2, 1]
depths = [3, 6, 18, 3]
drop_rate = 0
drop_path_rate = 0.1




class Upsample(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, padding = 0, stride = 1): 
        super().__init__()
        self.convT = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride = stride, padding = padding)
        self.bn = nn.GroupNorm(1, out_channels, eps = 1e-6)

    def forward(self, x):
        x = self.convT(x)
        x = self.bn(x)
        
        return x




class Downsample(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0): 
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride = stride, padding = padding)
        self.bn = nn.GroupNorm(1, out_channels, eps = 1e-6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        return x




class HMHSA(nn.Module):
    def __init__(self, dim, head, grid_size, ds_ratio, drop = 0.): 
        super().__init__()
        self.num_heads = dim // head
        self.grid_size = grid_size
        self.head = head
        self.dim = dim
        
        assert (self.num_heads * head == dim), "Dim needs to be divisible by Head."
        
        self.qkv = nn.Conv3d(dim, dim * 3, 1) 
        self.proj = nn.Conv3d(dim, dim, 1) 
        self.norm = nn.GroupNorm(1, dim, eps = 1e-6)
        self.drop = nn.Dropout3d(drop, inplace = True)

        if self.grid_size > 1:
            self.attention_norm = nn.GroupNorm(1, dim, eps = 1e-6)
            self.avg_pool = nn.AvgPool3d(ds_ratio, stride = ds_ratio)
            self.ds_norm = nn.GroupNorm(1, dim, eps = 1e-6)
            self.q = nn.Conv3d(dim, dim, 1)
            self.kv = nn.Conv3d(dim, dim * 2, 1)
        
    def forward(self, x): 
        N, C, H, W, D = x.shape 
        qkv = self.qkv(self.norm(x))

        if self.grid_size > 1:
            grid_h, grid_w, grid_d = H // self.grid_size, W // self.grid_size, D // self.grid_size 
            qkv = qkv.reshape(N, 3, self.num_heads, self.head, grid_h, self.grid_size, grid_w, self.grid_size, grid_d, self.grid_size)
            qkv = qkv.permute(1, 0, 2, 4, 6, 8, 5, 7, 9, 3)
            qkv = qkv.reshape(3, -1, self.grid_size * self.grid_size * self.grid_size, self.head)
            query, key, value = qkv[0], qkv[1], qkv[2]
        
            attention = (query / (self.dim ** (1/2))) @ key.transpose(-2, -1)
            attention = attention.softmax(dim = -1)

            attention_x = (attention @ value).reshape(N, self.num_heads, grid_h, grid_w, grid_d, self.grid_size, self.grid_size, self.grid_size, self.head)
            attention_x = attention_x.permute(0, 1, 8, 2, 5, 3, 6, 4, 7).reshape(N, C, H, W, D)
            attention_x = self.attention_norm(x + attention_x)

            kv = self.kv(self.ds_norm(self.avg_pool(attention_x)))

            query = self.q(attention_x).reshape(N, self.num_heads, self.head, -1)
            query = query.transpose(-2, -1) 
            kv = kv.reshape(N, 2, self.num_heads, self.head, -1)
            kv = kv.permute(1, 0, 2, 4, 3) 
            key, value = kv[0], kv[1]

        else:
            qkv = qkv.reshape(N, 3, self.num_heads, self.head, -1)
            qkv = qkv.permute(1, 0, 2, 4, 3) 
            query, key, value = qkv[0], qkv[1], qkv[2]  

        attention = (query / (self.dim ** (1/2))) @ key.transpose(-2, -1)
        attention = attention.softmax(dim = -1)

        global_attention_x = (attention @ value).transpose(-2, -1).reshape(N, C, H, W, D) 

        if self.grid_size > 1:
            global_attention_x = global_attention_x + attention_x

        x = self.drop(self.proj(global_attention_x))

        return x




class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, kernel_size, drop = 0., act_layer=nn.SiLU):
        expanded_channels = in_channels * expansion
        out_channels = out_channels
        padding = (kernel_size - 1) // 2
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(1, in_channels, eps = 1e-6),
            nn.Conv3d(in_channels, expanded_channels, kernel_size = 1, padding = 0, bias = False),
            act_layer(inplace = True)
        )              
        self.conv2 = nn.Sequential(
            nn.Conv3d(expanded_channels, expanded_channels, kernel_size = kernel_size, padding = padding, groups = expanded_channels, bias = False),
            act_layer(inplace = True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(expanded_channels, out_channels, kernel_size = 1, padding = 0, bias = False),
            nn.GroupNorm(1, out_channels, eps = 1e-6)
        )
        self.drop = nn.Dropout2d(drop, inplace = True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)

        return x



class Block(nn.Module):
    def __init__(self, dim, head, grid_size = 1, ds_ratio = 1, expansion = 4, drop = 0., drop_path = 0., kernel_size = 3, act_layer = nn.SiLU):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = HMHSA(dim, head, grid_size = grid_size, ds_ratio = ds_ratio, drop = drop)
        self.conv = MBConv(in_channels = dim, out_channels = dim, expansion = expansion, kernel_size = kernel_size, drop = drop, act_layer = act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.conv(x))
        return x




class HAT_Net(nn.Module):
    def __init__(self, img_size = 224, in_chans = 3, num_classes = 1000, act_layer = nn.SiLU):
        super(HAT_Net, self).__init__()
        self.depths = depths

        # sequential CNNs
        self.CNN = nn.Sequential(
            nn.ConvTranspose3d(in_channels = 3, out_channels = 3, kernel_size = 4, padding = 2, stride = 2),
            nn.GroupNorm(3, 3, eps = 1e-6),
            act_layer(inplace = True),
            nn.ConvTranspose3d(in_channels = 3, out_channels = 3, kernel_size = 4, padding = 2, stride = 2),
            nn.GroupNorm(3, 3, eps = 1e-6),
            act_layer(inplace = True),
            nn.Conv3d(in_channels = 3, out_channels = 16, kernel_size = 4, stride = 2, padding = 2),
            nn.GroupNorm(1, 16, eps = 1e-6),
            act_layer(inplace = True),
            nn.Conv3d(in_channels = 16, out_channels = 64, kernel_size = 2, stride = 2, padding = 0),
            )

        # block -> H-MSHA + MLP
        self.blocks = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for stage in range(len(dims)):
            self.blocks.append(nn.ModuleList([Block(
                dim = dims[stage], head = head, kernel_size = kernel_sizes[stage], expansion = expansions[stage],
                grid_size = grid_sizes[stage], ds_ratio = ds_ratios[stage], drop = drop_rate, drop_path = dpr[sum(depths[:stage]) + i])
                for i in range(depths[stage])])) 
        self.blocks = nn.ModuleList(self.blocks)

        # downsamples
        self.ds1 = Downsample(in_channels = dims[0], out_channels = dims[1], kernel_size = 5)
        self.ds2 = Downsample(in_channels = dims[1], out_channels = dims[2], kernel_size = 3)
        self.ds3 = Downsample(in_channels = dims[2], out_channels = dims[3], kernel_size = 2, stride = 2)

        # upsamples
        self.us1 = Upsample(in_channels = dims[3], out_channels = dims[2], kernel_size = 2, stride = 2)
        self.us2 = Upsample(in_channels = dims[2], out_channels = dims[1], kernel_size = 3)
        self.us3 = Upsample(in_channels = dims[1], out_channels = dims[0], kernel_size = 5)

        self.TCNN = nn.Sequential(
            nn.ConvTranspose3d(in_channels = dims[0], out_channels = 16, kernel_size = 2, stride = 2, padding = 0),
            nn.GroupNorm(1, 16, eps = 1e-6),
            act_layer(inplace = True),
            nn.ConvTranspose3d(in_channels = 16, out_channels = 1, kernel_size = 4, stride = 2, padding = 2),
            nn.GroupNorm(1, 1, eps = 1e-6),
            act_layer(inplace = True),
            nn.Conv3d(in_channels = 1, out_channels = 1, kernel_size = 4, stride = 2, padding = 2),
            nn.GroupNorm(1, 1, eps = 1e-6),
            act_layer(inplace = True),
            nn.Conv3d(in_channels = 1, out_channels = 1, kernel_size = 4, stride = 2, padding = 2),
        )
                
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std = .02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        print(x.shape)
        x = self.CNN(x)
        print(x.shape)
        for block in self.blocks[0]:
            x = block(x)
        x = self.ds1(x)
        for block in self.blocks[1]:
            x = block(x)
        x = self.ds2(x)
        for block in self.blocks[2]:
            x = block(x)
        x = self.ds3(x)
        for block in self.blocks[3]:
            x = block(x)
        x =self.us1(x)
        x =self.us2(x)
        x =self.us3(x)
        x =self.TCNN(x)

        return x
