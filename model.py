import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from torchvision import transforms

# Defines a simple gating mechanism that multiplies corresponding elements of two halves of a tensor
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# Implements a depthwise separable convolution block, reducing computational cost for convolution operations
class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

# Custom implementation of Layer Normalization for 2D feature maps
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        # Normalization logic
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # Backpropagation logic
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

# Wrapper for LayerNormFunction to use it as a standard nn.Module
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# Defines a Channel Attention Layer, focusing the model on relevant channels
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel//16, 1, padding=0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel//16, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)   
class CCSB(nn.Module):
    def __init__(self, channels, reduction = 4, bias = True, bn = False):
        super(CCSB, self).__init__()
        self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding = 1, bias = bias),
                nn.ReLU(inplace = True),
                nn.Conv2d(channels, channels, 3, padding = 1, bias = bias),
        )
        self.ca = CALayer(channels)
        self.sa = SpatialAttention()
        #add 0809
    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        res = self.sa(res)
        out = x + res
        return out
class CDCA(nn.Module):
    def __init__(self, channels, reduction = 4, bias = True, bn = False):
        super(CDCA, self).__init__()
        self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding = 1, bias = bias),
                nn.ReLU(inplace = True),
                nn.Conv2d(channels, channels, 3, padding = 1, bias = bias),
        )
        self.ca = CALayer(channels)
        self.sa = StripPooling(channels, 50)
        #add 0809
    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        res = self.sa(res)
        out = x + res
        return out
# Defines Strip Pooling, which pools features across the spatial dimensions in stripes, allowing for efficient global context aggregation
class StripPooling(nn.Module):
    def __init__(self, channels, pool_size, norm_layer=nn.BatchNorm2d):
        super(StripPooling, self).__init__()
        # Initialization and definition of pooling and convolution layers for strip pooling
        self.pool_size = pool_size
        self.channels = channels
        self.horizontal_pool = nn.AdaptiveAvgPool2d((pool_size, 1))
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, pool_size))
        self.horizontal_conv = nn.Conv2d(channels, channels, 1)
        self.vertical_conv = nn.Conv2d(channels, channels, 1)
        self.horizontal_bn = norm_layer(channels)
        self.vertical_bn = norm_layer(channels)
        self.fusion_conv = nn.Conv2d(2 * channels, channels, 1)
        self.fusion_bn = norm_layer(channels)

    def forward(self, x):
        # Forward pass through the strip pooling layers, with normalization and fusion of pooled features
        _, _, h, w = x.size()
        hp = self.horizontal_pool(x)
        vp = self.vertical_pool(x)
        hp = F.interpolate(hp, size=(h, w), mode='nearest')
        vp = F.interpolate(vp, size=(h, w), mode='nearest')
        hp = self.horizontal_conv(hp)
        vp = self.vertical_conv(vp)
        hp = self.horizontal_bn(hp)
        vp = self.vertical_bn(vp)
        fusion = self.fusion_conv(torch.cat([hp, vp], dim=1))
        fusion = self.fusion_bn(fusion)
        output = torch.sigmoid(fusion) * x
        return output

    def forward(self, x):
        _, _, h, w = x.size()

        # Apply strip pooling
        hp = self.horizontal_pool(x)
        vp = self.vertical_pool(x)

        # Expand the pooled features back to the original size
        hp = F.interpolate(hp, size=(h, w), mode='nearest')
        vp = F.interpolate(vp, size=(h, w), mode='nearest')

        # 1x1 convolutions
        hp = self.horizontal_conv(hp)
        vp = self.vertical_conv(vp)

        # Batch normalization
        hp = self.horizontal_bn(hp)
        vp = self.vertical_bn(vp)

        # Fusion of features
        fusion = self.fusion_conv(torch.cat([hp, vp], dim=1))
        fusion = self.fusion_bn(fusion)

        # Final fused feature
        output = torch.sigmoid(fusion) * x  # Use torch.sigmoid instead of F.sigmoid
        return output
class one_conv(nn.Module):
    """Applies a convolution followed by a LeakyReLU activation and concatenates the input with the output.
    
    Parameters:
    - G0 (int): Number of input channels.
    - G (int): Number of output channels.
    """
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)  # Concatenate along the channel dimension



class RDB(nn.Module):
    """Implements a Residual Dense Block (RDB) for effective feature learning.
    
    Parameters:
    - G0 (int): Number of input channels to the block.
    - C (int): The growth rate or how many filters to add per one_conv layer.
    - G (int): Number of output channels from each one_conv layer.
    """
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = [one_conv(G0 + i * G, G) for i in range(C)]
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0 + C * G, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x  # Residual connection



class RDG(nn.Module):
    """Groups multiple RDB blocks for deeper feature extraction.
    
    Parameters:
    - G0 (int): Number of input channels.
    - C (int): Growth rate in RDBs.
    - G (int): Number of output channels in each one_conv within RDBs.
    - n_RDB (int): Number of RDBs in the group.
    """
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        del temp
        return buffer_cat

class SAGM(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = depthwise_separable_conv(in_channels=c, out_channels=dw_channel, kernel_size=3, padding=1, stride=1)
        self.conv2= depthwise_separable_conv(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca= CDCA(dw_channel // 2)
        
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4= depthwise_separable_conv(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1)
        self.conv5= depthwise_separable_conv(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1)       
        self.norm1= nn.InstanceNorm2d(c)
        self.norm2= nn.InstanceNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class Net(nn.Module):
    """
    A neural network designed for image processing tasks that require upscaling and feature enhancement.
    
    Parameters:
    - upscale_factor (int): The factor by which images will be upscaled.
    """
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        # Define the initial feature extraction module
        self.init_feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=True),  # Initial convolution
            nn.LeakyReLU(0.1, inplace=True),  # Activation function
            SAGM(64),  # Spatial Attention Gate Module (assumed to be defined elsewhere)
        )
        # Setup Residual Dense Groups (RDGs) for both initial and intermediate processing
        self.deep_features = nn.ModuleList([RDG(64, 4, 24, 4)])  # Initial RDGs
        self.deep_features.extend([RDG(64, 4, 24, 4) for _ in range(1)])  # Intermediate RDGs
        
        # Convolutional Vision-Attention Blocks (CVIABs) for enhanced feature attention
        self.CVIABs = nn.ModuleList([CVIAB(64, 4)])  # Assuming each CVIAB uses features from RDGs
        self.CVIABs.extend([CVIAB(64, 4) for _ in range(1)])  # Additional CVIABs
        
        # Convolution layers to combine features from RDGs before passing to CVIABs
        self.Convs = nn.ModuleList([nn.Conv2d(64*4, 64, 1)])
        self.Convs.extend([nn.Conv2d(64*4, 64, 1) for _ in range(1)])
        
        # Fusion modules to integrate features from CVIABs
        self.Fusions = nn.ModuleList([Fusion() for _ in range(1 + 1)])  # Final fusion for all blocks
        
        # Final feature fusion and upscaling
        self.fusion_final = nn.Sequential(
            nn.Conv2d(64*(1+1), 64, 1),  # Combine all features
            nn.Conv2d(64, 64, 3, 1, 1))  # Refine combined features
        self.upscale = nn.Sequential(  # Upscale to target resolution
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1))

    def forward(self, x_left, x_right, is_training):
        # Preprocess and upscale input images
        x_left_upscale = F.interpolate(x_left, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x_right_upscale = F.interpolate(x_right, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        
        # Initial feature extraction
        buffer_left, buffer_right = self.init_feature(x_left), self.init_feature(x_right)
        
        # Process features through RDGs and CVIABs
        for i in range(len(self.CVIABs)):
            catfea_left, catfea_right = self.deep_features[i](buffer_left), self.deep_features[i](buffer_right)
            buffer_left, buffer_right = self.Convs[i](catfea_left), self.Convs[i](catfea_right)
            if is_training:
                buffer_leftT, buffer_rightT, (M_right_to_left, M_left_to_right), (V_left, V_right) = self.CVIABs[i](buffer_left, buffer_right, catfea_left, catfea_right, is_training)
            else:
                buffer_leftT, buffer_rightT = self.CVIABs[i](buffer_left, buffer_right, catfea_left, catfea_right, is_training)
            
            # Fuse features after processing through CVIABs
            buffer_left, buffer_right = self.Fusions[i](torch.cat([buffer_left, buffer_leftT], dim=1)), self.Fusions[i](torch.cat([buffer_right, buffer_rightT], dim=1))
        
        # Final fusion and upscaling
        buffer_leftF, buffer_rightF = self.fusion_final(torch.cat([buffer_left, buffer_leftT], dim=1)), self.fusion_final(torch.cat([buffer_right, buffer_rightT], dim=1))
        out_left, out_right = self.upscale(buffer_leftF) + x_left_upscale, self.upscale(buffer_rightF) + x_right_upscale
        
        if is_training:
            return out_left, out_right, (M_right_to_left, M_left_to_right), (V_left, V_right)
        return out_left, out_right

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True),
            RDB(G0=64, C=4, G=32),
            CALayer(64))

    def forward(self, x):
        x = self.fusion(x)
        return x

class SARM(nn.Module):
    def __init__(self, channels, dilation=1):
        super(SARM, self).__init__()
        self.body = nn.Sequential(
            # Depthwise separable convolution
            nn.Conv2d(channels, channels, 3, 1, dilation, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(channels),  # Add batch normalization for stability
            nn.LeakyReLU(0.1, inplace=True),
            
            # Another depthwise separable convolution
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(channels),  # Add batch normalization for stability
            nn.LeakyReLU(0.1, inplace=True),
            
            # Optional: Dilated convolution for larger receptive field
            # Adjust the dilation rate to control the receptive field size
            nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation, bias=True),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        out = self.body(x)
        return out + x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class grouped_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, groups=4):
        super(grouped_conv, self).__init__()
        self.grouped = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

    def forward(self, x):
        return self.grouped(x)
    
class CVIAB(nn.Module):
    """
    Convolutional Vision-Attention Block that applies channel and spatial attentions
    to enhance features between stereo image pairs.

    Parameters:
    - channels (int): Number of input channels.
    - n_RDBs (int): Number of Residual Dense Blocks to consider for attention calculation.
    """
    def __init__(self, channels, n_RDBs):
        super(CVIAB, self).__init__()
        # Grouped convolution operations for generating attention queries and keys
        self.bq = grouped_conv(n_RDBs * channels, channels, 3, 1, 1)
        self.bs = grouped_conv(n_RDBs * channels, channels, 3, 1, 1)
        
        # Softmax for attention normalization
        self.softmax = nn.Softmax(dim=-1)
        
        # Spatial Attention and Residual Block
        self.rb = SARM(n_RDBs * channels)
        
        # Layer normalization scaled for 4x channels
        self.bn = LayerNorm2d(4 * channels)
        
        # Channel attention module
        self.channel_attention = ChannelAttention(channels)
        
        # Depthwise separable convolution for efficient spatial processing
        self.depthwise_separable_conv = depthwise_separable_conv(channels, channels, 3, 1, 1)
        
        # Custom Convolutional Spatial Block for further enhancement
        self.CCSB = CCSB(channels)

    def __call__(self, x_left, x_right, catfea_left, catfea_right, is_training):
        # Apply channel attention to both left and right features
        x_left = self.channel_attention(x_left)
        x_right = self.channel_attention(x_right)
        b, c0, h0, w0 = x_left.shape
        # Generate attention queries and keys using grouped convolutions
        Q = self.bq(self.rb(self.bn(catfea_left)))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(catfea_right)))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)
        # Compute similarity scores and normalize with softmax
        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))                    # (B*H) * C * Wr
        M_right_to_left = self.softmax(score)
                                                           # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(score.permute(0, 2, 1))                                  # (B*H) * Wr * Wl

        M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        # Generate visibility masks
        V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
                           M_left_to_right.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                           ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
                            M_right_to_left.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                                  ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        V_left_tanh = torch.tanh(5 * V_left)
        V_right_tanh = torch.tanh(5 * V_right)

        # Transform features based on attention and enhance them
        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2) 
        x_leftT=self.depthwise_separable_conv(x_leftT)
        x_rightT=self.depthwise_separable_conv(x_rightT)                             #  B, C0, H0, W0
        # Apply visibility masks and enhance features
        out_left = x_left * (1 - V_left_tanh.repeat(1, c0, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c0, 1, 1)
        out_left=self.CCSB(out_left)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c0, 1, 1)) +  x_rightT * V_right_tanh.repeat(1, c0, 1, 1)
        out_right=self.CCSB(out_right)

        if is_training == 1:
            return out_left, out_right, \
                   (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)),\
                   (V_left_tanh, V_right_tanh)
        if is_training == 0:
            return out_left, out_right


def M_Relax(M, num_pixels):
    """
    Applies relaxation to an attention matrix by spatially dilating its focus.

    Parameters:
    - M (Tensor): The attention matrix of shape (batch_size, height, width).
    - num_pixels (int): The number of pixels to dilate the attention focus.

    Returns:
    - Tensor: The relaxed attention matrix.
    """
    _, u, v = M.shape  # Extract the dimensions of the attention matrix
    M_list = [M.unsqueeze(1)]  # Initialize the list with the original matrix, adding a dimension for concatenation
    
    # For each pixel in the specified range, apply padding to shift the focus
    # and accumulate these shifted matrices to dilate the attention focus.
    for i in range(num_pixels):
        # Padding and shifting towards the bottom
        pad_bottom = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
        pad_M_bottom = pad_bottom(M[:, :-1-i, :])  # Avoid padding the last elements
        M_list.append(pad_M_bottom.unsqueeze(1))

        # Padding and shifting towards the right
        pad_right = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
        pad_M_right = pad_right(M[:, i+1:, :])  # Avoid padding the first elements
        M_list.append(pad_M_right.unsqueeze(1))
    
    # Concatenate and sum up all the shifted matrices to get a single relaxed matrix
    M_relaxed = torch.sum(torch.cat(M_list, dim=1), dim=1)
    return M_relaxed



if __name__ == "__main__":
    net = Net(upscale_factor=4)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
