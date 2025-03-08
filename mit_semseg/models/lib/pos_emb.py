import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional

def apply_rotatory_emb(x, pos_emb):
    x_out = torch.cat((pos_emb[:,:,:,1::2]*x[:,:,:,1::2] - \
                       pos_emb[:,:,:,::2]*x[:,:,:,::2], 
                       pos_emb[:,:,:,1::2]*x[:,:,:,1::2] + \
                       pos_emb[:,:,:,::2]*x[:,:,:,::2]), dim=-1)
    return x_out

class Image2DPositionalEncoding(nn.Module):
    """
    Learnable 2D positional encoding for images with interpolation support and visualization tools.
    
    Args:
        base_h (int): Base height for positional embeddings
        base_w (int): Base width for positional embeddings
        channels (int): Number of channels in the positional encoding
        dropout (float): Dropout probability (default: 0.1)
        interpolation_mode (str): Mode for interpolation (default: 'bilinear')
    """
    def __init__(
        self, 
        base_h: int, 
        base_w: int, 
        channels: int, 
        interpolation_mode: str = 'bilinear'
    ):
        super().__init__()
        
        if channels % 2 != 0:
            raise ValueError("Channels must be even for paired sine-cosine initialization")
            
        self.base_h = base_h
        self.base_w = base_w
        self.channels = channels
        self.interpolation_mode = interpolation_mode
        
        # Initialize using 2D sine/cosine patterns
        # h_pos = torch.arange(base_h).unsqueeze(1)
        # w_pos = torch.arange(base_w).unsqueeze(1)
        
        # div_term = 1.0 / (10000 ** (torch.arange(0, self.channels//2, 2).float() / (self.channels//2)))

        # Create separate learnable embeddings for height and width
        self.h_embedding = nn.Parameter(torch.empty(base_h, channels//4))
        self.w_embedding = nn.Parameter(torch.empty(base_w, channels//4))

        torch.nn.init.normal_(self.h_embedding.data)
        torch.nn.init.normal_(self.w_embedding.data)
        
    def create_2d_positional_encoding(self, h: int, w: int) -> torch.Tensor:
        """
        Creates 2D positional encoding by combining height and width embeddings,
        with interpolation support for different spatial dimensions.
        
        Args:
            h (int): Target height
            w (int): Target width
            
        Returns:
            torch.Tensor: Combined positional encoding of shape (1, channels, h, w)
        """
        # Get base embeddings
        h_emb = torch.flatten(
            torch.stack(
                (self.h_embedding.sin(),self.h_embedding.cos()),
                 dim=-1
            ),
            -2, -1
        ) # (base_h, channels//2)
        w_emb = torch.flatten(
            torch.stack(
                (self.w_embedding.sin(),self.w_embedding.cos()),
                 dim=-1
            ),
            -2, -1
        ) # (base_w, channels//2)
        
        # Create 2D positional encoding at base resolution
        h_emb = h_emb.unsqueeze(1).expand(-1, self.base_w, -1)
        w_emb = w_emb.unsqueeze(0).expand(self.base_h, -1, -1)
        
        pos_encoding = torch.cat([h_emb, w_emb], dim=-1)
        pos_encoding = pos_encoding.permute(2, 0, 1).unsqueeze(0)
        
        if h != self.base_h or w != self.base_w:
            pos_encoding = F.interpolate(
                pos_encoding,
                size=(h, w),
                mode=self.interpolation_mode,
                align_corners=True if self.interpolation_mode != 'nearest' else None
            )
        
        return torch.flatten(pos_encoding, -2, -1).transpose(2,1).unsqueeze(0)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        pos_encoding = self.create_2d_positional_encoding(h, w)
        return apply_rotatory_emb(x, pos_encoding)

    def visualize_encodings(
        self,
        h: Optional[int] = None,
        w: Optional[int] = None,
        channels_to_show: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (15, 10),
        cmap: str = 'viridis'
    ) -> None:
        """
        Visualizes the positional encodings as heatmaps.
        
        Args:
            h (int, optional): Height for visualization. Defaults to base_h.
            w (int, optional): Width for visualization. Defaults to base_w.
            channels_to_show (List[int], optional): List of channel indices to visualize.
                Defaults to first 6 channels.
            figsize (Tuple[int, int]): Figure size for the plot.
            cmap (str): Colormap for the heatmaps.
        """
        h = h or self.base_h
        w = w or self.base_w
        channels_to_show = channels_to_show or list(range(min(6, self.channels)))
        
        # Get positional encodings
        with torch.no_grad():
            pos_enc = self.create_2d_positional_encoding(h, w)
            pos_enc = pos_enc.squeeze(0)
            pos_enc = pos_enc.squeeze(0)
            pos_enc = pos_enc.transpose(0,1).reshape(self.channels,h,w)
        
        # Create subplot grid
        n_channels = len(channels_to_show)
        n_cols = min(3, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)
        
        # Plot each channel
        for idx, channel in enumerate(channels_to_show):
            row = idx // n_cols
            col = idx % n_cols
            
            channel_data = pos_enc[channel].cpu().numpy()
            sns.heatmap(
                channel_data,
                ax=axes[row, col],
                cmap=cmap,
                cbar=True
            )
            axes[row, col].set_title(f'Channel {channel}')
            axes[row, col].set_xlabel('Width')
            axes[row, col].set_ylabel('Height')
        
        # Remove empty subplots
        for idx in range(n_channels, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.show()

    def visualize_similarity_matrix(
        self,
        h: Optional[int] = None,
        w: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 5)
    ) -> None:
        """
        Visualizes similarity matrices between positions based on their encodings.
        
        Args:
            h (int, optional): Height for visualization. Defaults to base_h.
            w (int, optional): Width for visualization. Defaults to base_w.
            figsize (Tuple[int, int]): Figure size for the plot.
        """
        h = h or self.base_h
        w = w or self.base_w
        
        # Get positional encodings
        with torch.no_grad():
            pos_enc = self.create_2d_positional_encoding(h, w)
            pos_enc = pos_enc.squeeze(0)
            pos_enc = pos_enc.squeeze(0)
            pos_enc = pos_enc.transpose(0,1).reshape(self.channels,h,w)
        
        # Reshape to (h*w, channels)
        pos_enc = pos_enc.permute(1, 2, 0).reshape(-1, self.channels)
        
        # Compute similarity matrix
        similarity = F.cosine_similarity(pos_enc.unsqueeze(1), pos_enc.unsqueeze(0), dim=2)
        
        # Plot similarity matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot full similarity matrix
        sns.heatmap(similarity.cpu().numpy(), ax=ax1, cmap='viridis')
        ax1.set_title('Full Position Similarity Matrix')
        ax1.set_xlabel('Position Index')
        ax1.set_ylabel('Position Index')
        
        # Plot center row similarity
        center_idx = (h//2 - 1) * w + (h//2 - 1)
        center_similarity = similarity[center_idx].reshape(h, w)
        sns.heatmap(center_similarity.cpu().numpy(), ax=ax2, cmap='viridis')
        ax2.set_title('Similarity to Center Position')
        ax2.set_xlabel('Width')
        ax2.set_ylabel('Height')
        
        plt.tight_layout()
        plt.show()

class RelativePositionalEncoding(nn.Module):
    def __init__(
        self, 
        base_h: int, 
        base_w: int, 
        channels: int,
        sr_ratio: int,
    ):
        super().__init__()
        self.base_h = base_h
        self.base_w = base_w
        self.channels = channels
        self.sr_ratio = sr_ratio
        base_h_sr = base_h // sr_ratio
        base_w_sr = base_w // sr_ratio
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(((base_h + base_h_sr) - 1) * ((base_w + base_w_sr) - 1), channels))  # (h+h_sr)-1 * (w+w_sr)-1, channels

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(base_h)
        coords_w = torch.arange(base_w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
        coords_flatten = torch.flatten(coords, 1)  # 2, h*w
        coords_h_sr = torch.arange(base_h_sr)
        coords_w_sr = torch.arange(base_w_sr)
        coords_sr = torch.stack(torch.meshgrid([coords_h_sr, coords_w_sr]))  # 2, h, w
        coords_flatten_sr = torch.flatten(coords_sr, 1)  # 2, h*w
        relative_coords = coords_flatten[:, :, None] - coords_flatten_sr[:, None, :]  # 2, h*w, h_sr*w_sr
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h_sr*w_sr, 2
        relative_coords[:, :, 0] += base_h_sr - 1  # shift to start from 0
        relative_coords[:, :, 1] += base_w_sr - 1
        relative_coords[:, :, 0] *= (base_w + base_w_sr) - 1
        relative_position_index = relative_coords.sum(-1)  # h*w, h_sr*w_sr
        self.register_buffer("relative_position_index", relative_position_index)
    def forward(self, attn):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.base_h * self.base_w, self.base_h // self.sr_ratio * self.base_w // self.sr_ratio, -1)  # h*w,h_sr*w_sr,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, h*w, h_sr*w_sr
        attn = attn + relative_position_bias.unsqueeze(0)
        return attn