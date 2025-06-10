import torch
import torch.nn as nn
from utils import LoadConfig
from model import Attention
from log_tracker import Logger


class VitBlock(nn.Module):
    """
    Vision Transformer (ViT) Block consisting of:
    - Multi-head self-attention with pre-normalization and residual connection
    - Feed-forward MLP block with pre-normalization and residual connection
    - Dropout for regularization

    Configurations such as embedding dimension, MLP hidden size, and dropout rate
    are loaded from a YAML configuration file.

    Attributes:
        attention_block (Attention): Multi-head self-attention module.
        pre_attention_norm (nn.LayerNorm): LayerNorm applied before attention.
        pre_mlp_norm (nn.LayerNorm): LayerNorm applied before MLP.
        mlp_block (nn.Sequential): Feed-forward MLP block with two linear layers and GELU activation.
    """
    def __init__(self):
        """
        Initialize the ViT block.
        Loads configurations and sets up attention, normalization, and MLP layers.
        """
        super().__init__()
        self.config = LoadConfig.load_config("configs/vit_block_config.yaml")
        self.logger = Logger.get_logger()

        self.mlp_size = self.config['mlp_size']
        self.embedding_dimension = self.config['embedding_dimension']
        self.drop_rate = self.config['drop_rate']

        # Initialize Attention block
        self.attention_block = Attention()

        # Separate LayerNorms before attention and MLP per ViT design
        self.pre_attention_norm = nn.LayerNorm(self.embedding_dimension)
        self.pre_mlp_norm = nn.LayerNorm(self.embedding_dimension)

        # MLP block with two linear layers, GELU activation, and dropout
        self.mlp_block = self.create_mlp_block()

        self.logger.info(f"Initialized VitBlock with embedding_dimension={self.embedding_dimension}, "
                         f"mlp_size={self.mlp_size}, drop_rate={self.drop_rate}")


    def create_mlp_block(self):
        """
        Create the MLP feed-forward block used after attention in the ViT block.

        Structure:
            Linear (embedding_dimension -> mlp_size)
            GELU activation
            Dropout
            Linear (mlp_size -> embedding_dimension)
            Dropout

        Returns:
            nn.Sequential: The MLP block as a sequential module.
        """
        return nn.Sequential(
            nn.Linear(self.embedding_dimension, self.mlp_size),
            nn.GELU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.mlp_size, self.embedding_dimension),
            nn.Dropout(self.drop_rate),
        )

    def forward(self, x):
        """
        Forward pass for the ViT block applying attention and MLP with residual connections.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dimension)

        Returns:
            torch.Tensor: Output tensor of the same shape after attention and MLP layers.
        """
        self.logger.info(f"Input to VitBlock forward: {x.shape}")

        # Step 1: Normalize input before attention (Pre-Norm)
        x_norm_input = self.pre_attention_norm(x)
        self.logger.info(f"After pre-attention LayerNorm: {x_norm_input.shape}")

        # Step 2: Apply multi-head self-attention to normalized input
        attn_output = self.attention_block(x_norm_input)
        self.logger.info(f"Output of attention block: {attn_output.shape}")

        # Step 3: Add residual connection (input + attention output)
        x = x + attn_output
        self.logger.info(f"After adding attention residual: {x.shape}")

        # Step 4: Normalize input before MLP (Pre-Norm)
        x_norm_mlp_input = self.pre_mlp_norm(x)
        self.logger.info(f"After pre-MLP LayerNorm: {x_norm_mlp_input.shape}")

        # Step 5: Apply MLP block
        mlp_output = self.mlp_block(x_norm_mlp_input)
        self.logger.info(f"Output of MLP block: {mlp_output.shape}")

        # Step 6: Add residual connection (input + MLP output)
        x = x + mlp_output
        self.logger.info(f"After adding MLP residual: {x.shape}")

        return x
