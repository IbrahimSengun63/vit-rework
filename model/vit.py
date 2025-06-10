import torch
import torch.nn as nn
from utils import LoadConfig
from model import Attention
from log_tracker import Logger
from model import PatchEmbedding
from model import VitBlock


class VIT(nn.Module):
    """
    Vision Transformer (ViT) model.

    This model consists of:
    - A Patch Embedding module to convert input images into patch tokens with positional encoding and a classification (CLS) token.
    - A stack of Transformer encoder layers (VitBlocks), each applying multi-head self-attention and MLP blocks with residual connections.
    - A classification head that normalizes the CLS token output and maps it to class logits.

    Configurations like number of layers, embedding dimension, dropout rate, and number of classes
    are loaded from an external YAML configuration file.

    Attributes:
        patch_embedding (PatchEmbedding): Converts images into patch embeddings.
        transformer_layers (nn.ModuleList): List of VitBlock transformer encoder layers.
        classifier (nn.Sequential): Final classification head (LayerNorm + Linear).
    """
    def __init__(self):
        """
        Initialize the Vision Transformer model.

        Loads configurations and sets up the patch embedding, transformer blocks, and classification head.
        """
        super().__init__()
        # Load model configuration parameters from yaml file
        self.config = LoadConfig.load_config("configs/vit_config.yaml")

        # Initialize logger instance
        self.logger = Logger.get_logger()

        # Extract hyperparameters from config
        self.n_layers = self.config['n_layers']  # Number of transformer blocks
        self.embedding_dimension = self.config['embedding_dimension']  # Embedding size
        self.drop_rate = self.config['drop_rate']  # Dropout rate (not used here directly)
        self.num_classes = self.config['num_classes']  # Number of classification output classes

        # Initialize PatchEmbedding module to convert images to patch tokens
        self.patch_embedding = PatchEmbedding()
        self.logger.info(f"PatchEmbedding module initialized with embedding dimension {self.embedding_dimension}")

        # Create transformer encoder layers (stack of VitBlocks)
        self.transformer_layers = self.create_transformer_layers()
        self.logger.info(f"Created {self.n_layers} transformer layers (VitBlock)")

        # Create classification head (LayerNorm + Linear)
        self.classifier = self.create_dense_layer()
        self.logger.info(f"Classification head created for {self.num_classes} output classes")

    def create_transformer_layers(self):
        """
        Create the transformer encoder layers by stacking VitBlocks.

        Returns:
            nn.ModuleList: List containing n_layers of VitBlock modules.
        """
        transformer_layers = nn.ModuleList([
            VitBlock() for _ in range(self.n_layers)
        ])
        return transformer_layers

    def create_dense_layer(self):
        """
        Create the final classification head for the CLS token.

        Structure:
            - LayerNorm to normalize CLS token embedding
            - Linear layer to produce class logits

        Returns:
            nn.Sequential: Classification head module.
        """
        dense_layer = nn.Sequential(
            nn.LayerNorm(self.embedding_dimension),
            nn.Linear(self.embedding_dimension, self.num_classes)
        )
        return dense_layer

    def forward(self, x):
        """
        Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Class logits with shape (batch_size, num_classes)
        """
        self.logger.info(f"Starting forward pass with input tensor shape: {x.shape}")

        # Step 1: Convert images to patch embeddings with positional encoding & CLS token
        x = self.patch_embedding(x)
        self.logger.info(f"Patch embeddings created with shape: {x.shape} (B, num_patches+1, embedding_dim)")

        # Step 2: Pass patch embeddings through each transformer block
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            self.logger.info(f"Output shape after transformer layer {i}: {x.shape}")

        # Step 3: Extract the CLS token representation (first token)
        cls_token = x[:, 0]
        self.logger.info(f"CLS token extracted with shape: {cls_token.shape}")

        # Step 4: Pass CLS token through classifier head to get logits
        logits = self.classifier(cls_token)
        self.logger.info(f"Logits generated with shape: {logits.shape}")

        self.logger.info("Forward pass completed successfully.")

        return logits
