import torch
import torch.nn as nn
from utils import LoadConfig
from log_tracker import Logger


class PatchEmbedding(nn.Module):
    def __init__(self):
        """
        Initializes the PatchEmbedding module by loading configuration,
        setting image/patch sizes, embedding dimensions, and creating layers
        such as positional embeddings, CLS token, patch embedding linear layer,
        and dropout.
        """
        super().__init__()
        self.config = LoadConfig.load_config("configs/patch_embedding_config.yaml")
        self.logger = Logger.get_logger()

        self.img_height = self.config['img_height']
        self.img_width = self.config['img_width']
        self.img_channel = self.config['img_channel']
        self.patch_height = self.config['patch_height']
        self.patch_width = self.config['patch_width']
        self.embedding_dimension = self.config['embedding_dimension']
        self.drop_rate = self.config['drop_rate']

        self.num_tokens = self.calculate_number_of_tokens()
        self.logger.info(f"Calculated number of patch tokens: {self.num_tokens}")

        self.positional_embedding = self.create_positional_embedding()
        self.logger.info(f"Created positional embedding of shape: {self.positional_embedding.shape}")

        self.cls_token = self.create_cls_token()
        self.logger.info(f"Initialized CLS token of shape: {self.cls_token.shape}")

        self.patch_embedding = self.create_patch_embedding_layer()
        self.logger.info(f"Created patch embedding projection layer: {self.patch_embedding}")

        self.drop_out = nn.Dropout(self.drop_rate)

    def calculate_number_of_tokens(self):
        """
        Calculates the number of patch tokens for the input image.
        Formula: (image_height * image_width) // (patch_height * patch_width)
        Returns:
            int: Number of patch tokens (excluding CLS token).
        """
        # Calculates how many patch tokens the image will be divided into
        # Formula: (H * W) / (patch_H * patch_W)
        # Example: (224 * 224) / (16 * 16) = 196 patches
        total_tokens = (self.img_height * self.img_width) // (self.patch_height * self.patch_width)
        self.logger.info(f"calculate_number_of_tokens: {total_tokens}")
        return total_tokens

    def calculate_patch_dimension(self):
        """
        Calculates the flattened size of a single patch.
        Each patch of shape (C, patch_H, patch_W) is flattened into a vector.
        Returns:
            int: Flattened patch dimension = channels * patch_height * patch_width.
        """
        # Calculates flattened size of one patch (patch_dim)
        # Each patch is (C, patch_H, patch_W) → flatten → C * patch_H * patch_W
        # Example: 3 * 16 * 16 = 768
        patch_dimension = self.img_channel * self.patch_height * self.patch_width
        self.logger.info(f"calculate_patch_dimension: {patch_dimension}")
        return patch_dimension

    def create_positional_embedding(self):
        """
        Creates a learnable positional embedding tensor.
        Shape is (1, num_tokens + 1, embedding_dimension) to account for CLS token.
        Returns:
            nn.Parameter: Positional embedding tensor.
        """
        # Creates learnable positional embeddings of shape:
        # (1, num_tokens + 1, embedding_dim)
        # +1 for the class (CLS) token
        # Example: (1, 197, 768)
        pos = nn.Parameter(torch.zeros(1, self.calculate_number_of_tokens() + 1, self.embedding_dimension))
        self.logger.info("Created positional embedding tensor.")
        return pos

    def create_cls_token(self):
        """
        Creates a learnable class (CLS) token.
        Shape is (1, 1, embedding_dimension) and expanded during forward pass.
        Returns:
            nn.Parameter: CLS token tensor.
        """
        # Creates the learnable class token [CLS]
        # Shape: (1, 1, embedding_dim)
        # Will be expanded to (B, 1, embedding_dim) during training
        cls = nn.Parameter(torch.randn(1, 1, self.embedding_dimension))
        self.logger.info("Initialized learnable CLS token.")
        return cls

    def add_pos_token(self, x):
        """
        Adds positional embeddings to the token sequence.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N+1, embedding_dimension),
                              where CLS token is already prepended.
        Returns:
            torch.Tensor: Tensor with positional embeddings added element-wise.
        """
        # Adds positional embedding to the patch embeddings
        # x: (B, N+1, embedding_dim)
        # Assumes [CLS] token is already prepended
        # Element-wise sum of same shape
        self.logger.info(f"Adding positional embedding to token sequence of shape {x.shape}")
        x = x + self.positional_embedding
        self.logger.info(f"Output shape after adding positional embedding: {x.shape}")
        return x

    def add_cls_token(self, x):
        """
        Prepends the CLS token to the patch embeddings.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, embedding_dimension).
        Returns:
            torch.Tensor: Tensor of shape (B, N+1, embedding_dimension) with CLS token prepended.
        """
        # Prepends the CLS token to the patch embeddings
        # x: (B, N, embedding_dim)

        # Create CLS token and expand to batch size
        # Shape: (1, 1, D) → (B, 1, D)
        self.logger.info(f"Adding CLS token to input of shape {x.shape}")
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        # Concatenate along token (sequence) dimension
        # Output: (B, N+1, D)
        x = torch.cat((cls_tokens, x), dim=1)
        self.logger.info(f"Output shape after adding CLS token: {x.shape}")
        return x

    def create_patches(self, x):
        """
        Splits input images into flattened patches.
        Args:
            x (torch.Tensor): Input images tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Flattened patches of shape (B, num_patches, patch_dim).
        """
        self.logger.info(f"Creating patches from input image of shape: {x.shape}")
        # Assume input x is shape: (B, C, H, W), e.g., (8, 3, 224, 224)

        stride_height = self.patch_height
        stride_width = self.patch_width

        # Step 1: Unfold height dimension (dim=2)
        # Formula: num_patches_H = (H - patch_height) // stride_height + 1
        # This creates sliding windows of size patch_height with stride stride_height
        # Output shape becomes: (B, C, num_patches_H, patch_H, W)

        # Step 2: Unfold width dimension (dim=3)
        # This creates sliding windows of size patch_width with stride stride_width
        # Output shape becomes: (B, C, num_patches_H, num_patches_W, patch_H, patch_W)
        # (B, C, H, W) -> (B, C, num_patches_H, num_patches_W, patch_H, patch_W)

        x = x.unfold(2, self.patch_height, stride_height).unfold(3, self.patch_width, stride_width)
        self.logger.info(f"Shape after unfolding: {x.shape}")

        # Step 3: Rearrange dimensions to group each patch as a single unit
        # From: (B, C, num_patches_H, num_patches_W, patch_H, patch_W)
        # To:   (B, num_patches_H, num_patches_W, C, patch_H, patch_W)
        # This moves channel and patch contents to the end
        x = x.permute(0, 2, 3, 1, 4, 5)
        self.logger.info(f"Shape after permuting for patch grouping: {x.shape}")

        # Step 4: Flatten each patches
        # Each patch is now a (C, patch_H, patch_W) tensor, we flatten it into a vector
        # Reshape to: (B, num_patches_H * num_patches_W, patch_dim)
        # Where patch_dim = C * patch_H * patch_W
        # Flatten patches: (B, num_patches, patch_dim)
        # Example: 14 × 14 = 196 patches, patch_dim = 3 × 16 × 16 = 768 → (B, 196, 768)
        x = x.reshape(x.shape[0], -1, self.calculate_patch_dimension())
        self.logger.info(f"Final patch tensor shape: {x.shape}")

        return x

    def create_patch_embedding_layer(self):
        """
        Creates a linear layer to project flattened patches to embedding dimension.
        Returns:
            nn.Linear: Linear layer for patch embedding.
        """
        # Calculate the flattened dimension of each patch
        # This is: number of channels * patch height * patch width
        # Example: 3 * 16 * 16 = 768
        patch_dim = self.calculate_patch_dimension()

        # Create a linear layer that projects each flattened patch vector
        # from patch_dim (e.g., 768) to the embedding dimension (e.g., 768 or 512)
        # This converts raw patch pixels into a learnable feature embedding

        self.logger.info(f"Creating linear layer for patch embedding from {patch_dim} to {self.embedding_dimension}")
        layer = nn.Linear(patch_dim, self.embedding_dimension)
        self.logger.info(f"Created patch embedding layer: {layer}")
        return layer

    def forward(self, x):
        """
        Forward pass of the PatchEmbedding module.
        Steps:
          - Create flattened patches from images
          - Project patches to embeddings
          - Prepend CLS token
          - Add positional embeddings
          - Apply dropout
        Args:
            x (torch.Tensor): Input image batch (B, C, H, W).
        Returns:
            torch.Tensor: Embedded patch tokens with CLS and positional embeddings.
        """

        x = self.create_patches(x)

        x = self.patch_embedding(x)

        x = self.add_cls_token(x)

        x = self.add_pos_token(x)

        x = self.drop_out(x)

        return x
