import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LoadConfig
from log_tracker import Logger


class Attention(nn.Module):
    def __init__(self):
        """
        Initialize the multi-head self-attention module.
        Loads configuration parameters such as number of heads, embedding dimension, and dropout rate.
        Creates projection layers for query/key/value and output, and calculates per-head dimension.
        """
        super().__init__()
        self.config = LoadConfig.load_config("configs/attention_config.yaml")
        self.logger = Logger.get_logger()

        self.n_heads = self.config['n_heads']
        self.embedding_dimension = self.config['embedding_dimension']
        self.drop_rate = self.config['drop_rate']

        self.head_dim = self.calculate_head_dimension()
        self.projection_layer = self.create_projection_layer()
        self.output_layer = self.create_output_layer()

        self.logger.info(f"Initialized Attention module with config: "
                         f"n_heads={self.n_heads}, embedding_dim={self.embedding_dimension}, "
                         f"dropout={self.drop_rate}, head_dim={self.head_dim}")

    def calculate_head_dimension(self):
        """
        Calculates the dimension of each attention head.
        Returns:
            int: head_dim = embedding_dimension // n_heads
        """
        head_dim = self.embedding_dimension // self.n_heads
        self.logger.info(f"Calculated head_dim: {head_dim}")
        return head_dim

    def create_projection_layer(self):
        """
        Creates a linear layer that projects the input embeddings
        into concatenated queries, keys, and values for all heads.
        Returns:
            nn.Linear: Linear layer with output dimension 3 * embedding_dimension.
        """
        layer = nn.Linear(self.embedding_dimension, 3 * self.embedding_dimension, bias=False)
        self.logger.info(f"Created projection layer: {layer}")
        return layer

    def create_output_layer(self):
        """
        Creates the output projection layer followed by dropout.
        This layer projects the concatenated attention outputs back to the embedding dimension.
        Returns:
            nn.Sequential: Output layer with linear and dropout.
        """
        layer = nn.Sequential(
            nn.Linear(self.embedding_dimension, self.embedding_dimension),
            nn.Dropout(self.drop_rate)
        )
        self.logger.info(f"Created output layer: {layer}")
        return layer

    def calculate_attention(self, x):
        """
        Compute multi-head self-attention for input tensor x.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            attention_output: Output tensor of shape (batch_size, seq_len, embedding_dim)
                              after applying multi-head self-attention.
        """

        # Step 1: Project input x to combined query, key, and value vectors
        # qkv shape: (batch_size, seq_len, 3 * embedding_dim)
        self.logger.info(f"Input x shape: {x.shape}")
        qkv = self.projection_layer(x)

        # Step 2: Split qkv into q, k, v each with shape (batch_size, seq_len, embedding_dim)
        # This corresponds to:
        # q = Q matrix, k = K matrix, v = V matrix for self-attention
        q, k, v = qkv.chunk(3, dim=-1)
        self.logger.info(f"Split into q, k, v with shapes: q={q.shape}, k={k.shape}, v={v.shape}")

        # Step 3: Reshape q, k, v to separate attention heads:
        # Each now shape (batch_size, seq_len, n_heads, head_dim)
        # where embedding_dim = n_heads * head_dim
        q = q.view(q.shape[0], q.shape[1], self.n_heads, self.head_dim)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, self.head_dim)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.head_dim)
        self.logger.info(f"Reshaped q, k, v to multi-head shapes: q={q.shape}, k={k.shape}, v={v.shape}")

        # Step 4: Transpose to bring n_heads before seq_len for batched matmul:
        # New shape: (batch_size, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        self.logger.info(f"Transposed q, k, v for attention: q={q.shape}, k={k.shape}, v={v.shape}")

        # Step 5: Calculate scaled dot-product attention scores:
        # Formula: Attention(Q, K) = Softmax((Q @ K^T) / sqrt(head_dim))
        # q shape: (batch_size, n_heads, seq_len, head_dim)
        # k shape: (batch_size, n_heads, seq_len, head_dim)
        # k.transpose(-2, -1) shape: (batch_size, n_heads, head_dim, seq_len)
        # Resulting attention_scores shape: (batch_size, n_heads, seq_len, seq_len)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        self.logger.info(f"Computed attention scores shape: {attention_scores.shape}")

        # Step 6: Apply softmax along last dimension to get attention probabilities:
        # Shape remains (batch_size, n_heads, seq_len, seq_len)
        attention_probs = F.softmax(attention_scores, dim=-1)
        self.logger.info(f"Applied softmax, attention_probs shape: {attention_probs.shape}")

        # Step 7: Apply dropout on attention probabilities for regularization during training
        attention_probs = F.dropout(attention_probs, p=self.drop_rate, training=self.training)
        self.logger.info(f"Applied dropout to attention_probs")

        # Step 8: Weight values (v) by attention probabilities:
        # attention_probs shape: (batch_size, n_heads, seq_len, seq_len)
        # v shape: (batch_size, n_heads, seq_len, head_dim)
        # Resulting attention_output shape: (batch_size, n_heads, seq_len, head_dim)
        attention_output = torch.matmul(attention_probs, v)
        self.logger.info(f"Attention output after weighting: {attention_output.shape}")

        # Step 9: Transpose attention_output to shape (batch_size, seq_len, n_heads, head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous()
        self.logger.info(f"Transposed attention output back: {attention_output.shape}")

        # Step 10: Concatenate all heads by reshaping to (batch_size, seq_len, embedding_dim)
        attention_output = attention_output.view(
            qkv.shape[0],  # batch_size
            qkv.shape[1],  # seq_len
            self.n_heads * self.head_dim  # embedding_dim
        )
        self.logger.info(f"Final attention_output shape: {attention_output.shape}")

        # Return final multi-head self-attention output
        return attention_output

    def forward(self, x):
        """
        Forward pass through the attention layer.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len, embedding_dim)
        """
        self.logger.info(f"Forward pass started with input shape: {x.shape}")
        x = self.calculate_attention(x)
        self.logger.info(f"Shape after attention: {x.shape}")
        x = self.output_layer(x)
        self.logger.info(f"Output shape after output layer: {x.shape}")
        return x
