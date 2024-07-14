import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, feature_dim, head_dim, scale):
        super(SelfAttention, self).__init__()
        self.scale = scale
        self.query = nn.Linear(feature_dim, head_dim)
        self.key = nn.Linear(feature_dim, head_dim)
        self.value = nn.Linear(feature_dim, head_dim)

    def forward(self, x):
        Q = self.query(x) # batch_size * num_tokens * head_dim
        K = self.key(x) # batch_size * num_tokens * head_dim
        V = self.value(x) # batch_size * num_tokens * head_dim

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale # batch_size * num_tokens * num_tokens
        attn_probs = F.softmax(attn_scores, dim=-1) # batch_size * num_tokens * num_tokens
        attn_output = torch.matmul(attn_probs, V) # batch_size * num_tokens * head_dim

        return attn_output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_features, num_heads=8, dropout_p=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = in_features
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim ** 0.5

        self.attention_heads = nn.ModuleList([
            SelfAttention(in_features, self.head_dim, self.scale) for _ in range(num_heads)
        ])
        self.output = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        attn_outputs = [attn_head(x) for attn_head in self.attention_heads]
        multi_head_output = torch.cat(attn_outputs, dim=-1) # batch_size * num_tokens * feature_dim
        
        output = self.dropout(self.output(multi_head_output)) # batch_size * num_tokens * feature_dim

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, in_features, hidden_dim, num_heads=8, dropout_p=0.):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.attention = MultiHeadSelfAttention(in_features, num_heads, dropout_p)
        self.norm2 = nn.LayerNorm(in_features)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, in_features),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        attn_out = x + self.attention(self.norm1(x)) # batch_size * num_tokens * feature_dim
        ff_out = attn_out + self.feed_forward(self.norm2(attn_out)) # batch_size * num_tokens * feature_dim
        return ff_out


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        image_size=32,
        patch_size=4,
        dropout_p=0.,
        num_layers=7,
        hidden_dim=384,
        mlp_dim=1536,
        num_heads=8
    ):
        super(VisionTransformer, self).__init__()
        
        self.patch_size = patch_size
        patch_dim = patch_size ** 2 * input_channels # The dimension of one patch vector
        self.num_patches = (image_size // patch_size) ** 2
        token_count = self.num_patches + 1

        self.patch_embedding = nn.Linear(patch_dim, hidden_dim) # Embed the patch vectors
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, token_count, hidden_dim))

        self.transformer_encoders = nn.Sequential(
            *[TransformerEncoderLayer(hidden_dim, mlp_dim, num_heads, dropout_p) for _ in range(num_layers)]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self._split_patches(x)
        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.position_embedding
        x = self.transformer_encoders(x)

        x = x[:, 0]

        x = self.classifier(x)
        return x

    def _split_patches(self, x):
        # batch_size, channels, height, width = x.size()
        # patch_size = self.patch_size
        # patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches = patches.contiguous().view(batch_size, channels, -1, patch_size * patch_size).permute(0, 3, 2, 1)
        # patches = patches.reshape(batch_size, -1, patch_size * patch_size * channels)

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        patches = patches.reshape(x.size(0), self.num_patches, -1)
        return patches