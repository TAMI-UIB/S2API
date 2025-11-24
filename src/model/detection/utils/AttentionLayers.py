from torch import nn
from torch.nn.functional import fold

class SAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, patch_size=1, expansion=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        embed_dim = in_channels * patch_size * patch_size

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, y=None):
        B, C, H, W = x.size()

        patches_x = self.unfold(x).transpose(1, 2)

        if y is not None:
            patches_y = self.unfold(y).transpose(1, 2)
            attn_in_x = self.norm1(patches_x)
            attn_in_y = self.norm1(patches_y)
            attn_out, _ = self.attn(query=attn_in_x, key=attn_in_y, value=attn_in_y)
        else:
            attn_in = self.norm1(patches_x)
            attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        patches = patches_x + attn_out

        ffn_in = self.norm2(patches)
        ffn_out = self.ffn(ffn_in)
        patches = patches + ffn_out

        patches = patches.transpose(1, 2)
        x = fold(patches, output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)

        return x