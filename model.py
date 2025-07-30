import torch
import torch.nn as nn

class VideoCaptioningModel(nn.Module):
    def __init__(self, feature_dim=1280, hidden_dim=512, vocab_size=7052, num_layers=2, dropout=0.3, pad_idx=0, max_len=50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        # 🔸 Project input features to GRU hidden size
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # 🔸 Word Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)

        # 🔸 GRU Decoder (unidirectional)
        self.gru = nn.GRU(input_size=hidden_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout)

        # 🔸 Output layer: hidden state → vocab distribution
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, video_feat, caption_input):
        # video_feat: [B, 1280]
        # caption_input: [B, T]

        B, T = caption_input.size()

        # --- Initial hidden state ---
        h0 = self.feature_proj(video_feat).unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, B, hidden_dim]

        # --- Embed caption tokens ---
        emb = self.embedding(caption_input)  # [B, T, hidden_dim]

        # --- GRU forward ---
        gru_out, _ = self.gru(emb, h0)  # [B, T, hidden_dim]

        # --- Output logits ---
        logits = self.output_layer(gru_out)  # [B, T, vocab_size]
        return logits

