import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import TreebankWordTokenizer

class VideoCaptionDataset(Dataset):
    def __init__(self, split, feature_dir='features', vocab_dir='vocab', max_len=50):
        self.split = split  # 'train', 'val', or 'test'
        self.max_len = max_len

        # Load vocabulary
        with open(os.path.join(vocab_dir, 'word2idx.json')) as f:
            self.word2idx = json.load(f)

        # Load tokenized captions
        with open(os.path.join(vocab_dir, 'tokenized_captions.json')) as f:
            self.tokenized_data = json.load(f)

        # Load list of video_ids from split JSON
        split_json_path = os.path.join(split, f"{split}.json")
        with open(split_json_path) as f:
            split_data = json.load(f)
        self.video_ids = list(split_data.keys())

        # Feature paths (Only CNN features here)
        self.cnn_dir = os.path.join(feature_dir, 'cnn', split)

        # Special token indices
        self.PAD = self.word2idx['<pad>']
        self.SOS = self.word2idx['<sos>']
        self.EOS = self.word2idx['<eos>']
        self.UNK = self.word2idx['<unk>']

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]

        # Load CNN features (Only CNN features are loaded here)
        cnn_feat = np.load(os.path.join(self.cnn_dir, f"{video_id}.npy"))
        video_tensor = torch.tensor(cnn_feat, dtype=torch.float32)

        # Randomly select a caption for the video
        tokens = self.tokenized_data[video_id][np.random.randint(len(self.tokenized_data[video_id]))]
        caption = [self.SOS] + [self.word2idx.get(w, self.UNK) for w in tokens] + [self.EOS]

        # Truncate and pad caption to max length
        caption = caption[:self.max_len]
        caption += [self.PAD] * (self.max_len - len(caption))

        # Split caption into input and target
        caption_input = torch.tensor(caption[:-1])  # [T-1] (all except EOS)
        caption_target = torch.tensor(caption[1:])  # [T-1] (all except SOS)

        return video_tensor, caption_input, caption_target


# --- Collate function for batching ---
def collate_fn(batch):
    videos, cap_inputs, cap_targets = zip(*batch)
    return (
        torch.stack(videos),         # [B, 2048] CNN features
        torch.stack(cap_inputs),     # [B, T]
        torch.stack(cap_targets)     # [B, T]
    )


# --- Dataloader builder (optional) ---
def get_loaders(batch_size=32, feature_dir='features', vocab_dir='vocab'):
    # Create datasets for train, validation, and test splits
    train_set = VideoCaptionDataset('train', feature_dir, vocab_dir)
    val_set = VideoCaptionDataset('val', feature_dir, vocab_dir)
    test_set = VideoCaptionDataset('test', feature_dir, vocab_dir)

    # Create DataLoaders for each dataset
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
