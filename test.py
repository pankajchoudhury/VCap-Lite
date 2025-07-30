import os
import json
import torch
from tqdm import tqdm
from dataloader import get_loaders
from model import VideoCaptioningModel
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import pandas as pd
import random

# --- Load vocab ---
with open("vocab/word2idx.json") as f:
    word2idx = json.load(f)
with open("vocab/idx2word.json") as f:
    idx2word = json.load(f)

PAD_IDX = word2idx["<pad>"]
SOS_IDX = word2idx["<sos>"]
EOS_IDX = word2idx["<eos>"]
VOCAB_SIZE = len(word2idx)
MAX_LEN = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load test data ---
_, _, test_loader = get_loaders(batch_size=32)

# --- Load trained LSTM model ---
model = VideoCaptioningModel(
    feature_dim=1280,
    hidden_dim=512,
    vocab_size=VOCAB_SIZE,
    num_layers=2,
    dropout=0.3,
    pad_idx=PAD_IDX,
    max_len=MAX_LEN
).to(device)

model.load_state_dict(torch.load("best_model_msrvtt_gru_mobilenetv2.pt", map_location=device))
model.eval()

# --- Decode output tokens ---
def decode_caption(indices):
    words = []
    for idx in indices:
        word = idx2word.get(str(idx.item()), "<unk>")
        if word == "<eos>":
            break
        if word not in ["<sos>", "<pad>"]:
            words.append(word)
    return " ".join(words)

# --- Beam Search Decoder ---
def beam_search(video_feat, beam_size=5, max_len=50):
    sequences = [[torch.tensor([SOS_IDX], device=device), 0.0]]

    for _ in range(max_len):
        candidates = []
        for seq, score in sequences:
            if seq[-1].item() == EOS_IDX:
                candidates.append((seq, score))
                continue

            input_seq = seq.unsqueeze(0)  # [1, t]
            with torch.no_grad():
                logits = model(video_feat, input_seq)  # [1, t, vocab]
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # [1, vocab]
            topk_log_probs, topk_indices = log_probs.topk(beam_size)

            for i in range(beam_size):
                new_seq = torch.cat([seq, topk_indices[0, i].unsqueeze(0)], dim=0)
                new_score = score + topk_log_probs[0, i].item()
                candidates.append((new_seq, new_score))

        sequences = sorted(candidates, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_size]

    return sequences[0][0]

# --- Evaluation over beam sizes ---
all_results = []
beam=5
with open("beamwise_metrics.txt", "w") as metric_log:
    # for beam in range(1, 11):
    print("-" * 20, f"beam size = {beam}", "-" * 20)
    gts, res = {}, {}
    captions_log = []

    with torch.no_grad():
        for batch_idx, (video_feats, _, _) in enumerate(tqdm(test_loader, desc="Generating captions")):
            video_feats = video_feats.to(device)

            for i in range(video_feats.size(0)):
                video_id = test_loader.dataset.video_ids[batch_idx * test_loader.batch_size + i]
                ref_caps = test_loader.dataset.tokenized_data[video_id]

                feat = video_feats[i:i+1]  # [1, 2560]
                pred_seq = beam_search(feat, beam_size=beam, max_len=MAX_LEN)
                caption = decode_caption(pred_seq)

                gts[video_id] = [{"caption": " ".join(ref)} for ref in ref_caps]
                res[video_id] = [{"caption": caption}]
                captions_log.append(f"[{video_id}]\nPRED : {caption}\nGT   : " + "; ".join([" ".join(x) for x in ref_caps]) + "\n")

    # --- COCO Evaluation ---
    tokenizer = PTBTokenizer()
    gts_tok = tokenizer.tokenize(gts)
    res_tok = tokenizer.tokenize(res)

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Cider(), "CIDEr"),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]

    scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts_tok, res_tok)
        if isinstance(method, list):
            for m, s in zip(method, score):
                scores[m] = s
        else:
            scores[method] = score
    scores["beam_size"] = beam
    all_results.append(scores)

    # Save 10 random sample outputs
   # random_samples = random.sample(captions_log, min(10, len(captions_log)))
   # with open(f"samples_beam_{beam}.txt", "w") as f:
       # f.writelines(line + "\n" for line in random_samples)

    # Save metrics
   # metric_log.write(f"\nBeam Size: {beam}\n")
   # for k, v in scores.items():
        #if k != "beam_size":
           # metric_log.write(f"{k}: {v:.4f}\n")
   # metric_log.write("-" * 50 + "\n")

    # Print on console
   # print("\n📌 Sample Predictions:")
   # for entry in random_samples:
      #  print(entry)
    #print("📊 Evaluation Metrics:")
    for k, v in scores.items():
        if k != "beam_size":
            print(f"{k}: {v:.4f}")
    print("✅ Beam", beam, "done.\n")

# --- Save all beam results ---
results = pd.DataFrame(all_results)
results = (results * 100).round(1)
results.to_csv("results_for_all_beam.csv", index=False)
print("📁 All beam-wise results saved to results_for_all_beam.csv")
