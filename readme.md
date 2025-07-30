
# VCap‑Lite: A Computationally Efficient Framework for Single‑Sentence Video Captioning

## 🔍 Overview

VCap‑Lite is a lightweight deep learning framework designed for Single-Sentence Video Captioning (SSVC), where the goal is to generate concise natural language descriptions summarizing the content of a given video. Traditional methods achieve high performance using heavy models like transformers, but they are unsuitable for deployment on edge devices due to high computational costs. 

**VCap‑Lite** addresses this gap by combining:
- **MobileNetV2** as a lightweight visual encoder
- **Gated Recurrent Unit (GRU)** as a caption generator

Despite its reduced complexity, VCap‑Lite achieves **competitive results** on standard benchmarks (MSVD and MSR-VTT) while requiring significantly fewer resources.

---

### 📊 Performance Comparison with State-of-the-Art Methods

| **Method**           | **MSVD B4** | **MSVD M**    | **MSVD R**    | **MSVD C**    | **MSR-VTT B4** | **MSR-VTT M**    | **MSR-VTT R**    | **MSR-VTT C**    | **GFLOPs** | **Params (M)** |
| -------------------- | ----------- | -------- | -------- | -------- | -------------- | -------- | -------- | -------- | ---------- | -------------- |
| GRU-EVE (Aafaq2019)  | 47.9        | 35.0     | 71.5     | 78.1     | 38.3           | 28.4     | 60.7     | 48.1     | 844.2      | 208.29         |
| MGSA (Chen2019)      | 53.4        | 35.0     | -        | 86.7     | 42.4           | 27.6     | -        | 47.5     | 694.4      | 146.29         |
| OA-BTG (Zhang2019)   | 56.9        | 36.2     | -        | 90.6     | 41.4           | 28.2     | -        | 46.9     | 750.2      | 118.6          |
| OpenBook (Zhang2021) | -           | -        | -        | -        | 33.9           | 23.7     | 50.2     | 52.9     | 694.4      | 146.29         |
| ORG-TRL (Zhang2020)  | 54.3        | 36.4     | 73.9     | 95.2     | 43.6           | 28.8     | 62.1     | 50.9     | 994.4      | 190.33         |
| PickNet (Chen2018)   | 52.3        | 33.3     | 69.6     | 76.5     | 41.3           | 27.7     | 59.8     | 44.1     | 92.6       | 72.2           |
| PMI-CAP (Chen2020)   | 54.6        | 36.4     | -        | 95.1     | 42.1           | 28.7     | -        | 49.4     | 694.4      | 146.29         |
| POS+CG (Wang2019)    | 52.5        | 34.1     | 71.3     | 88.7     | 42.0           | 28.2     | 61.6     | 48.7     | 450.4      | 69.97          |
| POS+VCT (Hou2019)    | 52.8        | 36.1     | 71.8     | 87.8     | 42.3           | 29.7     | 62.8     | 49.1     | 694.4      | 146.29         |
| SAAT (Zheng2020)     | 46.5        | 33.5     | 69.4     | 81.0     | 39.9           | 27.7     | 61.2     | 51.0     | 694.4      | 146.29         |
| SibNet (Liu2020)     | 54.2        | 34.8     | 71.7     | 88.2     | 40.9           | 27.5     | 60.2     | 47.5     | 45.4       | 18.98          |
| STG-KD (Pan2020)     | 52.2        | 36.9     | 73.9     | 93.0     | 40.5           | 28.3     | 60.9     | 47.1     | 870.4      | 112.66         |
| SWINBERT (Lin2022)   | 66.3        | 42.4     | 80.9     | 149.4    | 45.4           | 30.6     | 64.1     | 55.9     | 207.0      | 138.0          |
| **VCap-Lite (Ours)** | **51.5**    | **34.6** | **71.3** | **90.2** | **34.8**       | **25.9** | **56.1** | **41.1** | **9.2**    | **14.5**       |


> 📌 **Key Insight:** VCap‑Lite achieves near state-of-the-art accuracy with drastically reduced GFLOPs and parameter count, making it suitable for low-resource and real-time applications.

---

## 🛠️ Setup Instructions

### 1. Conda Environment Setup

```bash
conda env create -f environment.yml
conda activate vcap-lite
````

---

## 🚀 Testing with Pretrained Weights

### Step 1: Extract and Organize Data

Extract `DATA.tar.xz`. It should contain the following structure:

```
├── msrvtt
│   ├── features
│   │   ├── test/
│   │   ├── train/
│   │   └── val/
│   ├── test/
│   ├── train/
│   ├── val/
│   ├── vocab/
│   └── best_model_msrvtt_gru_mobilenetv2.pt
├── msvd
│   ├── features
│   │   ├── test/
│   │   ├── train/
│   │   └── val/
│   ├── test/
│   ├── train/
│   ├── val/
│   ├── vocab/
│   └── best_model_msvd_gru_mobilenetv2.pt
```

Copy all contents into the root directory of the repository.

### Step 2: Modify Model Path

In `test.py`, replace the model loading line with:

```python
model.load_state_dict(torch.load("best_model_msrvtt_gru_mobilenetv2.pt", map_location=device))
```

### Step 3: Run Testing

```bash
python test.py
```

### 🔎 Evaluation Results

**MSR-VTT**

```
BLEU-1:   0.7348
BLEU-2:   0.5821
BLEU-3:   0.4516
BLEU-4:   0.3480
CIDEr:    0.4109
METEOR:   0.2588
ROUGE_L:  0.5613
```

**MSVD**

```
BLEU-1:   0.8048
BLEU-2:   0.6940
BLEU-3:   0.6030
BLEU-4:   0.5155
CIDEr:    0.9020
METEOR:   0.3463
ROUGE_L:  0.7134
```



