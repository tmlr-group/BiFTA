# Let's Roll a BiFTA: Bi-refinement for Fine-grained Text-visual Alignment in Vision-Language Models

This repository is the official PyTorch implementation of the paper: Let's Roll a BiFTA:
Bi-refinement for Fine-grained Text-visual Alignment in Vision-Language Models

**Abstract:**
Recent research has shown that aligning fine-grained text descriptions with localized image
patches can significantly improve the zero-shot performance of pre-trained vision-language
models (e.g., CLIP). However, we find that both fine-grained text descriptions and localized
image patches often contain redundant information, making text-visual alignment less
effective. In this paper, we tackle this issue from two perspectives: view refinement and
description refinement, termed as Bi-refinement for Fine-grained Text-visual Alignment
(BiFTA). View refinement removes redundant image patches with high Intersection over
Union (IoU) ratios, resulting in more distinctive visual samples. Description refinement
removes redundant text descriptions with high pairwise cosine similarity, ensuring greater
diversity in the remaining descriptions. BiFTA achieves superior zero-shot performance on 6
benchmark datasets for both ViT-based and ResNet-based CLIP, justifying the necessity to
remove redundant information in visual-text alignment.

## Environment
- Python (3.12.3)
- Cuda (12.2.0)
- Pytorch (2.3.0)


## Installation
```
    conda create -n bifta
    conda activate bifta
    pip install -r requirements.txt
```

## Dataset Preparation
To implement the results, please modify the `data_path = ""` in `config` of this repository to where your data is stored.


## Step 2.1: Running Code for Baselines
```
  cd BiFTA
  python main --dataseed [seed] --dataset [dataset] --model_size [model_size]
```


## Acknowledgements

This repo is built upon these previous works:
- **Masked**

