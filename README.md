# VideoEvent

This is the official implementation repository for VideoEvent.

## 1. Environment Setup

All required dependencies for this project are listed in the `environment.yml` file.

You can create and activate the environment using Conda:

```bash
# Using conda
conda env create -f environment.yml
conda activate videoevent
```

## 2. Usage

### Training

The training script is located at `scripts/video/train`.

### Evaluation

The evaluation scripts are located at `scripts/video/eval`.

## 3. Dataset

All datasets used in this project are publicly available. You can download them from their respective official sources:

* [WebVid](https://github.com/m-bain/webvid)
* [LLaVA-558K](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)
* [Video-ChatGPT-99K](https://github.com/mbzuai-oryx/Video-ChatGPT)