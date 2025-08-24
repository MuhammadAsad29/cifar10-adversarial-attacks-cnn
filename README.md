# CIFAR-10 Adversarial Attacks

This project implements a complete pipeline for:

Training a CNN on CIFAR-10

Evaluating adversarial robustness (FGSM & PGD)

Adversarial training as a defense

Explainability (Grad-CAM & Integrated Gradients)

⚙️ Setup
python -m venv .venv
## Linux/Mac
source .venv/bin/activate
## Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt


💡 Optional: If you have a GPU, install CUDA-enabled PyTorch from pytorch.org
.

## 🏋️ Train a Clean (Standard) Model
python train.py --epochs 50 --batch-size 128 --lr 0.001 --weight-decay 5e-4 --opt adam


Best checkpoint → checkpoints/clean_best.pt

Logs → show running accuracy and loss

## 📊 Evaluate Clean Model
python eval.py --ckpt checkpoints/clean_best.pt

## ⚡ Adversarial Attacks
FGSM
python attacks.py --ckpt checkpoints/clean_best.pt --attack fgsm --eps 0.0314 --eval

PGD
python attacks.py --ckpt checkpoints/clean_best.pt --attack pgd --eps 0.0314 --alpha 0.0078 --steps 10 --eval


## Outputs:

Adversarial accuracy

Example grids saved in outputs/

## 🛡️ Adversarial Training (Defense)

PGD-based adversarial training:

python defense_advtrain.py --epochs 30 --batch-size 128 --eps 0.0314 --alpha 0.0078 --steps 4


Saves → checkpoints/robust_best.pt

Prints → both clean & adversarial validation accuracy

## 🔍 Explainability (Grad-CAM & Integrated Gradients)

Run explanations on a trained model (clean or robust):

python explainability.py --ckpt checkpoints/clean_best.pt --samples 8 --attack none


Against adversarial examples:

python explainability.py --ckpt checkpoints/clean_best.pt --samples 8 --attack pgd --eps 0.0314 --alpha 0.0078 --steps 10


Saved figures in outputs/:

gradcam_clean.png, gradcam_adv.png

ig_clean.png, ig_adv.png

## 💡 Tips

Increase --epochs for higher accuracy (80%+ achievable).

Tune --eps, --alpha, --steps for stronger/weaker attacks.

On CPU → use smaller batch size (--batch-size 64) for faster training.

## 📂 File Map

cifar_cnn.py — CNN with BN/Dropout + GAP head

utils.py — Dataloaders, transforms, helpers

train.py — Train clean model

eval.py — Evaluate clean model

attacks.py — FGSM/PGD attacks & evaluation

defense_advtrain.py — PGD adversarial training

explainability.py — Grad-CAM & Integrated Gradients

report_template.md — Report template

## 🔁 Reproducibility

Fix seed: --seed 42

Use --workers 2 on low-RAM systems to reduce dataloader overhead


