CIFAR-10 Adversarial Attacks 

This project implements a complete pipeline for training a CNN on CIFAR-10, evaluating adversarial robustness (FGSM & PGD), applying adversarial training as a defense, and exploring explainability (Grad-CAM & Integrated Gradients).

0) Setup
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
Optional: if you have a GPU, install CUDA-enabled PyTorch following the instructions at https://pytorch.org/ .

1) Train a clean (standard) model
python train.py --epochs 50 --batch-size 128 --lr 0.001 --weight-decay 5e-4 --opt adam
• Best checkpoint is saved at checkpoints/clean_best.pt.
• Logs show running accuracy and loss.

2) Evaluate clean model on test data
python eval.py --ckpt checkpoints/clean_best.pt

3) Evaluate clean model under adversarial attacks
FGSM
python attacks.py --ckpt checkpoints/clean_best.pt --attack fgsm --eps 0.0314 --eval
PGD
python attacks.py --ckpt checkpoints/clean_best.pt --attack pgd --eps 0.0314 --alpha 0.0078 --steps 10 --eval
Outputs: adversarial accuracy and sample grids in outputs/.

4) Adversarial training (defense)
PGD-based adversarial training:
python defense_advtrain.py --epochs 30 --batch-size 128 --eps 0.0314 --alpha 0.0078 --steps 4
Saves checkpoints/robust_best.pt and prints both clean & adversarial validation accuracy.

5) Explainability (Grad-CAM & Integrated Gradients)
Run explanations on a trained model (clean or robust):
python explainability.py --ckpt checkpoints/clean_best.pt --samples 8 --attack none
# or with adversarial examples:
python explainability.py --ckpt checkpoints/clean_best.pt --samples 8 --attack pgd --eps 0.0314 --alpha 0.0078 --steps 10
Figures saved under outputs/:
• gradcam_clean.png, gradcam_adv.png
• ig_clean.png, ig_adv.png

6) Tips
• Increase --epochs for higher accuracy (80%+ is achievable with this model).
• Tune --eps, --alpha, and --steps for stronger/weaker attacks.
• On CPU, lower --batch-size (e.g., 64) to reduce runtime.

File Map
• cifar_cnn.py — CNN with BN/Dropout + GAP head
• utils.py — Dataloaders, transforms, training/eval helpers
• train.py — Train standard model
• eval.py — Evaluate on clean test data
• attacks.py — FGSM/PGD attacks + adversarial evaluation
• defense_advtrain.py — PGD adversarial training loop
• explainability.py — Grad-CAM & Integrated Gradients visualizations
• report_template.md — Report template

Repro tips
• Use --seed 42 (or any integer) for reproducible results.
• Use --workers 2 on low-RAM systems to reduce dataloader overhead.


