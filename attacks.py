import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from cifar_cnn import build_model
from utils import get_dataloaders, get_device, load_checkpoint, eps_pixel_to_normalized, clamp_to_image

def fgsm_attack(model, images, targets, eps_pixel):
    model.eval()
    images = images.clone().detach().requires_grad_(True)
    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()
    grad_sign = images.grad.data.sign()
    eps_norm = eps_pixel_to_normalized(eps_pixel, images.device)
    adv = images + eps_norm * grad_sign
    adv = clamp_to_image(adv).detach()
    return adv

def pgd_attack(model, images, targets, eps_pixel, alpha_pixel, steps=10, random_start=True):
    model.eval()
    x = images.clone().detach()
    eps_norm = eps_pixel_to_normalized(eps_pixel, images.device)
    alpha_norm = eps_pixel_to_normalized(alpha_pixel, images.device)

    if random_start:
        # random noise within epsilon-ball in normalized space
        x = x + (torch.rand_like(x)*2 - 1.0) * eps_norm
        x = clamp_to_image(x)

    x = x.detach().requires_grad_(True)
    for _ in range(steps):
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()
        with torch.no_grad():
            x = x + alpha_norm * x.grad.sign()
            # project into epsilon-ball around original image (in normalized space)
            delta = torch.clamp(x - images, min=-eps_norm, max=eps_norm)
            x = images + delta
            x = clamp_to_image(x)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)
    return x.detach()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for images, targets in tqdm(loader, desc='Eval', leave=False):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        pred = logits.argmax(1)
        correct += (pred == targets).float().sum().item()
        n += images.size(0)
    return correct / n

def evaluate_under_attack(model, loader, device, attack='fgsm', eps=8/255, alpha=2/255, steps=10):
    model.eval()
    correct, n = 0, 0
    for images, targets in tqdm(loader, desc=f'Attack:{attack}', leave=False):
        images, targets = images.to(device), targets.to(device)
        if attack == 'fgsm':
            adv = fgsm_attack(model, images, targets, eps_pixel=eps)
        else:
            adv = pgd_attack(model, images, targets, eps_pixel=eps, alpha_pixel=alpha, steps=steps)
        logits = model(adv)
        pred = logits.argmax(1)
        correct += (pred == targets).float().sum().item()
        n += images.size(0)
    return correct / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--attack', type=str, default='fgsm', choices=['fgsm','pgd'])
    ap.add_argument('--eps', type=float, default=8/255)
    ap.add_argument('--alpha', type=float, default=2/255)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--eval', action='store_true', help='Also print clean accuracy')
    args = ap.parse_args()

    device = get_device()
    _, test_loader = get_dataloaders(batch_size=args.batch_size, workers=args.workers)
    model = build_model().to(device)
    state = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(state['model'])

    if args.eval:
        clean_acc = evaluate(model, test_loader, device)
        print(f'Clean accuracy: {clean_acc*100:.2f}%')

    adv_acc = evaluate_under_attack(model, test_loader, device, attack=args.attack, eps=args.eps, alpha=args.alpha, steps=args.steps)
    print(f'Adversarial accuracy ({args.attack.upper()} eps={args.eps}): {adv_acc*100:.2f}%')

if __name__ == '__main__':
    main()
