import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from cifar_cnn import build_model
from utils import get_dataloaders, get_device, load_checkpoint

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    top1, n = 0.0, 0
    for images, targets in tqdm(loader, desc='Eval', leave=False):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        top1 += (logits.argmax(1) == targets).float().sum().item()
        n += images.size(0)
    return top1 / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    device = get_device()
    _, test_loader = get_dataloaders(batch_size=args.batch_size, workers=args.workers)

    model = build_model().to(device)
    state = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(state['model'])

    acc = evaluate(model, test_loader, device)
    print(f'Clean test accuracy: {acc*100:.2f}%')

if __name__ == '__main__':
    main()
