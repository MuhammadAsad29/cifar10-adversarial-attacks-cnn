import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from cifar_cnn import build_model
from utils import get_dataloaders, build_optimizer, accuracy, set_seed, get_device, save_checkpoint
from attacks import pgd_attack

def train_adv_epoch(model, loader, optimizer, device, criterion, eps, alpha, steps):
    model.train()
    running_acc, running_loss = 0.0, 0.0
    for images, targets in tqdm(loader, desc='AdvTrain', leave=False):
        images, targets = images.to(device), targets.to(device)
        # generate adversarial examples on-the-fly
        with torch.enable_grad():
            adv = pgd_attack(model, images, targets, eps_pixel=eps, alpha_pixel=alpha, steps=steps, random_start=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(adv)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += (logits.argmax(1) == targets).float().sum().item()

    n = len(loader.dataset)
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for images, targets in tqdm(loader, desc='Eval', leave=False):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        correct += (logits.argmax(1) == targets).float().sum().item()
        n += images.size(0)
    return correct / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--opt', type=str, default='adam', choices=['adam','sgd'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt', type=str, default='checkpoints/robust_best.pt')
    ap.add_argument('--eps', type=float, default=8/255)
    ap.add_argument('--alpha', type=float, default=2/255)
    ap.add_argument('--steps', type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f'Using device: {device}')

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, workers=args.workers)
    model = build_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, opt=args.opt, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_clean, best_epoch = 0.0, -1
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_adv_epoch(model, train_loader, optimizer, device, criterion, eps=args.eps, alpha=args.alpha, steps=args.steps)
        clean_acc = evaluate(model, test_loader, device)
        scheduler.step()
        print(f'Epoch {epoch:03d}/{args.epochs} | AdvTrain loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | Clean acc {clean_acc*100:.2f}%')

        if clean_acc > best_clean:
            best_clean = clean_acc
            best_epoch = epoch
            save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'acc': clean_acc}, args.ckpt)
            print(f'  -> Saved new best robust model to {args.ckpt} (clean acc={clean_acc*100:.2f}%)')

    print(f'Best clean accuracy after adversarial training: {best_clean*100:.2f}% (epoch {best_epoch})')

if __name__ == '__main__':
    main()
