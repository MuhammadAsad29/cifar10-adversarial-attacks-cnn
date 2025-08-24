import argparse, time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from cifar_cnn import build_model
from utils import get_dataloaders, build_optimizer, accuracy, set_seed, get_device, save_checkpoint

def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    running_acc, running_loss = 0.0, 0.0
    for images, targets in tqdm(loader, desc='Train', leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_acc += (logits.argmax(1) == targets).float().sum().item()
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    running_acc, running_loss = 0.0, 0.0
    for images, targets in tqdm(loader, desc='Eval', leave=False):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        running_loss += loss.item() * images.size(0)
        running_acc += (logits.argmax(1) == targets).float().sum().item()
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--opt', type=str, default='adam', choices=['adam','sgd'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt', type=str, default='checkpoints/clean_best.pt')
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f'Using device: {device}')

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, workers=args.workers)
    model = build_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, opt=args.opt, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, criterion)
        te_loss, te_acc = evaluate(model, test_loader, device, criterion)
        scheduler.step()

        print(f'Epoch {epoch:03d}/{args.epochs} | Train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | Test loss {te_loss:.4f} acc {te_acc*100:.2f}%')

        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'acc': te_acc}, args.ckpt)
            print(f'  -> Saved new best to {args.ckpt} (acc={te_acc*100:.2f}%)')

    print(f'Best test accuracy: {best_acc*100:.2f}%')

if __name__ == '__main__':
    main()
