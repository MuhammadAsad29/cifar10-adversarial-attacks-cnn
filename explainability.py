import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from cifar_cnn import build_model, CifarCNN
from utils import get_dataloaders, get_device, load_checkpoint, clamp_to_image
from attacks import fgsm_attack, pgd_attack

# ---------- Grad-CAM ----------

class GradCAM:
    def __init__(self, model: CifarCNN, target_layer_name: str = 'features.2.block.4'):
        self.model = model
        self.model.eval()
        # pick a conv layer inside last ConvBlock; default BN layer index then we grab preceding conv features
        self._activations = None
        self._gradients = None

        # fetch target layer module
        layer = self._get_layer_by_name(self.model, target_layer_name)
        if layer is None:
            raise ValueError(f"Layer '{target_layer_name}' not found. Try a different name.")
        layer.register_forward_hook(self._forward_hook)
        layer.register_full_backward_hook(self._backward_hook)

    def _get_layer_by_name(self, model, name: str):
        # name like 'features.2.block.4'
        cur = model
        for attr in name.split('.'):
            if attr.isdigit():
                cur = cur[int(attr)]
            else:
                cur = getattr(cur, attr)
        return cur

    def _forward_hook(self, module, inp, out):
        self._activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, images: torch.Tensor, targets: torch.Tensor):
        # Forward & backward to get gradients for target class
        images = images.clone().detach().requires_grad_(True)
        logits = self.model(images)
        # pick the logit of the target class for each sample
        selected = logits.gather(1, targets.view(-1,1)).sum()
        self.model.zero_grad(set_to_none=True)
        selected.backward()

        # weights: global-average of gradients over spatial dims
        weights = self._gradients.mean(dim=(2,3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = torch.relu(cam)
        # normalize to [0,1] per-sample
        B = cam.size(0)
        cam_ = cam.view(B, -1)
        cam_min = cam_.min(dim=1, keepdim=True)[0]
        cam_max = cam_.max(dim=1, keepdim=True)[0]
        cam_norm = ((cam_ - cam_min) / (cam_max - cam_min + 1e-8)).view_as(cam)
        return cam_norm  # shape (B,1,H,W)

# ---------- Integrated Gradients ----------

def integrated_gradients(model, images, targets, baseline=None, steps: int = 50):
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(images)  # black image in normalized space
    # interpolate between baseline and image
    alphas = torch.linspace(0.0, 1.0, steps=steps, device=images.device).view(-1,1,1,1)
    attributions = torch.zeros_like(images)

    for alpha in alphas:
        x = baseline + alpha * (images - baseline)
        x.requires_grad_(True)
        logits = model(x)
        selected = logits.gather(1, targets.view(-1,1)).sum()
        model.zero_grad(set_to_none=True)
        selected.backward()
        grads = x.grad.detach()
        attributions += grads

    attributions /= steps
    # multiply by (input - baseline)
    attributions = (images - baseline) * attributions
    return attributions

# ---------- Visualization helpers ----------

def denorm_to_pixel(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(1,3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=x.device).view(1,3,1,1)
    return torch.clamp(x * std + mean, 0, 1)

def overlay_cam_on_image(cam, img, alpha=0.5):
    # cam: (B,1,H,W) normalized [0,1]; img: (B,3,H,W) pixel space [0,1]
    B, _, Hi, Wi = img.shape
    cam_resized = F.interpolate(cam, size=(Hi, Wi), mode='bilinear', align_corners=False)
    cam_rgb = cam_resized.repeat(1,3,1,1) # make 3-channel
    return torch.clamp((1 - alpha) * img + alpha * cam_rgb, 0, 1)

def select_examples(model, loader, device, n=8):
    """Return n correctly classified and n misclassified indices from the first few batches."""
    model.eval()
    correct_imgs, correct_lbls = [], []
    wrong_imgs, wrong_lbls, wrong_pred = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            preds = logits.argmax(1)
            mask_correct = preds == targets
            mask_wrong = ~mask_correct

            if mask_correct.any():
                correct_imgs.append(images[mask_correct])
                correct_lbls.append(targets[mask_correct])
            if mask_wrong.any():
                wrong_imgs.append(images[mask_wrong])
                wrong_lbls.append(targets[mask_wrong])
                wrong_pred.append(preds[mask_wrong])

            if sum(x.size(0) for x in correct_imgs) >= n and sum(x.size(0) for x in wrong_imgs) >= n:
                break

    def stack_first_k(tensors, k):
        if not tensors:
            return None
        cat = torch.cat(tensors, dim=0)
        return cat[:k]

    return (stack_first_k(correct_imgs, n), stack_first_k(correct_lbls, n), 
            stack_first_k(wrong_imgs, n), stack_first_k(wrong_lbls, n))

def grid_save(tensors, path, nrow=8):
    grid = make_grid(tensors, nrow=nrow, padding=2)
    save_image(grid, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--attack', type=str, default='none', choices=['none','fgsm','pgd'])
    ap.add_argument('--eps', type=float, default=8/255)
    ap.add_argument('--alpha', type=float, default=2/255)
    ap.add_argument('--steps', type=int, default=10)
    args = ap.parse_args()

    device = get_device()
    _, test_loader = get_dataloaders(batch_size=args.batch_size, workers=args.workers)

    model = build_model().to(device)
    state = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(state['model'])

    corr_imgs, corr_lbls, wrong_imgs, wrong_lbls = select_examples(model, test_loader, device, n=args.samples)

    # Choose a batch to visualize (prefer correctly classified first)
    if corr_imgs is None:
        print('No correctly classified examples found in sampled data; using misclassified.')
        corr_imgs, corr_lbls = wrong_imgs, wrong_lbls

    images, targets = corr_imgs, corr_lbls

    # Optionally craft adversarial counterparts
    if args.attack != 'none':
        if args.attack == 'fgsm':
            adv = fgsm_attack(model, images, targets, eps_pixel=args.eps)
        else:
            adv = pgd_attack(model, images, targets, eps_pixel=args.eps, alpha_pixel=args.alpha, steps=args.steps)
    else:
        adv = None

    # ---- Grad-CAM ----
    cam = GradCAM(model, target_layer_name='features.2.block.4')
    cams_clean = cam(images, targets)  # (B,1,H,W)
    imgs_px = denorm_to_pixel(images)
    overlay_clean = overlay_cam_on_image(cams_clean, imgs_px, alpha=0.5)
    grid_save(overlay_clean, 'outputs/gradcam_clean.png')
    print('Saved Grad-CAM for clean samples -> outputs/gradcam_clean.png')

    if adv is not None:
        cams_adv = cam(adv, targets)
        adv_px = denorm_to_pixel(adv)
        overlay_adv = overlay_cam_on_image(cams_adv, adv_px, alpha=0.5)
        grid_save(overlay_adv, 'outputs/gradcam_adv.png')
        print('Saved Grad-CAM for adversarial samples -> outputs/gradcam_adv.png')

    # ---- Integrated Gradients ----
    ig_clean = integrated_gradients(model, images, targets, steps=32)
    # Normalize IG attributions to [0,1] for visualization (absolute)
    ig_clean_vis = (ig_clean.abs() / (ig_clean.abs().amax(dim=(1,2,3), keepdim=True) + 1e-8)).clamp(0,1)
    grid_save(ig_clean_vis, 'outputs/ig_clean.png')
    print('Saved Integrated Gradients for clean samples -> outputs/ig_clean.png')

    if adv is not None:
        ig_adv = integrated_gradients(model, adv, targets, steps=32)
        ig_adv_vis = (ig_adv.abs() / (ig_adv.abs().amax(dim=(1,2,3), keepdim=True) + 1e-8)).clamp(0,1)
        grid_save(ig_adv_vis, 'outputs/ig_adv.png')
        print('Saved Integrated Gradients for adversarial samples -> outputs/ig_adv.png')

if __name__ == '__main__':
    main()
