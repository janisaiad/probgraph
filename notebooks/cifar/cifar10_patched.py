"""
cifar10 patch reconstruction using stochastic interpolants

this implementation:
- trains on dog class only (cifar10 class 5)
- uses mask-conditioned u-net architecture
- learns to reconstruct masked patches while keeping visible pixels fixed
- conditions on mask during both training and generation
- forces visible pixels to remain constant in output

approach:
  x0 = masked_image * mask + noise * (1-mask)  # we start with visible pixels + noise in masked regions
  x1 = original_image                           # we target full image
  model learns interpolant: x0 -> x1 conditioned on mask
  during generation: output = model_output * (1-mask) + original * mask  # we force visible pixels fixed
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('../../')
from typing import Tuple, Any

import interflow as itf
import interflow.stochastic_interpolant as stochastic_interpolant

if torch.cuda.is_available():
    print('cuda available, setting default tensor residence to gpu')
    itf.util.set_torch_device('cuda')
else:
    print('no cuda device found')
print(itf.util.get_torch_device())

print("torch version:", torch.__version__)

# we define utility functions
def grab(var):
    """we take a tensor off the gpu and convert it to a numpy array on the cpu"""
    return var.detach().cpu().numpy()

# we load cifar10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root="../../data/cifar10", train=True, 
                                        download=True, transform=transform)

# we filter dataset to keep only dog class (class 5)
# cifar10 classes: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck
target_class = 5  # we select dog class
dog_indices = [i for i in range(len(trainset)) if trainset.targets[i] == target_class]
print(f"\nfiltered dataset to class 'dog': {len(dog_indices)} images (out of {len(trainset)} total)")

# we create data iterator that only samples from dog images
def get_cifar_batch(bs):
    """we get a batch of cifar10 dog images only"""
    indices = torch.randint(0, len(dog_indices), (bs,))
    imgs = torch.stack([trainset[dog_indices[i]][0] for i in indices])
    return imgs.to(itf.util.get_torch_device())

# we create masking function for patches
def create_patch_mask(bs, patch_size=8, num_patches=4):
    """we create random patch masks, 1 for visible pixels, 0 for masked patches"""
    mask = torch.ones(bs, 3, 32, 32)
    for i in range(bs):
        for _ in range(num_patches):
            x = torch.randint(0, 32 - patch_size, (1,)).item()
            y = torch.randint(0, 32 - patch_size, (1,)).item()
            mask[i, :, x:x+patch_size, y:y+patch_size] = 0  # we mask the patch
    return mask.to(itf.util.get_torch_device())

# we define u-net style convolutional denoiser for image reconstruction
class UNetDenoiser(nn.Module):
    """we use u-net architecture with skip connections for image reconstruction, conditioned on mask"""
    def __init__(self, in_channels=5, out_channels=3, base_channels=64):  # we add 1 channel for mask conditioning
        super().__init__()
        
        # we define encoder (downsampling path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU()
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU()
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU(),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU()
        )
        
        # we define bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels*8),
            nn.ReLU(),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1),
            nn.GroupNorm(8, base_channels*8),
            nn.ReLU()
        )
        
        # we define decoder (upsampling path) with skip connections
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU()
        )
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU(),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU()
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU()
        )
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU()
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU()
        )
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU()
        )
        
        # we define final output layer
        self.final = nn.Conv2d(base_channels, out_channels, 1)
    
    def forward(self, x_with_t):
        """we forward pass with skip connections"""
        # we encode
        e1 = self.enc1(x_with_t)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # we process bottleneck
        b = self.bottleneck(e3)
        
        # we decode with skip connections
        d3 = self.dec3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3_conv(d3)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2_conv(d2)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1_conv(d1)
        
        # we output final reconstruction
        out = self.final(d1)
        return out

# we define wrapper for eta network to match expected interface
class EtaNetwork(nn.Module):
    """we wrap unet to accept concatenated [x, t, mask] input"""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.mask = None  # we store mask for conditioning
    
    def set_mask(self, mask):
        """we set the mask for conditioning"""
        self.mask = mask
    
    def forward(self, xt_concat):
        """we expect input of shape [bs, 3*32*32 + 1]"""
        bs = xt_concat.shape[0]
        x = xt_concat[:, :-1].reshape(bs, 3, 32, 32)  # we reshape to image
        t = xt_concat[:, -1:]  # we extract time
        
        # we expand time to match spatial dimensions
        t_channel = t.view(bs, 1, 1, 1).expand(bs, 1, 32, 32)
        
        # we add mask conditioning (use first channel of mask for simplicity)
        if self.mask is not None:
            mask_channel = self.mask[:bs, 0:1, :, :]  # we take first channel of mask [bs, 1, 32, 32]
        else:
            mask_channel = torch.ones(bs, 1, 32, 32).to(x.device)  # we default to all visible
        
        x_with_t_mask = torch.cat([x, t_channel, mask_channel], dim=1)  # we concatenate [x, t, mask]
        
        # we process through unet
        out = self.unet(x_with_t_mask)
        
        # we flatten output
        return out.reshape(bs, -1)

# we define velocity field wrapper
class VelocityNetwork(nn.Module):
    """we wrap unet for velocity field b, conditioned on mask"""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.mask = None  # we store mask for conditioning
    
    def set_mask(self, mask):
        """we set the mask for conditioning"""
        self.mask = mask
    
    def forward(self, xt_concat):
        """we expect input of shape [bs, 3*32*32 + 1]"""
        bs = xt_concat.shape[0]
        x = xt_concat[:, :-1].reshape(bs, 3, 32, 32)  # we reshape to image
        t = xt_concat[:, -1:]  # we extract time
        
        # we expand time to match spatial dimensions
        t_channel = t.view(bs, 1, 1, 1).expand(bs, 1, 32, 32)
        
        # we add mask conditioning (use first channel of mask for simplicity)
        if self.mask is not None:
            mask_channel = self.mask[:bs, 0:1, :, :]  # we take first channel of mask [bs, 1, 32, 32]
        else:
            mask_channel = torch.ones(bs, 1, 32, 32).to(x.device)  # we default to all visible
        
        x_with_t_mask = torch.cat([x, t_channel, mask_channel], dim=1)  # we concatenate [x, t, mask]
        
        # we process through unet
        out = self.unet(x_with_t_mask)
        
        # we flatten output
        return out.reshape(bs, -1)

# we define training step function
def train_step(
    bs: int,
    interpolant: stochastic_interpolant.Interpolant,
    opt_b: Any,
    opt_eta: Any,
    sched_b: Any,
    sched_eta: Any,
    patch_size: int,
    num_patches: int,
    mask_loss_weight: float = 10.0
):
    """we take a single step of optimization on the training set"""
    opt_b.zero_grad()
    opt_eta.zero_grad()
    
    # we construct batch of real images
    x1s_img = get_cifar_batch(bs)  # we get [bs, 3, 32, 32]
    
    # we create masks
    masks = create_patch_mask(bs, patch_size=patch_size, num_patches=num_patches)  # we get [bs, 3, 32, 32]
    
    # we create masked images + noise in masked regions as starting point
    noise = torch.randn_like(x1s_img) * (1 - masks)  # we add noise only in masked regions
    x0s_img = x1s_img * masks + noise  # we combine masked image and noise
    
    # we flatten for interpolant
    x0s = x0s_img.reshape(bs, -1)  # we flatten to [bs, 3072]
    x1s = x1s_img.reshape(bs, -1)  # we flatten to [bs, 3072]
    masks_flat = masks.reshape(bs, -1)  # we flatten mask too
    
    # we sample random times
    ts = torch.rand(size=(bs,)).to(itf.util.get_torch_device())
    
    # we set masks for conditioning in both networks
    b.set_mask(masks)
    eta.set_mask(masks)
    
    # we compute the losses
    loss_start = time.perf_counter()
    loss_b_full = loss_fn_b(b, x0s, x1s, ts, interpolant)
    loss_eta_full = loss_fn_eta(eta, x0s, x1s, ts, interpolant)
    
    # we weight the loss to focus on masked regions (multiply by mask_loss_weight for masked pixels)
    # we compute per-pixel weight: visible pixels get weight 1.0, masked pixels get weight mask_loss_weight
    loss_weights = masks_flat + mask_loss_weight * (1 - masks_flat)  # we create per-pixel weights
    
    # we apply weighted loss (approximation: we multiply total loss by average weight)
    avg_weight = loss_weights.mean()
    loss_b = loss_b_full * avg_weight
    loss_eta = loss_eta_full * avg_weight
    
    loss_val = loss_b + loss_eta
    loss_end = time.perf_counter()
    
    # we compute the gradient
    backprop_start = time.perf_counter()
    loss_b.backward()
    loss_eta.backward()
    b_grad = torch.tensor([torch.nn.utils.clip_grad_norm_(b.parameters(), float('inf'))])
    eta_grad = torch.tensor([torch.nn.utils.clip_grad_norm_(eta.parameters(), float('inf'))])
    backprop_end = time.perf_counter()
    
    # we perform the update
    update_start = time.perf_counter()
    opt_b.step()
    opt_eta.step()
    sched_b.step()
    sched_eta.step()
    update_end = time.perf_counter()
    
    if counter < 5:
        print(f'[loss: {loss_end - loss_start:.4f}s], [backprop: {backprop_end-backprop_start:.4f}s], [update: {update_end-update_start:.4f}s]')
    
    return loss_val.detach(), loss_b.detach(), loss_eta.detach(), b_grad.detach(), eta_grad.detach()

# we define visualization function
def make_plots(
    b: torch.nn.Module,
    eta: torch.nn.Module,
    interpolant: stochastic_interpolant.Interpolant,
    counter: int,
    data_dict: dict,
    patch_size: int,
    num_patches: int
):
    """we make plots to visualize reconstruction results"""
    print(f"\nepoch: {counter}")
    
    # we get a batch for visualization
    vis_bs = 8
    x1s_img = get_cifar_batch(vis_bs)
    masks = create_patch_mask(vis_bs, patch_size=patch_size, num_patches=num_patches)
    
    # we create masked images
    noise = torch.randn_like(x1s_img) * (1 - masks)
    x0s_img = x1s_img * masks + noise
    
    # we reconstruct using probability flow
    x0s = x0s_img.reshape(vis_bs, -1)
    x1s = x1s_img.reshape(vis_bs, -1)
    
    # we use simple forward integration
    with torch.no_grad():
        # we set masks for conditioning during generation
        b.set_mask(masks)
        eta.set_mask(masks)
        
        s = stochastic_interpolant.SFromEta(eta, interpolant.a)
        pflow = stochastic_interpolant.PFlowIntegrator(
            b=b, method='dopri5', interpolant=interpolant, n_step=10
        )
        xfs_pflow, _ = pflow.rollout(x0s)
        xf_pflow_raw = xfs_pflow[-1].reshape(vis_bs, 3, 32, 32)
        
        # we force visible pixels to remain fixed (only reconstruct masked regions)
        xf_pflow = xf_pflow_raw * (1 - masks) + x1s_img * masks  # we keep original pixels where mask=1
    
    # we plot results
    fig, axes = plt.subplots(3, vis_bs, figsize=(vis_bs*2, 6))
    
    for i in range(vis_bs):
        # we denormalize images for visualization
        def denorm(img):
            return img * 0.5 + 0.5
        
        # we show original image
        axes[0, i].imshow(np.transpose(grab(denorm(x1s_img[i])), (1, 2, 0)))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('original', fontsize=10)
        
        # we show masked image
        axes[1, i].imshow(np.transpose(grab(denorm(x0s_img[i])), (1, 2, 0)))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('masked', fontsize=10)
        
        # we show reconstructed image
        axes[2, i].imshow(np.transpose(grab(denorm(xf_pflow[i])), (1, 2, 0)))
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('reconstructed', fontsize=10)
    
    plt.suptitle(f'dog patch reconstruction - epoch {counter}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'dog_reconstruction_epoch_{counter}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # we plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = np.arange(len(data_dict['losses'])) * metrics_freq
    
    # we plot losses
    axes[0].plot(epochs, data_dict['losses'], label='total loss', linewidth=2)
    axes[0].plot(epochs, data_dict['b_losses'], label='b loss', alpha=0.7)
    axes[0].plot(epochs, data_dict['eta_losses'], label='eta loss', alpha=0.7)
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].set_title('training loss (dog class)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # we plot gradients
    axes[1].plot(epochs, data_dict['b_grads'], label='b grad norm', linewidth=2)
    axes[1].plot(epochs, data_dict['eta_grads'], label='eta grad norm', linewidth=2)
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('gradient norm')
    axes[1].set_title('gradient norms')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # we plot learning rates
    axes[2].plot(epochs, data_dict['lrs'], label='learning rate', linewidth=2)
    axes[2].set_xlabel('epoch')
    axes[2].set_ylabel('learning rate')
    axes[2].set_title('learning rate schedule')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'dog_training_curves_epoch_{counter}.png', dpi=150, bbox_inches='tight')
    plt.show()

# we define main training setup
if __name__ == "__main__":
    # we set hyperparameters
    base_lr = 1e-4
    batch_size = 32
    n_epochs = 5000
    patch_size = 8
    num_patches = 4
    metrics_freq = 100
    plot_freq = 500
    
    print(f"\nhyperparameters:")
    print(f"  class: dog (cifar10 class 5)")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {base_lr}")
    print(f"  n_epochs: {n_epochs}")
    print(f"  patch_size: {patch_size}")
    print(f"  num_patches: {num_patches}")
    print(f"\nconditioning approach:")
    print(f"  - networks conditioned on binary mask (visible=1, masked=0)")
    print(f"  - visible pixels kept fixed during generation")
    print(f"  - only masked regions are reconstructed")
    
    # we define interpolant (one-sided linear interpolation)
    path = 'one-sided-linear'
    interpolant = stochastic_interpolant.Interpolant(path=path, gamma_type=None)
    print(f"\nusing interpolant: {path}")
    
    # we define loss functions
    loss_fn_b = stochastic_interpolant.make_loss(
        method='shared', interpolant=interpolant, loss_type='one-sided-b'
    )
    loss_fn_eta = stochastic_interpolant.make_loss(
        method='shared', interpolant=interpolant, loss_type='one-sided-eta'
    )
    
    # we create networks
    print("\ncreating u-net architectures with mask conditioning...")
    unet_b = UNetDenoiser(in_channels=5, out_channels=3, base_channels=64)  # we add mask channel
    unet_eta = UNetDenoiser(in_channels=5, out_channels=3, base_channels=64)  # we add mask channel
    
    b = VelocityNetwork(unet_b).to(itf.util.get_torch_device())
    eta = EtaNetwork(unet_eta).to(itf.util.get_torch_device())
    
    # we count parameters
    n_params_b = sum(p.numel() for p in b.parameters() if p.requires_grad)
    n_params_eta = sum(p.numel() for p in eta.parameters() if p.requires_grad)
    print(f"b network parameters: {n_params_b:,}")
    print(f"eta network parameters: {n_params_eta:,}")
    
    # we create optimizers and schedulers
    opt_b = torch.optim.Adam(b.parameters(), lr=base_lr)
    opt_eta = torch.optim.Adam(eta.parameters(), lr=base_lr)
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=n_epochs, eta_min=base_lr*0.01)
    sched_eta = torch.optim.lr_scheduler.CosineAnnealingLR(opt_eta, T_max=n_epochs, eta_min=base_lr*0.01)
    
    # we initialize data dictionary
    data_dict = {
        'losses': [],
        'b_losses': [],
        'eta_losses': [],
        'b_grads': [],
        'eta_grads': [],
        'lrs': []
    }
    
    # we train the model
    print("\nstarting training...\n")
    counter = 1
    for epoch in range(n_epochs):
        loss, b_loss, eta_loss, b_grad, eta_grad = train_step(
            batch_size, interpolant, opt_b, opt_eta, sched_b, sched_eta,
            patch_size, num_patches
        )
        
        # we log metrics
        if (counter - 1) % metrics_freq == 0:
            data_dict['losses'].append(grab(loss).item())
            data_dict['b_losses'].append(grab(b_loss).item())
            data_dict['eta_losses'].append(grab(eta_loss).item())
            data_dict['b_grads'].append(grab(b_grad).item())
            data_dict['eta_grads'].append(grab(eta_grad).item())
            data_dict['lrs'].append(opt_b.param_groups[0]['lr'])
            
            print(f"epoch {counter}: loss={grab(loss).item():.4f}, b_loss={grab(b_loss).item():.4f}, eta_loss={grab(eta_loss).item():.4f}")
        
        # we make plots
        if (counter - 1) % plot_freq == 0:
            make_plots(b, eta, interpolant, counter, data_dict, patch_size, num_patches)
            
            # we save checkpoints
            torch.save({
                'epoch': counter,
                'b_state_dict': b.state_dict(),
                'eta_state_dict': eta.state_dict(),
                'opt_b_state_dict': opt_b.state_dict(),
                'opt_eta_state_dict': opt_eta.state_dict(),
                'data_dict': data_dict,
                'class': 'dog',
                'class_id': 5
            }, f'dog_checkpoint_epoch_{counter}.pt')
            print(f"saved checkpoint at epoch {counter}")
        
        counter += 1
    
    print("\ntraining complete!")