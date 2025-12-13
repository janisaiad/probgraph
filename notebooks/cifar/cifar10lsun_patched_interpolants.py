import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
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

# we load lsun dataset (bedroom class)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # we resize lsun images to 32x32
    transforms.ToTensor(),  # we convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # we normalize channels
])

trainset = torchvision.datasets.LSUN(
    root="../../data/lsun",  # we set lsun root directory
    classes=["bedroom_train"],  # we use bedroom training subset
    transform=transform  # we apply preprocessing transforms
)
print(f"\nloaded lsun bedroom_train: {len(trainset)} images")  # we log lsun subset size

# we create data iterator that only samples from lsun bedroom images
def get_cifar_batch(bs):
    """we get a batch of lsun bedroom images only"""
    indices = torch.randint(0, len(trainset), (bs,))  # we sample random indices
    imgs = torch.stack([trainset[i][0] for i in indices])  # we stack selected images
    return imgs.to(itf.util.get_torch_device())  # we move batch to device

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

class SinusoidalTimeEmbedding(nn.Module):
    """we build sinusoidal positional embeddings for scalar time"""  # we describe time embedding
    def __init__(self, embedding_dim: int, max_period: float = 10000.0) -> None:
        super().__init__()  # we call parent constructor
        self.embedding_dim = embedding_dim  # we store embedding dimension
        self.max_period = max_period  # we store maximum period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """we map scalar timesteps to sinusoidal embeddings"""  # we describe forward
        if not isinstance(t, torch.Tensor):  # we convert non-tensor to tensor
            t_tensor = torch.tensor(t, dtype=torch.float32)  # we build tensor from scalar or array
        else:
            t_tensor = t.float()  # we cast to float
        if t_tensor.dim() == 0:  # we handle scalar time
            t_tensor = t_tensor[None]  # we add batch dimension
        if t_tensor.dim() == 2 and t_tensor.shape[1] == 1:  # we squeeze singleton feature dimension
            t_tensor = t_tensor[:, 0]  # we reduce to shape [bs]
        device = t_tensor.device  # we get device
        half_dim = self.embedding_dim // 2  # we compute half dimension
        if half_dim < 1:  # we guard against invalid configuration
            raise ValueError("we expect embedding_dim to be at least 2")  # we raise error for tiny dims
        exponent = -torch.log(torch.tensor(self.max_period, device=device)) / float(half_dim - 1)  # we compute exponent
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * exponent)  # we build frequencies
        args = t_tensor.view(-1, 1) * freqs.view(1, -1)  # we build outer product of time and frequencies
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # we concatenate sin and cos
        if self.embedding_dim % 2 == 1:  # we pad for odd dimension
            emb = torch.nn.functional.pad(emb, (0, 1))  # we add one zero dimension
        return emb  # we return embeddings


class TimeResBlock(nn.Module):
    """we define residual block modulated by time embedding"""  # we describe residual block
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, stride: int = 1) -> None:
        super().__init__()  # we call parent constructor
        self.in_channels = in_channels  # we store input channels
        self.out_channels = out_channels  # we store output channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)  # we define first conv
        self.norm1 = nn.GroupNorm(8, out_channels)  # we define first group norm
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)  # we define second conv
        self.norm2 = nn.GroupNorm(8, out_channels)  # we define second group norm
        self.act = nn.ReLU()  # we define activation
        self.time_mlp = nn.Linear(time_dim, out_channels)  # we define linear layer for time embedding
        if stride != 1 or in_channels != out_channels:  # we check if skip connection must project
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride)  # we define skip projection
        else:
            self.skip = nn.Identity()  # we define identity skip
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """we apply residual block given time embedding"""  # we describe forward
        h = self.conv1(x)  # we apply first convolution
        h = self.norm1(h)  # we normalize
        h = self.act(h)  # we activate
        time_out = self.time_mlp(t_emb).view(t_emb.shape[0], self.out_channels, 1, 1)  # we project time embedding
        h = h + time_out  # we inject time information
        h = self.conv2(h)  # we apply second convolution
        h = self.norm2(h)  # we normalize
        h = self.act(h)  # we activate
        return h + self.skip(x)  # we add residual connection


# we define u-net style convolutional denoiser for image reconstruction
class UNetDenoiser(nn.Module):
    """we use u-net architecture with time-conditioned residual blocks for image reconstruction"""  # we describe unet
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64, time_dim: int = 256) -> None:
        super().__init__()  # we call parent constructor
        
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)  # we build sinusoidal time encoder
        self.time_mlp = nn.Sequential(  # we refine time embedding
            nn.Linear(time_dim, time_dim),  # we apply linear projection
            nn.SiLU(),  # we apply nonlinearity
            nn.Linear(time_dim, time_dim),  # we apply second projection
        )  # we define time mlp
        
        # we define encoder (downsampling path) with time-conditioned residual blocks
        self.enc1 = TimeResBlock(in_channels, base_channels, time_dim, stride=1)  # we define first encoder block
        self.enc2 = TimeResBlock(base_channels, base_channels * 2, time_dim, stride=2)  # we define second encoder block
        self.enc3 = TimeResBlock(base_channels * 2, base_channels * 4, time_dim, stride=2)  # we define third encoder block
        
        # we define bottleneck
        self.bottleneck = TimeResBlock(base_channels * 4, base_channels * 8, time_dim, stride=2)  # we define bottleneck block
        
        # we define decoder (upsampling path) with skip connections and time-conditioned residual blocks
        self.dec3_up = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, 3, stride=2, padding=1, output_padding=1
        )  # we upsample from bottleneck
        self.dec3_block = TimeResBlock(base_channels * 8, base_channels * 4, time_dim, stride=1)  # we refine with skip from enc3
        
        self.dec2_up = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 3, stride=2, padding=1, output_padding=1
        )  # we upsample
        self.dec2_block = TimeResBlock(base_channels * 4, base_channels * 2, time_dim, stride=1)  # we refine with skip from enc2
        
        self.dec1_up = nn.ConvTranspose2d(
            base_channels * 2, base_channels, 3, stride=2, padding=1, output_padding=1
        )  # we upsample
        self.dec1_block = TimeResBlock(base_channels * 2, base_channels, time_dim, stride=1)  # we refine with skip from enc1
        
        # we define final output layer
        self.final = nn.Conv2d(base_channels, out_channels, 1)  # we project to rgb
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """we forward pass with time-conditioned residual blocks"""  # we describe forward
        t_emb = self.time_embedding(t.to(x.device))  # we build sinusoidal time embedding
        t_emb = self.time_mlp(t_emb)  # we refine time embedding
        
        # we encode
        e1 = self.enc1(x, t_emb)  # we apply first encoder block
        e2 = self.enc2(e1, t_emb)  # we apply second encoder block
        e3 = self.enc3(e2, t_emb)  # we apply third encoder block
        
        # we process bottleneck
        b = self.bottleneck(e3, t_emb)  # we apply bottleneck block
        
        # we decode with skip connections
        d3 = self.dec3_up(b)  # we upsample from bottleneck
        d3 = torch.cat([d3, e3], dim=1)  # we concatenate skip connection from encoder level 3
        d3 = self.dec3_block(d3, t_emb)  # we apply decoder block 3
        
        d2 = self.dec2_up(d3)  # we upsample
        d2 = torch.cat([d2, e2], dim=1)  # we concatenate skip from encoder level 2
        d2 = self.dec2_block(d2, t_emb)  # we apply decoder block 2
        
        d1 = self.dec1_up(d2)  # we upsample
        d1 = torch.cat([d1, e1], dim=1)  # we concatenate skip from encoder level 1
        d1 = self.dec1_block(d1, t_emb)  # we apply decoder block 1
        
        # we output final reconstruction
        out = self.final(d1)  # we map to rgb
        return out  # we return reconstructed image

# we define wrapper for eta network to match expected interface
class EtaNetwork(nn.Module):
    """we wrap unet to accept flattened x and scalar time inputs"""  # we describe eta wrapper
    def __init__(self, unet: nn.Module) -> None:
        super().__init__()  # we call parent constructor
        self.unet = unet  # we store underlying unet
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """we expect x of shape [bs, 3*32*32] and scalar or batched t"""  # we describe forward
        bs = x.shape[0]  # we get batch size
        x_img = x.reshape(bs, 3, 32, 32)  # we reshape flat image to 4d tensor
        out = self.unet(x_img, t)  # we run time-conditioned unet
        return out.reshape(bs, -1)  # we flatten output

# we define velocity field wrapper
class VelocityNetwork(nn.Module):
    """we wrap unet for velocity field b with flattened x and scalar time inputs"""  # we describe velocity wrapper
    def __init__(self, unet: nn.Module) -> None:
        super().__init__()  # we call parent constructor
        self.unet = unet  # we store underlying unet
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """we expect x of shape [bs, 3*32*32] and scalar or batched t"""  # we describe forward
        bs = x.shape[0]  # we get batch size
        x_img = x.reshape(bs, 3, 32, 32)  # we reshape flat image to 4d tensor
        out = self.unet(x_img, t)  # we run time-conditioned unet
        return out.reshape(bs, -1)  # we flatten output

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
    
    # we compute the losses
    loss_start = time.perf_counter()
    loss_b = loss_fn_b(b, x0s, x1s, ts, interpolant)
    loss_eta = loss_fn_eta(eta, x0s, x1s, ts, interpolant)
    
    # we add extra weight to loss on masked regions
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
        s = stochastic_interpolant.SFromEta(eta, interpolant.a)
        pflow = stochastic_interpolant.PFlowIntegrator(
            b=b, method='dopri5', interpolant=interpolant, n_step=10
        )
        xfs_pflow, _ = pflow.rollout(x0s)
        xf_pflow = xfs_pflow[-1].reshape(vis_bs, 3, 32, 32)
    
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
    results_dir = os.path.join("vresults", "cifarpatchedinterpolants")  # we define directory for saving figures
    os.makedirs(results_dir, exist_ok=True)  # we ensure results directory exists
    recon_path = os.path.join(results_dir, f"dog_reconstruction_epoch_{counter}.png")  # we build reconstruction path
    plt.savefig(recon_path, dpi=150, bbox_inches="tight")
    plt.show()
    
    # we plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = np.arange(len(data_dict["losses"])) * metrics_freq
    
    # we plot losses
    axes[0].plot(epochs, data_dict["losses"], label="total loss", linewidth=2)
    axes[0].plot(epochs, data_dict["b_losses"], label="b loss", alpha=0.7)
    axes[0].plot(epochs, data_dict["eta_losses"], label="eta loss", alpha=0.7)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("training loss (dog class)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # we plot gradients
    axes[1].plot(epochs, data_dict["b_grads"], label="b grad norm", linewidth=2)
    axes[1].plot(epochs, data_dict["eta_grads"], label="eta grad norm", linewidth=2)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("gradient norm")
    axes[1].set_title("gradient norms")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # we plot learning rates
    axes[2].plot(epochs, data_dict["lrs"], label="learning rate", linewidth=2)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("learning rate")
    axes[2].set_title("learning rate schedule")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    path_name = interpolant.path  # we get interpolant path name
    gamma_name = getattr(interpolant, "gamma_type", "none")  # we get gamma schedule name
    
    plt.tight_layout()
    results_dir = os.path.join("vresults", "cifarpatchedinterpolants")  # we define directory for saving curves
    os.makedirs(results_dir, exist_ok=True)  # we ensure results directory exists
    curves_path = os.path.join(results_dir, f"dog_training_curves_{path_name}_{gamma_name}_epoch_{counter}.png")  # we build curves path
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.show()
    
# we define main training setup
if __name__ == "__main__":
    # we set hyperparameters
    base_lr = 1e-4  # we set base learning rate
    batch_size = 32  # we set batch size
    n_epochs = 5000  # we set number of epochs
    patch_size = 8  # we set patch size
    num_patches = 4  # we set number of patches
    metrics_freq = 100  # we set metrics logging frequency
    plot_freq = 500  # we set plotting frequency
    
    print("\nhyperparameters:")  # we print header for hyperparameters
    print(f"  class: dog (cifar10 class 5)")  # we print target class
    print(f"  batch_size: {batch_size}")  # we print batch size
    print(f"  learning_rate: {base_lr}")  # we print base learning rate
    print(f"  n_epochs: {n_epochs}")  # we print number of epochs
    print(f"  patch_size: {patch_size}")  # we print patch size
    print(f"  num_patches: {num_patches}")  # we print number of patches
    
    paths = ["linear", "trig", "encoding-decoding"]  # we define two-sided interpolant paths
    gamma_types = ["bsquared", "sinesquared", "sigmoid"]  # we define non-gaussian gamma schedules
    
    for path in paths:
        for gamma_type in gamma_types:
            print(f"\nusing interpolant: path={path}, gamma_type={gamma_type}")  # we log current interpolant config
            
            interpolant = stochastic_interpolant.Interpolant(path=path, gamma_type=gamma_type)  # we create interpolant
            interpolant.gamma_type = gamma_type  # we store gamma type on interpolant for logging
            
            # we define loss functions for two-sided stochastic interpolant
            loss_fn_b = stochastic_interpolant.make_loss(
                method="shared", interpolant=interpolant, loss_type="b"
            )
            loss_fn_eta = stochastic_interpolant.make_loss(
                method="shared", interpolant=interpolant, loss_type="eta"
            )
            
            # we create networks
            print("\ncreating u-net architectures...")  # we notify network instantiation
            unet_b = UNetDenoiser(in_channels=3, out_channels=3, base_channels=64)  # we create time-conditioned unet for b
            unet_eta = UNetDenoiser(in_channels=3, out_channels=3, base_channels=64)  # we create time-conditioned unet for eta
            
            b = VelocityNetwork(unet_b).to(itf.util.get_torch_device())  # we move b network to device
            eta = EtaNetwork(unet_eta).to(itf.util.get_torch_device())  # we move eta network to device
            
            # we count parameters
            n_params_b = sum(p.numel() for p in b.parameters() if p.requires_grad)  # we count trainable params of b
            n_params_eta = sum(p.numel() for p in eta.parameters() if p.requires_grad)  # we count trainable params of eta
            print(f"b network parameters: {n_params_b:,}")  # we print number of parameters of b
            print(f"eta network parameters: {n_params_eta:,}")  # we print number of parameters of eta
            
            # we create optimizers and schedulers
            opt_b = torch.optim.Adam(b.parameters(), lr=base_lr)  # we create optimizer for b
            opt_eta = torch.optim.Adam(eta.parameters(), lr=base_lr)  # we create optimizer for eta
            sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_b, T_max=n_epochs, eta_min=base_lr * 0.01
            )  # we create lr scheduler for b
            sched_eta = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_eta, T_max=n_epochs, eta_min=base_lr * 0.01
            )  # we create lr scheduler for eta
            
            # we initialize data dictionary
            data_dict = {
                "losses": [],
                "b_losses": [],
                "eta_losses": [],
                "b_grads": [],
                "eta_grads": [],
                "lrs": [],
            }  # we store training metrics
            
            # we train the model for this configuration
            print("\nstarting training...\n")  # we announce training start
            counter = 1  # we initialize iteration counter
            for epoch in range(n_epochs):
                loss, b_loss, eta_loss, b_grad, eta_grad = train_step(
                    batch_size, interpolant, opt_b, opt_eta, sched_b, sched_eta, patch_size, num_patches
                )  # we perform one training step
                
                # we log metrics
                if (counter - 1) % metrics_freq == 0:
                    data_dict["losses"].append(grab(loss).item())  # we log total loss
                    data_dict["b_losses"].append(grab(b_loss).item())  # we log b loss
                    data_dict["eta_losses"].append(grab(eta_loss).item())  # we log eta loss
                    data_dict["b_grads"].append(grab(b_grad).item())  # we log gradient norm of b
                    data_dict["eta_grads"].append(grab(eta_grad).item())  # we log gradient norm of eta
                    data_dict["lrs"].append(opt_b.param_groups[0]["lr"])  # we log learning rate
                    
                    print(
                        f"[path={path} | gamma_type={gamma_type}] epoch {counter}: "
                        f"loss={grab(loss).item():.4f}, "
                        f"b_loss={grab(b_loss).item():.4f}, "
                        f"eta_loss={grab(eta_loss).item():.4f}"
                    )  # we print training status
                
                # we make plots and save checkpoints
                if (counter - 1) % plot_freq == 0:
                    make_plots(b, eta, interpolant, counter, data_dict, patch_size, num_patches)  # we visualize results
                    
                    results_dir = os.path.join("vresults", "cifarpatchedinterpolants")  # we define directory for checkpoints
                    os.makedirs(results_dir, exist_ok=True)  # we ensure results directory exists
                    ckpt_name = os.path.join(
                        results_dir,
                        f"dog_checkpoint_{path}_{gamma_type}_epoch_{counter}.pt",
                    )  # we build checkpoint filename
                    torch.save(
                        {
                            "epoch": counter,
                            "b_state_dict": b.state_dict(),
                            "eta_state_dict": eta.state_dict(),
                            "opt_b_state_dict": opt_b.state_dict(),
                            "opt_eta_state_dict": opt_eta.state_dict(),
                            "data_dict": data_dict,
                            "class": "dog",
                            "class_id": 5,
                            "path": path,
                            "gamma_type": gamma_type,
                        },
                        ckpt_name,
                    )  # we save checkpoint
                    print(
                        f"saved checkpoint at epoch {counter} for path={path}, gamma_type={gamma_type}"
                    )  # we log checkpoint saving
                
                counter += 1  # we increment iteration counter
            
            print(f"\ntraining complete for path={path}, gamma_type={gamma_type}\n")  # we mark configuration as finished