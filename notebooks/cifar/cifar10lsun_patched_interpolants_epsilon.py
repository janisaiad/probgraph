import os  # we import os for filesystem operations
import sys  # we import sys to modify path
import time  # we import time for timing
from typing import Any  # we import Any for type hints

import matplotlib.pyplot as plt  # we import matplotlib for plotting
import numpy as np  # we import numpy for array operations
import torch  # we import torch for tensors
import torch.nn as nn  # we import torch.nn for neural networks
import torchvision  # we import torchvision for datasets
import torchvision.transforms as transforms  # we import transforms for data preprocessing
import urllib.request  # we import urllib to download files
import zipfile  # we import zipfile to extract archives

sys.path.append("../../")  # we add project root to python path

import interflow as itf  # we import interflow utilities
import interflow.stochastic_interpolant as stochastic_interpolant  # we import stochastic interpolant tools
from interflow import fabrics  # we import fabrics to build gamma schedules


if torch.cuda.is_available():  # we select cuda if available
    print("cuda available, setting default tensor residence to gpu")  # we log cuda availability
    itf.util.set_torch_device("cuda")  # we set torch device to cuda
else:
    print("no cuda device found")  # we log absence of cuda
print(itf.util.get_torch_device())  # we print currently selected device

print(f"torch version: {torch.__version__}")  # we print torch version


# we define utility functions
def grab(var: torch.Tensor) -> np.ndarray:
    """we take a tensor off the gpu and convert it to a numpy array on the cpu"""  # we document grab
    return var.detach().cpu().numpy()  # we convert tensor to numpy array


# we load lsun dataset (bedroom class)
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),  # we resize lsun images to 32x32
        transforms.ToTensor(),  # we convert pil image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # we normalize channels
    ]
)  # we compose transforms


def ensure_lsun_bedroom(lsun_root: str) -> None:
    """we download and extract lsun bedroom_train_lmdb if missing"""  # we document lsun helper
    bedroom_lmdb_dir = os.path.join(lsun_root, "bedroom_train_lmdb")  # we build expected lmdb directory
    if os.path.isdir(bedroom_lmdb_dir):  # we check if directory already exists
        print(f"found existing lsun bedroom_train_lmdb at {bedroom_lmdb_dir}")  # we log existing dataset
        return  # we skip download when present
    
    os.makedirs(lsun_root, exist_ok=True)  # we ensure root directory exists
    url = "https://dl.yf.io/lsun/scenes/bedroom_train_lmdb.zip"  # we set official lsun download url
    zip_path = os.path.join(lsun_root, "bedroom_train_lmdb.zip")  # we choose local zip path
    
    print(f"downloading lsun bedroom_train_lmdb from {url} to {zip_path} ...")  # we log download start
    urllib.request.urlretrieve(url, zip_path)  # we download zip archive
    print("download complete, extracting archive...")  # we log extraction start
    
    with zipfile.ZipFile(zip_path, "r") as zf:  # we open zip file
        zf.extractall(lsun_root)  # we extract all contents into root
    
    os.remove(zip_path)  # we remove zip file after extraction
    print(f"lsun bedroom_train_lmdb extracted to {bedroom_lmdb_dir}")  # we log extraction location


lsun_root = "../../data/lsun"  # we set lsun root directory
ensure_lsun_bedroom(lsun_root)  # we ensure bedroom lmdb is available

trainset = torchvision.datasets.LSUN(
    root=lsun_root, classes=["bedroom_train"], transform=transform
)  # we load lsun bedroom training set
print(f"\nloaded lsun bedroom_train: {len(trainset)} images")  # we log lsun subset size


# we create data iterator that only samples from lsun bedroom images
def get_cifar_batch(bs: int) -> torch.Tensor:
    """we get a batch of lsun bedroom images only"""  # we document get_cifar_batch
    indices = torch.randint(0, len(trainset), (bs,))  # we sample random indices
    imgs = torch.stack([trainset[i][0] for i in indices])  # we stack selected images
    return imgs.to(itf.util.get_torch_device())  # we move batch to device


# we create masking function for patches
def create_patch_mask(bs: int, patch_size: int = 8, num_patches: int = 4) -> torch.Tensor:
    """we create random patch masks, 1 for visible pixels, 0 for masked patches"""  # we document create_patch_mask
    mask = torch.ones(bs, 3, 32, 32)  # we start with all ones
    for i in range(bs):  # we iterate over batch
        for _ in range(num_patches):  # we add a number of patches
            x = torch.randint(0, 32 - patch_size, (1,)).item()  # we choose x coordinate
            y = torch.randint(0, 32 - patch_size, (1,)).item()  # we choose y coordinate
            mask[i, :, x : x + patch_size, y : y + patch_size] = 0  # we mask the patch
    return mask.to(itf.util.get_torch_device())  # we move mask to device


# we define u-net style convolutional denoiser for image reconstruction
class UNetDenoiser(nn.Module):
    """we use u-net architecture with skip connections for image reconstruction"""  # we describe unet

    def __init__(self, in_channels: int = 4, out_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()  # we call parent constructor

        # we define encoder (downsampling path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),  # we define first conv layer
            nn.GroupNorm(8, base_channels),  # we normalize features
            nn.ReLU(),  # we apply nonlinearity
            nn.Conv2d(base_channels, base_channels, 3, padding=1),  # we define second conv
            nn.GroupNorm(8, base_channels),  # we normalize again
            nn.ReLU(),  # we add nonlinearity
        )  # we build first encoder block

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),  # we downsample
            nn.GroupNorm(8, base_channels * 2),  # we normalize
            nn.ReLU(),  # we activate
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),  # we refine features
            nn.GroupNorm(8, base_channels * 2),  # we normalize
            nn.ReLU(),  # we activate
        )  # we build second encoder block

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),  # we downsample deeper
            nn.GroupNorm(8, base_channels * 4),  # we normalize
            nn.ReLU(),  # we activate
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),  # we refine
            nn.GroupNorm(8, base_channels * 4),  # we normalize
            nn.ReLU(),  # we activate
        )  # we build third encoder block

        # we define bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),  # we downsample to bottleneck
            nn.GroupNorm(8, base_channels * 8),  # we normalize
            nn.ReLU(),  # we activate
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),  # we refine
            nn.GroupNorm(8, base_channels * 8),  # we normalize
            nn.ReLU(),  # we activate
        )  # we build bottleneck block

        # we define decoder (upsampling path) with skip connections
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 8, base_channels * 4, 3, stride=2, padding=1, output_padding=1
            ),  # we upsample
            nn.GroupNorm(8, base_channels * 4),  # we normalize
            nn.ReLU(),  # we activate
        )  # we build first decoder upsample block
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),  # we fuse skip connection
            nn.GroupNorm(8, base_channels * 4),  # we normalize
            nn.ReLU(),  # we activate
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),  # we refine
            nn.GroupNorm(8, base_channels * 4),  # we normalize
            nn.ReLU(),  # we activate
        )  # we build first decoder conv block

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, 3, stride=2, padding=1, output_padding=1
            ),  # we upsample
            nn.GroupNorm(8, base_channels * 2),  # we normalize
            nn.ReLU(),  # we activate
        )  # we build second decoder upsample block
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),  # we fuse skip connection
            nn.GroupNorm(8, base_channels * 2),  # we normalize
            nn.ReLU(),  # we activate
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),  # we refine
            nn.GroupNorm(8, base_channels * 2),  # we normalize
            nn.ReLU(),  # we activate
        )  # we build second decoder conv block

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 3, stride=2, padding=1, output_padding=1),  # we upsample
            nn.GroupNorm(8, base_channels),  # we normalize
            nn.ReLU(),  # we activate
        )  # we build third decoder upsample block
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),  # we fuse skip connection
            nn.GroupNorm(8, base_channels),  # we normalize
            nn.ReLU(),  # we activate
            nn.Conv2d(base_channels, base_channels, 3, padding=1),  # we refine
            nn.GroupNorm(8, base_channels),  # we normalize
            nn.ReLU(),  # we activate
        )  # we build third decoder conv block

        # we define final output layer
        self.final = nn.Conv2d(base_channels, out_channels, 1)  # we map features to rgb channels

    def forward(self, x_with_t: torch.Tensor) -> torch.Tensor:
        """we forward pass with skip connections"""  # we describe forward
        # we encode
        e1 = self.enc1(x_with_t)  # we apply first encoder block
        e2 = self.enc2(e1)  # we apply second encoder block
        e3 = self.enc3(e2)  # we apply third encoder block

        # we process bottleneck
        b = self.bottleneck(e3)  # we compute bottleneck features

        # we decode with skip connections
        d3 = self.dec3(b)  # we upsample from bottleneck
        d3 = torch.cat([d3, e3], dim=1)  # we concatenate skip from encoder level 3
        d3 = self.dec3_conv(d3)  # we refine fused features

        d2 = self.dec2(d3)  # we upsample
        d2 = torch.cat([d2, e2], dim=1)  # we concatenate skip from encoder level 2
        d2 = self.dec2_conv(d2)  # we refine

        d1 = self.dec1(d2)  # we upsample
        d1 = torch.cat([d1, e1], dim=1)  # we concatenate skip from encoder level 1
        d1 = self.dec1_conv(d1)  # we refine

        # we output final reconstruction
        out = self.final(d1)  # we map to rgb
        return out  # we return reconstructed image


# we define wrapper for eta network to match expected interface
class EtaNetwork(nn.Module):
    """we wrap unet to accept separate (x, t) inputs"""  # we describe eta wrapper

    def __init__(self, unet: nn.Module) -> None:
        super().__init__()  # we call parent constructor
        self.unet = unet  # we store underlying unet

    def forward(self, x: torch.Tensor, t: Any) -> torch.Tensor:
        """we expect x of shape [bs, 3*32*32] and scalar or batched t"""  # we describe eta forward
        bs = x.shape[0]  # we get batch size
        x_img = x.reshape(bs, 3, 32, 32)  # we reshape flat image to 4d tensor

        if not isinstance(t, torch.Tensor):  # we convert non-tensor time to tensor
            t_tensor = torch.tensor(t, device=x.device, dtype=x.dtype)  # we build tensor from scalar or array
        else:
            t_tensor = t.to(x.device)  # we move time tensor to same device as x

        if t_tensor.dim() == 0:  # we handle scalar time
            t_batch = t_tensor.repeat(bs).unsqueeze(1)  # we repeat scalar t over batch
        elif t_tensor.dim() == 1:  # we handle vector time
            if t_tensor.shape[0] == 1 and bs > 1:  # we broadcast single time value to batch
                t_batch = t_tensor.repeat(bs).unsqueeze(1)  # we repeat along batch dimension
            else:
                t_batch = t_tensor.unsqueeze(1)  # we add feature dimension
        elif t_tensor.dim() == 2:  # we handle matrix-shaped time
            if t_tensor.shape[0] == 1 and bs > 1:  # we broadcast row to batch
                t_batch = t_tensor.repeat(bs, 1)  # we repeat row across batch
            else:
                t_batch = t_tensor  # we assume proper shape [bs, 1]
        else:
            raise ValueError("we expect time tensor of dimension at most 2")  # we guard against unsupported shapes

        t_channel = t_batch.view(bs, 1, 1, 1).expand(bs, 1, 32, 32)  # we broadcast time as extra channel
        x_with_t = torch.cat([x_img, t_channel], dim=1)  # we concatenate time as 4th channel

        out = self.unet(x_with_t)  # we run unet
        return out.reshape(bs, -1)  # we flatten output


# we define velocity field wrapper
class VelocityNetwork(nn.Module):
    """we wrap unet for velocity field b with separate (x, t) inputs"""  # we describe velocity wrapper

    def __init__(self, unet: nn.Module) -> None:
        super().__init__()  # we call parent constructor
        self.unet = unet  # we store underlying unet

    def forward(self, x: torch.Tensor, t: Any) -> torch.Tensor:
        """we expect x of shape [bs, 3*32*32] and scalar or batched t"""  # we describe velocity forward
        bs = x.shape[0]  # we get batch size
        x_img = x.reshape(bs, 3, 32, 32)  # we reshape flat image to 4d tensor

        if not isinstance(t, torch.Tensor):  # we convert non-tensor time to tensor
            t_tensor = torch.tensor(t, device=x.device, dtype=x.dtype)  # we build tensor from scalar or array
        else:
            t_tensor = t.to(x.device)  # we move time tensor to same device as x

        if t_tensor.dim() == 0:  # we handle scalar time
            t_batch = t_tensor.repeat(bs).unsqueeze(1)  # we repeat scalar t over batch
        elif t_tensor.dim() == 1:  # we handle vector time
            if t_tensor.shape[0] == 1 and bs > 1:  # we broadcast single time value to batch
                t_batch = t_tensor.repeat(bs).unsqueeze(1)  # we repeat along batch dimension
            else:
                t_batch = t_tensor.unsqueeze(1)  # we add feature dimension
        elif t_tensor.dim() == 2:  # we handle matrix-shaped time
            if t_tensor.shape[0] == 1 and bs > 1:  # we broadcast row to batch
                t_batch = t_tensor.repeat(bs, 1)  # we repeat row across batch
            else:
                t_batch = t_tensor  # we assume proper shape [bs, 1]
        else:
            raise ValueError("we expect time tensor of dimension at most 2")  # we guard against unsupported shapes

        t_channel = t_batch.view(bs, 1, 1, 1).expand(bs, 1, 32, 32)  # we broadcast time as extra channel
        x_with_t = torch.cat([x_img, t_channel], dim=1)  # we concatenate time as 4th channel

        out = self.unet(x_with_t)  # we run unet
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
    mask_loss_weight: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """we take a single step of optimization on the training set"""  # we document train_step
    opt_b.zero_grad()  # we reset gradients for b
    opt_eta.zero_grad()  # we reset gradients for eta

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
    _ = masks.reshape(bs, -1)  # we flatten mask too (unused but kept for clarity)

    # we sample random times
    ts = torch.rand(size=(bs,)).to(itf.util.get_torch_device())  # we sample times in [0,1]

    # we compute the losses
    loss_start = time.perf_counter()  # we record loss computation start
    loss_b = loss_fn_b(b, x0s, x1s, ts, interpolant)  # we compute b loss
    loss_eta = loss_fn_eta(eta, x0s, x1s, ts, interpolant)  # we compute eta loss

    # we add extra weight to loss on masked regions (currently unused, kept for compatibility)
    loss_val = loss_b + loss_eta  # we combine losses
    loss_end = time.perf_counter()  # we record loss computation end

    # we compute the gradient
    backprop_start = time.perf_counter()  # we record backprop start
    loss_b.backward()  # we backpropagate through b
    loss_eta.backward()  # we backpropagate through eta
    b_grad = torch.tensor(
        [torch.nn.utils.clip_grad_norm_(b.parameters(), float("inf"))]
    )  # we clip and record b gradient norm
    eta_grad = torch.tensor(
        [torch.nn.utils.clip_grad_norm_(eta.parameters(), float("inf"))]
    )  # we clip and record eta gradient norm
    backprop_end = time.perf_counter()  # we record backprop end

    # we perform the update
    update_start = time.perf_counter()  # we record update start
    opt_b.step()  # we update b
    opt_eta.step()  # we update eta
    sched_b.step()  # we step scheduler for b
    sched_eta.step()  # we step scheduler for eta
    update_end = time.perf_counter()  # we record update end

    if counter < 5:  # we print timings for first few steps
        print(
            f"[loss: {loss_end - loss_start:.4f}s], "
            f"[backprop: {backprop_end - backprop_start:.4f}s], "
            f"[update: {update_end - update_start:.4f}s]"
        )  # we log timing information

    return loss_val.detach(), loss_b.detach(), loss_eta.detach(), b_grad.detach(), eta_grad.detach()  # we return detached values


# we define visualization function
def make_plots(
    b: torch.nn.Module,
    eta: torch.nn.Module,
    interpolant: stochastic_interpolant.Interpolant,
    counter: int,
    data_dict: dict,
    patch_size: int,
    num_patches: int,
) -> None:
    """we make plots to visualize reconstruction results"""  # we document make_plots
    print(f"\nepoch: {counter}")  # we log epoch

    # we get a batch for visualization
    vis_bs = 8  # we set visualization batch size
    x1s_img = get_cifar_batch(vis_bs)  # we sample images
    masks = create_patch_mask(vis_bs, patch_size=patch_size, num_patches=num_patches)  # we create masks

    # we create masked images
    noise = torch.randn_like(x1s_img) * (1 - masks)  # we add noise only in masked regions
    x0s_img = x1s_img * masks + noise  # we create corrupted images

    # we reconstruct using probability flow
    x0s = x0s_img.reshape(vis_bs, -1)  # we flatten starting points
    _ = x1s_img.reshape(vis_bs, -1)  # we flatten targets (unused here)

    # we use simple forward integration
    with torch.no_grad():  # we disable gradients for sampling
        _ = stochastic_interpolant.SFromEta(eta, interpolant.a)  # we build score from eta (unused here but kept)
        pflow = stochastic_interpolant.PFlowIntegrator(
            b=b, method="dopri5", interpolant=interpolant, n_step=10
        )  # we build probability flow integrator
        xfs_pflow, _ = pflow.rollout(x0s)  # we rollout from x0s
        xf_pflow = xfs_pflow[-1].reshape(vis_bs, 3, 32, 32)  # we take final states and reshape

    # we plot results
    fig, axes = plt.subplots(3, vis_bs, figsize=(vis_bs * 2, 6))  # we create figure

    for i in range(vis_bs):  # we iterate over visualization batch
        # we denormalize images for visualization
        def denorm(img: torch.Tensor) -> torch.Tensor:
            return img * 0.5 + 0.5  # we revert normalization

        # we show original image
        axes[0, i].imshow(np.transpose(grab(denorm(x1s_img[i])), (1, 2, 0)))  # we plot original
        axes[0, i].axis("off")  # we hide axes
        if i == 0:  # we add title to first column
            axes[0, i].set_title("original", fontsize=10)  # we set title

        # we show masked image
        axes[1, i].imshow(np.transpose(grab(denorm(x0s_img[i])), (1, 2, 0)))  # we plot corrupted
        axes[1, i].axis("off")  # we hide axes
        if i == 0:  # we add title to first column
            axes[1, i].set_title("masked", fontsize=10)  # we set title

        # we show reconstructed image
        axes[2, i].imshow(np.transpose(grab(denorm(xf_pflow[i])), (1, 2, 0)))  # we plot reconstructions
        axes[2, i].axis("off")  # we hide axes
        if i == 0:  # we add title to first column
            axes[2, i].set_title("reconstructed", fontsize=10)  # we set title

    plt.suptitle(f"dog patch reconstruction - epoch {counter}", fontsize=14, y=1.02)  # we add global title
    plt.tight_layout()  # we adjust layout
    results_dir = os.path.join("vresults", "cifarpatchedinterpolants")  # we define directory for saving figures
    os.makedirs(results_dir, exist_ok=True)  # we ensure results directory exists
    recon_path = os.path.join(
        results_dir,
        f"dog_reconstruction_epoch_{counter}.png",
    )  # we build reconstruction path
    plt.savefig(recon_path, dpi=150, bbox_inches="tight")  # we save reconstruction figure
    plt.show()  # we display figure

    # we plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # we create figure for curves

    epochs = np.arange(len(data_dict["losses"])) * metrics_freq  # we build epoch axis

    # we plot losses
    axes[0].plot(epochs, data_dict["losses"], label="total loss", linewidth=2)  # we plot total loss
    axes[0].plot(epochs, data_dict["b_losses"], label="b loss", alpha=0.7)  # we plot b loss
    axes[0].plot(epochs, data_dict["eta_losses"], label="eta loss", alpha=0.7)  # we plot eta loss
    axes[0].set_xlabel("epoch")  # we label x axis
    axes[0].set_ylabel("loss")  # we label y axis
    axes[0].set_title("training loss (dog class)")  # we set title
    axes[0].legend()  # we add legend
    axes[0].grid(True, alpha=0.3)  # we add grid

    # we plot gradients
    axes[1].plot(epochs, data_dict["b_grads"], label="b grad norm", linewidth=2)  # we plot b gradient norm
    axes[1].plot(epochs, data_dict["eta_grads"], label="eta grad norm", linewidth=2)  # we plot eta gradient norm
    axes[1].set_xlabel("epoch")  # we label x axis
    axes[1].set_ylabel("gradient norm")  # we label y axis
    axes[1].set_title("gradient norms")  # we set title
    axes[1].legend()  # we add legend
    axes[1].grid(True, alpha=0.3)  # we add grid

    # we plot learning rates
    axes[2].plot(epochs, data_dict["lrs"], label="learning rate", linewidth=2)  # we plot learning rate
    axes[2].set_xlabel("epoch")  # we label x axis
    axes[2].set_ylabel("learning rate")  # we label y axis
    axes[2].set_title("learning rate schedule")  # we set title
    axes[2].legend()  # we add legend
    axes[2].grid(True, alpha=0.3)  # we add grid

    path_name = interpolant.path  # we get interpolant path name
    gamma_name = getattr(interpolant, "gamma_type", "none")  # we get gamma schedule name
    epsilon_name = getattr(interpolant, "epsilon", 1.0)  # we get epsilon value

    plt.tight_layout()  # we adjust layout
    os.makedirs(results_dir, exist_ok=True)  # we ensure results directory exists
    curves_path = os.path.join(
        results_dir,
        f"dog_training_curves_{path_name}_{gamma_name}_eps{epsilon_name}_epoch_{counter}.png",
    )  # we build curves path
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")  # we save curves figure
    plt.show()  # we display figure


# we define main training setup with epsilon sweep
if __name__ == "__main__":  # we enter main script
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
    epsilons = [0.1, 0.2, 0.5, 1.0]  # we define epsilon sweep values

    for epsilon in epsilons:  # we loop over epsilon values
        for path in paths:  # we loop over interpolant paths
            for gamma_type in gamma_types:  # we loop over gamma schedules
                print(
                    f"\nusing interpolant: path={path}, gamma_type={gamma_type}, epsilon={epsilon}"
                )  # we log current interpolant config

                base_gamma, base_gamma_dot, base_gg_dot = fabrics.make_gamma(
                    gamma_type=gamma_type
                )  # we get base gamma functions

                def scaled_gamma(t: torch.Tensor, g=base_gamma, eps: float = epsilon) -> torch.Tensor:
                    return eps * g(t)  # we scale gamma amplitude

                def scaled_gamma_dot(t: torch.Tensor, gd=base_gamma_dot, eps: float = epsilon) -> torch.Tensor:
                    return eps * gd(t)  # we scale gamma derivative amplitude

                def scaled_gg_dot(t: torch.Tensor, ggd=base_gg_dot, eps: float = epsilon) -> torch.Tensor:
                    return (eps**2) * ggd(t)  # we scale gamma*gamma_dot amplitude

                interpolant = stochastic_interpolant.Interpolant(
                    path=path,
                    gamma_type=None,
                    gamma=scaled_gamma,
                    gamma_dot=scaled_gamma_dot,
                    gg_dot=scaled_gg_dot,
                )  # we create interpolant with scaled non-gaussian gamma
                interpolant.gamma_type = gamma_type  # we store gamma type on interpolant for logging
                interpolant.epsilon = float(epsilon)  # we store epsilon on interpolant for logging

                # we define loss functions for two-sided stochastic interpolant
                loss_fn_b = stochastic_interpolant.make_loss(
                    method="shared", interpolant=interpolant, loss_type="b"
                )  # we build b loss
                loss_fn_eta = stochastic_interpolant.make_loss(
                    method="shared", interpolant=interpolant, loss_type="eta"
                )  # we build eta loss

                # we create networks
                print("\ncreating u-net architectures...")  # we notify network instantiation
                unet_b = UNetDenoiser(
                    in_channels=4, out_channels=3, base_channels=64
                )  # we create u-net for b
                unet_eta = UNetDenoiser(
                    in_channels=4, out_channels=3, base_channels=64
                )  # we create u-net for eta

                b = VelocityNetwork(unet_b).to(itf.util.get_torch_device())  # we move b network to device
                eta = EtaNetwork(unet_eta).to(itf.util.get_torch_device())  # we move eta network to device

                # we count parameters
                n_params_b = sum(p.numel() for p in b.parameters() if p.requires_grad)  # we count params of b
                n_params_eta = sum(
                    p.numel() for p in eta.parameters() if p.requires_grad
                )  # we count params of eta
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
                data_dict: dict[str, list[float]] = {
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
                for epoch in range(n_epochs):  # we iterate over epochs
                    (
                        loss,
                        b_loss,
                        eta_loss,
                        b_grad,
                        eta_grad,
                    ) = train_step(
                        batch_size,
                        interpolant,
                        opt_b,
                        opt_eta,
                        sched_b,
                        sched_eta,
                        patch_size,
                        num_patches,
                    )  # we perform one training step

                    # we log metrics
                    if (counter - 1) % metrics_freq == 0:  # we log at chosen frequency
                        data_dict["losses"].append(grab(loss).item())  # we log total loss
                        data_dict["b_losses"].append(grab(b_loss).item())  # we log b loss
                        data_dict["eta_losses"].append(grab(eta_loss).item())  # we log eta loss
                        data_dict["b_grads"].append(grab(b_grad).item())  # we log gradient norm of b
                        data_dict["eta_grads"].append(grab(eta_grad).item())  # we log gradient norm of eta
                        data_dict["lrs"].append(opt_b.param_groups[0]["lr"])  # we log learning rate

                        print(
                            f"[path={path} | gamma_type={gamma_type} | eps={epsilon}] "
                            f"epoch {counter}: "
                            f"loss={grab(loss).item():.4f}, "
                            f"b_loss={grab(b_loss).item():.4f}, "
                            f"eta_loss={grab(eta_loss).item():.4f}"
                        )  # we print training status

                    # we make plots and save checkpoints
                    if (counter - 1) % plot_freq == 0:  # we plot at chosen frequency
                        make_plots(
                            b, eta, interpolant, counter, data_dict, patch_size, num_patches
                        )  # we visualize results

                        results_dir = os.path.join(
                            "vresults", "cifarpatchedinterpolants"
                        )  # we define directory for checkpoints
                        os.makedirs(results_dir, exist_ok=True)  # we ensure results directory exists
                        ckpt_name = os.path.join(
                            results_dir,
                            f"dog_checkpoint_{path}_{gamma_type}_eps{epsilon}_epoch_{counter}.pt",
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
                                "epsilon": float(epsilon),
                            },
                            ckpt_name,
                        )  # we save checkpoint
                        print(
                            f"saved checkpoint at epoch {counter} for path={path}, gamma_type={gamma_type}, eps={epsilon}"
                        )  # we log checkpoint saving

                    counter += 1  # we increment iteration counter

                print(
                    f"\ntraining complete for path={path}, gamma_type={gamma_type}, eps={epsilon}\n"
                )  # we mark configuration as finished

