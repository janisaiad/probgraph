import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cifar'))  # we add cifar directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))  # we add project root to path

import interflow as itf
import interflow.stochastic_interpolant as stochastic_interpolant

# we import network architectures from training script
import importlib.util
cifar_module_path = os.path.join(os.path.dirname(__file__), '..', 'cifar', 'cifar10lsun_patched_interpolants.py')
spec = importlib.util.spec_from_file_location("cifar10lsun_patched_interpolants", cifar_module_path)
cifar_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cifar_module)

# we import constants and classes
UNetDenoiser = cifar_module.UNetDenoiser
EtaNetwork = cifar_module.EtaNetwork
VelocityNetwork = cifar_module.VelocityNetwork
NUM_CHANNELS = cifar_module.NUM_CHANNELS
IMAGE_HEIGHT = cifar_module.IMAGE_HEIGHT
IMAGE_WIDTH = cifar_module.IMAGE_WIDTH
IMAGE_SIZE = cifar_module.IMAGE_SIZE
DATASET_NAME = cifar_module.DATASET_NAME

# we define create_patch_mask locally since it uses constants
def create_patch_mask(bs, patch_size=6, num_patches=4):
    """we create random patch masks, 1 for visible pixels, 0 for masked patches"""
    mask = torch.ones(bs, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)  # we initialize mask for rgb images
    for i in range(bs):
        for _ in range(num_patches):
            x = torch.randint(0, max(1, IMAGE_HEIGHT - patch_size + 1), (1,)).item()  # we sample patch row
            y = torch.randint(0, max(1, IMAGE_WIDTH - patch_size + 1), (1,)).item()  # we sample patch column
            mask[i, :, x:x+patch_size, y:y+patch_size] = 0  # we mask the patch
    return mask.to(itf.util.get_torch_device())

# we set paths relative to project root
_script_dir = os.path.dirname(os.path.abspath(__file__))  # we get script directory
_project_root = os.path.join(_script_dir, "..", "..")  # we go up to project root
RESULTS_ROOT = os.path.join(_project_root, "vresults", "cifarpatchedinterpolants")  # we store root for saved results
COMPARISON_ROOT = os.path.join(_project_root, "vresults", "comparisons")  # we store comparison results
os.makedirs(COMPARISON_ROOT, exist_ok=True)  # we ensure comparison directory exists

if torch.cuda.is_available():  # we set device
    print('cuda available, setting default tensor residence to gpu')
    itf.util.set_torch_device('cuda')
else:
    print('no cuda device found')
print(itf.util.get_torch_device())

def grab(var):
    """we take a tensor off the gpu and convert it to a numpy array on the cpu"""
    return var.detach().cpu().numpy()

def find_latest_checkpoints(
    dataset_tag: str,
    path: str,
    gamma_type: str,
    epsilon_tag: str,
    epoch: Optional[int] = None
) -> Optional[str]:
    """we find the latest checkpoint for a given configuration"""
    ckpt_dir = os.path.join(RESULTS_ROOT, dataset_tag, "checkpoints")  # we look in checkpoints directory
    if not os.path.exists(ckpt_dir):  # we check if directory exists
        return None
    
    if epoch is not None:  # we look for specific epoch
        ckpt_pattern = os.path.join(ckpt_dir, f"cifar_checkpoint_{path}_{gamma_type}_epoch_{epoch}.pt")
        if os.path.exists(ckpt_pattern):  # we check if specific checkpoint exists
            return ckpt_pattern
        return None
    
    # we find all checkpoints for this configuration
    pattern = os.path.join(ckpt_dir, f"cifar_checkpoint_{path}_{gamma_type}_epoch_*.pt")
    checkpoints = glob.glob(pattern)  # we list all matching checkpoints
    
    if not checkpoints:  # we check if any found
        return None
    
    # we extract epoch numbers and return latest
    def extract_epoch(ckpt_path: str) -> int:
        basename = os.path.basename(ckpt_path)  # we get filename
        epoch_str = basename.split('_epoch_')[1].split('.pt')[0]  # we extract epoch number
        return int(epoch_str)
    
    checkpoints.sort(key=extract_epoch, reverse=True)  # we sort by epoch descending
    return checkpoints[0]  # we return latest

def load_model_from_checkpoint(ckpt_path: str, device) -> Tuple[nn.Module, nn.Module, Dict]:
    """we load b and eta networks from checkpoint"""
    print(f"loading checkpoint: {ckpt_path}")  # we log checkpoint path
    ckpt = torch.load(ckpt_path, map_location=device)  # we load checkpoint
    
    # we extract configuration
    path = ckpt.get("path", "linear")  # we get path type
    gamma_type = ckpt.get("gamma_type", "bsquared")  # we get gamma type
    
    # we create networks
    unet_b = UNetDenoiser(in_channels=3, out_channels=3, base_channels=64)  # we create unet for b
    unet_eta = UNetDenoiser(in_channels=3, out_channels=3, base_channels=64)  # we create unet for eta
    
    b = VelocityNetwork(unet_b).to(device)  # we move b to device
    eta = EtaNetwork(unet_eta).to(device)  # we move eta to device
    
    # we load state dicts
    b.load_state_dict(ckpt["b_state_dict"])  # we load b weights
    eta.load_state_dict(ckpt["eta_state_dict"])  # we load eta weights
    
    b.eval()  # we set to eval mode
    eta.eval()  # we set to eval mode
    
    config = {
        "path": path,
        "gamma_type": gamma_type,
        "epsilon": ckpt.get("epsilon"),
        "epsilon_tag": ckpt.get("epsilon_tag", "eps-none"),
        "epoch": ckpt.get("epoch", 0),
    }  # we store configuration
    
    return b, eta, config  # we return networks and config

def reconstruct_with_model(
    b: nn.Module,
    eta: nn.Module,
    interpolant: stochastic_interpolant.Interpolant,
    x0s: torch.Tensor,
    n_step: int = 50
) -> torch.Tensor:
    """we reconstruct images using probability flow"""
    with torch.no_grad():  # we disable gradients
        s = stochastic_interpolant.SFromEta(eta, interpolant.a)  # we create s from eta
        pflow = stochastic_interpolant.PFlowIntegrator(
            b=b, method='rk4', interpolant=interpolant, n_step=n_step, sample_only=True
        )  # we create integrator
        xfs_pflow, _ = pflow.rollout(x0s)  # we integrate
        return xfs_pflow[-1]  # we return final reconstruction

def create_comparison_plot(
    original_imgs: torch.Tensor,
    masked_imgs: torch.Tensor,
    reconstructions: Dict[str, torch.Tensor],
    save_path: str,
    patch_size: int = 6,
    num_patches: int = 4
):
    """we create a comparison plot showing original, masked, and all reconstructions (single row only)"""
    num_models = len(reconstructions)  # we count models
    # we use only the first image for single row comparison
    original_img = original_imgs[0:1]  # we take first image
    masked_img = masked_imgs[0:1]  # we take first masked image
    reconstructions_first = {k: v[0:1] for k, v in reconstructions.items()}  # we take first reconstruction for each model
    
    # we compute grid dimensions (single row)
    ncols = 2 + num_models  # original + masked + reconstructions
    nrows = 1  # single row only
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, 3.0))  # we create figure with single row
    
    def denorm(img):
        """we denormalize and clamp to [0,1] for visualization"""
        denormed = img * 0.5 + 0.5  # we denormalize from [-1,1] to [0,1]
        return torch.clamp(denormed, 0.0, 1.0)  # we clamp to valid range
    
    # we show original image
    img_orig = np.transpose(grab(denorm(original_img[0])), (1, 2, 0))  # we denormalize and transpose
    axes[0].imshow(img_orig)  # we display image
    axes[0].axis('off')  # we remove axes
    axes[0].set_title('original', fontsize=11, fontweight='bold')  # we set title
    
    # we show masked image
    img_masked = np.transpose(grab(denorm(masked_img[0])), (1, 2, 0))  # we denormalize and transpose
    axes[1].imshow(img_masked)  # we display image
    axes[1].axis('off')  # we remove axes
    axes[1].set_title(f'masked\n(patch={patch_size}x{patch_size}, np={num_patches})', fontsize=10)  # we set title
    
    # we show reconstructions and prepare legend labels
    sorted_models = sorted(reconstructions_first.items())  # we sort models for consistent ordering
    legend_labels = []  # we collect legend labels
    for idx, (model_name, recon) in enumerate(sorted_models):  # we iterate over models
        img_recon = np.transpose(grab(denorm(recon[0])), (1, 2, 0))  # we denormalize and transpose
        axes[2 + idx].imshow(img_recon)  # we display image
        axes[2 + idx].axis('off')  # we remove axes
        col_label = f"{idx + 1}. {model_name.replace('_', ' ')}"  # we create column label
        axes[2 + idx].set_title(f"#{idx + 1}", fontsize=10, fontweight='bold')  # we set column number
        legend_labels.append(col_label)  # we add to legend labels
    
    # we add legend at the bottom
    legend_text = " | ".join([f"{idx + 1}: {name.replace('_', ' ')}" for idx, (name, _) in enumerate(sorted_models)])  # we create legend text
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=9, family='monospace')  # we add legend at bottom
    
    plt.suptitle(f'reconstruction comparison', fontsize=13, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # we adjust layout to make room for legend
    plt.savefig(save_path, dpi=150, bbox_inches="tight")  # we save figure
    plt.close()  # we close figure
    print(f"saved comparison plot to {save_path}")  # we log save

def main():
    """we run comparison across all trained models"""
    print("=" * 80)
    print("starting reconstruction comparison")
    print("=" * 80)
    
    # we set parameters
    patch_size = 6  # we set patch size
    num_patches = 4  # we set number of patches
    num_test_samples = 8  # we set number of test images
    epoch = None  # we use latest checkpoint (set to int for specific epoch)
    
    # we load dataset for test images (CelebA like in training script)
    print(f"\nloading CelebA dataset for test images...")  # we log dataset loading
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),  # we resize to target size
        transforms.ToTensor(),  # we convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # we normalize
    ])
    celebA_root = os.path.join(_project_root, "data", "celeba")  # we set celebA root directory
    print(f"CelebA root: {os.path.abspath(celebA_root)}")  # we log celebA root
    try:  # we wrap in try-except for error handling
        trainset = torchvision.datasets.CelebA(
            root=celebA_root,  # we set celebA root directory
            split="train",  # we use training split
            download=True,  # we allow automatic download
            transform=transform  # we apply preprocessing transforms
        )  # we load celebA
        print(f"✓ CelebA loaded successfully: {len(trainset)} images")  # we log success
    except Exception as e:  # we catch loading errors
        import traceback
        print(f"✗ error loading CelebA: {e}")  # we log error
        traceback.print_exc()
        raise  # we re-raise to stop execution
    
    # we generate candidate test images and find the best one (lowest reconstruction error)
    print(f"\ngenerating candidate test images and finding best reconstruction...")  # we log process
    num_candidates = 50  # we test more candidates to find a good one
    torch.manual_seed(42)  # we set seed for reproducibility
    candidate_indices = torch.randint(0, len(trainset), (num_candidates,))  # we sample candidate indices
    candidate_imgs = torch.stack([trainset[i][0] for i in candidate_indices]).to(itf.util.get_torch_device())  # we get candidate batch
    
    # we create masks for candidates
    candidate_masks = create_patch_mask(num_candidates, patch_size=patch_size, num_patches=num_patches)  # we create masks
    candidate_noise = torch.randn_like(candidate_imgs) * (1 - candidate_masks)  # we create noise
    candidate_masked = candidate_imgs * candidate_masks + candidate_noise  # we combine masked image and noise
    candidate_x0s = candidate_masked.reshape(num_candidates, -1)  # we flatten for interpolant
    
    print(f"testing {num_candidates} candidate images to find best reconstruction...")  # we log testing
    
    # we load one model to test reconstruction quality (use first available model)
    best_idx = 0  # we default to first image if no models available
    best_error = float('inf')  # we initialize best error
    
    # we find first available checkpoint to test with
    paths = ["linear", "trig", "encoding-decoding"]  # we define paths
    gamma_types = ["bsquared", "sinesquared", "sigmoid"]  # we define gamma types
    epsilon_tags = ["eps-none", "eps-0.5", "eps-0.25"]  # we define epsilon tags
    dataset_tag_base = f"{DATASET_NAME}_patch{patch_size}x{patch_size}_np{num_patches}"
    
    test_model_found = False  # we track if we found a test model
    
    for epsilon_tag in epsilon_tags:  # we iterate over epsilon values
        if test_model_found:  # we break if we found a model
            break
        full_dataset_tag = f"{dataset_tag_base}_{epsilon_tag}"  # we build full dataset tag
        for path in paths:  # we iterate over paths
            if test_model_found:  # we break if we found a model
                break
            for gamma_type in gamma_types:  # we iterate over gamma types
                ckpt_path = find_latest_checkpoints(
                    full_dataset_tag, path, gamma_type, epsilon_tag, epoch=epoch
                )  # we find latest checkpoint
                if ckpt_path is not None:  # we use first available model
                    try:  # we try to load and test
                        b, eta, config = load_model_from_checkpoint(ckpt_path, itf.util.get_torch_device())  # we load model
                        interpolant = stochastic_interpolant.Interpolant(
                            path=config["path"], gamma_type=config["gamma_type"]
                        )  # we create interpolant
                        interpolant.gamma_type = config["gamma_type"]  # we store gamma type
                        
                        print(f"using {path}_{gamma_type}_{epsilon_tag} to find best image...")  # we log model used
                        
                        # we reconstruct all candidates
                        with torch.no_grad():  # we disable gradients
                            xf_recon = reconstruct_with_model(b, eta, interpolant, candidate_x0s, n_step=50)  # we reconstruct
                            xf_recon_img = xf_recon.reshape(num_candidates, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)  # we reshape
                            
                            # we compute mse for each candidate
                            mse_per_image = torch.mean((xf_recon_img - candidate_imgs) ** 2, dim=(1, 2, 3))  # we compute mse per image
                            best_idx = torch.argmin(mse_per_image).item()  # we find best image index
                            best_error = mse_per_image[best_idx].item()  # we get best error
                            
                            print(f"best image index: {best_idx}, mse: {best_error:.6f}")  # we log best image
                        
                        # we clean up
                        del b, eta, interpolant  # we delete models
                        if torch.cuda.is_available():  # we clear cuda cache
                            torch.cuda.empty_cache()
                        test_model_found = True  # we mark as found
                        break
                    except Exception as e:  # we catch errors
                        print(f"error testing with {path}_{gamma_type}_{epsilon_tag}: {e}")
                        continue
    
    if not test_model_found:  # we warn if no model found
        print("warning: no model found for testing, using first candidate image")
        best_idx = 0  # we use first image
    
    # we select the best image and create test batch with fixed seed for reproducibility
    print(f"\nusing best image (index {best_idx}, mse: {best_error:.6f}) for comparisons...")  # we log selection
    torch.manual_seed(42)  # we set seed for reproducibility
    selected_idx = candidate_indices[best_idx].item()  # we get selected index
    # we select the best image and create test batch
    best_image = candidate_imgs[best_idx:best_idx+1]  # we get best image
    # we replicate the best image for multiple test samples with different masks
    x1s_img = best_image.repeat(num_test_samples, 1, 1, 1)  # we replicate best image
    # we create different masks for each test sample (but same base image)
    masks = create_patch_mask(num_test_samples, patch_size=patch_size, num_patches=num_patches)  # we create masks
    
    # we create masked images
    noise = torch.randn_like(x1s_img) * (1 - masks)  # we create noise
    x0s_img = x1s_img * masks + noise  # we combine masked image and noise
    x0s = x0s_img.reshape(num_test_samples, -1)  # we flatten for interpolant
    
    print(f"test images shape: {x1s_img.shape}")  # we log shape
    print(f"masked images shape: {x0s_img.shape}")  # we log shape
    print(f"selected image index from dataset: {candidate_indices[best_idx].item()}, mse error: {best_error:.6f}")  # we log selection details
    
    # we define configurations to compare
    paths = ["linear", "trig", "encoding-decoding"]  # we define paths
    gamma_types = ["bsquared", "sinesquared", "sigmoid"]  # we define gamma types
    epsilon_tags = ["eps-none", "eps-0.5", "eps-0.25"]  # we define epsilon tags
    
    # we collect all reconstructions
    reconstructions = {}  # we store reconstructions by model name
    model_configs = {}  # we store model configurations
    
    dataset_tag = f"{DATASET_NAME}_patch{patch_size}x{patch_size}_np{num_patches}"
    
    for epsilon_tag in epsilon_tags:  # we iterate over epsilon values
        full_dataset_tag = f"{dataset_tag}_{epsilon_tag}"  # we build full dataset tag
        
        for path in paths:  # we iterate over paths
            for gamma_type in gamma_types:  # we iterate over gamma types
                # we find checkpoint
                ckpt_path = find_latest_checkpoints(
                    full_dataset_tag, path, gamma_type, epsilon_tag, epoch=epoch
                )  # we find latest checkpoint
                
                if ckpt_path is None:  # we skip if no checkpoint
                    print(f"no checkpoint found for {path}_{gamma_type}_{epsilon_tag}, skipping...")
                    continue
                
                # we load model
                try:  # we wrap in try-except
                    b, eta, config = load_model_from_checkpoint(ckpt_path, itf.util.get_torch_device())  # we load model
                    
                    # we create interpolant
                    interpolant = stochastic_interpolant.Interpolant(
                        path=config["path"], gamma_type=config["gamma_type"]
                    )  # we create interpolant
                    interpolant.gamma_type = config["gamma_type"]  # we store gamma type
                    
                    # we reconstruct
                    print(f"reconstructing with {path}_{gamma_type}_{epsilon_tag} (epoch {config['epoch']})...")
                    xf_pflow = reconstruct_with_model(b, eta, interpolant, x0s, n_step=50)  # we reconstruct
                    xf_pflow_img = xf_pflow.reshape(num_test_samples, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)  # we reshape
                    
                    # we store reconstruction
                    model_name = f"{path}_{gamma_type}_{epsilon_tag}"  # we build model name
                    reconstructions[model_name] = xf_pflow_img  # we store reconstruction
                    model_configs[model_name] = config  # we store config
                    
                    # we clear cache
                    del b, eta, interpolant  # we delete models
                    if torch.cuda.is_available():  # we clear cuda cache
                        torch.cuda.empty_cache()
                    
                except Exception as e:  # we catch errors
                    print(f"error loading/reconstructing with {path}_{gamma_type}_{epsilon_tag}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    if not reconstructions:  # we check if any reconstructions
        print("no reconstructions found, exiting...")
        return
    
    print(f"\ncomparing {len(reconstructions)} models on {num_test_samples} images")
    
    # we create comparison plot
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # we create timestamp
    save_path = os.path.join(
        COMPARISON_ROOT,
        f"comparison_{timestamp}_patch{patch_size}x{patch_size}_np{num_patches}.png"
    )  # we build save path
    
    create_comparison_plot(
        x1s_img, x0s_img, reconstructions, save_path,
        patch_size=patch_size, num_patches=num_patches
    )  # we create plot
    
    # we also create per-model detailed comparisons
    print("\ncreating per-model detailed comparisons...")
    for model_name, recon in reconstructions.items():  # we iterate over models
        config = model_configs[model_name]  # we get config
        
        # we create detailed plot for this model
        ncols = 3  # original, masked, reconstructed
        nrows = num_test_samples  # one row per sample
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))  # we create figure
        
        if nrows == 1:  # we handle single row case
            axes = axes[None, :]  # we add row dimension
        
        def denorm(img):
            """we denormalize and clamp to [0,1] for visualization"""
            denormed = img * 0.5 + 0.5  # we denormalize from [-1,1] to [0,1]
            return torch.clamp(denormed, 0.0, 1.0)  # we clamp to valid range
        
        for i in range(num_test_samples):  # we iterate over samples
            # we show original
            img_orig = np.transpose(grab(denorm(x1s_img[i])), (1, 2, 0))  # we denormalize and transpose
            axes[i, 0].imshow(img_orig)  # we display image
            axes[i, 0].axis('off')  # we remove axes
            if i == 0:
                axes[i, 0].set_title('original', fontsize=10)  # we set title
            
            # we show masked
            img_masked = np.transpose(grab(denorm(x0s_img[i])), (1, 2, 0))  # we denormalize and transpose
            axes[i, 1].imshow(img_masked)  # we display image
            axes[i, 1].axis('off')  # we remove axes
            if i == 0:
                axes[i, 1].set_title('masked', fontsize=10)  # we set title
            
            # we show reconstruction
            img_recon = np.transpose(grab(denorm(recon[i])), (1, 2, 0))  # we denormalize and transpose
            axes[i, 2].imshow(img_recon)  # we display image
            axes[i, 2].axis('off')  # we remove axes
            if i == 0:
                axes[i, 2].set_title('reconstructed', fontsize=10)  # we set title
        
        plt.suptitle(f'{model_name} - epoch {config["epoch"]}', fontsize=12, y=0.995)
        plt.tight_layout()
        
        model_save_path = os.path.join(
            COMPARISON_ROOT,
            f"comparison_{model_name}_{timestamp}.png"
        )  # we build model-specific save path
        plt.savefig(model_save_path, dpi=150, bbox_inches="tight")  # we save figure
        plt.close()  # we close figure
        print(f"saved detailed plot for {model_name} to {model_save_path}")  # we log save
    
    print("\n" + "=" * 80)
    print("comparison complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
