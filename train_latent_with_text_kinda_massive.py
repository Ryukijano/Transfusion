from shutil import rmtree
from pathlib import Path
import os
import logging

import torch
from torch import nn, tensor, Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from einops import rearrange

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from transfusion_pytorch import Transfusion, print_modality_sample
from transfusion_pytorch.transfusion import Transformer

# Import the FlowersDataset class
from transfusion_pytorch.datasets import FlowersDataset  # Correct import path

#n hf related

from datasets import load_dataset
from diffusers.models import AutoencoderKL

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder = "vae")

class Encoder(Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        with torch.no_grad():
            latent = self.vae.encode(image * 2 - 1)

        return 0.18215 * latent.latent_dist.sample()

class Decoder(Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        latents = (1 / 0.18215) * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        return (image / 2 + 0.5).clamp(0, 1)

# Initialize results folder
results_folder = Path('./results')
rmtree('./results', ignore_errors=True)
results_folder.mkdir(exist_ok=True, parents=True)

# Setup logging
log_file = results_folder / 'training.log'
logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Log training start
logging.info("Training started.")

# Constants
SAMPLE_EVERY = 100
CHECKPOINTS_DIR = Path('./checkpoints')
CHECKPOINTS_DIR.mkdir(exist_ok=True, parents=True)

# Load labels and create prompts
def load_labels(label_file: Path) -> list:
    with label_file.open('r') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

labels_path = Path('./data/flowers/labels.txt')
labels = load_labels(labels_path)
PROMPTS = [f"A beautiful {label} in full bloom." for label in labels]

# Define Transformer parameters
transformer = Transformer(
    dim=512,  # Example dimension, adjust as needed
    depth=6,  # Number of transformer layers
    dim_head=64,
    heads=8,
    dropout=0.1,
    ff_expansion_factor=4,
    attn_kwargs={},  # Add any specific attention kwargs if required
    ff_kwargs={},    # Add any specific feedforward kwargs if required
    attn_laser=True,
    unet_skips=True,
    use_flex_attn=True,
    num_residual_streams=4
)

# Initialize the Transfusion model with the transformer
model = Transfusion(
    num_text_tokens=256,
    transformer=transformer,  # Passing the Transformer instance
    dim_latent=4,
    channel_first_latent=True,
    modality_default_shape=(64, 64),
    # ... include other required parameters as needed
).cuda()

optimizer = Adam(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda')
ema_model = ...  # Initialize EMA model
best_loss = float('inf')

# Initialize the Dataset
dataset = FlowersDataset(image_size=512)  # Replace with appropriate arguments

# Initialize DataLoader
iter_dl = iter(DataLoader(dataset))  # Define your dataset and DataLoader

# Initialize progress bar
from tqdm import tqdm
progress = tqdm(range(100000), desc='Training')

# Training Loop
for step in progress:
    total_loss = 0
    # Gradient accumulation loop
    for acc_step in range(4):
        with torch.amp.autocast('cuda'):
            try:
                batch = next(iter_dl)
            except StopIteration:
                iter_dl = iter(DataLoader(dataset))  # Reset DataLoader
                batch = next(iter_dl)
            loss = model.forward(batch)
            loss = loss / 4
            total_loss += loss.item()
        scaler.scale(loss).backward()

    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Update EMA model
    ema_model.update()

    # Calculate average loss
    avg_loss = total_loss / 4
    running_loss = 0.9 * running_loss + 0.1 * avg_loss if step > 1 else avg_loss

    # Update progress bar
    progress.set_postfix({
        'loss': f'{avg_loss:.4f}',
        'avg_loss': f'{running_loss:.4f}',
        'gpu_mem': f'{torch.cuda.memory_reserved() / 1e9:.1f}GB'
    })

    # Save only the best checkpoint
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_loss,
        }
        torch.save(
            checkpoint,
            CHECKPOINTS_DIR / 'best_model.pt'
        )
        message = f'New best model saved with loss: {best_loss:.4f}'
        progress.write(message)
        logging.info(message)

    # Generate and save images based on prompts
    if step % SAMPLE_EVERY == 0:
        for prompt in PROMPTS:
            generate_image_for_prompt(prompt, step)
        log_message = f'Generated images for step {step}'
        logging.info(log_message)

from transformers import GPT2Tokenizer

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def encode_prompt(prompt: str) -> Tensor:
    
    encoded = tokenizer.encode(
        prompt,
        add_special_tokens=True,
        max_length=256,  # Adjust based on model
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoded.squeeze(0)

def generate_image_for_prompt(prompt: str, step: int):
    """
    Generates and saves an image based on the given text prompt.

    Args:
        prompt (str): The text prompt to generate the image from.
        step (int): The current training step, used for naming the saved image.
    """
    model.eval()  # Set model to evaluation mode
    try:
        with torch.no_grad():
            # Encode the prompt
            tokens = encode_prompt(prompt)  # Define this function based on your tokenizer
            tokens = tokens.unsqueeze(0).cuda()

            # Generate image using EMA model
            generated_image = ema_model.generate_from_text(tokens)

            # Post-process the image
            image = (generated_image / 2 + 0.5).clamp(0, 1)
            image = image.cpu()

            # Create a sanitized filename
            sanitized_prompt = prompt.replace(" ", "_").replace("/", "_")
            filename = results_folder / f'{step}_{sanitized_prompt}.png'

            # Save the image
            save_image(image, str(filename))
    except Exception as e:
        progress.write(f"Error generating image for prompt '{prompt}': {e}")
    finally:
        model.train()  # Revert to training mode

