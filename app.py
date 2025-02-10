# app.py

import torch
from torch import nn
from pathlib import Path
import gradio as gr
import torchvision.transforms as T
from torchvision.utils import save_image
from transfusion_pytorch.transfusion import Transfusion, print_modality_sample

# Define Encoder and Decoder (ensure consistency with training)
class Encoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        with torch.no_grad():
            latent = self.vae.encode(image * 2 - 1)
        return 0.18215 * latent.latent_dist.sample()

class Decoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        return (image / 2 + 0.5).clamp(0, 1)

# Function to set up the model
def setup_model():
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    model = Transfusion(
        num_text_tokens=256,
        dim_latent=4,
        channel_first_latent=True,
        modality_default_shape=(64, 64),  # Adjust based on your training
        modality_encoder=Encoder(vae),
        modality_decoder=Decoder(vae),
        pre_post_transformer_enc_dec=(
            nn.Sequential(
                nn.Conv2d(4, 640, 3, 2, 1),     # Adjust channels as per training
                nn.Conv2d(640, 1280, 3, 2, 1),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(1280, 640, 3, 2, 1, output_padding=1),
                nn.ConvTranspose2d(640, 4, 3, 2, 1, output_padding=1),
            )
        ),
        add_pos_emb=True,
        modality_num_dim=2,
        reconstruction_loss_weight=0.05,  # Match training
        transformer=dict(
            dim=1280,          # Match training
            depth=24,          # Match training
            dim_head=80,       # Match training
            heads=16,          # Match training
            dropout=0.1,
            ff_expansion_factor=2.0
        )
    ).cuda()

    return model

# Function to load the checkpoint partially
def load_checkpoint_partial(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model_dict = model.state_dict()
    # Filter out unnecessary keys
    checkpoint_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and model_dict[k].shape == v.shape}
    # Overwrite entries in the existing state dict
    model_dict.update(checkpoint_dict)
    # Load the new state dict
    model.load_state_dict(model_dict)
    print("Loaded checkpoint partially, some layers may not have been loaded.")
    return model

# Initialize the model and load the checkpoint
model = setup_model()
checkpoint_path = Path('./checkpoints/best_model.pt')
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

model = load_checkpoint_partial(model, checkpoint_path)
model.eval()  # Set model to evaluation mode

# Define the inference function
def infer_text(text_input):
    """
    Performs transfusion on the input text and returns the generated image.
    """
    import torch
    from transformers import GPT2Tokenizer

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Add padding and truncation
    encoded = tokenizer.encode(
        text_input,
        add_special_tokens=True,
        max_length=256,  # Match the num_text_tokens from model config
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).cuda()

    # Forward pass through the model with proper error handling
    try:
        with torch.no_grad():
            loss, embeddings = model.forward_text(encoded)
            
            # Generate image from embeddings
            generated_image = model.generate_from_embeddings(embeddings)
            
            # Post-process image
            image = generated_image.cpu().squeeze(0)
            image = (image * 255).clamp(0, 255).byte()
            return image.permute(1, 2, 0).numpy()
    except RuntimeError as e:
        print(f"Error during inference: {str(e)}")
        # Return a blank image or error image
        return numpy.zeros((64, 64, 3), dtype=numpy.uint8)

# Gradio interface setup
iface = gr.Interface(
    fn=infer_text,
    inputs=gr.Textbox(
        lines=2, 
        placeholder="Enter your text here...",
        label="Input Text"
    ),
    outputs=gr.Image(
        type="numpy",
        label="Generated Image"
    ),
    title="Transfusion Text-to-Image",
    description="Enter text to generate an image using the Transfusion model.",
    examples=[
        ["A beautiful sunrise over the mountains."],
        ["A futuristic cityscape at night."]
    ]
)

# Launch the app
if __name__ == "__main__":
    iface.launch()