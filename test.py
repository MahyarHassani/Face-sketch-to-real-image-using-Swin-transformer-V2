import modal
import torch
from pathlib import Path
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import gdown
import shutil
import zipfile

volume = modal.Volume.from_name("checkpoints")
MODEL_NAME = "microsoft/swinv2-base-patch4-window12-192-22k"
INPUT_SIZE = 192
GDRIVE_FILE_ID = "1URzlsAlOj92T9gIjr3ASIDmnry7pG6WV"  
image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "transformers", "matplotlib", "Pillow", "gdown"
)

app = modal.App(name="sketch-to-real-test", image=image)

class Decoder(nn.Module):
    def __init__(self, encoder_hidden_dim=1024):
        super().__init__()
        self.initial_size = INPUT_SIZE // 32
        
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(encoder_hidden_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)

@app.function(gpu="A100", volumes={"/my_vol": volume})
def test_model():
    """Test the trained model on test images from zip file"""
    from transformers import Swinv2Model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_temp_dir = Path("/tmp")
    zip_path = base_temp_dir / "test_images.zip"
    test_dir = base_temp_dir / "test_images"
    
    print("\nDownloading zip file...")
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
        str(zip_path),
        quiet=False
    )
    
    print("\nExtracting zip file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(test_dir)
    
    print("Loading encoder...")
    encoder = Swinv2Model.from_pretrained(MODEL_NAME).to(device)
    print("Loading decoder...")
    decoder = Decoder(encoder.config.hidden_size).to(device)
    
    checkpoint_path = Path("/my_vol/final_model.pth")
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    encoder.eval()
    decoder.eval()
    
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    results_dir = Path("/my_vol/test_results")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nProcessing images...")
    image_files = list(test_dir.glob("**/*.jpg"))  # Recursive search for jpg files
    
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        
        try:
            original_img = Image.open(img_path).convert("RGB")
            original_size = original_img.size  
            
            model_input = transform(original_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = encoder(model_input).last_hidden_state
                batch_size, seq_len, hidden_dim = features.shape
                h = w = int(seq_len ** 0.5)
                output = decoder(features.view(batch_size, hidden_dim, h, w))
            
            output = output.cpu().squeeze(0)
            output = (output + 1) / 2  
            output = transforms.ToPILImage()(output)
            output = output.resize(original_size, Image.Resampling.LANCZOS)
            
            input_save_path = results_dir / f"input_{img_path.name}"
            output_save_path = results_dir / f"output_{img_path.name}"
            
            original_img.save(input_save_path)
            output.save(output_save_path)
            print(f"Saved results to: {results_dir}")
            print(f"Image dimensions: {original_size}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    print("\nCleaning up temporary files...")
    shutil.rmtree(test_dir)
    os.remove(zip_path)
    
    print("\nProcessing complete!")
    return str(results_dir)

@app.function(volumes={"/my_vol": volume})
def list_results():
    """List all test results in the volume"""
    results_dir = Path("/my_vol/test_results")
    if results_dir.exists():
        print("\nTest results:")
        for file in sorted(results_dir.glob("*.jpg")):
            print(f"- {file.name}")
            print(f"  Size: {file.stat().st_size / 1024:.2f} KB")
    else:
        print("No test results found")

if __name__ == "__main__":
    with app.run():
        print("Starting test...")
        test_model.remote()
        
        list_results.remote()