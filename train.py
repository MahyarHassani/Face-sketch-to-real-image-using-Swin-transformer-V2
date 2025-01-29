import modal
import torch
from pathlib import Path
import os
from torch import nn
import lpips

volume = modal.Volume.from_name("checkpoints")

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "timm", "transformers",
    "scikit-image", "lpips", "pytorch-fid", "tqdm",
    "matplotlib", "Pillow", "requests", "gdown"
)

app = modal.App(name="sketch-to-real", image=image)

MODEL_NAME = "microsoft/swinv2-base-patch4-window12-192-22k"
BATCH_SIZE = 64
INPUT_SIZE = 192

class Decoder(nn.Module):
    def __init__(self, encoder_hidden_dim=1024):
        super().__init__()
        self.initial_size = INPUT_SIZE // 32
        
        self.initial = nn.Sequential(
            nn.Linear(encoder_hidden_dim, 512 * (self.initial_size ** 2)),
            nn.ReLU()
        )
        
        def res_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            res_block(256, 256),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            res_block(128, 128),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            res_block(64, 64),
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            res_block(32, 32),
            nn.ReLU()
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = x.view(-1, 512, self.initial_size, self.initial_size)
        
        x1 = self.up1(x)
        x2 = self.up2(x1)
        x3 = self.up3(x2)
        x4 = self.up4(x3)
        
        return self.final(x4)

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.lpips = lpips.LPIPS(net='alex').cuda()
        
    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        mse_loss = self.mse(pred, target)
        perceptual_loss = self.lpips(pred, target).mean()
        
        return l1_loss * 0.5 + mse_loss * 0.3 + perceptual_loss * 0.2

@app.function(volumes={"/checkpoints": volume})
def list_checkpoints():
    """List all checkpoints in the volume"""
    checkpoint_dir = Path("/checkpoints")
    if checkpoint_dir.exists():
        print("\nAvailable checkpoints:")
        for file in checkpoint_dir.glob("*.pth"):
            print(f"- {file.name}")
            print(f"  Size: {file.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("No checkpoint directory found")

@app.function(gpu="A100", timeout=8400, volumes={"/my_vol": volume}) # 8400 = 2.33 hours, you can adjus the GPU and timeout as needed
def train_model():
    import gdown
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image
    from tqdm import tqdm
    from transformers import Swinv2Model

    current_dir = Path.cwd()
    dataset_path = current_dir / "dataset"
    dataset_path.mkdir(exist_ok=True, parents=True)

    if not (dataset_path / "train").exists():
        print("Downloading dataset...")
        zip_path = current_dir / "dataset.zip"
        
        try:
            file_id = "1vrNfm7xdGi5IXp3ss--aGYS7tXoTWmHs"
            direct_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(direct_url, str(zip_path), quiet=False)
            
            if zip_path.exists():
                print("Dataset downloaded successfully!")
                print("Extracting dataset...")
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(current_dir)
                print("Dataset extracted successfully!")
                zip_path.unlink()
            else:
                raise Exception("Dataset download failed!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise

    save_dir = Path("/checkpoints")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n✓ Checkpoints will be saved to: {save_dir}")

    class SketchToRealDataset(torch.utils.data.Dataset):
        def __init__(self, sketch_dir, real_dir, transform=None):
            self.sketch_paths = sorted([p for p in sketch_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
            self.real_paths = sorted([p for p in real_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
            
            min_len = min(len(self.sketch_paths), len(self.real_paths))
            self.sketch_paths = self.sketch_paths[:min_len]
            self.real_paths = self.real_paths[:min_len]
            
            self.transform = transform or transforms.Compose([
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        def __len__(self):
            return len(self.sketch_paths)

        def __getitem__(self, idx):
            sketch = Image.open(self.sketch_paths[idx]).convert("RGB")
            real = Image.open(self.real_paths[idx]).convert("RGB")
            return self.transform(sketch), self.transform(real)

    train_dataset = SketchToRealDataset(
        dataset_path / "train" / "sketch",
        dataset_path / "train" / "real"
    )
    val_dataset = SketchToRealDataset(
        dataset_path / "val" / "sketch",
        dataset_path / "val" / "real"
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda")
    encoder = Swinv2Model.from_pretrained(MODEL_NAME).to(device)
    decoder = Decoder(encoder.config.hidden_size).to(device)
    
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW([
        {'params': encoder.parameters(), 'lr': 2e-4},  
        {'params': decoder.parameters(), 'lr': 2e-3}   
    ])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, 
        verbose=True, min_lr=1e-6
    )
    
    
    num_epochs = 20 
    best_val_loss = float('inf')
    patience = 5     
    no_improve = 0
    
    warmup_epochs = 2
    
    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            lr_scale = min(1., float(epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * lr_scale
        
        encoder.train()
        decoder.train()
        train_losses = []
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Learning rates: Encoder={optimizer.param_groups[0]['lr']:.6f}, Decoder={optimizer.param_groups[1]['lr']:.6f}")
        
        for sketches, reals in tqdm(train_loader):
            sketches, reals = sketches.cuda(), reals.cuda()
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                features = encoder(sketches).last_hidden_state
                outputs = decoder(features)
                loss = criterion(outputs, reals)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        val_loss = validate(encoder, decoder, val_loader, criterion)
        print(f"Validation loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            print(f"New best model! (val_loss: {val_loss:.4f})")
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, '/my_vol/best_model.pth')
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs")
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, f'/my_vol/checkpoint_epoch_{epoch+1}.pth')
        
        if no_improve >= patience:
            print("Early stopping triggered!")
            break

def download_checkpoints_local():
    """Download checkpoints from volume to local machine"""
    print("Downloading checkpoints to local machine...")
    
    local_dir = Path(__file__).parent / "checkpoints"
    local_dir.mkdir(exist_ok=True, parents=True)
    
    with volume.get() as vol:
        for checkpoint in ["best_model.pth", "final_model.pth"]:
            remote_path = f"/checkpoints/{checkpoint}"
            local_path = local_dir / checkpoint
            try:
                vol.get_file(remote_path, str(local_path))
                print(f"Downloaded {checkpoint} to: {local_path}")
            except Exception as e:
                print(f"Failed to download {checkpoint}: {e}")

if __name__ == "__main__":
    with app.run():
        list_checkpoints.remote()
        train_model.remote()
        download_checkpoints_local()