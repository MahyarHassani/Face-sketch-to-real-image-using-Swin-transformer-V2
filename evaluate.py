import modal
import torch
from pathlib import Path
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from pytorch_fid import fid_score
from skimage.metrics import structural_similarity as ssim
import cv2

volume = modal.Volume.from_name("checkpoints")

METRIC_SIZE = 256

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "numpy", "scikit-image", 
    "opencv-python-headless", "pytorch-fid"
)

app = modal.App(name="sketch-to-real-evaluate", image=image)

@app.function(volumes={"/my_vol": volume})
def calculate_metrics():
    results_dir = Path("/my_vol/test_results")
    if not results_dir.exists():
        print("No test results found!")
        return
    
    input_images = sorted(results_dir.glob("input_*.jpg"))
    output_images = sorted(results_dir.glob("output_*.jpg"))
    
    if len(input_images) == 0 or len(output_images) == 0:
        print("No image pairs found!")
        return
    
    print(f"Found {len(input_images)} image pairs")
    
    ssim_scores = []
    for in_path, out_path in zip(input_images, output_images):
        try:
            input_img = cv2.imread(str(in_path))
            output_img = cv2.imread(str(out_path))
            
            if input_img is None or output_img is None:
                print(f"Error reading images: {in_path.name}")
                continue
                
            input_img = cv2.resize(input_img, (METRIC_SIZE, METRIC_SIZE))
            output_img = cv2.resize(output_img, (METRIC_SIZE, METRIC_SIZE))
            
            input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            output_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
            
            score = ssim(input_gray, output_gray)
            ssim_scores.append(score)
            print(f"SSIM for {in_path.name}: {score:.4f}")
            
        except Exception as e:
            print(f"Error processing {in_path.name}: {e}")
            continue
    
    if not ssim_scores:
        print("No valid SSIM scores calculated!")
        return
        
    avg_ssim = np.mean(ssim_scores)
    print(f"\nAverage SSIM: {avg_ssim:.4f}")
    
    real_dir = Path("/tmp/real")
    fake_dir = Path("/tmp/fake")
    real_dir.mkdir(exist_ok=True, parents=True)
    fake_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nPreparing images for FID calculation...")
    for in_path, out_path in zip(input_images, output_images):
        try:
            input_img = Image.open(in_path).convert('RGB')
            output_img = Image.open(out_path).convert('RGB')
            
            input_img = input_img.resize((METRIC_SIZE, METRIC_SIZE), Image.Resampling.LANCZOS)
            output_img = output_img.resize((METRIC_SIZE, METRIC_SIZE), Image.Resampling.LANCZOS)
            
            input_save_path = real_dir / in_path.name
            output_save_path = fake_dir / out_path.name
            
            input_img.save(input_save_path)
            output_img.save(output_save_path)
            
        except Exception as e:
            print(f"Error preparing {in_path.name} for FID: {e}")
            continue
    
    try:
        print("\nCalculating FID score...")
        fid = fid_score.calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=50,
            device='cpu',
            dims=2048
        )
        print(f"FID Score: {fid:.4f}")
    except Exception as e:
        print(f"Error calculating FID: {e}")
        fid = None
    
    metrics_path = results_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        if fid is not None:
            f.write(f"FID Score: {fid:.4f}\n")
        f.write("\nIndividual SSIM Scores:\n")
        for img, score in zip(input_images, ssim_scores):
            f.write(f"{img.name}: {score:.4f}\n")
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    print("\nCleaning up temporary directories...")
    for dir_path in [real_dir, fake_dir]:
        if dir_path.exists():
            for file in dir_path.glob("*"):
                file.unlink()
            dir_path.rmdir()

@app.function(volumes={"/my_vol": volume})
def list_metrics():
    metrics_path = Path("/my_vol/test_results/metrics.txt")
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            print(f.read())
    else:
        print("No metrics file found")

if __name__ == "__main__":
    with app.run():
        print("Calculating metrics...")
        calculate_metrics.remote()
        
        print("\nResults:")
        list_metrics.remote()
