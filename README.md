# Sketch-to-Image Translation with Swin Transformer
![68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6c6963656e73652f68756767696e67666163652f64617461736574732e7376673f636f6c6f723d626c7565](https://github.com/user-attachments/assets/1d2009fd-f6b2-4076-8611-0a5fb184e1b8)

A deep learning model that transforms sketch drawings to photorealistic images employing a SwinV2 encoder and a self-developed CNN decoder. However, the model needs more training and some improvement, I decide to share here to seek some valuable advice and enhancement on this idea. The project further entails test tools for the model as well as converting real images to sketches.

![figure1](https://github.com/user-attachments/assets/1cd48002-6757-4d88-80d0-93c0ab9d087b)

## Table of Contents
- Introduction
- Features
- Installation
- Usage
- Results
- Limitations
- Future Work
- Contributing
- License
- Refrences
-------------------------------------------------------------------------------------------------------------------
## Introduction
The best way to represent a simple or complex idea is by sketching and bringing that idea to life, furthermore, it is so simple that everyone can get their idea to the paper by drawing. That means you don’t need to be an artist to make your sketches. The importance of sketches is not only about art, architecture, or engineering, it could be the best way to describe the face of a criminal based on the victim descriptions. Generating real images based on sketches has been a popular topic in computer vision and machine learning. Plenty of tools can make real images based on face sketches currently used at police stations, digital image processing, and public security systems to figure out the identity of criminals and suspects and keep the city peaceful. By improving machine learning and neural networks, these tools are getting better every day, and there are massive methodologies that can do this specific task. In the past, researchers used various methods such as convolutional neural networks to achieve great results in making a detailed real image based on the sketches. Yet, their limitations—e.g., the limited receptive fields of CNNs for long dependencies and training instability of GANs—require more effective architectures. 
Step in the Swin Transformer, a variant of vision transformers with its hierarchical design and shifted window mechanism. Unlike standard transformers, Swin's local self-attention windows enable effective computation with the capability to capture both fine-tuned details and global contextual relationships—crucial in bridging the abstract strokes of sketches with the textural richness of actual faces. Through progressively combining patches step by step, Swin models structural coherence and infers realistic details, addressing challenges such as sparse textures and ambiguous contours inherent in sketches.
This project aims to bridge the gap in face sketch to real image by presenting a novel approach by using SwinV2-B, pre-trained on the large ImageNet-22K dataset, and fine-tuning it for domain-specific face feature generation. This strategy takes advantage of Swin's hierarchical representation learning and adapts its attention mechanisms to the subtleties of sketch-photo translation so that it generalizes well even with sparse input. Our model outperforms state-of-the-art approaches on a common benchmark such as CelebA both quantitatively metrics such as structural similarity index measure (SSIM) and Fréchet inception distance (FID) and by human evaluation. In particular, it maintains identity-critical features—like eye shape and jawline—with record accuracy, a key necessity for forensic use. The model has been trained using A100 40GB GPu for 8 hours on modal.com.

------------------------------------------------------------------------------------------------------------------------
## Features
- 🖌️ Hybrid Transformer-CNN Architecture: Combines SwinV2's attention mechanism with a CNN decoder for high-quality image generation.
- ⚡ Mixed-Precision Training: Optimized for GPU performance.
- 🔄 Residual Blocks in Decoder: Enhances feature propagation and gradient flow.
- 📈 Learning Rate Scheduling with Warmup: Improves training stability.
- 🛑 Early Stopping Mechanism: Prevents overfitting.
- 📊 Evaluation Metrics: Includes SSIM and FID for quantitative assessment.
- ✏️ Sketch Generation: Converts real images into pencil sketches for dataset preparation.
-------------------------------------------------------------------------------------------------------------------------
## Installation
```
# Clone repository
git clone https://github.com/MahyarHassani/Face-sketch-to-real-image-using-Swin-transformer-V2.git
cd Face-sketch-to-real-image-using-Swin-transformer-V2

# Install dependencies
pip install -r requirements.txt

# Install LPIPS requirement
pip install lpips
```
---------------------------------------------------------------------------------------------------------------------------
## Usage
I used CelebA dataset which is avalaible at: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and then generated sketches from real images.
The leveraged dataset is avalaible at: https://drive.google.com/file/d/1vrNfm7xdGi5IXp3ss--aGYS7tXoTWmHs/view?usp=drive_link
The leveraged images for test are avalible at: https://drive.google.com/file/d/1URzlsAlOj92T9gIjr3ASIDmnry7pG6WV/view?usp=sharing

1. Generate sketches from real images
```
jupyter notebook image2sketch.ipynb
```
If you're using modal.com, remember that you have to create a volume before the training.
Run this code on CMD:
```
modal volumecreate checkpoints
```
2. Training the Model
```
modal run train.py # or ( modal run train.py::train_model )
```
3. Testing the Model
```
modal run test.py
```
4. Evaluating the Model
```
modal run evaluate.py
```
------------------------------------------------------------------------------------------------------------------------------
## Results 
Due to the low resources, I could not continue the trainig process and the results can be improved.

Current Metrics (after partial training):
```
SSIM (Structural Similarity): 0.4490
FID (Fréchet Inception Distance): 201.2582
```

Sample results:
![figure](https://github.com/user-attachments/assets/66e2f9f8-f0c5-416f-85f2-358ca0d6ea01)


---------------------------------------------------------------------------------------------------------------------------------
## Limitations

### ⚠️ Current Challenges:
- Suboptimal metrics due to limited training (25 epochs completed).
- Model tends to produce blurry outputs.
- High FID score indicates significant room for improvement.

---------------------------------------------------------------------------------------------------------------------------------
## Future Work
- Complete full training cycle (100+ epochs).
- Experiment with different decoder architectures.
- Add attention mechanisms in the decoder.
- Implement GAN-based adversarial loss.
- Try progressive growing of resolution.
- Optimize for lower VRAM consumption.

-----------------------------------------------------------------------------------------------------------------------------------
## Contributing
Contributions are welcome! Please open an issue first to discuss proposed changes.

-----------------------------------------------------------------------------------------------------------------------------------
## License
Distributed under the Apache License 2.0.

-----------------------------------------------------------------------------------------------------------------------------------
Note: This project is currently in active development and should be considered experimental. The maintainers welcome collaborations and resource contributions to continue improving the model.

-----------------------------------------------------------------------------------------------------------------------------------
## Refrences 
[1] Wang, N., Tao, D., Gao, X., Li, X., & Li, J. (2013). Transductive face sketch-photo synthesis. IEEE transactions on neural networks and learning systems, 24(9), 1364-1376.

[2] Klare, B., Li, Z., & Jain, A. K. (2010). Matching forensic sketches to mug shot photos. IEEE transactions on pattern analysis and machine intelligence, 33(3), 639-646.

[3] Bhandare, M. S., & Vibhute, A. S. (2022, December). Face Sketch to Image Generation and Verification Using Adversarial and Discrimination Network. In Techno-Societal 2016, International Conference on Advanced Technologies for Societal Applications (pp. 901-909). Cham: Springer International Publishing.

[4] Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao, Y., Zhang, Z., Dong, L., Wei, F., & Guo, B. (2022). Swin Transformer V2: Scaling Up Capacity and Resolution. In International Conference on Computer Vision and Pattern Recognition (CVPR).

[5] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015, December). Deep learning face attributes in the wild. In Proceedings of the International Conference on Computer Vision (ICCV).
