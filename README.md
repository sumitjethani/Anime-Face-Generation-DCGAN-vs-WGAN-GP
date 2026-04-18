# Anime Face Generation — DCGAN vs WGAN-GP

Generates anime faces using two GAN variants — **DCGAN** and **WGAN-GP** — trained on the Anime Faces dataset. The project tackles mode collapse by comparing both architectures side by side.

---

## Demo

🤗 **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/Sumit-Jethani/Anime-Face-Generation-DCGAN-vs-WGAN-GP)

---

## Model Architecture

### Generator (Shared)
- Input: Random noise vector (100-dim)
- 5 Transposed Convolution layers
- BatchNorm + ReLU activations
- Output: 64×64 RGB image with Tanh

### DCGAN Discriminator
- 5 Convolution layers
- BatchNorm + LeakyReLU
- Sigmoid output
- Loss: Binary Cross Entropy (BCE)

### WGAN-GP Critic
- 5 Convolution layers
- InstanceNorm + LeakyReLU
- No Sigmoid (outputs raw score)
- Loss: Wasserstein Distance + Gradient Penalty (λ=10)

---

## Training Details

| Parameter | DCGAN | WGAN-GP |
|---|---|---|
| Dataset | Anime Faces | Anime Faces |
| Image Size | 64×64 | 64×64 |
| Batch Size | 64 | 64 |
| Epochs | 50 | 50 |
| Optimizer | Adam | Adam |
| LR (Generator) | 0.0002 | 0.0001 |
| LR (Discriminator) | 0.0001 | 0.0001 |
| Betas | (0.5, 0.999) | (0.0, 0.9) |
| Critic Iterations | — | 5 |
| Gradient Penalty λ | — | 10 |
| Mixed Precision | AMP | — |

---

## Mode Collapse Techniques Used

- **Label Smoothing** — real labels set to 0.9 instead of 1.0 in DCGAN
- **WGAN-GP** — Wasserstein loss with gradient penalty for stable training
- **InstanceNorm** in WGAN-GP critic instead of BatchNorm
- **Lower Discriminator LR** to prevent it from overpowering the generator
- **Critic Iterations** — critic trained 5x more than generator in WGAN-GP

---

## Download Models

**DCGAN:**
```bash
wget https://huggingface.co/spaces/Sumit-Jethani/Anime-Face-Generation-DCGAN-vs-WGAN-GP/resolve/main/dcgan_checkpoint.pth
```

**WGAN-GP:**
```bash
wget https://huggingface.co/spaces/Sumit-Jethani/Anime-Face-Generation-DCGAN-vs-WGAN-GP/resolve/main/wgangp_checkpoint.pth
```

---

## Project Structure

```
├── app.py                                          # Gradio app for HuggingFace deployment
├── requirements.txt                                # Dependencies
├── dcgan_checkpoint.pth                            # Pretrained DCGAN weights
├── wgangp_checkpoint.pth                           # Pretrained WGAN-GP weights
└── tackling-mode-collapse-in-generative-adversarial.ipynb   # Training notebook
```

---

## Installation & Usage

```bash
# Clone the repo
git clone https://github.com/sumitjethani/Anime-Face-Generation-DCGAN-vs-WGAN-GP.git
cd Anime-Face-Generation-DCGAN-vs-WGAN-GP

# Install dependencies
pip install -r requirements.txt

# Download models
wget https://huggingface.co/spaces/Sumit-Jethani/Anime-Face-Generation-DCGAN-vs-WGAN-GP/resolve/main/dcgan_checkpoint.pth
wget https://huggingface.co/spaces/Sumit-Jethani/Anime-Face-Generation-DCGAN-vs-WGAN-GP/resolve/main/wgangp_checkpoint.pth

# Run the app
python app.py
```

---

## Requirements

```
torch==2.1.0
torchvision==0.16.0
gradio==4.44.0
Pillow==10.3.0
numpy==1.26.4
```

---

## Dataset

[Anime Faces Dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) — Kaggle

---

## Author

**Sumit Jethani**
- GitHub: [github.com/sumitjethani](https://github.com/sumitjethani)
- LinkedIn: [linkedin.com/in/sumit-jethani](https://linkedin.com/in/sumit-jethani)
- HuggingFace: [huggingface.co/Sumit-Jethani](https://huggingface.co/Sumit-Jethani)
