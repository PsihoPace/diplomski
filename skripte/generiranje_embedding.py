import torch
from torchvision import models
from PIL import Image
from pathlib import Path

# === Putanja do slike ===
IMAGE_PATH = Path("D:/Diplomski/test_slika/test_pozega.png")  # zamijeni po potrebi

# === Učitavanje ResNet18 modela ===
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # makni klasifikacijski sloj
model.eval()

# === Transformacija kao kod treniranja ===
transform = models.ResNet18_Weights.DEFAULT.transforms()

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor).squeeze().numpy()
    return embedding.flatten()  # [512]

# === Provjera postoji li slika ===
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"❌ Slika nije pronađena: {IMAGE_PATH}")

# === Generiranje embeddinga ===
embedding = get_image_embedding(IMAGE_PATH)

# === Ispis rezultata ===
print("✅ Embedding uspješno generiran!")
print(f"Prvih 10 vrijednosti: {embedding[:10]}")
