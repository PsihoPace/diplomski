import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import psycopg2
from tqdm import tqdm

# Putanja do tvoje data mape
BASE_DIR = Path("D:/Diplomski/data")

# Postavke za konekciju na bazu
DB_CONFIG = {
    "dbname": "image_embeddings_db",
    "user": "postgres",
    "password": "user123",  # zamijeni stvarnom lozinkom
    "host": "localhost",
    "port": 5434
}

# Priprema modela
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # ukloni klasifikator
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor).squeeze().numpy()
    return embedding.flatten()

# Spoji se na bazu
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Iteracija kroz sve podmape (npr. po videima)
for folder in BASE_DIR.iterdir():
    if not folder.is_dir():
        continue

    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        print(f"[!] Nema CSV-a u: {folder.name}")
        continue

    df = pd.read_csv(csv_files[0])

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Obrada {folder.name}"):
        image_path = folder / row["Image Name"]
        if not image_path.exists():
            print(f"[!] Slika ne postoji: {image_path}")
            continue

        embedding = get_embedding(image_path)

        # Spremi sve u bazu
        cur.execute("""
            INSERT INTO image_embeddings (image_name, latitude, longitude, timestamp, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            row["Image Name"],
            row["Latitude"],
            row["Longitude"],
            row["Timestamp (s)"],
            embedding.tolist()
        ))
        conn.commit()

print("✅ Svi podaci uspješno uneseni u bazu!")
cur.close()
conn.close()
