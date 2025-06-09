import torch
from torchvision import models
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import psycopg2

# Putanja do tvoje data mape
BASE_DIR = Path("D:/Diplomski/data")

# Postavke za konekciju na bazu
DB_CONFIG = {
    "dbname": "image_embeddings_db",
    "user": "postgres",
    "password": "user123",
    "host": "localhost",
    "port": 5434
}

# Priprema modela
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # bez klasifikacijskog sloja
model.eval()
transform = models.ResNet18_Weights.DEFAULT.transforms()

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor).reshape(-1).numpy()

    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm == 0:
        print(f"[!] Invalid embedding for {image_path.name} — preskačem.")
        return None

    return embedding / norm

# Spoji se na bazu
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Iteracija kroz podmape
for folder in BASE_DIR.iterdir():
    if not folder.is_dir():
        continue

    video_name = folder.name
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        print(f"[!] Nema CSV-a u: {video_name}")
        continue

    # Provjeri je li video već u bazi
    cur.execute("SELECT 1 FROM image_embeddings WHERE video_name = %s LIMIT 1", (video_name,))
    if cur.fetchone():
        print(f"[!!] {video_name} već postoji u bazi — preskačem.")
        continue

    df = pd.read_csv(csv_files[0])
    total = len(df)

    for i, row in df.iterrows():
        image_path = folder / row["Image Name"]
        if not image_path.exists():
            print(f"[!] Slika ne postoji: {image_path}")
            continue

        embedding = get_embedding(image_path)
        if embedding is None:
            continue  # preskoči ako je invalid

        print(f"[{video_name}] {i+1}/{total} → ubacujem")

        cur.execute("""
            INSERT INTO image_embeddings (image_name, latitude, longitude, timestamp, video_name, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            row["Image Name"],
            row["Longitude"],  
            row["Latitude"],
            row["Timestamp (s)"],
            row["Video Name"],
            embedding.tolist()
        ))
        conn.commit()

print("✅ Svi validni podaci uneseni u bazu.")
cur.close()
conn.close()
