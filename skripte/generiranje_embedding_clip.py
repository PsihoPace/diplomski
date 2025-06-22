import torch
from torchvision import models
from PIL import Image
import psycopg2
import numpy as np
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# --- Postavke ---
IMAGE_PATH = Path("D:/Diplomski/test_slika/test_bukovica.png")

DB_CONFIG = {
    "dbname": "image_embeddings_db",
    "user": "postgres",
    "password": "user123",
    "host": "localhost",
    "port": 5434
}

# --- Učitavanje modela ---
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
model.heads = torch.nn.Identity()
model.eval()
transform = models.ViT_B_16_Weights.DEFAULT.transforms()

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor).squeeze().numpy()
    return embedding.flatten()

# --- Geolokacija ---
geolocator = Nominatim(user_agent="geo_locator")

def reverse_geocode(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language="hr")
        return location.address if location else "Nepoznata lokacija"
    except GeocoderTimedOut:
        return "Geocoder timeout"

# --- Generiranje embeddinga slike ---
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"❌ Slika nije pronađena: {IMAGE_PATH}")

embedding = get_image_embedding(IMAGE_PATH)
embedding_cosine = embedding / np.linalg.norm(embedding)
embedding_str = "[" + ",".join(map(str, embedding_cosine)) + "]"

# --- Spoji se na bazu ---
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# --- Kosinusna udaljenost ---
cur.execute(f"""
    SELECT image_name, video_name, latitude, longitude, timestamp,
           embedding <=> '{embedding_str}'::vector AS distance
    FROM image_embeddings_clip
    ORDER BY distance ASC
    LIMIT 5;
""")
cosine_results = cur.fetchall()

print("\n📌 Top 5 rezultata prema Kosinusnoj udaljenosti:\n")
for i, row in enumerate(cosine_results, 1):
    name, video, lat, lon, ts, dist = row
    maps_link = f"https://www.google.com/maps?q={lat:.7f},{lon:.7f}"
    address = reverse_geocode(lat, lon)
    print(f"{i}. {name} | {video} | ({lat:.5f}, {lon:.5f}) | t={ts:.2f}s | udaljenost={dist:.4f}")
    print(f"   📍 Google Maps: {maps_link}")
    print(f"   🏠 Adresa: {address}\n")

# --- Euklidska udaljenost ---
cur.execute(f"""
    SELECT image_name, video_name, latitude, longitude, timestamp,
           embedding <-> '{embedding_str}'::vector AS distance
    FROM image_embeddings_clip
    ORDER BY distance ASC
    LIMIT 5;
""")
euclidean_results = cur.fetchall()

print("\n📌 Top 5 rezultata prema Euklidskoj udaljenosti:\n")
for i, row in enumerate(euclidean_results, 1):
    name, video, lat, lon, ts, dist = row
    maps_link = f"https://www.google.com/maps?q={lat:.7f},{lon:.7f}"
    address = reverse_geocode(lat, lon)
    print(f"{i}. {name} | {video} | ({lat:.5f}, {lon:.5f}) | t={ts:.2f}s | udaljenost={dist:.4f}")
    print(f"   📍 Google Maps: {maps_link}")
    print(f"   🏠 Adresa: {address}\n")

# --- Zatvori konekciju ---
cur.close()
conn.close()
