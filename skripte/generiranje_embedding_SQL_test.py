import torch
from torchvision import models
from PIL import Image
import numpy as np
import psycopg2
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from pathlib import Path

# --- Config ---
IMAGE_PATH = Path("D:/Diplomski/test_slika/test_bukovica2.png")
DB_CONFIG = {
    "dbname": "image_embeddings_db",
    "user": "postgres",
    "password": "user123",
    "host": "localhost",
    "port": 5434
}

# --- Load ResNet18 and prepare transform ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
transform = models.ResNet18_Weights.DEFAULT.transforms()

# --- Load and embed image ---
def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor).squeeze().numpy()
    return embedding.flatten()

# --- Reverse geocoding setup ---
geolocator = Nominatim(user_agent="geo_locator")

def reverse_geocode(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language="hr")
        return location.address if location else "Nepoznata lokacija"
    except GeocoderTimedOut:
        return "Geocoder timeout"

# --- Generate embedding and normalize for cosine ---
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"❌ Slika nije pronađena: {IMAGE_PATH}")

embedding = get_embedding(IMAGE_PATH)
embedding_cosine = embedding / np.linalg.norm(embedding)
print("✅ Embedding generated. First 10 values:")
print(embedding_cosine[:10])

# Convert to pgvector format
embedding_pgvector = "{" + ",".join(f"{x:.6f}" for x in embedding_cosine) + "}"

# --- Query top 5 from DB ---
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute(f"""
    SELECT image_name, video_name, latitude, longitude, timestamp,
           1 - (embedding <#> '{embedding_pgvector}'::vector) AS cosine_similarity
    FROM image_embeddings
    ORDER BY cosine_similarity DESC
    LIMIT 5;
""")

results = cur.fetchall()

# --- Print results ---
print("\n📌 Top 5 najbližih rezultata (Kosinusna sličnost):\n")
for i, row in enumerate(results, 1):
    name, video, lat, lon, ts, similarity = row
    print(f"{i}. {name} | {video} | ({lat:.5f}, {lon:.5f}) | t={ts:.2f}s | sličnost={similarity:.4f}")

# Google Maps link and address
best_lat, best_lon = results[0][2], results[0][3]
maps_link = f"https://www.google.com/maps?q={best_lat:.7f},{best_lon:.7f}"
address = reverse_geocode(best_lat, best_lon)

print(f"\n🌍 Najbliža lokacija: {maps_link}")
print(f"📌 Procijenjena adresa: {address}")

# --- Cleanup ---
cur.close()
conn.close()
