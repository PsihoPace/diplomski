import os
import requests
from pathlib import Path

# Putanje
BASE_DIR = Path("D:/Diplomski")
INPUT_DIR = BASE_DIR / "input"
RAW_DIR = BASE_DIR / "raw"

FOLDERS = {
    "video_files.txt": RAW_DIR / "videos",
    "json_files.txt": RAW_DIR / "json",
    "geojson_files.txt": RAW_DIR / "geojson",
}

# Kreiraj sve foldere ako ne postoje
for folder in FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

def download_file(url, dest_folder):
    filename = url.split("/")[-1]
    dest_path = dest_folder / filename

    if dest_path.exists():
        print(f"[✓] Already exists: {filename}")
        return

    try:
        print(f"↓ Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[+] Done: {filename}")
    except Exception as e:
        print(f"[!] Failed to download {filename}: {e}")

# Iteriraj po svim txt fajlovima
for txt_file, dest_folder in FOLDERS.items():
    txt_path = INPUT_DIR / txt_file

    if not txt_path.exists():
        print(f"[!] File not found: {txt_path}")
        continue

    with open(txt_path, "r") as f:
        links = [line.strip() for line in f if line.strip()]

    for link in links:
        download_file(link, dest_folder)
