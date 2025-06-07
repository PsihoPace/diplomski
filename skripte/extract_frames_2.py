import json
import cv2
import numpy as np
import os
import pandas as pd
from geopy.distance import geodesic
from pathlib import Path

# Putanje
BASE_DIR = Path("D:/Diplomski")
VIDEO_DIR = BASE_DIR / "raw/videos"
JSON_DIR = BASE_DIR / "raw/json"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Funkcija za izračun udaljenosti u metrima između dvije WGS84 točke
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).meters

def process_video(video_path, json_path, output_subdir):
    video_name = video_path.stem  # Dodaj ime videa
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Cannot open video: {video_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    previous_coordinates = None
    previous_time = None
    frame_data = []

    for i in range(1, len(data)):
        current_time = data[i].get('time', None)
        current_coordinates = data[i].get('coordinates', None)

        if current_coordinates:
            lat, lon = current_coordinates

            if previous_coordinates:
                distance = calculate_distance(previous_coordinates, (lat, lon))
                if distance > 35:  # Udaljenost u metrima
                    frame_position = int(current_time * frame_rate)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                    ret, frame = cap.read()

                    if ret:
                        frame_name = f"frame_{frame_number:04d}.jpg"
                        frame_path = output_subdir / frame_name
                        cv2.imwrite(str(frame_path), frame)

                        # Dodaj sve podatke u CSV
                        frame_data.append([frame_name, lat, lon, current_time, video_name])

                        previous_coordinates = (lat, lon)
                        previous_time = current_time
                        frame_number += 1
            else:
                previous_coordinates = (lat, lon)
                previous_time = current_time

    cap.release()

    # Spremi CSV
    if frame_data:
        df = pd.DataFrame(frame_data, columns=[
            "Image Name", "Latitude", "Longitude", "Timestamp (s)", "Video Name"
        ])
        csv_path = output_subdir / f"{video_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[✓] Processed {video_path.name}, saved to {output_subdir.name}")
    else:
        print(f"[!] No frames extracted from {video_path.name}")

# Glavni dio – obradi sve videe
for video_file in VIDEO_DIR.glob("*.mp4"):
    json_file = JSON_DIR / (video_file.stem + ".json")
    if not json_file.exists():
        print(f"[!] JSON not found for video: {video_file.name}")
        continue

    output_subdir = OUTPUT_DIR / video_file.stem
    output_subdir.mkdir(parents=True, exist_ok=True)

    process_video(video_file, json_file, output_subdir)
