import json
import cv2
import numpy as np
import os
import pandas as pd
from geopy.distance import geodesic
from pathlib import Path

# Putanje
BASE_DIR = Path("D:/Diplomski")
VIDEO_PATH = BASE_DIR / "raw/videos_1/GS010358_052022_0_.mp4"
OUTPUT_FRAME_DIR = BASE_DIR / "data/frames"
OUTPUT_FRAME_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUTPUT_PATH = BASE_DIR / "data/frames_coordinates.csv"

# Funkcija za izračun euklidske udaljenosti u metrima između dvije WGS84 točke
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).meters

# Učitaj .json file
with open(BASE_DIR / 'raw/json/GS010358_052022_0_.json', 'r') as f:
    data = json.load(f)

# Otvori video
cap = cv2.VideoCapture(str(VIDEO_PATH))

# Pokreni CSV zapis
frame_data = []

frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frame rate videozapisa
frame_number = 0

# Počni od prvog framea
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Postavi početnu referencu (početne koordinate i time)
previous_coordinates = None
previous_time = None

# Iteriraj kroz sve vremenske točke u jsonu
for i in range(1, len(data)):
    current_time = data[i].get('time', None)
    current_coordinates = data[i].get('coordinates', None)
    
    if current_coordinates:
        lat, lon = current_coordinates
        
        # Ako postoje prethodne koordinate, računaj udaljenost
        if previous_coordinates:
            # Izračunaj udaljenost između prethodnih i trenutnih koordinata
            distance = calculate_distance(previous_coordinates, (lat, lon))
            
            # Ako je udaljenost veća od 5 metara, uzmi frame na trenutnom vremenu
            if distance > 50:
                # Izračunaj frame koji odgovara vremenu (time)
                frame_position = int(current_time * frame_rate)  # Razlikovanje prema frame rate-u
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                ret, frame = cap.read()
                
                if ret:
                    frame_name = f"frame_{frame_number:04d}.jpg"
                    frame_path = OUTPUT_FRAME_DIR / frame_name
                    
                    # Spremi frame
                    cv2.imwrite(str(frame_path), frame)
                    
                    # Dodaj u CSV listu
                    frame_data.append([frame_name, lat, lon])
                    
                    # Novi frame postaje referentni frame za daljnje usporedbe
                    previous_coordinates = (lat, lon)
                    previous_time = current_time
                    frame_number += 1
        else:
            # Početne koordinate
            previous_coordinates = (lat, lon)
            previous_time = current_time

# Zatvori video
cap.release()

# Spremi podatke u CSV
df = pd.DataFrame(frame_data, columns=["Image Name", "Latitude", "Longitude"])
df.to_csv(CSV_OUTPUT_PATH, index=False)

print(f"Frames saved to {OUTPUT_FRAME_DIR}")
print(f"CSV with coordinates saved to {CSV_OUTPUT_PATH}")
