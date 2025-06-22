# master_test.py
from pathlib import Path
from test_resnet_module import test_image_similarity as test_resnet
from test_vit_module import test_image_similarity as test_vit
import csv
from datetime import datetime

RESULTS_DIR = Path("D:/Diplomski/Rezultati")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#"D:\Diplomski\test_slika\test_0.jpg"

def main():
    print("🧠 Model Tester za ViT i ResNet")
    image_path = input("Unesi putanju do slike (ili drag & drop):\n> ").strip()
    image_path = Path(image_path.strip('"'))

    if not image_path.exists():
        print(f"❌ Slika ne postoji: {image_path}")
        return

    print("\n🔍 Testiranje s ResNet-18:")
    print("-" * 40)
    resnet_results_cos, resnet_results_euc = test_resnet(image_path, return_results=True)

    print("\n🔍 Testiranje s ViT-B/16:")
    print("-" * 40)
    vit_results_cos, vit_results_euc = test_vit(image_path, return_results=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = RESULTS_DIR / f"Test-{timestamp}.csv"

    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write the image name at the top
        writer.writerow([f"Izvorna slika: {image_path.name}"])
        writer.writerow([])

        # ResNet Cosine
        writer.writerow(["Rezultati - ResNet (Kosinusna udaljenost)"])
        writer.writerow(["Rank", "Image", "Lat", "Lon", "Dist", "Adresa", "Google Maps"])
        for i, row in enumerate(resnet_results_cos, 1):
            writer.writerow([i] + row)

        writer.writerow([])

        # ResNet Euclidean
        writer.writerow(["Rezultati - ResNet (Euklidska udaljenost)"])
        writer.writerow(["Rank", "Image", "Lat", "Lon", "Dist", "Adresa", "Google Maps"])
        for i, row in enumerate(resnet_results_euc, 1):
            writer.writerow([i] + row)

        writer.writerow([])

        # ViT Cosine
        writer.writerow(["Rezultati - ViT (Kosinusna udaljenost)"])
        writer.writerow(["Rank", "Image", "Lat", "Lon", "Dist", "Adresa", "Google Maps"])
        for i, row in enumerate(vit_results_cos, 1):
            writer.writerow([i] + row)

        writer.writerow([])

        # ViT Euclidean
        writer.writerow(["Rezultati - ViT (Euklidska udaljenost)"])
        writer.writerow(["Rank", "Image", "Lat", "Lon", "Dist", "Adresa", "Google Maps"])
        for i, row in enumerate(vit_results_euc, 1):
            writer.writerow([i] + row)

    print(f"\n✅ Rezultati zapisani u: {csv_path}")

if __name__ == "__main__":
    main()
