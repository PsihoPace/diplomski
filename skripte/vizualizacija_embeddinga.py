import psycopg2
import pandas as pd
import numpy as np
import umap
import seaborn as sns
import matplotlib.pyplot as plt

# Konfiguracija baze
DB_CONFIG = {
    "dbname": "image_embeddings_db",
    "user": "postgres",
    "password": "user123",  # zamijeni stvarnom lozinkom
    "host": "localhost",
    "port": 5434
}

# Konekcija
conn = psycopg2.connect(**DB_CONFIG)

# Učitaj podatke iz baze
df = pd.read_sql("""
    SELECT image_name, video_name, latitude, longitude, timestamp, embedding
    FROM image_embeddings
    LIMIT 2000;
""", conn)

# Konverzija stringa u listu ako je potrebno
df["embedding"] = df["embedding"].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Priprema embeddinga za UMAP
X = np.vstack(df["embedding"].values)

# Smanjenje dimenzionalnosti u 2D
reducer = umap.UMAP(random_state=42)
X_2d = reducer.fit_transform(X)

# Dodaj u DataFrame
df["x"] = X_2d[:, 0]
df["y"] = X_2d[:, 1]

# Crtaj scatter plot s bojama po video_name
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x="x", y="y", hue="video_name", palette="tab10", s=30)

plt.title("UMAP vizualizacija embeddinga grupiranih po video_name")
plt.xlabel("UMAP Dim 1")
plt.ylabel("UMAP Dim 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Video Name")
plt.grid(True)
plt.tight_layout()
plt.show()

# Zatvori konekciju
conn.close()
