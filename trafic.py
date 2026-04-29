import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import TensorDataset, DataLoader

# Appareil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Appareil utilisé : {device}")

# ─── ÉTAPE 1 : Charger les données ───────────────────────────────────────────
print("Chargement des données...")
df = pd.read_csv("chicago_traffic.csv")

print(f"✅ Données chargées !")
print(f"   - Nombre de lignes : {len(df)}")
print(f"   - Colonnes : {list(df.columns)}")
print("\nAperçu des données :")
print(df.head())

# ─── ÉTAPE 2 : Nettoyer les données ──────────────────────────────────────────
print("\nNettoyage des données...")

df = df[['TIME', 'SEGMENTID', 'SPEED']]
df = df[df['SPEED'] > 0]
df['TIME'] = pd.to_datetime(df['TIME'], format='%m/%d/%Y %I:%M:%S %p')
df = df[(df['TIME'].dt.year >= 2013) & (df['TIME'].dt.year <= 2018)]

print(f"✅ Données nettoyées !")
print(f"   - Lignes restantes : {len(df)}")
print(f"   - Période : {df['TIME'].min()} → {df['TIME'].max()}")
print(f"   - Nombre de capteurs : {df['SEGMENTID'].nunique()}")

# ─── ÉTAPE 3 : Structurer les données ────────────────────────────────────────
print("\nStructuration des données...")

top_sensors = df['SEGMENTID'].value_counts().head(200).index
df = df[df['SEGMENTID'].isin(top_sensors)]

df_pivot = df.pivot_table(index='TIME', columns='SEGMENTID', values='SPEED', aggfunc='mean')
df_pivot = df_pivot.ffill().bfill()

data = df_pivot.values.astype(np.float32)

print(f"✅ Matrice créée !")
print(f"   - Shape : {data.shape}")
print(f"   - Capteurs : {data.shape[1]}")
print(f"   - Pas de temps : {data.shape[0]}")

# ─── ÉTAPE 4 : Construire le graphe ──────────────────────────────────────────
print("\nConstruction du graphe...")
N = data.shape[1]

edge_index = []
for i in range(N):
    for j in range(N):
        if abs(i - j) == 1:
            edge_index.append([i, j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

print(f"✅ Graphe construit !")
print(f"   - Nœuds : {N}")
print(f"   - Arêtes : {edge_index.shape[1]}")

# ─── ÉTAPE 5 : Préparer entrées / sorties ────────────────────────────────────
print("\nPréparation des séquences...")

IN_STEPS  = 12  # 12 pas en entrée (historique)
OUT_STEPS = 3   # 3 pas en sortie (prédiction)

# Normalisation
data_mean = data.mean()
data_std  = data.std()
data_norm = (data - data_mean) / data_std

X, Y = [], []
for t in range(len(data_norm) - IN_STEPS - OUT_STEPS):
    X.append(data_norm[t : t + IN_STEPS])
    Y.append(data_norm[t + IN_STEPS : t + IN_STEPS + OUT_STEPS])

X = torch.tensor(np.array(X), dtype=torch.float32)
Y = torch.tensor(np.array(Y), dtype=torch.float32)

# Découpage train / test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

print(f"✅ Séquences prêtes !")
print(f"   - Entraînement : {len(X_train)} séquences")
print(f"   - Test         : {len(X_test)} séquences")

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ─── ÉTAPE 6 : Définir le modèle STGNN ───────────────────────────────────────
class STGNN(nn.Module):
    def __init__(self, num_nodes, in_steps, out_steps, hidden=64):
        super(STGNN, self).__init__()
        # Module spatial (GCN)
        self.gcn1 = GCNConv(in_steps, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        # Module temporel (GRU)
        self.gru  = nn.GRU(hidden, hidden, batch_first=True)
        # Couche de sortie
        self.fc   = nn.Linear(hidden, out_steps)

    def forward(self, x, edge_index):
        # x : [batch, in_steps, num_nodes]
        batch = x.shape[0]
        # Appliquer GCN sur chaque pas de temps
        x = x.permute(0, 2, 1)  # [batch, num_nodes, in_steps]
        out = []
        for i in range(x.shape[0]):
            h = F.relu(self.gcn1(x[i], edge_index))
            h = F.relu(self.gcn2(h, edge_index))
            out.append(h)
        x = torch.stack(out)     # [batch, num_nodes, hidden]
        # Appliquer GRU
        x, _ = self.gru(x)
        x = self.fc(x)           # [batch, num_nodes, out_steps]
        return x

model = STGNN(num_nodes=200, in_steps=IN_STEPS, out_steps=OUT_STEPS)
print(f"\n✅ Modèle STGNN créé !")
print(f"   - Paramètres : {sum(p.numel() for p in model.parameters()):,}")


# ─── ÉTAPE 7 : Entraînement ───────────────────────────────────────────────────
print("\nEntraînement du modèle...")

BATCH_SIZE = 64
EPOCHS     = 10

train_loader = DataLoader(
    TensorDataset(X_train, Y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch, edge_index)
        loss   = criterion(y_pred, y_batch.permute(0, 2, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"   Epoch {epoch+1}/{EPOCHS} - Loss : {avg_loss:.4f}")

print("✅ Entraînement terminé !")