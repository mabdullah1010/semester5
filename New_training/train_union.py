import argparse, glob, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

FEATS = ["heading","tracking","speed","aim","shot_alert","front_dist","back_dist",
         "back_left_dist","back_right_dist","closest_dist","furthest_dist",
         "closest_angle","furthest_angle","heading_tracking_diff","closest_diff",
         "furthest_diff","aim_diff"]
ACTS = ["thrust", "turnLeft", "turnRight", "shoot"]

parser = argparse.ArgumentParser()
parser.add_argument("-glob", nargs="+", required=True,
                    help="one or more globs of State CSVs. "
                         "UNION: pass all three GA trees. CONTROL: pass one GA's tree.")
parser.add_argument("-name", required=True, help="output tag, e.g. union or ga_nav")
parser.add_argument("-epochs", type=int, default=10)
parser.add_argument("-bs", type=int, default=4096)
parser.add_argument("-lr", type=float, default=1e-3)
args = parser.parse_args()

class ShooterModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(in_features, 256), nn.Linear(256, 256), nn.Linear(256, 128)
        self.out = nn.Linear(128, 4)
    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); x = F.relu(self.fc3(x))
        return self.out(x)

# --- Load + pool ---
files = []
for g in args.glob: files += glob.glob(g, recursive=True)
files = sorted(set(files))
if not files:
    raise SystemExit("No CSVs matched. Check your -glob.")
print(f"[{args.name}] loading {len(files)} files...")
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
print(f"[{args.name}] {len(df):,} total state-action rows")

X = df[FEATS].to_numpy(dtype=np.float32)
Y = df[ACTS].to_numpy(dtype=np.float32)

# Diagnostic: how often do near-identical states carry conflicting labels across the pool?
# (Cheap proxy: label entropy per action = how split the targets are overall.)
pos = Y.mean(axis=0)
print(f"[{args.name}] action positive-rates:",
      {a: round(float(r), 3) for a, r in zip(ACTS, pos)})

scaler = StandardScaler().fit(X)
Xs = scaler.transform(X).astype(np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ShooterModel(17).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.BCEWithLogitsLoss()

ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(Y))
dl = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True)

print(f"[{args.name}] training on {device}, {args.epochs} epochs")
for ep in range(1, args.epochs + 1):
    model.train(); tot, nb = 0.0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    # per-action accuracy at 0.5 on a sample
    model.eval()
    with torch.no_grad():
        idx = np.random.choice(len(Xs), size=min(200000, len(Xs)), replace=False)
        pr = torch.sigmoid(model(torch.from_numpy(Xs[idx]).to(device))).cpu().numpy()
        acc = ((pr >= 0.5).astype(np.float32) == Y[idx]).mean(axis=0)
    print(f"  epoch {ep:2d}  loss {tot/nb:.4f}  acc " +
          " ".join(f"{a}={v:.3f}" for a, v in zip(ACTS, acc)))

out = f"{args.name}.pth"
torch.save({"model_state_dict": model.state_dict(), "scaler": scaler}, out)
print(f"[{args.name}] saved {out}")
