import argparse, glob, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import sklearn.preprocessing._data as _skdata

StandardScaler = _skdata.StandardScaler
torch.serialization.add_safe_globals([StandardScaler])

FEATS = ["heading","tracking","speed","aim","shot_alert","front_dist","back_dist",
         "back_left_dist","back_right_dist","closest_dist","furthest_dist",
         "closest_angle","furthest_angle","heading_tracking_diff","closest_diff",
         "furthest_diff","aim_diff"]

parser = argparse.ArgumentParser()
parser.add_argument("-glob", nargs="+",
                    default=["gen_nav/**/States_*.csv",
                             "gen_shooter/**/States_*.csv",
                             "gen_thruster/**/States_*.csv"],
                    help="globs of logged real states (the 17 feature columns are used)")
parser.add_argument("-synthetic", type=int, default=0,
                    help="if >0, ignore CSVs and sample this many uniform-random states instead")
parser.add_argument("-seed", type=int, default=0)
args = parser.parse_args()

class ShooterModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(in_features, 256), nn.Linear(256, 256), nn.Linear(256, 128)
        self.out = nn.Linear(128, 4)
    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); x = F.relu(self.fc3(x))
        return self.out(x)

NAMES = ["Nav", "Shooter", "Thruster"]
models, scalers = [], []
for fn in ["nav.pth", "shooter.pth", "thruster.pth"]:
    ck = torch.load(fn, map_location="cpu", weights_only=False)
    s = ck["scaler"]; m = ShooterModel(s.mean_.shape[0])
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    models.append(m); scalers.append(s)

# --- Build X: real logged states (default) or synthetic fallback ---
if args.synthetic > 0:
    rng = np.random.default_rng(args.seed)
    LOW  = np.array([0,0,0,0, 0, 0,0,0,0,0,0, 0,0, -1,-1,-1,-1], dtype=np.float32)
    HIGH = np.array([1,1,1.55,1, 1, 1,1,1,1,1,1, 1,1,  1, 1, 1, 1], dtype=np.float32)
    X = rng.uniform(LOW, HIGH, size=(args.synthetic, 17)).astype(np.float32)
    print(f"\n[SYNTHETIC] {args.synthetic} uniform-random states "
          f"(off-distribution; inflates disagreement — not paper-grade)")
else:
    files = []
    for g in args.glob: files += glob.glob(g, recursive=True)
    files = sorted(set(files))
    if not files:
        raise SystemExit("No state CSVs matched. Log the GAs first, or pass -synthetic N.")
    df = pd.concat((pd.read_csv(f)[FEATS] for f in files), ignore_index=True)
    X = df.to_numpy(dtype=np.float32)
    print(f"\n[REAL] {len(files)} files, {len(X):,} on-distribution states (paper-grade)")

N = len(X)

# Binarized decisions per expert: (3 experts, N states, 4 actions)
dec = np.zeros((3, N, 4), dtype=np.int8)
with torch.no_grad():
    for i in range(3):
        xs = scalers[i].transform(X)
        p = torch.sigmoid(models[i](torch.tensor(xs, dtype=torch.float32))).numpy()
        dec[i] = (p >= 0.5).astype(np.int8)

ACTIONS = ["thrust", "turnLeft", "turnRight", "shoot"]
votes = dec.sum(axis=0)
disagree = (votes > 0) & (votes < 3)
majority = votes >= 2

print(f"\n{'action':<10}{'disagree%':>11}{'any-want%':>11}{'majority%':>11}{'unanimous-yes%':>16}")
for a, name in enumerate(ACTIONS):
    print(f"{name:<10}{disagree[:,a].mean()*100:>10.1f}%{(votes[:,a]>0).mean()*100:>10.1f}%"
          f"{majority[:,a].mean()*100:>10.1f}%{(votes[:,a]==3).mean()*100:>15.1f}%")

shooter_fire = dec[1,:,3] == 1
if shooter_fire.sum() > 0:
    print(f"\nWhen Shooter wants to fire ({shooter_fire.mean()*100:.1f}% of states):")
    print(f"  Nav also fires:      {(dec[0, shooter_fire, 3]==1).mean()*100:.1f}%")
    print(f"  Thruster also fires: {(dec[2, shooter_fire, 3]==1).mean()*100:.1f}%")
    print(f"  -> hard-vote (2/3) fires in {(majority[shooter_fire,3]).mean()*100:.1f}% of these")
else:
    print("\nShooter never wanted to fire in this sample.")

both_turn = (dec[:,:,1].max(0) == 1) & (dec[:,:,2].max(0) == 1)
print(f"\nStates where experts split left-vs-right: {both_turn.mean()*100:.1f}%")
print("(high here = contradictory movement labels -> union net can't fit both)\n")