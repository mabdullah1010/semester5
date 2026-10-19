import libpyAI as ai
import sys, os, time, csv, math, random, atexit

# --- ARGS ---
PORT = int(sys.argv[1])
CHRO_STR = sys.argv[2]
INDEX = sys.argv[3]
LOG_DIR = sys.argv[4]
GA_NAME = sys.argv[5] if len(sys.argv) > 5 else "ga"   # nav / shooter / thruster

def binary_list_to_int(lst):
    return int(''.join(map(str, lst)), 2)

def chromo_decoder(flat_chromo):
    genes = [flat_chromo[i:i+5] for i in range(0, 55, 5)]
    p = [binary_list_to_int(list(gene)) for gene in genes]
    return [
        p[0] * 20, p[1], p[2] * 5, p[3], p[4], p[5] * 3,
        p[6] * 4, p[7] * 3, p[8] * 4, p[9], p[10]
    ]

gold_params = chromo_decoder(CHRO_STR)
os.chdir(LOG_DIR)

# --- Normalizers: IDENTICAL to the NN pipeline so GA states pool with NN states ---
def norm_distance(x): return x / 1500
def norm_speed(x): return x / 20
def norm_angle(x): return x / 360
def norm_angle_diff(x): return x / 180
def norm_shot_alert(x):
    if x is None or x >= 200: return 0
    if x <= 80: return 1
    return (200 - x) / 120

FEATS = ["heading","tracking","speed","aim","shot_alert","front_dist","back_dist",
         "back_left_dist","back_right_dist","closest_dist","furthest_dist",
         "closest_angle","furthest_angle","heading_tracking_diff","closest_diff",
         "furthest_diff","aim_diff"]

class StateLogger:
    """Per-tick (17 features + 4 GA actions). Separate file from BotStats."""
    def __init__(self, idx):
        self.path = f"States_{GA_NAME}_{idx}.csv"
        self.fh = open(self.path, "w", newline="")
        self.w = csv.writer(self.fh)
        self.w.writerow(FEATS + ["thrust", "turnLeft", "turnRight", "shoot"])
        self.n = 0
        atexit.register(self.close)
    def write(self, feats, acts):
        self.w.writerow([round(v, 6) for v in feats] + [int(a) for a in acts])
        self.n += 1
        if self.n % 2000 == 0: self.fh.flush()
    def close(self):
        try: self.fh.flush(); self.fh.close()
        except Exception: pass

class BotStats:
    def __init__(self, idx):
        self.path = f"Bot_{idx}_performance.csv"
        self.done_path = f"Bot_{idx}.done"
        self.ls, self.le = 0, 0
        self.lx, self.ly = None, None
        self.t_dist, self.t_thrust = 0.0, 0
        self.ws_t, self.we_t, self.mut_t, self.k_t, self.d_t = 0, 0, 0, 0, 0
        self.ws, self.we, self.mut, self.k, self.d = 0, 0, 0, 0, 0
        self.log_count = 0
        self.last_log = time.time()
        with open(self.path, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Log_Index", "Total_Dist", "Total_Thrusts",
                                     "Wall_Crash_Self", "Wall_Crash_Enemy", "Mutual_Crash",
                                     "Kills", "Deaths", "Current_Score"])

    def tick(self, x, y, ss, es):
        if self.lx is not None:
            d = math.sqrt((x - self.lx)**2 + (y - self.ly)**2)
            if d < 300: self.t_dist += d
        ds, de = ss - self.ls, es - self.le
        if -5.0 <= ds <= -2.5 and not (-5.0 <= de <= -2.5): self.ws += 1; self.ws_t += 1
        elif -5.0 <= de <= -2.5 and not (-5.0 <= ds <= -2.5): self.we += 1; self.we_t += 1
        elif (-5.0 <= ds <= -2.5) and (-5.0 <= de <= -2.5): self.mut += 1; self.mut_t += 1
        if (15 <= ds <= 25): self.k += 1; self.k_t += 1
        if (15 <= de <= 25): self.d += 1; self.d_t += 1
        self.lx, self.ly, self.ls, self.le = x, y, ss, es

    def log_row(self, score):
        self.log_count += 1
        row = [time.strftime("%H:%M:%S"), self.log_count, round(self.t_dist, 2), self.t_thrust,
               self.ws, self.we, self.mut, self.k, self.d, score]
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        self.ws = self.we = self.mut = self.k = self.d = 0
        if self.log_count >= 240:
            total = ["TOTAL", self.log_count, round(self.t_dist, 2), self.t_thrust,
                     self.ws_t, self.we_t, self.mut_t, self.k_t, self.d_t, score]
            with open(self.path, 'a', newline='') as f:
                csv.writer(f).writerow(total)
            state_log.close()
            open(self.done_path, 'w').close()
            sys.exit(0)

stats = BotStats(INDEX)
state_log = StateLogger(INDEX)

def AI_loop():
    if ai.selfAlive() != 1:
        stats.lx, stats.ly = None, None
        return

    rh = ai.selfHeadingDeg()
    cx, cy = ai.selfX(), ai.selfY()
    if rh is None or cx == 0: return

    ss, es = ai.selfScore(), ai.enemyScoreId(2)
    stats.tick(cx, cy, ss, es)

    # --- FULL CHROMOSOME PARAMETERS ---
    p = gold_params
    f_dist_th, m_speed_th, b_th = p[0], p[1], p[2]
    f_diff_th1, s_al_th, e_dist_th = p[4], p[5], p[6]
    c_dist_th1, c_dist_th2, f_diff_th2, a_diff_th = p[7], p[8], p[9], p[10]

    # --- SENSORS ---
    h = int(rh)
    t = int(ai.selfTrackingDeg())
    s = int(ai.selfSpeed())
    aim = int(ai.aimdir(0))
    alert = int(ai.shotAlert(0))

    ai.setTurnSpeedDeg(20)

    feelers = [ai.wallFeeler(1500, (h + i) % 360) for i in range(360)]
    front_dist = feelers[0]
    back_dist, bl_dist, br_dist = feelers[180], feelers[175], feelers[185]
    closest_dist, furthest_dist = min(feelers), max(feelers)

    ad = lambda a1, a2: ((a1 - a2 + 180) % 360) - 180
    closest_angle = (h + feelers.index(closest_dist)) % 360
    furthest_angle = (h + feelers.index(furthest_dist)) % 360
    ht_diff = ad(t, h)
    c_diff = ad(closest_angle, h)
    f_diff = ad(furthest_angle, h)
    aim_diff = ad(aim, h) if aim != -1 else 0   # guarded, for GA control logic

    # --- STATE VECTOR: featurized EXACTLY like the NN pipeline (raw aim, unguarded diff) ---
    raw_inputs = [
        norm_angle(h), norm_angle(t), norm_speed(s), norm_angle(aim),
        norm_shot_alert(alert), norm_distance(front_dist), norm_distance(back_dist),
        norm_distance(bl_dist), norm_distance(br_dist), norm_distance(closest_dist),
        norm_distance(furthest_dist), norm_angle(closest_angle), norm_angle(furthest_angle),
        norm_angle_diff(ht_diff), norm_angle_diff(c_diff),
        norm_angle_diff(f_diff), norm_angle_diff(ad(aim, h))
    ]

    action_thrust, action_left, action_right, action_shoot = 0, 0, 0, 0

    # --- THRUST LOGIC ---
    goingBackwards = abs(ht_diff) > 80
    backdanger = (back_dist < b_th or bl_dist < b_th or br_dist < b_th) and goingBackwards
    if front_dist > f_dist_th and s <= m_speed_th and abs(f_diff) < f_diff_th1:
        action_thrust = 1
    elif backdanger:
        action_thrust = 1
    elif 0 < alert < s_al_th:
        action_thrust = 1
    if s == 0 and random.randint(0, 50) == 4:
        action_thrust = 1

    # --- TURN LOGIC ---
    if ai.enemyDistance(0) < e_dist_th and s < 2:
        if aim_diff > 0: action_left = 1
        else: action_right = 1
    elif closest_dist < c_dist_th1 and s != 0:
        if c_diff > 0: action_right = 1
        else: action_left = 1
    elif closest_dist < c_dist_th2 or (0 < alert < s_al_th):
        if f_diff > 0: action_left = 1
        else: action_right = 1
    elif aim != -1 and abs(aim_diff) > a_diff_th:
        if aim_diff > 0: action_left = 1
        else: action_right = 1
    elif abs(f_diff) > f_diff_th1 and f_diff > 0:
        action_left = 1
    elif abs(f_diff) > f_diff_th2 and f_diff <= 0:
        action_right = 1

    # --- FIRING LOGIC (capture the decision, then act) ---
    if abs(aim_diff) <= 5 and aim != -1 and ai.wallFeeler(1500, aim) > ai.enemyDistance(0):
        action_shoot = 1

    # --- LOG STATE + GA ACTIONS (this tick) ---
    state_log.write(raw_inputs, [action_thrust, action_left, action_right, action_shoot])

    # --- EXECUTION ---
    ai.thrust(action_thrust)
    if action_thrust: stats.t_thrust += 1
    ai.turnLeft(action_left)
    ai.turnRight(action_right)
    if action_shoot: ai.fireShot()

    if time.time() - stats.last_log >= 1.0:
        stats.log_row(ss)
        stats.last_log = time.time()

ai.start(AI_loop, ["-name", f"GA_{INDEX}", "-port", str(PORT), "-join", "localhost"])
