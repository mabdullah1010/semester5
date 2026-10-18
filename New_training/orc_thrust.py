import subprocess, os, time, sys, shutil

# --- SETTINGS ---
X_RUNS = 25
Y_INSTANCES = 16
XP_PATH = "./xpilots"
BOT_SCRIPT = "agent_worker_thrust.py"
BASE_PORT = 10001

# Which GA you are logging this pass. CHANGE THIS and RAW_CHRO together for each GA.
GA_NAME = "nav"                 # "nav" | "shooter" | "thruster"

RESULTS_DIR = f"gen_{GA_NAME}/states_{GA_NAME}"

# GA Chromosome (11 genes x 5 bits). PASTE THE CHROMOSOME FOR *THIS* GA.

RAW_CHRO= [
            [
                0,
                0,
                1,
                0,
                1
            ],
            [
                1,
                0,
                0,
                1,
                0
            ],
            [
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                0,
                1
            ],
            [
                1,
                1,
                0,
                0,
                0
            ],
            [
                1,
                0,
                0,
                0,
                0
            ],
            [
                0,
                0,
                1,
                0,
                0
            ],
            [
                0,
                1,
                1,
                0,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1
            ],
            [
                0,
                1,
                1,
                1,
                1
            ],
            [
                0,
                1,
                1,
                1,
                1
            ]
        ]
        


FLAT_CHRO = "".join(["".join(map(str, g)) for g in RAW_CHRO])

def hard_cleanup():
    subprocess.run(["pkill", "-9", "xpilots"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", BOT_SCRIPT], stderr=subprocess.DEVNULL)
    time.sleep(2)

if __name__ == "__main__":
    if os.path.exists(RESULTS_DIR): shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR)
    print(f"Logging GA '{GA_NAME}' -> {RESULTS_DIR}/")

    for run in range(1, X_RUNS + 1):
        print(f"\n--- STARTING RUN {run}/{X_RUNS} ---")
        run_dir = os.path.join(RESULTS_DIR, f"run_{run}")
        os.makedirs(run_dir)

        hard_cleanup()

        for i in range(Y_INSTANCES):
            port = BASE_PORT + i
            subprocess.Popen([XP_PATH, "-map", "lifeless.xp", "-port", str(port), "-headless", "-noquit", "-switchbase", "1"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(5)

        for i in range(Y_INSTANCES):
            port = BASE_PORT + i
            # 5th arg = GA_NAME, so the worker names its state CSV States_<ga>_<idx>.csv
            subprocess.Popen(["python3", BOT_SCRIPT, str(port), FLAT_CHRO, str(i), run_dir, GA_NAME])
            time.sleep(0.2)

        while True:
            done_files = [f for f in os.listdir(run_dir) if f.endswith(".done")]
            sys.stdout.write(f"\rProgress: {len(done_files)}/{Y_INSTANCES} instances complete...")
            sys.stdout.flush()
            if len(done_files) >= Y_INSTANCES:
                print("\nRun complete.")
                break
            time.sleep(2)

    hard_cleanup()
    print(f"\nDONE. State CSVs under {RESULTS_DIR}/run_*/States_{GA_NAME}_*.csv")
