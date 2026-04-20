import time
import random
import pandas as pd
import matplotlib.pyplot as plt

from games.kuhn import KuhnRootChanceGameState
from cfr import CFRPlus
from mcts import MCTS


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_ITERATIONS = 50000
STEP = 500

GAMES_PER_STEP = 1000     # 🔥 verhoogd (was 200)
NUM_RUNS = 3              # 🔥 nieuw (voor averaging)

MCTS_ITERATIONS = 1000    # 🔥 FIX: constant houden!

card_combinations = ["12", "13", "21", "23", "31", "32"]


# ─────────────────────────────────────────────
# HELPER FUNCTIES
# ─────────────────────────────────────────────
def sample_action(strategy):
    r = random.random()
    cumulative = 0
    for action, prob in strategy.items():
        cumulative += prob
        if r <= cumulative:
            return action
    return list(strategy.keys())[0]


def play_game(root, cfr, mcts):
    state = root.sample_one()

    cfr_time = 0.0
    mcts_time = 0.0

    while not state.is_terminal():

        if state.to_move == 1:
            start = time.perf_counter()
            strategy = cfr.nash_equilibrium[state.inf_set()]
            action = sample_action(strategy)
            cfr_time += time.perf_counter() - start

        else:
            start = time.perf_counter()
            action = mcts.search(state)
            mcts_time += time.perf_counter() - start

        state = state.play(action)

    return state.evaluation(), cfr_time, mcts_time


# ─────────────────────────────────────────────
# EXPERIMENT
# ─────────────────────────────────────────────
def run_experiment():
    results = []

    print("Experiment gestart...")

    for it in range(STEP, MAX_ITERATIONS + 1, STEP):
        print(f"Iteraties: {it}")

        root = KuhnRootChanceGameState(actions=card_combinations)

        total_cfr_wins = 0
        total_mcts_wins = 0

        total_cfr_time = 0.0
        total_mcts_time = 0.0
        total_train_time = 0.0

        # 🔁 meerdere runs
        for run in range(NUM_RUNS):

            # ─── CFR TRAINING ───
            start = time.perf_counter()
            cfr = CFRPlus(root)
            cfr.run(iterations=it)
            cfr.compute_nash_equilibrium()
            train_time = time.perf_counter() - start

            total_train_time += train_time

            # vaste MCTS (niet opnieuw maken per move)
            mcts = MCTS(iterations=MCTS_ITERATIONS)

            # verdeel games over runs
            games_per_run = GAMES_PER_STEP // NUM_RUNS

            for _ in range(games_per_run):
                result, cfr_time, mcts_time = play_game(root, cfr, mcts)

                total_cfr_time += cfr_time
                total_mcts_time += mcts_time

                if result > 0:
                    total_cfr_wins += 1
                elif result < 0:
                    total_mcts_wins += 1

        results.append({
            "iterations": it,
            "cfr_winrate": total_cfr_wins / GAMES_PER_STEP,
            "mcts_winrate": total_mcts_wins / GAMES_PER_STEP,
            "cfr_time": total_cfr_time / GAMES_PER_STEP,
            "mcts_time": total_mcts_time / GAMES_PER_STEP,
            "cfr_train_time": total_train_time / NUM_RUNS
        })

    df = pd.DataFrame(results)

    # Extra metrics
    df["winrate_diff"] = df["cfr_winrate"] - df["mcts_winrate"]
    df["avg_time_diff"] = df["cfr_time"] - df["mcts_time"]

    df["cfr_total_time_per_game"] = df["cfr_time"] + (
        df["cfr_train_time"] / GAMES_PER_STEP
    )

    return df


# ─────────────────────────────────────────────
# GRAFIEKEN
# ─────────────────────────────────────────────
def make_plots(df):

    # 🔥 betere smoothing
    df["cfr_smooth"] = df["cfr_winrate"].rolling(5).mean()
    df["mcts_smooth"] = df["mcts_winrate"].rolling(5).mean()

    # ─── Convergentie ───
    plt.figure(figsize=(8, 5))
    plt.plot(df["iterations"], df["cfr_smooth"], marker='o', label="CFR+")
    plt.plot(df["iterations"], df["mcts_smooth"], marker='s', label="MCTS")
    plt.xlabel("Aantal iteraties")
    plt.ylabel("Winrate")
    plt.title("Convergentie en strategiekwaliteit")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergencea.png", dpi=300)
    plt.close()

    # ─── Tijd ───
    plt.figure(figsize=(8, 5))
    plt.plot(df["iterations"], df["cfr_time"], marker='o', label="CFR+ (decision)")
    plt.plot(df["iterations"], df["mcts_time"], marker='s', label="MCTS")
    plt.plot(df["iterations"], df["cfr_train_time"], marker='^', label="CFR+ (training)")
    plt.plot(df["iterations"], df["cfr_total_time_per_game"], linestyle='--', label="CFR+ (total/game)")
    plt.xlabel("Aantal iteraties")
    plt.ylabel("Tijd (s)")
    plt.title("Tijd: training vs beslissing")
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("timea.png", dpi=300)
    plt.close()

    # ─── Winrate verschil ───
    plt.figure(figsize=(8, 5))
    diff = df["winrate_diff"]
    plt.plot(df["iterations"], diff, marker='o', color="purple")
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Aantal iteraties")
    plt.ylabel("Winrate verschil")
    plt.title("Voordeel CFR+ t.o.v. MCTS")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("differencea.png", dpi=300)
    plt.close()

    print("Grafieken opgeslagen.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Start experiment...\n")

    df = run_experiment()

    df = df.round(6)

    with open("experiment_results.csv", "w") as f:
        f.write("EXPERIMENT INFO\n")
        f.write(f"MAX_ITERATIONS,{MAX_ITERATIONS}\n")
        f.write(f"STEP,{STEP}\n")
        f.write(f"GAMES_PER_STEP,{GAMES_PER_STEP}\n")
        f.write(f"NUM_RUNS,{NUM_RUNS}\n")
        f.write(f"MCTS_ITERATIONS,{MCTS_ITERATIONS}\n")
        f.write("\n")
        df.to_csv(f, index=False)

    print("\nCSV opgeslagen als: experiment_results.csv")

    make_plots(df)

    print("\nKlaar!")