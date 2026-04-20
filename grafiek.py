import time
import random
import matplotlib.pyplot as plt

from games.kuhn import KuhnRootChanceGameState
from cfr import CFRPlus
from mcts import MCTS


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_ITERATIONS = 10000
STEP = 1000
GAMES_PER_STEP = 200


# ─────────────────────────────────────────────
# GAME SETUP
# ─────────────────────────────────────────────
card_combinations = ["12", "13", "21", "23", "31", "32"]


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

    while not state.is_terminal():

        if state.to_move == 1:
            strategy = cfr.nash_equilibrium[state.inf_set()]
            action = sample_action(strategy)
        else:
            action = mcts.search(state)

        state = state.play(action)

    return state.evaluation()


# ─────────────────────────────────────────────
# EXPERIMENT
# ─────────────────────────────────────────────
iterations_list = []
cfr_winrates = []
mcts_winrates = []

print("Experiment gestart...\n")

for it in range(STEP, MAX_ITERATIONS + 1, STEP):

    print(f"Iteraties: {it}")

    root = KuhnRootChanceGameState(actions=card_combinations)

    # CFR+
    cfr = CFRPlus(root)
    cfr.run(iterations=it)
    cfr.compute_nash_equilibrium()

    # MCTS
    mcts = MCTS(iterations=it)

    cfr_wins = 0
    mcts_wins = 0

    for _ in range(GAMES_PER_STEP):
        result = play_game(root, cfr, mcts)

        if result > 0:
            cfr_wins += 1
        elif result < 0:
            mcts_wins += 1

    cfr_winrate = cfr_wins / GAMES_PER_STEP
    mcts_winrate = mcts_wins / GAMES_PER_STEP

    iterations_list.append(it)
    cfr_winrates.append(cfr_winrate)
    mcts_winrates.append(mcts_winrate)


# ─────────────────────────────────────────────
# GRAFIEK (PROPER)
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))

plt.plot(iterations_list, cfr_winrates,
         label="CFR+", linewidth=2)

plt.plot(iterations_list, mcts_winrates,
         label="MCTS", linewidth=2)

plt.title("Convergentiesnelheid en strategiekwaliteit", fontsize=14)
plt.xlabel("Aantal iteraties", fontsize=12)
plt.ylabel("Winrate", fontsize=12)

plt.xlim(0, MAX_ITERATIONS)
plt.ylim(0, 1)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig("convergence.png", dpi=300)
plt.show()


print("\nKlaar. Grafiek opgeslagen als 'convergence.png'")

import matplotlib.pyplot as plt
import networkx as nx

# ─────────────────────────────────────────────
# 1. KUHN POKER FLOW
# ─────────────────────────────────────────────
def plot_kuhn_poker():
    G = nx.DiGraph()

    edges = [
        ("Start\n(ante)", "Speler A"),
        ("Speler A", "Bet"),
        ("Speler A", "Check"),
        ("Bet", "Speler B"),
        ("Check", "Speler B"),
        ("Speler B", "Call"),
        ("Speler B", "Fold"),
        ("Speler B", "Bet"),
        ("Call", "Showdown"),
        ("Fold", "A wint"),
        ("Bet", "A beslist opnieuw"),
    ]

    G.add_edges_from(edges)

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=2500, font_size=8)

    plt.title("Kuhn Poker spelverloop")
    plt.savefig("kuhn_poker.png", dpi=300)
    plt.close()


# ─────────────────────────────────────────────
# 2. MCTS BOOM + STAPPEN
# ─────────────────────────────────────────────
def plot_mcts_tree():
    G = nx.DiGraph()

    edges = [
        ("Root", "Selectie"),
        ("Selectie", "Expansie"),
        ("Expansie", "Simulatie"),
        ("Simulatie", "Backpropagatie"),
        ("Backpropagatie", "Update waarden"),
    ]

    G.add_edges_from(edges)

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=9)

    plt.title("MCTS proces (4 stappen)")
    plt.savefig("mcts_process.png", dpi=300)
    plt.close()


# ─────────────────────────────────────────────
# 3. CFR+ CONCEPT (REGRET → STRATEGIE)
# ─────────────────────────────────────────────
def plot_cfr_flow():
    G = nx.DiGraph()

    edges = [
        ("Actie keuze", "Uitkomst"),
        ("Uitkomst", "Regret berekenen"),
        ("Regret berekenen", "Positieve regret behouden"),
        ("Positieve regret behouden", "Strategie update"),
        ("Strategie update", "Nieuwe actie keuze"),
    ]

    G.add_edges_from(edges)

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=7)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightgreen", font_size=9)

    plt.title("CFR+ leerproces")
    plt.savefig("cfr_process.png", dpi=300)
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    plot_kuhn_poker()
    plot_mcts_tree()
    plot_cfr_flow()

    print("Extra afbeeldingen gemaakt:")
    print("- kuhn_poker.png")
    print("- mcts_process.png")
    print("- cfr_process.png")


if __name__ == "__main__":
    main()