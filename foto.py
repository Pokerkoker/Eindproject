import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# ─────────────────────────────────────────────
# 1. PERFECT VS IMPERFECTE INFO
# ─────────────────────────────────────────────
def plot_information_types():
    labels = ["Perfecte info (Schaken)", "Imperfecte info (Poker)"]
    values = [1, 0]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Perfecte vs Imperfecte informatie")
    plt.ylabel("Volledige informatie (1 = ja, 0 = nee)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("info_types.png")
    plt.close()


# ─────────────────────────────────────────────
# 2. GAME TREE (ALGEMEEN)
# ─────────────────────────────────────────────
def plot_game_tree():
    G = nx.DiGraph()

    G.add_edges_from([
        ("Start", "Kaart 1"),
        ("Start", "Kaart 2"),
        ("Kaart 1", "Bet"),
        ("Kaart 1", "Check"),
    ])

    plt.figure()
    nx.draw(G, with_labels=True)
    plt.title("Extensive-form game (vereenvoudigd)")
    plt.savefig("game_tree.png")
    plt.close()


# ─────────────────────────────────────────────
# 3. KUHN POKER BOOM
# ─────────────────────────────────────────────
def plot_kuhn_tree():
    G = nx.DiGraph()

    edges = [
        ("Start", "A: Bet"),
        ("Start", "A: Check"),
        ("A: Bet", "B: Call"),
        ("A: Bet", "B: Fold"),
        ("A: Check", "B: Bet"),
        ("A: Check", "B: Check"),
    ]

    G.add_edges_from(edges)

    plt.figure()
    nx.draw(G, with_labels=True)
    plt.title("Kuhn Poker spelboom")
    plt.savefig("kuhn_tree.png")
    plt.close()


# ─────────────────────────────────────────────
# 4. NASH EVENWICHT MATRIX
# ─────────────────────────────────────────────
def plot_nash():
    matrix = np.array([[2, 0],
                       [0, 1]])

    plt.figure()
    plt.imshow(matrix)
    plt.colorbar()
    plt.title("Payoff matrix (Nash-evenwicht)")
    plt.savefig("nash.png")
    plt.close()


# ─────────────────────────────────────────────
# 5. CFR+ CONVERGENTIE (voorbeeld)
# ─────────────────────────────────────────────
def plot_cfr_convergence():
    iterations = [0, 100, 500, 1000, 2000]
    regret = [1.0, 0.5, 0.2, 0.1, 0.05]

    plt.figure()
    plt.plot(iterations, regret)
    plt.xlabel("Iteraties")
    plt.ylabel("Regret")
    plt.title("CFR+ convergentie")
    plt.savefig("cfr_convergence.png")
    plt.close()


# ─────────────────────────────────────────────
# 6. MCTS FLOW
# ─────────────────────────────────────────────
def plot_mcts_flow():
    steps = ["Selectie", "Expansie", "Simulatie", "Backpropagatie"]

    plt.figure()
    plt.plot(steps, [1, 1, 1, 1], marker='o')
    plt.yticks([])
    plt.title("MCTS stappen")
    plt.savefig("mcts_flow.png")
    plt.close()


# ─────────────────────────────────────────────
# 7. VERGELIJKING CFR+ vs MCTS
# (gebruik hier echte data indien mogelijk)
# ─────────────────────────────────────────────
def plot_comparison():
    iterations = [100, 500, 1000, 2000]
    cfr = [0.4, 0.55, 0.65, 0.7]
    mcts = [0.5, 0.58, 0.6, 0.61]

    plt.figure()
    plt.plot(iterations, cfr, label="CFR+")
    plt.plot(iterations, mcts, label="MCTS")

    plt.xlabel("Iteraties")
    plt.ylabel("Winrate")
    plt.title("CFR+ vs MCTS vergelijking")
    plt.legend()
    plt.grid()

    plt.savefig("comparison.png")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    plot_information_types()
    plot_game_tree()
    plot_kuhn_tree()
    plot_nash()
    plot_cfr_convergence()
    plot_mcts_flow()
    plot_comparison()

    print("Alle afbeeldingen zijn gegenereerd en opgeslagen.")


if __name__ == "__main__":
    main()