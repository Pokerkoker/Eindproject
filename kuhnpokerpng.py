import networkx as nx
import matplotlib.pyplot as plt

# ─────────────────────────────
# Kleine MCTS-boom (begin spel)
# ─────────────────────────────
small_edges = [
    ("Root\nN=10", "Pass\nN=7"),
    ("Root\nN=10", "Bet\nN=3"),
    ("Pass\nN=7", "P2: Pass\nN=3"),
    ("Pass\nN=7", "P2: Bet\nN=4"),
]

small_pos = {
    "Root\nN=10": (0, 4),
    "Pass\nN=7": (-2, 3),
    "Bet\nN=3": (2, 3),
    "P2: Pass\nN=3": (-3, 2),
    "P2: Bet\nN=4": (-1, 2),
}

G_small = nx.DiGraph()
G_small.add_edges_from(small_edges)


# ─────────────────────────────
# Grote MCTS-boom (later in spel)
# ─────────────────────────────
large_edges = [
    ("Root\nN=100", "Pass\nN=60"),
    ("Root\nN=100", "Bet\nN=40"),

    ("Pass\nN=60", "P2: Pass\nN=20"),
    ("Pass\nN=60", "P2: Bet\nN=40"),

    ("P2: Bet\nN=40", "Call\nN=25"),
    ("P2: Bet\nN=40", "Fold\nN=15"),

    ("Bet\nN=40", "P2: Call\nN=30"),
    ("Bet\nN=40", "P2: Fold\nN=10"),
]

large_pos = {
    "Root\nN=100": (6, 4),
    "Pass\nN=60": (4, 3),
    "Bet\nN=40": (8, 3),
    "P2: Pass\nN=20": (3, 2),
    "P2: Bet\nN=40": (5, 2),
    "Call\nN=25": (5, 1),
    "Fold\nN=15": (3, 1),
    "P2: Call\nN=30": (9, 2),
    "P2: Fold\nN=10": (7, 2),
}

G_large = nx.DiGraph()
G_large.add_edges_from(large_edges)


# ─────────────────────────────
# Tekenen
# ─────────────────────────────
plt.figure(figsize=(14, 8))

nx.draw(G_small, pos=small_pos, with_labels=True, node_size=2500, node_color="lightblue", font_size=10, arrows=True)
nx.draw(G_large, pos=large_pos, with_labels=True, node_size=2500, node_color="lightgreen", font_size=10, arrows=True)

plt.title("MCTS-bomen: Klein (links) vs Groot (rechts)", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig("mcts_two_trees.png", dpi=300)
plt.show()