import math
import random


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = list(state.actions) if not state.is_terminal() else []

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def uct_score(self, c=math.sqrt(2)):
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self):
        return max(self.children, key=lambda n: n.uct_score())


class MCTS:
    def __init__(self, iterations=1000):
        self.iterations = iterations

    def search(self, state):
        root = MCTSNode(state)

        for _ in range(self.iterations):
            node = self._select(root)
            if not node.state.is_terminal():
                node = self._expand(node)
            result = self._simulate(node)
            self._backpropagate(node, result)

        return max(root.children, key=lambda n: n.visits).action

    def _select(self, node):
        while not node.state.is_terminal() and node.is_fully_expanded():
            node = node.best_child()
        return node

    def _expand(self, node):
        action = random.choice(node.untried_actions)
        new_state = node.state.play(action)
        child = MCTSNode(new_state, parent=node, action=action)
        node.untried_actions.remove(action)
        node.children.append(child)
        return child

    def _simulate(self, node):
        state = node.state
        starting_player = node.state.to_move

        while not state.is_terminal():
            state = state.play(random.choice(state.actions))

        result = state.evaluation()

        # perspectief correct maken
        return result if starting_player == 1 else -result

    def _backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent