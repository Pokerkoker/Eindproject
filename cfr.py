from common.constants import A
from common.utils import init_sigma, init_empty_node_maps


class CounterfactualRegretMinimizationBase:

    def __init__(self, root, chance_sampling=False):
        self.root = root
        self.sigma = init_sigma(root)
        self.cumulative_regrets = init_empty_node_maps(root)
        self.cumulative_sigma = init_empty_node_maps(root)
        self.nash_equilibrium = init_empty_node_maps(root)
        self.chance_sampling = chance_sampling

    def _update_sigma(self, i):

        # alleen positieve regrets gebruiken
        positive_regrets = [max(r, 0) for r in self.cumulative_regrets[i].values()]
        rgrt_sum = sum(positive_regrets)

        for a in self.cumulative_regrets[i]:
            if rgrt_sum > 0:
                self.sigma[i][a] = max(self.cumulative_regrets[i][a], 0) / rgrt_sum
            else:
                self.sigma[i][a] = 1.0 / len(self.cumulative_regrets[i])

    def compute_nash_equilibrium(self):
        self.__compute_ne_rec(self.root)

    def __compute_ne_rec(self, node):

        if node.is_terminal():
            return

        i = node.inf_set()

        if node.is_chance():
            self.nash_equilibrium[i] = {a: node.chance_prob() for a in node.actions}

        else:
            sigma_sum = sum(self.cumulative_sigma[i].values())

            if sigma_sum > 0:
                self.nash_equilibrium[i] = {
                    a: self.cumulative_sigma[i][a] / sigma_sum for a in node.actions
                }
            else:
                self.nash_equilibrium[i] = {
                    a: 1.0 / len(node.actions) for a in node.actions
                }

        for k in node.children:
            self.__compute_ne_rec(node.children[k])

    # CFR+ REGRET UPDATE
    def _cumulate_cfr_regret(self, information_set, action, regret):

        new_regret = self.cumulative_regrets[information_set][action] + regret

        # CFR+ verschil: negatieve regrets worden 0
        self.cumulative_regrets[information_set][action] = max(new_regret, 0)

    def _cumulate_sigma(self, information_set, action, prob):
        self.cumulative_sigma[information_set][action] += prob

    def run(self, iterations):
        raise NotImplementedError("Please implement run method")

    def value_of_the_game(self):
        return self.__value_of_the_game_state_recursive(self.root)

    def _cfr_utility_recursive(self, state, reach_a, reach_b):

        children_states_utilities = {}

        if state.is_terminal():
            return state.evaluation()

        if state.is_chance():

            if self.chance_sampling:
                return self._cfr_utility_recursive(
                    state.sample_one(), reach_a, reach_b
                )

            else:
                chance_outcomes = {state.play(action) for action in state.actions}

                return state.chance_prob() * sum(
                    [
                        self._cfr_utility_recursive(outcome, reach_a, reach_b)
                        for outcome in chance_outcomes
                    ]
                )

        value = 0

        for action in state.actions:

            child_reach_a = reach_a * (
                self.sigma[state.inf_set()][action] if state.to_move == A else 1
            )

            child_reach_b = reach_b * (
                self.sigma[state.inf_set()][action] if state.to_move == -A else 1
            )

            child_state_utility = self._cfr_utility_recursive(
                state.play(action), child_reach_a, child_reach_b
            )

            value += self.sigma[state.inf_set()][action] * child_state_utility

            children_states_utilities[action] = child_state_utility

        (cfr_reach, reach) = (reach_b, reach_a) if state.to_move == A else (reach_a, reach_b)

        for action in state.actions:

            action_cfr_regret = state.to_move * cfr_reach * (
                children_states_utilities[action] - value
            )

            self._cumulate_cfr_regret(state.inf_set(), action, action_cfr_regret)

            self._cumulate_sigma(
                state.inf_set(),
                action,
                reach * self.sigma[state.inf_set()][action],
            )

        if self.chance_sampling:
            self._update_sigma(state.inf_set())

        return value


class CFRPlus(CounterfactualRegretMinimizationBase):

    def __init__(self, root):
        super().__init__(root=root, chance_sampling=False)

    def run(self, iterations=1):

        for _ in range(iterations):

            self._cfr_utility_recursive(self.root, 1, 1)

            self.__update_sigma_recursively(self.root)

    def __update_sigma_recursively(self, node):

        if node.is_terminal():
            return

        if not node.is_chance():
            self._update_sigma(node.inf_set())

        for k in node.children:
            self.__update_sigma_recursively(node.children[k])