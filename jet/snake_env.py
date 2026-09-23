"""Fully observable Snake control task: the memory-neutral arm of the experiment.

The observation text carries the complete game state (body, heading, food,
score, budget) -- no hidden information exists, so episodic memory has no
theoretical advantage here. If the KV-memory model matches the memoryless
model on Snake while beating it on Maze, the memory claim is clean.

Map/food RNG/step logic reuse the repository's snake_game module unchanged.
"""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import snake_game  # noqa: E402

DIRS = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}
ORDER = ["north", "east", "south", "west"]


class SnakeEnv:
    def __init__(self, size, seed, target_food=3, budget=80):
        self.state = snake_game.make_snake(size, seed)
        self.size, self.seed = size, seed
        self.target_food, self.budget = target_food, budget
        self.t = 0

    @property
    def done(self):
        return self.state["done"] or self.t >= self.budget or self.state["score"] >= self.target_food

    @property
    def success(self):
        return self.state["score"] >= self.target_food or self.state["outcome"] == "win"

    def observation(self):
        text = snake_game.render_request(self.state)["state"]
        return (text + f"\nTarget: eat {self.target_food} food items. "
                f"Remaining moves: {self.budget - self.t}. Eaten so far: {self.state['score']}.")

    def candidates(self):
        return {a: f"Move the head {a}." for a in snake_game.valid_actions(self.state)}

    def step(self, action):
        before = self.state["score"]
        self.state = snake_game.step(self.state, action)
        self.t += 1
        if self.state["outcome"] in ("wall_collision", "self_collision"):
            return self.state["outcome"]
        if self.state["score"] > before:
            return "ate"
        return "moved"

    # ---- teacher: BFS to food, flood-fill survival fallback ----
    def _safe(self, action):
        return snake_game.one_step_safe(self.state, action)

    def teacher_action(self):
        actions = snake_game.valid_actions(self.state)
        if not actions:
            return None
        safe = [a for a in actions if self._safe(a)]
        if not safe:
            return actions[0]
        food, body, size = self.state["food"], self.state["body"], self.size
        if food is None:
            return safe[0]
        blocked = {tuple(c) for c in body[:-1]}
        start = tuple(body[0])
        prev, seen, queue = {}, {start}, deque([start])
        while queue:
            cell = queue.popleft()
            if cell == tuple(food):
                break
            for d, (dr, dc) in DIRS.items():
                nb = (cell[0] + dr, cell[1] + dc)
                if 0 <= nb[0] < size and 0 <= nb[1] < size and nb not in blocked and nb not in seen:
                    seen.add(nb)
                    prev[nb] = cell
                    queue.append(nb)
        if tuple(food) in seen:
            cell, path = tuple(food), []
            while cell != start:
                path.append(cell)
                cell = prev[cell]
            nxt = path[-1]
            for d, (dr, dc) in DIRS.items():
                if (start[0] + dr, start[1] + dc) == nxt and d in safe:
                    return d
        # survival fallback: pick the safe action maximizing reachable free space
        def reach(action):
            dr, dc = DIRS[action]
            head = (body[0][0] + dr, body[0][1] + dc)
            occ = {tuple(c) for c in body[:-1]}
            q, seen2 = deque([head]), {head}
            while q:
                c = q.popleft()
                for dr2, dc2 in DIRS.values():
                    nb = (c[0] + dr2, c[1] + dc2)
                    if 0 <= nb[0] < size and 0 <= nb[1] < size and nb not in occ and nb not in seen2:
                        seen2.add(nb)
                        q.append(nb)
            return len(seen2)
        return max(safe, key=reach)


def run_episode(size, seed, epsilon=0.0, rng=None, target_food=3, budget=80):
    import random as _random
    rng = rng or _random.Random(seed)
    env = SnakeEnv(size, seed, target_food=target_food, budget=budget)
    steps = []
    while not env.done:
        action = env.teacher_action()
        if rng.random() < epsilon:
            action = rng.choice(list(env.candidates()))
        obs, cands, pos = env.observation(), env.candidates(), list(env.state["body"][0])
        event = env.step(action)
        steps.append({"t": len(steps) + 1, "pos": pos, "obs": obs, "candidates": cands,
                      "action": action, "event": event, "remaining_before": env.budget - env.t + 1})
    return {"size": size, "seed": seed, "epsilon": epsilon, "success": env.success,
            "n_steps": len(steps), "target_food": target_food, "budget": budget, "steps": steps}
