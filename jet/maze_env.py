"""Full-control maze POMDP for the episodic-memory experiment.

Contract (this file):
- The model controls every move: candidates are always the four directions.
- Observation is the agent-centered 5x5 ASCII window plus coordinates, goal,
  and remaining budget. No code-maintained memory is included in the raw form;
  the code-memory arm builds its text from the same recorded trajectory.
- Moving into a wall leaves the agent in place and consumes budget (non-fatal);
  the event is observable and becomes part of the episode transcript.
- Success is reaching the goal before the budget expires.

Maps come from the repository's scaled_maze generator (deterministic by seed).
"""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scaled_maze import make_maze, validate_state  # noqa: E402
from evaluate_composed_maze import render_local_request  # noqa: E402

DIRS = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}
ORDER = ["north", "east", "south", "west"]


class MazeEnv:
    def __init__(self, size, seed, topology="tree", budget=None, start=None):
        source = make_maze(size, seed, topology)
        validate_state(source)
        self.size = source["size"]
        self.walls = set(map(tuple, source["walls"]))
        self.goal = tuple(source["goal"])
        self.budget = budget if budget is not None else size * size
        self.position = tuple(start) if start is not None else tuple(source["position"])
        self.t = 0
        self.done = self.position == self.goal
        self.success = self.done

    def _open(self, pos, direction):
        dr, dc = DIRS[direction]
        r, c = pos[0] + dr, pos[1] + dc
        return (0 <= r < self.size and 0 <= c < self.size and (r, c) not in self.walls), (r, c)

    def observation(self):
        world = {"game": "scaled_maze", "size": self.size, "walls": sorted(map(list, self.walls)),
                 "position": list(self.position), "goal": list(self.goal)}
        local = render_local_request(world, window_size=5)["state"]
        return (local + f"\nGoal coordinate: ({self.goal[0]},{self.goal[1]}). "
                f"Remaining attempts: {self.budget - self.t}.")

    def candidates(self):
        pos = self.position
        return {d: f"Move {d} to ({pos[0] + DIRS[d][0]},{pos[1] + DIRS[d][1]})." for d in ORDER}

    def step(self, direction):
        if self.done:
            raise RuntimeError("step after terminal state")
        if direction not in DIRS:
            raise ValueError(f"unknown direction {direction}")
        self.t += 1
        ok, target = self._open(self.position, direction)
        if not ok:
            event = "blocked"
        else:
            self.position = target
            event = "goal" if self.position == self.goal else "moved"
        self.done = event == "goal" or self.t >= self.budget
        self.success = event == "goal"
        return event

    def bfs_direction(self):
        """First direction of a shortest open-cell path to the goal (oracle)."""
        if self.position == self.goal:
            return None
        start, frontier, seen = self.position, deque(), {self.position: None}
        for _d in ORDER:
            ok, nb = self._open(start, _d)
            if ok:
                frontier.append((nb, _d))
                seen.setdefault(nb, _d)
        while frontier:
            cell, first = frontier.popleft()
            if cell == self.goal:
                return first
            for _d in ORDER:
                ok, nb = self._open(cell, _d)
                if ok and nb not in seen:
                    seen[nb] = first
                    frontier.append((nb, first))
        raise RuntimeError("goal unreachable")


def run_episode(size, seed, epsilon=0.0, rng=None, topology="tree", budget=None):
    """Roll one episode with the epsilon-greedy BFS teacher; record everything."""
    rng = rng or __import__("random").Random(seed)
    env = MazeEnv(size, seed, topology=topology, budget=budget)
    steps, blocked_pairs = [], set()
    while not env.done:
        action = env.bfs_direction()
        if rng.random() < epsilon:
            action = rng.choice(ORDER)
        obs, cands, pos, remaining = env.observation(), env.candidates(), env.position, env.budget - env.t
        repeat = (pos, action) in blocked_pairs
        event = env.step(action)
        if event == "blocked":
            blocked_pairs.add((pos, action))
        steps.append({"t": len(steps) + 1, "pos": list(pos), "obs": obs, "candidates": cands,
                      "action": action, "event": event, "remaining_before": remaining,
                      "teacher_repeat_collision": repeat})
    return {"size": size, "seed": seed, "epsilon": epsilon, "success": env.success,
            "n_steps": len(steps), "teacher_blocked_pairs": len(blocked_pairs), "steps": steps}
