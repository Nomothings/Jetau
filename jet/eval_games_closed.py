#!/usr/bin/env python3
"""Closed-loop game evaluation: completion rate within budget + steps-to-complete.

Runs a policy interactively in the real game environment (maze: reach the goal
within the attempt budget; snake: eat the food target within the step budget;
pokemon: win the 3v3 battle within max turns). Metrics: success rate, mean
steps (all episodes and successful only). This is the game-side analogue of
the original benchmarks' interactive protocols.
"""
import argparse
import json
import random
import time
from pathlib import Path

from jet.jet_policy import JetPolicy, NoMemPolicy, PromptMemPolicy

def make_env(task, seed):
    if task == "maze":
        from jet.maze_env import MazeEnv
        return MazeEnv(8, seed, budget=64)
    if task == "snake":
        from jet.snake_env import SnakeEnv
        return SnakeEnv(8, seed, target_food=3, budget=80)
    from jet.pokemon_env import Battle, SPECIES
    roster = sorted(SPECIES)
    rng = random.Random(seed)
    return Battle(rng.sample(roster, 3), rng.sample(roster, 3), rng=rng)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["maze", "snake", "pokemon"], required=True)
    p.add_argument("--policy", choices=["nomem", "promptmem", "jet"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-name", default="", help="label for the output record")
    p.add_argument("--note-window", type=int, default=6)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed-base", type=int, default=30000)
    p.add_argument("--out", required=True)
    p.add_argument("--mem-fraction", type=float, default=0.4)
    a = p.parse_args()

    if a.policy == "nomem":
        pol = NoMemPolicy(a.checkpoint, a.task, mem_fraction=a.mem_fraction)
    elif a.policy == "promptmem":
        pol = PromptMemPolicy(a.checkpoint, a.task, mem_fraction=a.mem_fraction)
    else:
        pol = JetPolicy(a.checkpoint, a.task, note_window=a.note_window,
                        mem_fraction=a.mem_fraction)
    print(f"policy={a.policy} model={a.model_name} task={a.task} window="
          f"{getattr(pol, 'note_window', '-')}", flush=True)

    succ, steps_all, steps_succ = 0, [], []
    t0 = time.perf_counter()
    for i in range(a.episodes):
        env = make_env(a.task, a.seed_base + i)
        guard = 0
        while not env.done and guard < 200:
            obs = env.observation()
            cands = env.candidates()
            action = pol.act(obs, cands)
            event = env.step(action)
            if hasattr(pol, "remember"):
                pol.remember(obs, action, str(event))
            guard += 1
        ok = bool(env.success)
        succ += ok
        steps_all.append(env.t if hasattr(env, "t") else guard)
        if ok:
            steps_succ.append(steps_all[-1])
        if hasattr(pol, "reset"):
            pol.reset()
    wall = time.perf_counter() - t0
    n = max(a.episodes, 1)
    result = {"task": a.task, "policy": a.policy, "model": a.model_name or a.checkpoint,
              "episodes": a.episodes, "success_rate": succ / n,
              "mean_steps_all": sum(steps_all) / n,
              "mean_steps_success": (sum(steps_succ) / len(steps_succ)) if steps_succ else None,
              "sec_per_episode": wall / n}
    print(json.dumps(result, indent=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
