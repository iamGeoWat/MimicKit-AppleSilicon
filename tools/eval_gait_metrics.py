"""Gait-metric evaluation instrument (openksp NM-T1, the survey §5-pin assertion form).

Rolls a trained checkpoint in the mujoco engine and measures WHAT KIND OF GAIT it is:
speed, cadence, duty factor, aerial fraction, Froude number, uprightness. These are the
numbers the 1g gate reads against the human-walk bands, and the instrument the 1.62g
emergence gate reuses (lope signature: duty < 0.5, aerial fraction > 0, Fr ~ 0.36 suited).

Usage (from MimicKit root, venv active):
  python tools/eval_gait_metrics.py --model output/amp_walk_1g_mujoco_v2/model.pt \
      [--steps 900] [--envs 8] [--gravity -1.62] [--out metrics.json]
"""
import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))
import util.arg_parser as arg_parser
import util.mp_util as mp_util
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder

ROOT = os.path.join(os.path.dirname(__file__), "..")

# human walk reference bands [decision: literature values, the clip-measured versions land
# with the emergence-gate work]
WALK_BANDS = {
    "speed_mps": (0.9, 1.8),
    "cadence_steps_per_s": (1.5, 2.2),
    "duty_factor": (0.55, 0.75),
    "aerial_fraction": (0.0, 0.05),
    "mean_root_height": (0.75, 1.0),
}
CONTACT_N = 20.0   # |ground force| above this = stance

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--gravity", type=float, default=None, help="z gravity override, e.g. -1.62")
    p.add_argument("--out", default="")
    return p.parse_args()

def build(nenvs, gravity):
    args = arg_parser.ArgParser()
    args.load_args([
        "--engine_config", os.path.join(ROOT, "data/engines/mujoco_engine.yaml"),
        "--env_config", os.path.join(ROOT, "data/envs/amp_humanoid_walk_env.yaml"),
        "--agent_config", os.path.join(ROOT, "data/agents/amp_humanoid_agent.yaml"),
    ])
    mp_util.init(0, 1, "cpu", "6000")
    import yaml
    eng_file = args.parse_string("engine_config")
    if gravity is not None:
        with open(eng_file) as f:
            cfg = yaml.safe_load(f)
        cfg["gravity"] = [0.0, 0.0, gravity]
        eng_file = "/tmp/openksp_eval_engine.yaml"
        with open(eng_file, "w") as f:
            yaml.safe_dump(cfg, f)
        args._table["engine_config"] = eng_file
    env = env_builder.build_env(args.parse_string("env_config"), eng_file,
                                nenvs, "cpu", visualize=False)
    agent = agent_builder.build_agent(args.parse_string("agent_config"), env, "cpu")
    return env, agent

def main():
    a = parse()
    env, agent = build(a.envs, a.gravity)
    agent.load(a.model)
    agent.eval()
    eng = env._engine
    names = eng.get_obj_body_names(0)
    feet = [names.index("right_foot"), names.index("left_foot")]
    g = abs(float(eng.get_gravity()[2]))

    obs, info = env.reset()
    N, T = a.envs, a.steps
    dt = eng.get_timestep()
    speed = np.zeros((T, N))
    height = np.zeros((T, N))
    contact = np.zeros((T, N, 2), dtype=bool)
    done_mask = np.zeros((T, N), dtype=bool)
    with torch.no_grad():
        for t in range(T):
            action, _ = agent._decide_action(obs, info)
            obs, r, done, info = env.step(action)
            rv = eng.get_root_vel(0).numpy()
            speed[t] = np.linalg.norm(rv[:, :2], axis=-1)
            height[t] = eng.get_root_pos(0).numpy()[:, 2]
            gf = eng.get_ground_contact_forces(0).numpy()
            contact[t, :, 0] = np.linalg.norm(gf[:, feet[0]], axis=-1) > CONTACT_N
            contact[t, :, 1] = np.linalg.norm(gf[:, feet[1]], axis=-1) > CONTACT_N
            done_mask[t] = done.numpy() != 0
            if done_mask[t].any():
                obs, info = env.reset(torch.nonzero(done.flatten()).flatten())

    # metrics over ALIVE frames only (post-reset transients included -- acceptable v1)
    alive = ~done_mask
    m = {}
    m["alive_fraction"] = float(alive.mean())
    m["speed_mps"] = float(speed[alive].mean())
    m["mean_root_height"] = float(height[alive].mean())
    stance = contact.any(axis=-1)
    m["duty_factor"] = float(contact[alive].mean())          # per-foot stance fraction
    m["aerial_fraction"] = float((~stance)[alive].mean())
    # cadence: rising edges of either foot per second, averaged over envs
    edges = (contact[1:] & ~contact[:-1]).sum(axis=(0, 2))   # [N] total footfalls
    m["cadence_steps_per_s"] = float((edges / (T * dt)).mean())
    L = 0.91
    m["froude"] = m["speed_mps"] ** 2 / (g * L)
    m["gravity"] = g

    print(f"\n=== GAIT METRICS ({a.model}, g={g:.2f}) ===")
    verdicts = {}
    for k, v in m.items():
        band = WALK_BANDS.get(k)
        tag = ""
        if band:
            ok = band[0] <= v <= band[1]
            verdicts[k] = ok
            tag = f"  walk-band {band} {'OK' if ok else 'OUT'}"
        print(f"  {k:>22s} = {v:8.3f}{tag}")
    in_band = sum(verdicts.values())
    print(f"  walk-band verdict: {in_band}/{len(verdicts)} in band; "
          f"alive {m['alive_fraction']:.1%}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(m, f, indent=1)
        print("  written:", a.out)

if __name__ == "__main__":
    main()
