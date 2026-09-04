"""Name WHICH body ends each episode, and how the character was oriented when it did.

The failure rule is "any non-foot body with ground contact force > 0.1 N". That says an
episode ended; it does not say whether the character FELL. A lunar lope legitimately puts a
hand down (Apollo footage is full of it), and a rule written for Earth locomotion would score
that as a fall -- so before spending another training arm on "it falls", the fall has to be
attributed to a body part and a trunk attitude.

Reports, per episode: the terminating body, the trunk pitch at that moment (signed, + = the
trunk has rotated BACKWARD past vertical), and the root height, so a genuine topple can be
separated from a touch.

    python tools/probe_fall_body.py --model CKPT.pt [--gravity -1.62] [--steps 600]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))
import util.arg_parser as arg_parser
import util.mp_util as mp_util
import envs.env_builder as env_builder
import envs.base_env as base_env
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent

ROOT = os.path.join(os.path.dirname(__file__), "..")


def trunk_pitch_deg(engine, char_id=0):
    """Signed pitch of the torso's up axis away from world up, + = leaning BACKWARD.

    Backward is -x in the MJCF humanoid's frame (it faces +x), so the sign comes from the
    torso up-axis' x component rather than from the magnitude alone -- a magnitude cannot
    tell a forward crouch from a backward topple, which is the whole question here.
    """
    rot = engine.get_body_rot(char_id)[0]          # (num_bodies, 4) xyzw
    q = rot[1].cpu().numpy() if rot.shape[0] > 1 else rot[0].cpu().numpy()
    x, y, z, w = q
    # world-space image of the body's local +z (up for this rig)
    up = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    tilt = np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))
    return -np.sign(up[0]) * tilt if abs(up[0]) > 1e-6 else tilt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--gravity", type=float, default=None)
    ap.add_argument("--env_config", default="data/envs/amp_steering_lowg_env.yaml")
    ap.add_argument("--agent_config", default="data/agents/amp_task_humanoid_agent.yaml")
    ap.add_argument("--engine_config", default="data/engines/mujoco_cpu_engine.yaml")
    a = ap.parse_args()

    eng_file = os.path.join(ROOT, a.engine_config)
    if a.gravity is not None:
        import yaml
        cfg = yaml.safe_load(open(eng_file))
        cfg["gravity"] = [0.0, 0.0, a.gravity]
        eng_file = "/tmp/openksp_fallprobe_engine.yaml"
        yaml.safe_dump(cfg, open(eng_file, "w"))

    args = arg_parser.ArgParser()
    args.load_args(["--env_config", os.path.join(ROOT, a.env_config),
                    "--agent_config", os.path.join(ROOT, a.agent_config)])
    mp_util.init(0, 1, "cpu", "7141")
    env = env_builder.build_env(args.parse_string("env_config"), eng_file, 1, "cpu", visualize=False)
    agent = agent_builder.build_agent(args.parse_string("agent_config"), env, "cpu")
    agent.load(a.model)
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)

    eng = env._engine
    names = eng.get_body_names() if hasattr(eng, "get_body_names") else None
    if names is None:
        names = env._kin_char_model.get_body_names()
    dt = eng.get_timestep()

    obs, info = env.reset()
    t0, episodes = 0, []
    with torch.no_grad():
        for t in range(a.steps):
            act, _ = agent._decide_action(obs, info)
            obs, r, done, info = env.step(act)
            dv = int(done.flatten()[0])
            if dv == 0:
                continue
            f = eng.get_ground_contact_forces(0)[0].detach().cpu().numpy()   # (bodies, 3)
            mag = np.abs(f).max(axis=-1)
            # the failure rule ZEROES the contact bodies (the feet) before testing, so the
            # attribution has to exclude them as well -- listing a foot as a "terminating
            # body" is the report contradicting the rule it is reporting on.
            excluded = set(env._contact_body_ids.cpu().numpy().tolist())
            hits = [(names[i], float(mag[i])) for i in np.argsort(-mag)[:3]
                    if mag[i] > 0.1 and i not in excluded]
            episodes.append({
                "flag": base_env.DoneFlags(dv).name,
                "dur_s": (t - t0 + 1) * dt,
                "pitch": trunk_pitch_deg(eng),
                "root_h": float(eng.get_root_pos(0).numpy()[0, 2]),
                "hits": hits,
            })
            t0 = t + 1
            obs, info = env.reset()

    print(f"\n=== FALL ATTRIBUTION  {os.path.basename(a.model)}  g={a.gravity}  "
          f"{a.steps} steps ({a.steps * dt:.0f} s) ===")
    print(f"{'flag':<6} {'dur s':>6} {'trunk pitch':>12} {'root h':>7}   bodies in contact (non-foot excluded by the rule)")
    for e in episodes:
        hs = ", ".join(f"{n} {v:.1f}N" for n, v in e["hits"]) or "(none above 0.1 N)"
        print(f"{e['flag']:<6} {e['dur_s']:>6.2f} {e['pitch']:>11.1f}d {e['root_h']:>7.3f}   {hs}")
    if episodes:
        fails = [e for e in episodes if e["flag"] == "FAIL"]
        print(f"\n{len(fails)}/{len(episodes)} episodes ended in FAIL; "
              f"mean life {np.mean([e['dur_s'] for e in episodes]):.2f} s")
        if fails:
            p = np.array([e["pitch"] for e in fails])
            # NEVER average this signed quantity: +100 and -100 are both "flat on the ground",
            # and their mean is 0, which reads as "upright". Magnitude answers "did it fall";
            # the sign split answers "in one direction or many".
            back, fwd = int((p > 0).sum()), int((p < 0).sum())
            h = np.array([e["root_h"] for e in fails])
            print(f"trunk tilt at failure: |pitch| mean {np.abs(p).mean():.0f} deg "
                  f"(90 = flat), root height {h.mean():.2f} m of a 0.88 m rest stance")
            print(f"direction: {back} backward / {fwd} forward -- "
                  f"{'ONE direction' if min(back, fwd) == 0 else 'MIXED, not a directional topple'}")
            from collections import Counter
            c = Counter(n for e in fails for n, _ in e["hits"])
            print("terminating bodies:", dict(c))


if __name__ == "__main__":
    main()
