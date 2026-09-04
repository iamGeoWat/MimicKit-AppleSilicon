"""NM-GAME-PATH tier B: bake a trained policy into a crew-rig animation clip.

The architecture (owner ruling 2026-09-03) is: trained policies -> baked clips -> game
animation, so that no gait is hand-authored. This is the baker. It rolls a checkpoint,
retargets each frame onto the crew rig with the alignment table, and writes a clip the game
can play through CrewGait's existing (dist, speed, g, ...) seam.

The clip is indexed by GROUND DISTANCE, not by time: CrewGait is a pure function of distance
travelled, so a distance-indexed table stays a pure function too -- the determinism law holds
with no runtime inference in the view layer.

    python tools/bake_clip.py --model output/CKPT.pt --rig crew_rig.json --out walk_1g.json \
        [--gravity -1.62] [--speed 1.28] [--cycles 4]
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))
import util.arg_parser as arg_parser
import util.mp_util as mp_util
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent
from retarget_to_crew import BONE_MAP, build_alignment, retarget_frame, quat_mul, quat_conj

ROOT = os.path.join(os.path.dirname(__file__), "..")


def local_quats(model, data):
    """Each mapped body's LOCAL rotation relative to its parent, xyzw."""
    out = {}
    for src in BONE_MAP:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, src)
        pid = model.body_parentid[bid]
        def q(i):
            w, x, y, z = data.xquat[i]
            return np.array([x, y, z, w])
        out[src] = quat_mul(quat_conj(q(pid)), q(bid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gravity", type=float, default=None)
    ap.add_argument("--speed", type=float, default=None, help="commanded speed (steering envs)")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--env_config", default="data/envs/amp_humanoid_walk_env.yaml")
    ap.add_argument("--agent_config", default="data/agents/amp_humanoid_agent.yaml")
    ap.add_argument("--engine_config", default="data/engines/mujoco_cpu_engine.yaml")
    a = ap.parse_args()

    eng_file = os.path.join(ROOT, a.engine_config)
    if a.gravity is not None:
        import yaml
        cfg = yaml.safe_load(open(eng_file))
        cfg["gravity"] = [0.0, 0.0, a.gravity]
        eng_file = "/tmp/openksp_bake_engine.yaml"
        yaml.safe_dump(cfg, open(eng_file, "w"))

    args = arg_parser.ArgParser()
    args.load_args(["--env_config", os.path.join(ROOT, a.env_config),
                    "--agent_config", os.path.join(ROOT, a.agent_config)])
    mp_util.init(0, 1, "cpu", "7100")
    env = env_builder.build_env(args.parse_string("env_config"), eng_file, 1, "cpu", visualize=False)
    agent = agent_builder.build_agent(args.parse_string("agent_config"), env, "cpu")
    agent.load(a.model)
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)

    align = build_alignment(a.rig, os.path.join(ROOT, "data/assets/humanoid/humanoid.xml"))
    eng = env._engine
    model, data = eng._model, eng._datas[0]
    g = abs(float(eng.get_gravity()[2]))
    dt = eng.get_timestep()

    obs, info = env.reset()
    frames, dist_acc, dist = [], 0.0, 0.0
    prev_xy = eng.get_root_pos(0).numpy()[0, :2].copy()
    with torch.no_grad():
        for _ in range(a.frames):
            if a.speed is not None and hasattr(env, "_tar_speed"):
                env._tar_speed[:] = a.speed
            act, _ = agent._decide_action(obs, info)
            obs, r, done, info = env.step(act)
            xy = eng.get_root_pos(0).numpy()[0, :2]
            dist += float(np.linalg.norm(xy - prev_xy))
            prev_xy = xy.copy()
            frames.append({
                "dist": dist,
                "root_h": float(eng.get_root_pos(0).numpy()[0, 2]),
                "bones": {k: [round(x, 6) for x in v]
                          for k, v in retarget_frame(local_quats(model, data), align).items()},
            })
            dv = int(done.flatten()[0])
            if dv:
                import envs.base_env as _be
                nm = _be.DoneFlags(dv).name
                if nm == "FAIL":              # a FALL ends the usable clip; a TIMEOUT does not
                    print(f"clip ended early at frame {len(frames)}: the policy FELL")
                    break
                obs, info = env.reset()

    speed = dist / (len(frames) * dt) if frames else 0.0
    clip = {
        "source": os.path.basename(a.model),
        "gravity": g,
        "fps": round(1.0 / dt, 3),
        "frames": len(frames),
        "distance_m": round(dist, 4),
        "mean_speed_mps": round(speed, 4),
        "bone_order": sorted({b for f in frames for b in f["bones"]}),
        "unmapped_crew_bones": ["toe_l", "toe_r"],
        "note": "distance-indexed: CrewGait is a pure function of ground distance, so playing "
                "this table by distance keeps the view layer deterministic with no inference.",
        "keys": frames,
    }
    json.dump(clip, open(a.out, "w"))
    print(f"BAKED {len(frames)} frames  g={g:.2f}  dist={dist:.2f} m  mean speed={speed:.2f} m/s "
          f"-> {a.out} ({os.path.getsize(a.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
