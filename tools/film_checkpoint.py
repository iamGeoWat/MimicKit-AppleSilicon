"""Film a trained checkpoint (openksp NM-T1 FORM pass).

MimicKit's recorder exists but run.py only hands its video to the logger; this saves an mp4.
Uses the NEWTON arm's headless ViewerGL (the mujoco arm is compute-only), which is legitimate
because the two engines were proven behaviourally equal (cross-engine transfer, 5/5 bands).

  python tools/film_checkpoint.py --model output/CKPT_amp_walk_1g_125M.pt \
      --out /tmp/walk_1g.mp4 [--steps 300] [--gravity -1.62] [--cam side|front|tq]
"""
import argparse
import os
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))
import util.arg_parser as arg_parser
import util.mp_util as mp_util
import envs.env_builder as env_builder
import envs.base_env as base_env
import learning.agent_builder as agent_builder

ROOT = os.path.join(os.path.dirname(__file__), "..")
CAMS = {
    "side":  (np.array([0.0, -4.0, 1.2]),  np.array([0.0, 0.0, 0.9])),
    "front": (np.array([4.0, 0.0, 1.4]),   np.array([0.0, 0.0, 0.9])),
    "tq":    (np.array([-3.0, -3.0, 2.0]), np.array([0.0, 0.0, 0.9])),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--gravity", type=float, default=None)
    p.add_argument("--cam", default="side", choices=list(CAMS))
    p.add_argument("--env_config", default="data/envs/amp_humanoid_walk_env.yaml")
    p.add_argument("--agent_config", default="data/agents/amp_humanoid_agent.yaml")
    a = p.parse_args()

    eng_file = os.path.join(ROOT, "data/engines/newton_engine.yaml")
    if a.gravity is not None:
        with open(eng_file) as f:
            cfg = yaml.safe_load(f)
        cfg["gravity"] = [0.0, 0.0, a.gravity]
        eng_file = "/tmp/openksp_film_engine.yaml"
        with open(eng_file, "w") as f:
            yaml.safe_dump(cfg, f)

    args = arg_parser.ArgParser()
    args.load_args(["--env_config", os.path.join(ROOT, a.env_config),
                    "--agent_config", os.path.join(ROOT, a.agent_config)])
    mp_util.init(0, 1, "cpu", "6100")

    env = env_builder.build_env(args.parse_string("env_config"), eng_file, 1, "cpu",
                                visualize=False, record_video=True)
    agent = agent_builder.build_agent(args.parse_string("agent_config"), env, "cpu")
    agent.load(a.model)
    agent.eval()

    eng = env._engine
    assert eng.enabled_record_video(), "engine did not enable recording"
    cam_pos, cam_tgt = CAMS[a.cam]
    rec = eng._video_recorder
    rec._cam_pos, rec._cam_target = cam_pos, cam_tgt
    # macOS retina: the GL backing store is 2x the requested window, so the recorder's
    # resolution assert fires on the first frame. Adopt the ACTUAL framebuffer size (the
    # assert then still guards frame-to-frame consistency, which is what it is for).
    probe = rec._record_frame()
    if (probe.shape[1], probe.shape[0]) != tuple(rec._resolution):
        print(f"film: framebuffer {probe.shape[1]}x{probe.shape[0]} != requested "
              f"{rec._resolution[0]}x{rec._resolution[1]} (retina scale) -- adopting actual")
        rec._resolution = (probe.shape[1], probe.shape[0])

    # per-env gravity path also works here; the yaml sets the scene value
    if a.gravity is not None and hasattr(eng, "set_env_gravity"):
        eng.set_env_gravity(np.array([0.0, 0.0, a.gravity]))

    # BOTH modes: the agent's TEST mode drops exploration noise (its set_mode also puts the
    # env in TEST, which starts the recorder). Filming with the agent left in TRAIN films the
    # EXPLORATION policy, not the learned one -- which is what made the first film fall.
    import learning.base_agent as _ba
    agent.set_mode(_ba.AgentMode.TEST)
    obs, info = env.reset()
    events = []
    with torch.no_grad():
        for _t in range(a.steps):
            action, _ = agent._decide_action(obs, info)
            obs, r, done, info = env.step(action)
            dv = done.flatten().numpy()
            for k in np.flatnonzero(dv):
                events.append((len(events), int(dv[k]), _t))
            if (dv != 0).any():
                obs, info = env.reset(torch.nonzero(done.flatten()).flatten())
    eng.stop_video_recording()
    import envs.base_env as _be
    print("film terminations:", [(t, _be.DoneFlags(v).name) for _, v, t in events])

    vid = eng.get_video_recording()
    assert vid is not None and vid.get_num_frames() > 0, "no frames captured"
    vid.save(a.out)
    print(f"FILM: {a.out}  frames={vid.get_num_frames()}  fps={vid.get_fps()}  "
          f"res={vid.get_resolution()}")


if __name__ == "__main__":
    main()
