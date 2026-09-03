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
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))
import util.arg_parser as arg_parser
import util.mp_util as mp_util
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder

ROOT = os.path.join(os.path.dirname(__file__), "..")

# human walk reference bands [decision: literature values, the clip-measured versions land
# with the emergence-gate work]
# Bands ANCHORED ON THE REFERENCE CLIP measured through THIS criterion (--demo:
# speed 1.276, cadence 1.90, duty 0.808, aerial 0.000, root height 0.845). Literature bands
# measured by other means judged the clip itself OUT -- a yardstick has to be cut from the
# same stock as the thing it measures. The physics-grounded halves are the walk/run
# boundary (duty > 0.5, aerial ~ 0), which is what the 1.62g emergence gate inverts.
WALK_BANDS = {
    "speed_mps": (0.9, 1.8),
    "mean_root_height": (0.75, 1.0),
    "cadence_g": (1.5, 2.3),
    "duty_factor_g": (0.55, 0.90),
    "aerial_fraction_g": (0.0, 0.05),
    # a reproduced walk stays up: the episode cap is 10 s, so a healthy policy is censored
    # (never falls) and reads at the cap; falling every few seconds is not a walk
    "mean_episode_s": (8.0, 1e9),
    "falls_per_min": (0.0, 1.0),
}
CONTACT_N = 20.0     # |ground force| above this = stance (diagnostic only -- chatters)
CORNER_Z = 0.02      # a foot BOX corner below this = stance (absolute, rotation-aware)

def foot_geoms(asset):
    """(offset, half-extents) of each foot's box geom, straight from the MJCF."""
    m_ = mujoco.MjModel.from_xml_path(asset)
    gpos, gsize = [], []
    for nm in ("right_foot", "left_foot"):
        bid = mujoco.mj_name2id(m_, mujoco.mjtObj.mjOBJ_BODY, nm)
        gi = [gg for gg in range(m_.ngeom) if m_.geom_bodyid[gg] == bid][0]
        gpos.append(m_.geom_pos[gi].copy())
        gsize.append(m_.geom_size[gi].copy())
    return gpos, gsize


def foot_corner_z(body_pos, body_rot, gpos, gsize):
    """Lowest world-z among a foot box's 8 corners. body_pos [N,3], body_rot [N,4] xyzw.
    ABSOLUTE criterion: independent of penetration depth, of the rollout's own minimum, and
    identical across arms and engines (the self-referential `min(foot_h)+3cm` threshold it
    replaces made the DeepMimic arm look bouncier than AMP purely through a lower floor)."""
    sx, sy, sz = gsize
    c = np.array([[dx * sx, dy * sy, dz * sz]
                  for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]) + np.asarray(gpos)
    c = c[None, :, :]                      # [1,8,3]
    q = np.asarray(body_rot)
    u = q[:, None, :3]                     # [N,1,3]
    w = q[:, None, 3:4]                    # [N,1,1]
    t = 2.0 * np.cross(u, c)               # [N,8,3]
    world = np.asarray(body_pos)[:, None, :] + c + w * t + np.cross(u, t)
    return world[:, :, 2].min(axis=1)      # [N]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="")
    p.add_argument("--demo", action="store_true",
                   help="measure the REFERENCE CLIP itself (kinematic replay) through the "
                        "identical criterion + metric code -- the apples-to-apples yardstick")
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--gravity", type=float, default=None, help="z gravity override, e.g. -1.62")
    p.add_argument("--engine_config", default="data/engines/mujoco_cpu_engine.yaml")
    p.add_argument("--out", default="")
    p.add_argument("--env_config", default="data/envs/amp_humanoid_walk_env.yaml")
    p.add_argument("--agent_config", default="data/agents/amp_humanoid_agent.yaml")
    return p.parse_args()

def build(nenvs, gravity, env_cfg, agent_cfg, engine_cfg="data/engines/mujoco_cpu_engine.yaml"):
    args = arg_parser.ArgParser()
    args.load_args([
        "--engine_config", os.path.join(ROOT, engine_cfg),
        "--env_config", os.path.join(ROOT, env_cfg),
        "--agent_config", os.path.join(ROOT, agent_cfg),
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

def demo_series(nsteps, dt):
    """Kinematic replay of the reference clip -> the same (speed, height, corner_z) series
    the policy path produces, so the SAME metric code judges both."""
    import anim.motion_lib as motion_lib
    import anim.mjcf_char_model as mm
    import mujoco as mj_
    asset = os.path.join(ROOT, "data/assets/humanoid/humanoid.xml")
    cm = mm.MJCFCharModel("cpu")
    cm.load(asset)
    ml = motion_lib.MotionLib(os.path.join(ROOT, "data/motions/humanoid/humanoid_walk.pkl"),
                              cm, "cpu")
    dur = float(ml.get_motion_length(torch.tensor([0])))
    times = torch.remainder(torch.arange(nsteps, dtype=torch.float64) * dt, dur).float()
    ids = torch.zeros(nsteps, dtype=torch.long)
    root_pos, root_rot, root_vel, _, joint_rot, _ = ml.calc_motion_frame(ids, times)
    bpos, brot = cm.forward_kinematics(root_pos, root_rot, joint_rot)
    names = [cm.get_body_name(i) for i in range(bpos.shape[-2])]
    feet = [names.index("right_foot"), names.index("left_foot")]
    gpos, gsize = foot_geoms(asset)
    T = nsteps
    speed = np.linalg.norm(root_vel[:, :2].numpy(), axis=-1)[:, None]
    height = root_pos[:, 2].numpy()[:, None]
    corner = np.zeros((T, 1, 2))
    for f_i, b in enumerate(feet):
        corner[:, 0, f_i] = foot_corner_z(bpos[:, b].numpy(), brot[:, b].numpy(),
                                          gpos[f_i], gsize[f_i])
    return speed, height, corner


def debounce(c, minrun=2):
    """Drop stance/swing runs shorter than minrun frames (66 ms) -- raw contact chatter
    inflated cadence 3x on the first checkpoint (morphological open+close)."""
    c = c.copy()
    for n in range(c.shape[1]):
        for f in range(c.shape[2]):
            x = c[:, n, f]
            for val in (True, False):
                run = 0
                for t in range(len(x) + 1):
                    if t < len(x) and x[t] == val:
                        run += 1
                    else:
                        if 0 < run < minrun:
                            x[t - run:t] = not val
                        run = 0
    return c


def report(a, label, speed, height, corner_z, contact, foot_speed, done_mask, dt, g):
    T = speed.shape[0]
    alive = ~done_mask
    m = {}
    # STABILITY (primary -- the FORM film caught alive_fraction lying: it counts the single
    # `done` FRAME, so a policy falling every 5 s still reads 0.997). Episode length is the
    # honest measure: how long the character stays up.
    ep_lens = []
    for n in range(done_mask.shape[1]):
        idx = np.flatnonzero(done_mask[:, n])
        prev = -1
        for i in idx:
            ep_lens.append((i - prev) * dt)
            prev = i
        if prev < len(done_mask) - 1:          # censored tail (still standing at cutoff)
            ep_lens.append((len(done_mask) - 1 - prev) * dt)
    m["mean_episode_s"] = float(np.mean(ep_lens)) if ep_lens else float(T * dt)
    m["falls_per_min"] = float(done_mask.sum() / (T * dt * done_mask.shape[1]) * 60.0)
    m["alive_fraction"] = float(alive.mean())
    m["speed_mps"] = float(speed[alive].mean())
    m["mean_root_height"] = float(height[alive].mean())

    # PRIMARY criterion: lowest foot-box corner below CORNER_Z -- absolute and rotation-aware.
    gc = debounce(corner_z < CORNER_Z)
    gstance = gc.any(axis=-1)
    m["duty_factor_g"] = float(gc[alive].mean())
    m["aerial_fraction_g"] = float((~gstance)[alive].mean())
    gedges = (gc[1:] & ~gc[:-1]).sum(axis=(0, 2))
    m["cadence_g"] = float((gedges / (T * dt)).mean())

    L = 0.91
    m["froude"] = m["speed_mps"] ** 2 / (g * L)
    m["gravity"] = g

    # DIAGNOSTIC only: the contact-force criterion chatters under implicitfast micro-bounce
    # (it manufactured a "trot" verdict for three readings before the geometric criterion
    # landed). Kept to expose that chatter, never to judge.
    if contact.any():
        fc = debounce(contact)
        m["diag_duty_force"] = float(fc[alive].mean())
        m["diag_aerial_force"] = float((~fc.any(axis=-1))[alive].mean())
        fedges = (fc[1:] & ~fc[:-1]).sum(axis=(0, 2))
        m["diag_cadence_force"] = float((fedges / (T * dt)).mean())
        slips = []
        for f_i in range(2):
            st = fc[:, :, f_i] & alive
            if st.any():
                slips.append(foot_speed[:, :, f_i][st].mean())
        m["diag_ankle_speed_stance"] = float(np.mean(slips)) if slips else 0.0

    print(f"\n=== GAIT METRICS ({label}, g={g:.2f}) ===")
    verdicts = {}
    for k, v in m.items():
        band = WALK_BANDS.get(k)
        tag = ""
        if band:
            ok = band[0] <= v <= band[1]
            verdicts[k] = ok
            tag = f"  walk-band {band} {'OK' if ok else 'OUT'}"
        print(f"  {k:>24s} = {v:8.3f}{tag}")
    print(f"  WALK VERDICT: {sum(verdicts.values())}/{len(verdicts)} primary bands in; "
          f"alive {m['alive_fraction']:.1%}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(m, f, indent=1)
        print("  written:", a.out)
    return m


def main():
    a = parse()
    if a.demo:
        dt = 1.0 / 30.0
        T = a.steps
        speed, height, corner_z = demo_series(T, dt)
        z = np.zeros((T, 1, 2))
        report(a, "REFERENCE CLIP humanoid_walk.pkl", speed, height, corner_z,
               z.astype(bool), z, np.zeros((T, 1), dtype=bool), dt, 9.81)
        return

    env, agent = build(a.envs, a.gravity, a.env_config, a.agent_config, a.engine_config)
    agent.load(a.model)
    agent.eval()
    eng = env._engine
    names = eng.get_obj_body_names(0)
    feet = [names.index("right_foot"), names.index("left_foot")]
    g = abs(float(eng.get_gravity()[2]))
    # foot geometry from the MJCF ITSELF, never from the engine -- the instrument must judge
    # any engine (newton arm included) by the same absolute criterion
    gpos, gsize = foot_geoms(os.path.join(ROOT, "data/assets/humanoid/humanoid.xml"))

    obs, info = env.reset()
    N, T = a.envs, a.steps
    dt = eng.get_timestep()
    speed = np.zeros((T, N))
    height = np.zeros((T, N))
    contact = np.zeros((T, N, 2), dtype=bool)
    foot_speed = np.zeros((T, N, 2))
    corner_z = np.zeros((T, N, 2))
    done_mask = np.zeros((T, N), dtype=bool)
    with torch.no_grad():
        for t in range(T):
            action, _ = agent._decide_action(obs, info)
            obs, r, done, info = env.step(action)
            speed[t] = np.linalg.norm(eng.get_root_vel(0).numpy()[:, :2], axis=-1)
            height[t] = eng.get_root_pos(0).numpy()[:, 2]
            bv = eng.get_body_vel(0).numpy()
            bp = eng.get_body_pos(0).numpy()
            br = eng.get_body_rot(0).numpy()
            gf = eng.get_ground_contact_forces(0).numpy()
            for f_i, b in enumerate(feet):
                foot_speed[t, :, f_i] = np.linalg.norm(bv[:, b, :2], axis=-1)
                corner_z[t, :, f_i] = foot_corner_z(bp[:, b], br[:, b], gpos[f_i], gsize[f_i])
                contact[t, :, f_i] = np.linalg.norm(gf[:, b], axis=-1) > CONTACT_N
            done_mask[t] = done.numpy() != 0
            if done_mask[t].any():
                obs, info = env.reset(torch.nonzero(done.flatten()).flatten())

    report(a, a.model, speed, height, corner_z, contact, foot_speed, done_mask, dt, g)


if __name__ == "__main__":
    main()
