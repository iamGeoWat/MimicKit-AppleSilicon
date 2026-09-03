"""Per-env gravity probe (openksp NM-T1, the g-channel prerequisite).

G1 uniform: set 1.62 for every env -> a dropped body falls at 1.62 (free-fall fit).
G2 SPLIT: env 0 at 9.81 and env 1 at 1.62 IN THE SAME BATCH -> the two fall distances
   differ by the gravity ratio. This is the whole point: one batch spanning gravities is
   what a CONDITIONED policy (g in the observation) trains on.
G3 readback: get_env_gravity reports what was set, per env.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))
import engines.engine as engine
import engines.newton_engine as ne

ASSET = os.path.join(os.path.dirname(__file__), "..", "data", "assets", "humanoid", "humanoid.xml")
CFG = {"engine_name": "newton", "control_mode": "none", "control_freq": 30,
       "sim_freq": 240, "env_spacing": 5}


def build(n):
    e = ne.NewtonEngine(CFG, n, "cpu", visualize=False)
    for i in range(n):
        e.create_env()
        e.create_obj(i, engine.ObjType.articulated, ASSET, "character",
                     start_pos=np.array([0.0, 0.0, 3.0]))
    e.initialize_sim()
    return e


def drop(e, steps=15):
    """Free-fall from a fixed height; returns per-env drop distance."""
    n = e.get_num_envs()
    e.set_root_pos(None, 0, torch.tensor([[0.0, 0.0, 3.0]]).repeat(n, 1))
    e.set_root_rot(None, 0, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1))
    e.set_root_vel(None, 0, torch.zeros(n, 3))
    e.set_root_ang_vel(None, 0, torch.zeros(n, 3))
    e.set_dof_vel(None, 0, torch.zeros(n, e.get_obj_num_dofs(0)))
    z0 = e.get_root_pos(0)[:, 2].clone().numpy()
    for _ in range(steps):
        e.step()
    z1 = e.get_root_pos(0)[:, 2].clone().numpy()
    return z0 - z1, steps * e.get_timestep()


def main():
    e = build(2)
    rc = 0

    e.set_env_gravity(np.array([0.0, 0.0, -1.62]))
    d, t = drop(e)
    want = 0.5 * 1.62 * t * t
    ok = np.all(np.abs(d - want) < 0.25 * want)
    print(f"G1 uniform 1.62: drop {np.round(d, 3)} m over {t:.2f} s, free-fall {want:.3f}  "
          f"{'OK' if ok else 'FAIL'}")
    rc |= 0 if ok else 1

    e.set_env_gravity(np.array([[0.0, 0.0, -9.81], [0.0, 0.0, -1.62]]))
    d, t = drop(e)
    ratio = d[0] / max(d[1], 1e-9)
    ok = 4.0 < ratio < 8.0     # 9.81/1.62 = 6.06, drag-free but contact-free too
    print(f"G2 SPLIT batch: env0(9.81) {d[0]:.3f} m  env1(1.62) {d[1]:.3f} m  ratio {ratio:.2f} "
          f"(expect ~6.06)  {'OK' if ok else 'FAIL'}")
    rc |= 0 if ok else 1

    g = e.get_env_gravity()
    ok = abs(g[0, 2] + 9.81) < 1e-5 and abs(g[1, 2] + 1.62) < 1e-5
    print(f"G3 readback: {np.round(g[:, 2], 3)}  {'OK' if ok else 'FAIL'}")
    rc |= 0 if ok else 1

    print("PER-ENV GRAVITY:", "ALL GREEN" if rc == 0 else "RED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
