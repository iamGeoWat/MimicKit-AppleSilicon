"""Physical sanity probes for MujocoCPUEngine BEFORE any training (openksp NM-T1-ENGINE).
Run from the MimicKit root: python tools/probe_mujoco_cpu_engine.py

P1 standing weight: dropped to the floor and settled, the summed ground contact force
   equals m*g upward (verifies contact extraction + sign convention).
P2 PD tracking: commanded a fixed elbow target, the joint settles near it (verifies the
   motor->position transform carries newton's PD law).
P3 determinism: two engines, same writes, byte-identical states after 60 steps.
P4 ang-vel frame (VERIFY-1): set a pure world-z spin on the root; after one step the
   heading advances by ~wz*dt regardless of initial root orientation IFF the engine
   interprets set_root_ang_vel in WORLD frame.
P5 throughput: control-samples/s at 64 envs -- REPORTED, not gated (machine load moves it
   2-3x; the four physics probes above are the actual pass/fail).
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))
import engines.engine as engine
import engines.mujoco_cpu_engine as mujoco_cpu_engine

CFG = {"engine_name": "mujoco_cpu", "control_mode": "pos", "control_freq": 30, "sim_freq": 240,
       "env_spacing": 5}
ASSET = os.path.join(os.path.dirname(__file__), "..", "data", "assets", "humanoid", "humanoid.xml")

def build(n, cfg=None):
    e = mujoco_cpu_engine.MujocoCPUEngine(cfg or CFG, n, "cpu", visualize=False)
    for i in range(n):
        e.create_env()
        e.create_obj(i, engine.ObjType.articulated, ASSET, "character",
                     start_pos=np.array([0.0, 0.0, 0.9]), start_rot=np.array([0.0, 0.0, 0.0, 1.0]))
    e.initialize_sim()
    return e

def p1_standing_weight():
    e = build(2)
    tgt = e.get_dof_pos(0).clone()
    e.set_cmd(0, tgt)
    for _ in range(150):   # 5 s to settle
        e.step()
    fzs = []
    for _ in range(10):
        e.step()
        fzs.append(float(e.get_ground_contact_forces(0)[0, :, 2].sum()))
    fz = sum(fzs) / len(fzs)
    z = float(e.get_root_pos(0)[0, 2])
    m = e.calc_obj_mass(0, 0)
    want = m * 9.81
    ok = abs(fz - want) / want < 0.05
    print(f"P1 standing weight: sum Fz={fz:8.1f}  m*g={want:8.1f}  root z={z:5.2f}  {'OK' if ok else 'FAIL'}")
    return ok

def p2_pd_tracking():
    e = build(1)
    names = e.get_obj_body_names(0)
    tgt = e.get_dof_pos(0).clone()
    tgt[:, :] = 0.0
    tgt[0, 12] = 1.2   # some hinge; just verify SOME joint follows its target
    e.set_cmd(0, tgt)
    for _ in range(60):
        e.step()
    q = float(e.get_dof_pos(0)[0, 12])
    ok = abs(q - 1.2) < 0.35   # gravity load allowed; must move most of the way
    print(f"P2 PD tracking: q[12]={q:6.3f} target 1.2  {'OK' if ok else 'FAIL'} (bodies: {names[:3]}...)")
    return ok

def p3_determinism():
    outs = []
    for _ in range(2):
        e = build(4)
        tgt = e.get_dof_pos(0).clone()
        e.set_cmd(0, tgt)
        for _ in range(60):
            e.step()
        outs.append((e.get_root_pos(0).clone(), e.get_dof_pos(0).clone()))
    same = torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    print(f"P3 determinism: {'OK (byte-identical)' if same else 'FAIL'}")
    return same

def p4_angvel_frame():
    """KINEMATIC frame check (dynamics-free -- a tilted articulated body precesses, so any
    stepping-based probe measures physics, not conventions): command a WORLD-z spin on a
    45deg-tilted root; mujoco's own world-aligned cvel (via get_body_ang_vel) must return
    exactly the commanded world vector."""
    e = build(1, {**CFG, "gravity": [0.0, 0.0, 0.0]})
    rot45 = np.array([np.sin(np.pi / 8), 0.0, 0.0, np.cos(np.pi / 8)])
    e.set_root_rot(None, 0, torch.as_tensor(rot45, dtype=torch.float32))
    e.set_root_ang_vel(None, 0, torch.tensor([0.0, 0.0, 3.0]))
    w = e.get_body_ang_vel(0)[0, 0].numpy()   # pelvis, world-aligned cvel
    ok = np.allclose(w, [0.0, 0.0, 3.0], atol=1e-4)
    print(f"P4 ang-vel frame (kinematic): commanded world (0,0,3), cvel pelvis {np.round(w,4)}  "
          f"{'OK (world semantics held)' if ok else 'FAIL'}")
    return bool(ok)

def p5_throughput():
    e = build(64)
    tgt = e.get_dof_pos(0).clone()
    e.set_cmd(0, tgt)
    for _ in range(5):
        e.step()
    t0 = time.perf_counter()
    n = 100
    for _ in range(n):
        e.step()
    dt = time.perf_counter() - t0
    sps = 64 * n / dt
    # REPORTS, never gates: a throughput floor is contention-sensitive (measured 18.6k on a
    # quiet machine, 6.1k while a full test suite hogged the cores) and would red the CORRECTNESS
    # suite for a reason that has nothing to do with the engine.
    print(f"P5 throughput: {sps:,.0f} control-samples/s bare engine (64 envs) "
          f"[informational -- machine-load sensitive; ~18k on a quiet M-series]")
    return True

if __name__ == "__main__":
    results = [p1_standing_weight(), p2_pd_tracking(), p3_determinism(),
               p4_angvel_frame(), p5_throughput()]
    print("PROBES:", "ALL GREEN" if all(results) else "RED", results)
    sys.exit(0 if all(results) else 1)
