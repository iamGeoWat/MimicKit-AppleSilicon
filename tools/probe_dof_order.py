"""Discriminating probe: do newton and mujoco order DOFs identically? (cross-arm C1 fallout)
Zero-dof FK must agree (permutation-immune); then wiggle one dof at a time and record which
body moves most in each arm -> the mapping table."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))
import engines.engine as engine

ASSET = os.path.join(os.path.dirname(__file__), "..", "data", "assets", "humanoid", "humanoid.xml")
CFG = {"control_mode": "pos", "control_freq": 30, "sim_freq": 240, "env_spacing": 5}

def build(kind):
    if kind == "mujoco":
        import engines.mujoco_cpu_engine as m
        e = m.MujocoCPUEngine(dict(CFG, engine_name="mujoco"), 1, "cpu", visualize=False)
    else:
        import engines.newton_engine as n
        e = n.NewtonEngine(dict(CFG, engine_name="newton"), 1, "cpu", visualize=False)
    e.create_env()
    e.create_obj(0, engine.ObjType.articulated, ASSET, "character",
                 start_pos=np.array([0.0, 0.0, 0.9]))
    e.initialize_sim()
    return e

def refresh(e):
    if hasattr(e, "_sim_state"):
        e._sim_state.eval_fk()

def fk_bodies(e, dof):
    nd = e.get_obj_num_dofs(0)
    e.set_root_pos(None, 0, torch.tensor([[0.0, 0.0, 1.5]]))
    e.set_root_rot(None, 0, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    e.set_dof_pos(None, 0, dof.view(1, nd))
    e.set_dof_vel(None, 0, torch.zeros(1, nd))
    refresh(e)
    return e.get_body_pos(0)[0].clone()

em = build("mujoco")
en = build("newton")
names = em.get_obj_body_names(0)
nd = em.get_obj_num_dofs(0)

z = torch.zeros(nd)
b0m = fk_bodies(em, z)
b0n = fk_bodies(en, z)
d0 = float((b0m - b0n).norm(dim=-1).max())
print(f"zero-dof FK: max |dpos| = {d0:.2e} m  {'AGREE' if d0 < 1e-3 else 'DISAGREE (deeper than dof order!)'}")

print(f"{'dof':>3} {'mujoco moves':>18} {'newton moves':>18}  match")
mismatches = 0
for k in range(nd):
    dof = torch.zeros(nd)
    dof[k] = 0.6
    bm = (fk_bodies(em, dof) - b0m).norm(dim=-1)
    bn = (fk_bodies(en, dof) - b0n).norm(dim=-1)
    im, iN = int(bm.argmax()), int(bn.argmax())
    ok = im == iN and abs(float(bm.max()) - float(bn.max())) < 5e-3
    if not ok:
        mismatches += 1
        print(f"{k:>3} {names[im]:>18} {names[iN]:>18}  {'ok' if ok else 'MISMATCH'} "
              f"({float(bm.max()):.3f} vs {float(bn.max()):.3f} m)")
print(f"{nd} dofs, {mismatches} mismatches")
