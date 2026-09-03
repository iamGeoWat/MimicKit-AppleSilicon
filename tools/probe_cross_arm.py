"""Cross-arm comparability probe: newton vs mujoco engine on identical states (NM-T1-ENGINE
acceptance item b). Run: python tools/probe_cross_arm.py

C1 FK agreement: identical root+dof writes -> body_pos/body_rot from both engines agree to
   ~1e-4 (same MJCF, same kinematics; disagreement = frame/order/convention bug and the
   running walk bake is judging garbage).
C2 velocity semantics vs newton: INFORMATIONAL. Adjudicated 2026-09-02 by C2b: the mujoco
   arm matches the KIN convention to 9e-4 m/s while newton deviates up to 0.5 m/s -- newton's
   spherical dof_vel frame disagrees with kin_char_model's own motion convention (upstream
   MimicKit inconsistency; the kin model is the authority the obs/demo pipeline speaks).
C3 dynamics ballpark: same PD targets, 10 control steps from the same state -> trajectories
   diverge only at the solver-difference level (root pos within ~5 cm, no frame flips).
Divergence GROWTH is expected (implicitfast vs newton solver); systematic frame errors are
what this probe exists to catch.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))
import engines.engine as engine

ASSET = os.path.join(os.path.dirname(__file__), "..", "data", "assets", "humanoid", "humanoid.xml")
CFG = {"control_mode": "pos", "control_freq": 30, "sim_freq": 240, "env_spacing": 5}
N = 2

def build(kind):
    if kind == "mujoco":
        import engines.mujoco_engine as m
        e = m.MujocoEngine(dict(CFG, engine_name="mujoco"), N, "cpu", visualize=False)
    else:
        import engines.newton_engine as n
        e = n.NewtonEngine(dict(CFG, engine_name="newton"), N, "cpu", visualize=False)
    for i in range(N):
        e.create_env()
        e.create_obj(i, engine.ObjType.articulated, ASSET, "character",
                     start_pos=np.array([0.0, 0.0, 0.9]))
    e.initialize_sim()
    return e

def write_state(e, seed):
    g = torch.Generator().manual_seed(seed)
    nd = e.get_obj_num_dofs(0)
    rot = torch.tensor([0.06, -0.04, 0.10, 0.0])
    rot[3] = float(torch.sqrt(1.0 - (rot[:3] ** 2).sum()))
    e.set_root_pos(None, 0, torch.tensor([0.1, -0.2, 1.1]).expand(N, 3).clone())
    e.set_root_rot(None, 0, rot.expand(N, 4).clone())
    e.set_root_vel(None, 0, torch.tensor([0.4, 0.1, 0.2]).expand(N, 3).clone())
    e.set_root_ang_vel(None, 0, torch.tensor([0.2, -0.3, 0.5]).expand(N, 3).clone())
    dof = 0.25 * torch.randn(nd, generator=g)
    e.set_dof_pos(None, 0, dof.expand(N, nd).clone())
    e.set_dof_vel(None, 0, (0.3 * torch.randn(nd, generator=g)).expand(N, nd).clone())
    return dof

def q_dist(a, b):
    """Quaternion distance robust to sign (|dot| -> angle)."""
    d = torch.abs((a * b).sum(dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(d)

def main():
    em = build("mujoco")
    en = build("newton")
    # engines may order BODIES differently -- align by name
    names_m = em.get_obj_body_names(0)
    names_n = en.get_obj_body_names(0)
    assert set(names_m) == set(names_n), f"body sets differ: {names_m} vs {names_n}"
    idx = [names_n.index(nm) for nm in names_m]
    print(f"bodies: {len(names_m)} (order aligned by name; newton order matches: {idx == list(range(len(idx)))})")

    dof = write_state(em, 7)
    write_state(en, 7)
    # newton serves body tensors from buffers that do NOT auto-refresh on joint_q writes
    # (the env seeds them via kin-model FK -- the set_body_pos pattern); the probe refreshes
    # via the engine's own FK so both arms serve derived state
    en._sim_state.pre_step_update()   # exp-map dof writes -> joint_q (sphere-joint path)
    en._sim_state.eval_fk()

    # C1: FK
    bp_m = em.get_body_pos(0)[0]
    bp_n = en.get_body_pos(0)[0][idx]
    br_m = em.get_body_rot(0)[0]
    br_n = en.get_body_rot(0)[0][idx]
    dp = (bp_m - bp_n).norm(dim=-1)
    dr = q_dist(br_m, br_n)
    print(f"C1 FK: max |dpos| = {float(dp.max()):.2e} m  max qangle = {float(dr.max()):.2e} rad "
          f"{'OK' if dp.max() < 1e-3 and dr.max() < 1e-2 else 'FAIL'}")
    if dp.max() >= 1e-3:
        worst = int(dp.argmax())
        print(f"   worst body: {names_m[worst]}  mujoco {bp_m[worst].numpy()}  newton {bp_n[worst].numpy()}")

    # C2: velocities
    bv_m = em.get_body_vel(0)[0]
    bv_n = en.get_body_vel(0)[0][idx]
    bw_m = em.get_body_ang_vel(0)[0]
    bw_n = en.get_body_ang_vel(0)[0][idx]
    dv = (bv_m - bv_n).norm(dim=-1)
    dw = (bw_m - bw_n).norm(dim=-1)
    print(f"C2 vel: max |dvel| = {float(dv.max()):.2e}  max |dang| = {float(dw.max()):.2e} "
          f"{'OK' if dv.max() < 1e-2 and dw.max() < 1e-2 else 'FAIL'}")
    if dv.max() >= 1e-2:
        worst = int(dv.argmax())
        print(f"   worst body: {names_m[worst]}  mujoco {bv_m[worst].numpy()}  newton {bv_n[worst].numpy()}")

    # C2b ADJUDICATION: the kin model is the convention authority (motion data + obs speak
    # it). Finite-diff its FK under the SAME written state -> reference body velocities;
    # whichever arm disagrees with the kin oracle carries the convention bug.
    import anim.mjcf_char_model as mm
    import util.torch_util as tu
    cm = mm.MJCFCharModel("cpu")
    cm.load(ASSET)
    root_pos = em.get_root_pos(0)[0:1].double()
    root_rot = em.get_root_rot(0)[0:1].double()
    root_vel = em.get_root_vel(0)[0:1].double()
    root_ang = em.get_root_ang_vel(0)[0:1].double()
    dofp = em.get_dof_pos(0)[0:1].double()
    dofv = em.get_dof_vel(0)[0:1].double()
    h = 1e-3   # float32 at 1e-5 quantized the dof advance to noise -- f64 + 1e-3 is clean
    b_a, _ = cm.forward_kinematics(root_pos.float(), root_rot.float(), cm.dof_to_rot(dofp.float()).double().float())
    b_a = b_a.double()
    # advance root: pos + v*h, rot by world ang*h (left-multiply); dofs via per-joint update
    dq = tu.exp_map_to_quat(root_ang * h)
    rot2 = tu.quat_mul(dq, root_rot)
    dofp2 = dofp.clone()
    for j in range(1, cm.get_num_joints()):
        o = cm.get_joint_dof_idx(j)
        dim = cm.get_joint_dof_dim(j)
        if dim == 3:   # spherical: q2 = q * exp(w_local*h)  (right-multiply = child frame)
            q1 = tu.exp_map_to_quat(dofp[:, o:o + 3])
            q2 = tu.quat_mul(q1, tu.exp_map_to_quat(dofv[:, o:o + 3] * h))
            dofp2[:, o:o + 3] = tu.quat_to_exp_map(q2)
        elif dim == 1:
            dofp2[:, o] = dofp[:, o] + dofv[:, o] * h
    b_b, _ = cm.forward_kinematics((root_pos + root_vel * h).float(), rot2.float(), cm.dof_to_rot(dofp2.float()))
    v_kin = (b_b.double() - b_a)[0] / h
    kin_names = [cm.get_body_name(i) for i in range(v_kin.shape[0])]
    kidx = [kin_names.index(nm) for nm in names_m]
    v_kin = v_kin[kidx]
    dm = (bv_m - v_kin).norm(dim=-1)
    dn = (bv_n - v_kin).norm(dim=-1)
    print(f"C2b kin-oracle: max |mujoco - kin| = {float(dm.max()):.2e}   max |newton - kin| = {float(dn.max()):.2e}")
    for i, nm in enumerate(names_m):
        if float(dm[i]) > 0.05 or float(dn[i]) > 0.05:
            print(f"    {nm:>16s}  |m-kin| {float(dm[i]):.3f}  |n-kin| {float(dn[i]):.3f}   kin {v_kin[i].numpy().round(3)}  m {bv_m[i].numpy().round(3)}  n {bv_n[i].numpy().round(3)}")
    print(f"    verdict: {'MUJOCO matches the kin convention' if dm.max() < dn.max() else 'NEWTON matches the kin convention'}"
          f" (the other arm carries the convention delta)")

    # C3: dynamics ballpark, PD holding the written pose
    tgt = dof.expand(N, -1).clone()
    em.set_cmd(0, tgt)
    en.set_cmd(0, tgt)
    for k in range(10):
        em.step()
        en.step()
        if k in (0, 4, 9):
            drift = float((em.get_root_pos(0)[0] - en.get_root_pos(0)[0]).norm())
            print(f"C3 dynamics: after step {k+1:2d}  root divergence = {drift:.4f} m")
    ok3 = float((em.get_root_pos(0)[0] - en.get_root_pos(0)[0]).norm()) < 0.05
    print(f"C3 verdict: {'OK (solver-level divergence only)' if ok3 else 'CHECK (over 5 cm at 10 steps)'}")
    return

if __name__ == "__main__":
    main()
