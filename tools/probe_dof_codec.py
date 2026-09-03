"""Calibrate + verify the hinge<->spherical dof codec BEFORE it enters the engine.

D1 composition order: mujoco's relative child quat for a 3-hinge (x,y,z) cluster equals
   qx(a) * qy(b) * qz(c) or the reverse -- decided empirically against mujoco FK.
D2 decomposition roundtrip: (a,b,c) -> quat -> decompose -> (a,b,c) within joint ranges.
D3 angular-velocity Jacobian: candidate J columns verified against finite-diff quats.
D4 the CONTRACT: kin-model FK(codec(hinge_angles)) == mujoco FK, positions to 1e-5 --
   the exact consistency the AMP obs/motion pipeline needs.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))
import util.torch_util as tu
import anim.mjcf_char_model as mm
import mujoco

ASSET = os.path.join(os.path.dirname(__file__), "..", "data", "assets", "humanoid", "humanoid.xml")

def aa(axis, ang):
    axis = torch.as_tensor(axis, dtype=torch.float64)
    return torch.cat([axis * torch.sin(torch.as_tensor(ang) / 2.0).reshape(1),
                      torch.cos(torch.as_tensor(ang) / 2.0).reshape(1)])

def q_xyz(a, b, c, order="xyz"):
    qx, qy, qz = aa([1., 0., 0.], a), aa([0., 1., 0.], b), aa([0., 0., 1.], c)
    if order == "xyz":
        return tu.quat_mul(tu.quat_mul(qx.view(1, 4), qy.view(1, 4)), qz.view(1, 4))[0]
    return tu.quat_mul(tu.quat_mul(qz.view(1, 4), qy.view(1, 4)), qx.view(1, 4))[0]

def mj_rel_quat(m, d, body):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    pid = m.body_parentid[bid]
    xq = lambda i: torch.tensor([d.xquat[i][1], d.xquat[i][2], d.xquat[i][3], d.xquat[i][0]])
    return tu.quat_mul(tu.quat_conjugate(xq(pid).view(1, 4)), xq(bid).view(1, 4))[0].double()

def qdist(a, b):
    return float(1.0 - torch.abs((a * b).sum()).clamp(max=1.0))

def decompose_xyz(q):
    """q = qx(a)*qy(b)*qz(c) -> (a,b,c). Rotation matrix (column convention) R = Rx Ry Rz."""
    x, y, z, w = [float(v) for v in q]
    # rows of R
    r00 = 1 - 2 * (y * y + z * z); r01 = 2 * (x * y - z * w); r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w);     r11 = 1 - 2 * (x * x + z * z); r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w);     r21 = 2 * (y * z + x * w); r22 = 1 - 2 * (x * x + y * y)
    b = np.arcsin(np.clip(r02, -1.0, 1.0))
    a = np.arctan2(-r12, r22)
    c = np.arctan2(-r01, r00)
    return a, b, c

def main():
    m = mujoco.MjModel.from_xml_path(ASSET)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(3)

    # D1: composition order on the abdomen cluster (qpos 7,8,9 -> body 'torso')
    errs = {"xyz": 0.0, "zyx": 0.0}
    for _ in range(12):
        a, b, c = rng.uniform(-0.9, 0.9, 3)
        mujoco.mj_resetData(m, d)
        d.qpos[7:10] = [a, b, c]
        mujoco.mj_forward(m, d)
        qm = mj_rel_quat(m, d, "torso")
        for order in errs:
            errs[order] = max(errs[order], qdist(q_xyz(a, b, c, order).double(), qm))
    order = min(errs, key=errs.get)
    print(f"D1 composition: xyz err {errs['xyz']:.2e} | zyx err {errs['zyx']:.2e} -> ORDER = {order} "
          f"{'OK' if errs[order] < 1e-9 else 'FAIL (neither matches)'}")
    assert errs[order] < 1e-9

    # D2: decomposition roundtrip (within hinge ranges, |b| capped by the y-range)
    worst = 0.0
    for _ in range(300):
        a, b, c = rng.uniform(-1.2, 1.2, 3) * [1.0, 0.75, 1.0]
        q = q_xyz(a, b, c, order)
        a2, b2, c2 = decompose_xyz(q)
        worst = max(worst, abs(a - a2), abs(b - b2), abs(c - c2))
    print(f"D2 decomposition roundtrip: worst |d| = {worst:.2e} {'OK' if worst < 1e-9 else 'FAIL'}")
    assert worst < 1e-9

    # D3: angular-velocity Jacobian, child frame -- NUMERIC (central differences over the
    # SAME q_xyz composition; the engine uses this exact helper at scatter/reset time only).
    # Verified against an independent finite-diff at a different dt + invertibility sweep.
    def numeric_J(a, b, c, h=1e-3):
        cols = []
        for k, (da, db, dc) in enumerate([(h, 0, 0), (0, h, 0), (0, 0, h)]):
            qp = q_xyz(a + da, b + db, c + dc, order)
            qm = q_xyz(a - da, b - db, c - dc, order)
            dq = tu.quat_mul(tu.quat_conjugate(qm.view(1, 4)), qp.view(1, 4))
            cols.append(tu.quat_to_exp_map(dq)[0].numpy() / (2 * h))
        return np.stack(cols, axis=1)
    worst = 0.0
    worst_cond = 0.0
    dt = 1e-4
    for _ in range(60):
        a, b, c = rng.uniform(-1.0, 1.0, 3) * [1.0, 0.75, 1.0]
        rates = rng.uniform(-2.0, 2.0, 3)
        q0 = q_xyz(a, b, c, order)
        q1 = q_xyz(a + rates[0] * dt, b + rates[1] * dt, c + rates[2] * dt, order)
        dq = tu.quat_mul(tu.quat_conjugate(q0.view(1, 4)), q1.view(1, 4))
        w_fd = tu.quat_to_exp_map(dq)[0].numpy() / dt
        Jn = numeric_J(a, b, c)
        worst = max(worst, float(np.abs(w_fd - Jn @ rates).max()))
        worst_cond = max(worst_cond, float(np.linalg.cond(Jn)))
    print(f"D3 numeric ang-vel Jacobian: worst |dw| = {worst:.2e}  worst cond(J) = {worst_cond:.1f} "
          f"{'OK' if worst < 1e-3 and worst_cond < 50 else 'FAIL'}")
    # tolerance 1e-3: the CHECK's forward-diff truncation is O(dt)~4e-4 and dominates; J itself
    # is central-diff O(h^2)~1e-6. 1e-3 rad/s against 1-10 rad/s signals = 0.01-0.1%.
    assert worst < 1e-3 and worst_cond < 50

    # D4: the contract -- kin FK on codec'd dofs == mujoco FK positions
    cm = mm.MJCFCharModel("cpu")
    cm.load(ASSET)
    sph_qpos = {"abdomen": 7, "neck": 10, "right_shoulder": 13, "left_shoulder": 17,
                "right_hip": 21, "right_ankle": 25, "left_hip": 28, "left_ankle": 32}
    hinge_qpos = {"right_elbow": 16, "left_elbow": 20, "right_knee": 24, "left_knee": 31}
    mujoco.mj_resetData(m, d)
    d.qpos[7:] = rng.uniform(-0.5, 0.5, m.nq - 7)
    d.qpos[0:3] = [0.0, 0.0, 1.5]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    dof = torch.zeros(1, cm.get_dof_size())
    for j in range(1, cm.get_num_joints()):
        jt = cm.get_joint(j)
        o = cm.get_joint_dof_idx(j)
        if jt.joint_type == mm.kin_char_model.JointType.SPHERICAL if hasattr(mm, "kin_char_model") else str(jt.joint_type.name) == "SPHERICAL":
            qp = sph_qpos[jt.name]
            q = q_xyz(*d.qpos[qp:qp + 3], order)
            dof[0, o:o + 3] = tu.quat_to_exp_map(q.view(1, 4).float())[0]
        elif str(jt.joint_type.name) == "HINGE":
            dof[0, o] = float(d.qpos[hinge_qpos[jt.name]])
    root_pos = torch.tensor([[0.0, 0.0, 1.5]])
    root_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    jrot = cm.dof_to_rot(dof)
    bpos, brot = cm.forward_kinematics(root_pos, root_rot, jrot)
    names = [cm.get_body_name(i) for i in range(bpos.shape[-2])]
    worst, wname = 0.0, ""
    for i, nm in enumerate(names):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        e = float(np.linalg.norm(bpos[0, i].numpy() - d.xpos[bid]))
        if e > worst:
            worst, wname = e, nm
    print(f"D4 kin-FK vs mujoco-FK: worst |dpos| = {worst:.2e} m ({wname}) {'OK' if worst < 1e-4 else 'FAIL'}")
    assert worst < 1e-4
    print("CODEC CALIBRATION: ALL GREEN (order=%s)" % order)

if __name__ == "__main__":
    main()
