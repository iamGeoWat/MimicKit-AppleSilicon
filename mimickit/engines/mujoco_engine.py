"""Native C MuJoCo engine for MimicKit -- openksp NM-T1-ENGINE (2026-09-02).

Why: mujoco_warp's CPU fallback runs GPU-shaped kernels and measures 365x slower than native
C MuJoCo on the same MJCF and machine (368 vs 134,376 control-samples/s; see
openksp/docs/research/neural-motion/t1-local-training-recipe.md). This engine implements the
MimicKit Engine interface over the `mujoco` python bindings with one MjData per env and a
persistent thread pool (mj_step releases the GIL; measured 1.07M physics steps/s x15 threads).

Semantic mirror of newton_engine (the reference arm), verified choice by choice:
  * State is served as torch tensors [num_envs, ...] on the engine device; quaternions XYZW
    (mujoco stores WXYZ -- converted at the scatter/gather boundary).
  * ControlMode.pos == newton's _build_controls: the MJCF's joint `stiffness`/`damping` are
    REPURPOSED as PD gains toward target_pos and the passive terms zeroed. Here: each <motor>
    actuator is replaced by a <position kp=stiffness kv=damping forcerange=actuatorfrcrange>
    servo on the same joint (same order), and the joint's passive stiffness/damping zeroed --
    mujoco then applies the identical PD law per SUBSTEP in C (closed-loop, like newton).
  * Envs are independent MjData -> no cross-env collisions by construction (newton needs
    env_spacing for this).
  * Gravity is configurable per run: engine config key `gravity: [x, y, z]` (the NM-T1
    g-channel seam). Default = the MJCF's own (-9.81 z).

V1 scope (asserted, not silently wrong): headless only; ONE articulated non-visual char per
env; hinge/slide joints only (the sphere-joint exp-map machinery of newton_engine is NOT
mirrored -- humanoid.xml is all-hinge); control modes none/pos/torque.

VERIFY-1 (open probe): root/body angular velocities are served in WORLD frame (mujoco free
joints store body-frame -- converted here). The cross-arm comparability run against newton
must confirm; if newton serves body-frame, flip ANG_VEL_WORLD.
"""

import concurrent.futures as futures
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET

import mujoco
import mujoco.rollout as mj_rollout

import engines.engine as engine
import util.torch_util as torch_util
from util.logger import Logger

ANG_VEL_WORLD = True     # newton-arm semantics served to envs (VERIFY-1)
MJ_FREE_ANG_LOCAL = True   # PROVEN by the kinematic probe 2026-09-02: mujoco free-joint
                           # angular qvel is BODY-frame (commanded world (0,0,3) under a 45deg
                           # tilt surfaced in cvel as R_x(45)z*3) -- convert both ways


def _quat_wxyz_to_xyzw(q):
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _quat_xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def _quat_rotate(q_xyzw, v, inverse=False):
    """Rotate vectors v [..., 3] by quaternions q [..., 4] (xyzw), numpy."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    u = q[..., :3]
    w = q[..., 3:4]
    if inverse:
        u = -u
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


class MujocoEngine(engine.Engine):
    def __init__(self, config, num_envs, device, visualize, record_video=False):
        super().__init__(visualize=visualize)
        assert not visualize, "MujocoEngine v1 is headless-only (use the newton arm to view)"

        self._device = device
        self._num_envs = num_envs

        sim_freq = config.get("sim_freq", 60)
        control_freq = config.get("control_freq", 10)
        assert sim_freq >= control_freq and sim_freq % control_freq == 0
        self._timestep = 1.0 / control_freq
        self._sim_steps = int(sim_freq / control_freq)
        self._sim_timestep = 1.0 / sim_freq
        self._sim_step_count = 0

        self._gravity_cfg = config.get("gravity", None)   # [x,y,z] or None = MJCF default
        self._num_threads = int(config.get("num_threads", min(num_envs, os.cpu_count() or 8)))

        if "control_mode" in config:
            self._control_mode = engine.ControlMode[config["control_mode"]]
        else:
            self._control_mode = engine.ControlMode.none
        assert self._control_mode in (engine.ControlMode.none, engine.ControlMode.pos,
                                      engine.ControlMode.torque), \
            "MujocoEngine v1: control_mode none/pos/torque only"

        self._env_count = 0
        self._char_asset = None
        self._start_pos = None
        self._start_rot = None
        self._record_video = record_video
        return

    # ---- construction ------------------------------------------------------------------

    def get_name(self):
        return "mujoco"

    def create_env(self):
        env_id = self._env_count
        assert env_id < self._num_envs
        self._env_count += 1
        return env_id

    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False,
                   enable_self_collisions=True, fix_root=False, start_pos=None,
                   start_rot=None, color=None, disable_motors=False):
        assert obj_type == engine.ObjType.articulated and not is_visual and not fix_root, \
            "MujocoEngine v1: one articulated dynamic char per env"
        if self._char_asset is not None:
            assert self._char_asset == asset_file, "MujocoEngine v1: one shared char asset"
            return 0
        self._char_asset = asset_file
        self._start_pos = np.array([0.0, 0.0, 0.0]) if start_pos is None else np.asarray(start_pos, dtype=np.float64)
        self._start_rot = np.array([0.0, 0.0, 0.0, 1.0]) if start_rot is None else np.asarray(start_rot, dtype=np.float64)
        return 0

    def _transform_mjcf(self, path):
        """XML-level mirror of newton _build_controls + _build_ground: floor plane added,
        <motor> -> <position kp kv forcerange> (pos mode), passive stiffness/damping zeroed."""
        tree = ET.parse(path)
        root = tree.getroot()

        wb = root.find("worldbody")
        floor = ET.SubElement(wb, "geom")
        floor.set("name", "engine_ground")
        floor.set("type", "plane")
        floor.set("size", "50 50 1")
        floor.set("pos", "0 0 0")

        joints = {}
        for j in root.iter("joint"):
            nm = j.get("name")
            if nm is not None:
                joints[nm] = j

        act = root.find("actuator")
        assert act is not None, "char MJCF has no <actuator> block"
        motors = list(act)
        by_joint = {m.get("joint"): m for m in motors}
        assert len(by_joint) == len(motors), "multiple actuators on one joint"
        # RE-EMIT in JOINT (tree/dof) order -- set_cmd arrives dof-ordered, and this MJCF's
        # hand-written actuator block swaps the hip y/z pairs
        joint_order = [j.get("name") for j in root.iter("joint")
                       if j.get("name") in by_joint]
        motors = [by_joint[jn] for jn in joint_order]
        self._act_joint_names = joint_order

        if self._control_mode == engine.ControlMode.pos:
            for m in list(act):
                act.remove(m)
            for m in motors:
                jn = m.get("joint")
                j = joints[jn]
                kp = float(j.get("stiffness", "0"))
                kd = float(j.get("damping", "0"))
                frng = j.get("actuatorfrcrange")
                p = ET.SubElement(act, "position")
                p.set("name", m.get("name", jn))
                p.set("joint", jn)
                p.set("kp", repr(kp))
                p.set("kv", repr(kd))
                if frng is not None:
                    p.set("forcerange", frng)
                # the MJCF default class carries ctrlrange [-1,1] -- newton's target_pos has
                # no such clamp, and PD targets are RADIANS; inherit nothing
                p.set("ctrllimited", "false")
                # newton zeroes the passive terms after stealing them as PD gains
                j.set("stiffness", "0")
                j.set("damping", "0")
        elif self._control_mode == engine.ControlMode.torque:
            for m in list(act):
                act.remove(m)
            for m in motors:   # keep the motors, in dof order
                act.append(m)
        elif self._control_mode == engine.ControlMode.none:
            for m in list(act):
                act.remove(m)

        opt = root.find("option")
        if opt is None:
            opt = ET.SubElement(root, "option")
        opt.set("timestep", repr(self._sim_timestep))
        # newton's SolverMuJoCo runs an implicit solver; mujoco's Euler at 240 Hz collapses
        # under the stiff repurposed PD (root z 0.89 -> 0.09 in 2 s, measured) -- implicitfast
        # is the matching integrator for damped stiff actuation
        opt.set("integrator", "implicitfast")
        if self._gravity_cfg is not None:
            g = self._gravity_cfg
            opt.set("gravity", "%r %r %r" % (g[0], g[1], g[2]))

        return ET.tostring(root, encoding="unicode")

    def initialize_sim(self):
        assert self._env_count == self._num_envs and self._char_asset is not None
        xml = self._transform_mjcf(self._char_asset)
        self._model = mujoco.MjModel.from_xml_string(xml)
        m = self._model

        # v1 scope guard: free root + scalar joints only (mirrors the non-sphere path of newton)
        assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        assert all(t in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
                   for t in m.jnt_type[1:]), "sphere joints not supported in v1"

        self._num_dofs = m.nv - 6
        # actuator order must equal dof order (set_cmd is dof-ordered)
        dof_joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                           for j in range(1, m.njnt)]
        if self._control_mode in (engine.ControlMode.pos, engine.ControlMode.torque):
            assert self._act_joint_names == dof_joint_names, \
                "actuator order != joint order; a permutation table is needed"

        # bodies: skip worldbody (id 0) -- newton's per-articulation body list has no world
        self._num_bodies = m.nbody - 1

        self._datas = [mujoco.MjData(m) for _ in range(self._num_envs)]
        self._pool = futures.ThreadPoolExecutor(max_workers=self._num_threads)

        N, nd, nb = self._num_envs, self._num_dofs, self._num_bodies
        f32, dev = torch.float32, self._device
        self._root_pos = torch.zeros((N, 3), dtype=f32, device=dev)
        self._root_rot = torch.zeros((N, 4), dtype=f32, device=dev)
        self._root_vel = torch.zeros((N, 3), dtype=f32, device=dev)
        self._root_ang_vel = torch.zeros((N, 3), dtype=f32, device=dev)
        self._dof_pos = torch.zeros((N, nd), dtype=f32, device=dev)
        self._dof_vel = torch.zeros((N, nd), dtype=f32, device=dev)
        self._body_pos = torch.zeros((N, nb, 3), dtype=f32, device=dev)
        self._body_rot = torch.zeros((N, nb, 4), dtype=f32, device=dev)
        self._body_vel = torch.zeros((N, nb, 3), dtype=f32, device=dev)
        self._body_ang_vel = torch.zeros((N, nb, 3), dtype=f32, device=dev)
        self._contact_forces = torch.zeros((N, nb, 3), dtype=f32, device=dev)
        self._ground_contact_forces = torch.zeros((N, nb, 3), dtype=f32, device=dev)
        self._dof_forces = torch.zeros((N, nd), dtype=f32, device=dev)
        self._targets = torch.zeros((N, nd), dtype=f32, device=dev)

        self._root_pos[:] = torch.as_tensor(self._start_pos, dtype=f32)
        self._root_rot[:] = torch.as_tensor(self._start_rot, dtype=f32)
        # (buffer init happens post-codec-build; qpos0 is all zeros for this MJCF and the
        # codec maps zero hinges -> zero exp-map, but encode anyway for generality)

        self._gravity = np.array(m.opt.gravity, dtype=np.float64)
        assert m.na == 0, "stateful actuators unsupported (FULLPHYSICS packing assumes na=0)"
        self._nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        assert self._nstate == 1 + m.nq + m.nv, "unexpected FULLPHYSICS layout"
        self._roll_state0 = np.zeros((self._num_envs, self._nstate))
        self._roll_ctrl = np.zeros((self._num_envs, self._sim_steps, m.nu))
        self._alloc_raw()
        self._build_dof_codec()
        self._dof_pos[:] = torch.as_tensor(
            self._encode_dof_pos(m.qpos0[7:][None, :].repeat(self._num_envs, axis=0)), dtype=f32)
        self._dirty = True
        self._flush()
        return

    # ---- dof codec: hinge angles <-> the kin-model's SPHERICAL exp-map convention -------
    # The kin char model (and newton's importer) consolidate each 3-hinge x,y,z series into
    # ONE spherical joint whose dofs are EXP-MAP coordinates; motions, observations and AMP
    # demo features all speak that language. The mujoco dynamics keep 28 hinges (fast, native
    # PD); this codec converts at the engine interface. Calibrated + verified by
    # tools/probe_dof_codec.py: composition q = qx(a)*qy(b)*qz(c) (machine precision vs
    # mujoco FK), decomposition roundtrip 1e-15, kin-FK == mujoco-FK to 3e-7 m.

    def _build_dof_codec(self):
        m = self._model
        from collections import defaultdict
        byb = defaultdict(list)
        for j in range(1, m.njnt):
            byb[int(m.jnt_bodyid[j])].append(j)
        cl_off, cl_body, cl_parent, hinge_dofs = [], [], [], []
        for bid, js in sorted(byb.items(), key=lambda kv: min(kv[1])):
            offs = [int(m.jnt_dofadr[j]) - 6 for j in js]
            if len(js) == 3:
                axes = np.stack([m.jnt_axis[j] for j in js])
                assert np.allclose(axes, np.eye(3)), \
                    "cluster axes not canonical x,y,z -- codec calibration does not apply"
                assert offs == list(range(offs[0], offs[0] + 3))
                cl_off.append(offs[0])
                cl_body.append(bid - 1)
                cl_parent.append(int(m.body_parentid[bid]) - 1)
            else:
                assert len(js) == 1, "unexpected joint grouping"
                hinge_dofs += offs
        self._cl_off = np.array(cl_off, dtype=int)
        self._cl_body = np.array(cl_body, dtype=int)
        self._cl_parent = np.array(cl_parent, dtype=int)
        assert (self._cl_parent >= 0).all(), "cluster body parented to world?"
        self._n_cl = len(cl_off)
        Logger.print("mujoco engine dof codec: %d spherical clusters, %d plain hinges"
                     % (self._n_cl, len(hinge_dofs)))
        return

    @staticmethod
    def _q_axis_batch(ang, axis_idx):
        """axis-angle quats about a canonical axis for a [...]-shaped angle array (xyzw)."""
        q = np.zeros(ang.shape + (4,))
        q[..., axis_idx] = np.sin(ang / 2.0)
        q[..., 3] = np.cos(ang / 2.0)
        return q

    @staticmethod
    def _q_mul(a, b):
        ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                         aw * by + ay * bw + az * bx - ax * bz,
                         aw * bz + az * bw + ax * by - ay * bx,
                         aw * bw - ax * bx - ay * by - az * bz], axis=-1)

    def _cluster_quats(self, angles):
        """angles [..., 3] hinge (a,b,c) -> quats [..., 4], q = qx*qy*qz (calibrated D1)."""
        return self._q_mul(self._q_mul(self._q_axis_batch(angles[..., 0], 0),
                                       self._q_axis_batch(angles[..., 1], 1)),
                           self._q_axis_batch(angles[..., 2], 2))

    @staticmethod
    def _decompose_xyz(q):
        """quats [..., 4] xyzw -> hinge angles [..., 3] for q = qx*qy*qz (probe D2)."""
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        r02 = 2 * (x * z + y * w)
        r12 = 2 * (y * z - x * w)
        r22 = 1 - 2 * (x * x + y * y)
        r01 = 2 * (x * y - z * w)
        r00 = 1 - 2 * (y * y + z * z)
        b = np.arcsin(np.clip(r02, -1.0, 1.0))
        a = np.arctan2(-r12, r22)
        c = np.arctan2(-r01, r00)
        return np.stack([a, b, c], axis=-1)

    @staticmethod
    def _quat_to_exp(q):
        """torch_util's exact exp-map convention (batched numpy in/out)."""
        t = torch.as_tensor(q.reshape(-1, 4), dtype=torch.float64)
        e = torch_util.quat_to_exp_map(t).numpy()
        return e.reshape(q.shape[:-1] + (3,))

    @staticmethod
    def _exp_to_quat(e):
        t = torch.as_tensor(e.reshape(-1, 3), dtype=torch.float64)
        q = torch_util.exp_map_to_quat(t).numpy()
        return q.reshape(e.shape[:-1] + (4,))

    def _encode_dof_pos(self, hinge_dof):
        """[N, nd] hinge angles -> kin-convention dof (clusters as exp-map)."""
        out = hinge_dof.copy()
        for k, o in enumerate(self._cl_off):
            out[:, o:o + 3] = self._quat_to_exp(self._cluster_quats(hinge_dof[:, o:o + 3]))
        return out

    def _decode_dof_pos(self, dof):
        """kin-convention dof -> [N, nd] hinge angles."""
        out = dof.copy()
        for k, o in enumerate(self._cl_off):
            out[:, o:o + 3] = self._decompose_xyz(self._exp_to_quat(dof[:, o:o + 3]))
        return out

    def _cluster_J(self, angles, h=1e-3):
        """numeric ang-vel Jacobians [..., 3, 3] (probe D3): columns = d(exp of relative
        quat)/d rate, central differences over the calibrated composition."""
        cols = []
        for k in range(3):
            dp = np.zeros(angles.shape)
            dp[..., k] = h
            qp = self._cluster_quats(angles + dp)
            qm = self._cluster_quats(angles - dp)
            qc = qm.copy()
            qc[..., :3] *= -1.0
            cols.append(self._quat_to_exp(self._q_mul(qc, qp)) / (2 * h))
        return np.stack(cols, axis=-1)

    def _encode_dof_vel(self, hinge_dof, hinge_vel):
        """hinge rates -> cluster local angular velocity (via J), hinges passthrough."""
        out = hinge_vel.copy()
        Js = None
        for k, o in enumerate(self._cl_off):
            J = self._cluster_J(hinge_dof[:, o:o + 3])
            out[:, o:o + 3] = np.einsum("nij,nj->ni", J, hinge_vel[:, o:o + 3])
        return out

    def _decode_dof_vel(self, hinge_dof, dof_vel):
        """cluster local angular velocity -> hinge rates (J solve at the decoded angles)."""
        out = dof_vel.copy()
        for k, o in enumerate(self._cl_off):
            J = self._cluster_J(hinge_dof[:, o:o + 3])
            out[:, o:o + 3] = np.linalg.solve(J, dof_vel[:, o:o + 3, None])[..., 0]
        return out

    # ---- state movement (chunked workers + cross-env vectorized math) ------------------
    # The naive path (64 pool tasks/step + per-env python math) costs ~6 ms/step against
    # ~0.4 ms of actual physics. Here: threads get CHUNKS of envs, per-env python touches
    # only raw field copies + the (small) contact loop, and all frame math runs once,
    # vectorized across envs.

    def _alloc_raw(self):
        m, N = self._model, self._num_envs
        self._raw_qpos = np.zeros((N, m.nq))
        self._raw_qvel = np.zeros((N, m.nv))
        self._raw_xpos = np.zeros((N, m.nbody, 3))
        self._raw_xquat = np.zeros((N, m.nbody, 4))
        self._raw_cvel = np.zeros((N, m.nbody, 6))
        self._raw_anchor = np.zeros((N, m.nbody, 3))
        self._raw_qfrc = np.zeros((N, m.nv))
        self._raw_cf = np.zeros((N, self._num_bodies, 3))
        self._raw_gf = np.zeros((N, self._num_bodies, 3))
        self._chunks = []
        base = 0
        for t in range(self._num_threads):
            n = N // self._num_threads + (1 if t < N % self._num_threads else 0)
            if n > 0:
                self._chunks.append((base, base + n))
            base += n
        return

    def _scatter_chunk(self, lo, hi):
        for i in range(lo, hi):
            d = self._datas[i]
            d.qpos[:] = self._sc_qpos[i]
            d.qvel[:] = self._sc_qvel[i]
            mujoco.mj_forward(self._model, d)
        return

    def _step_chunk(self, lo, hi):
        m, ns = self._model, self._sim_steps
        for i in range(lo, hi):
            d = self._datas[i]
            d.ctrl[:] = self._np_targets[i]
            for _ in range(ns):
                mujoco.mj_step(m, d)
        return

    def _collect_chunk(self, lo, hi, contacts):
        m = self._model
        buf = np.zeros(6)
        for i in range(lo, hi):
            d = self._datas[i]
            self._raw_qpos[i] = d.qpos
            self._raw_qvel[i] = d.qvel
            self._raw_xpos[i] = d.xpos
            self._raw_xquat[i] = d.xquat
            self._raw_cvel[i] = d.cvel
            self._raw_anchor[i] = d.subtree_com[m.body_rootid]
            self._raw_qfrc[i] = d.qfrc_actuator
            if contacts:
                cf = self._raw_cf[i]
                gf = self._raw_gf[i]
                cf[:] = 0.0
                gf[:] = 0.0
                for c in range(d.ncon):
                    con = d.contact[c]
                    mujoco.mj_contactForce(m, d, c, buf)
                    f_world = con.frame.reshape(3, 3).T @ buf[0:3]
                    b1 = m.geom_bodyid[con.geom1]
                    b2 = m.geom_bodyid[con.geom2]
                    # SIGN (empirical, probe P1): the world-rotated force acts on geom2;
                    # settled standing sums to +m*g upward on the char
                    if b2 > 0:
                        cf[b2 - 1] += f_world
                        if b1 == 0:
                            gf[b2 - 1] += f_world
                    if b1 > 0:
                        cf[b1 - 1] -= f_world
                        if b2 == 0:
                            gf[b1 - 1] -= f_world
        return

    def _gather_all(self, contacts):
        """Collect raw fields per env (threaded), then run every conversion ONCE across envs."""
        for lo, hi in self._chunks:
            self._collect_chunk(lo, hi, contacts)

        qpos, qvel = self._raw_qpos, self._raw_qvel
        rot = _quat_wxyz_to_xyzw(qpos[:, 3:7])
        w = qvel[:, 3:6]
        if ANG_VEL_WORLD and MJ_FREE_ANG_LOCAL:
            w = _quat_rotate(rot, w)

        t = torch.as_tensor
        f32 = torch.float32
        self._root_pos[:] = t(qpos[:, 0:3], dtype=f32)
        self._root_rot[:] = t(rot, dtype=f32)
        self._root_vel[:] = t(qvel[:, 0:3], dtype=f32)
        self._root_ang_vel[:] = t(w, dtype=f32)
        hinge_q = qpos[:, 7:]
        hinge_qd = qvel[:, 6:]
        self._dof_pos[:] = t(self._encode_dof_pos(hinge_q), dtype=f32)
        self._dof_vel[:] = t(self._encode_dof_vel(hinge_q, hinge_qd), dtype=f32)

        self._body_pos[:] = t(self._raw_xpos[:, 1:], dtype=f32)
        self._body_rot[:] = t(_quat_wxyz_to_xyzw(self._raw_xquat[:, 1:]), dtype=f32)
        ang = self._raw_cvel[:, 1:, 0:3]
        self._body_ang_vel[:] = t(ang, dtype=f32)
        lin = self._raw_cvel[:, 1:, 3:6] + np.cross(ang, self._raw_xpos[:, 1:] - self._raw_anchor[:, 1:])
        self._body_vel[:] = t(lin, dtype=f32)
        self._dof_forces[:] = t(self._raw_qfrc[:, 6:], dtype=f32)
        if contacts:
            self._contact_forces[:] = t(self._raw_cf, dtype=f32)
            self._ground_contact_forces[:] = t(self._raw_gf, dtype=f32)
        return

    def _flush(self):
        """After set_* writes: push torch state into every MjData, refresh kinematics."""
        if not self._dirty:
            return
        rp = self._root_pos.numpy().astype(np.float64)
        rr = self._root_rot.numpy().astype(np.float64)
        rv = self._root_vel.numpy().astype(np.float64)
        rw = self._root_ang_vel.numpy().astype(np.float64)
        if ANG_VEL_WORLD and MJ_FREE_ANG_LOCAL:
            rw = _quat_rotate(rr, rw, inverse=True)
        m = self._model
        self._sc_qpos = np.zeros((self._num_envs, m.nq))
        self._sc_qvel = np.zeros((self._num_envs, m.nv))
        self._sc_qpos[:, 0:3] = rp
        self._sc_qpos[:, 3:7] = _quat_xyzw_to_wxyz(rr)
        hinge_q = self._decode_dof_pos(self._dof_pos.numpy().astype(np.float64))
        self._sc_qpos[:, 7:] = hinge_q
        self._sc_qvel[:, 0:3] = rv
        self._sc_qvel[:, 3:6] = rw
        self._sc_qvel[:, 6:] = self._decode_dof_vel(hinge_q, self._dof_vel.numpy().astype(np.float64))
        if len(self._chunks) > 1:
            list(self._pool.map(lambda c: self._scatter_chunk(*c), self._chunks))
        else:
            self._scatter_chunk(0, self._num_envs)
        self._gather_all(contacts=False)
        self._dirty = False
        return

    # ---- stepping ----------------------------------------------------------------------

    def set_cmd(self, obj_id, cmd):
        if self._control_mode == engine.ControlMode.none:
            return
        tg = cmd.numpy().astype(np.float64) if torch.is_tensor(cmd) else np.asarray(cmd, dtype=np.float64)
        self._targets[:] = torch.as_tensor(self._decode_dof_pos(tg), dtype=torch.float32)
        return

    def _restore_chunk(self, lo, hi):
        m = self._model
        for i in range(lo, hi):
            d = self._datas[i]
            mujoco.mj_setState(m, d, self._roll_final[i], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            d.ctrl[:] = self._np_targets[i]
            mujoco.mj_forward(m, d)
        return

    def step(self):
        self._flush()
        m = self._model
        self._np_targets = self._targets.numpy().astype(np.float64)
        # pack FULLPHYSICS = [time, qpos, qvel] straight from the env datas' current state
        for i in range(self._num_envs):
            d = self._datas[i]
            self._roll_state0[i, 0] = d.time
            self._roll_state0[i, 1:1 + m.nq] = d.qpos
            self._roll_state0[i, 1 + m.nq:] = d.qvel
        self._roll_ctrl[:] = self._np_targets[:, None, :]
        # all substeps run in mujoco's C-side thread pool (no per-substep python, no GIL thrash)
        state, _ = mj_rollout.rollout(m, self._datas[:self._num_threads],
                                      self._roll_state0, self._roll_ctrl,
                                      nstep=self._sim_steps)
        self._roll_final = state[:, -1, :]
        # restore each env's data to its final state + forward for kinematics/contacts
        if len(self._chunks) > 1:
            list(self._pool.map(lambda c: self._restore_chunk(*c), self._chunks))
        else:
            self._restore_chunk(0, self._num_envs)
        self._gather_all(contacts=True)
        self._sim_step_count += 1
        return

    # ---- getters (torch views over the canonical buffers) ------------------------------

    def get_timestep(self):
        return self._timestep

    def get_num_envs(self):
        return self._num_envs

    def get_gravity(self):
        return self._gravity

    def get_root_pos(self, obj_id):
        self._flush()
        return self._root_pos

    def get_root_rot(self, obj_id):
        self._flush()
        return self._root_rot

    def get_root_vel(self, obj_id):
        self._flush()
        return self._root_vel

    def get_root_ang_vel(self, obj_id):
        self._flush()
        return self._root_ang_vel

    def get_dof_pos(self, obj_id):
        self._flush()
        return self._dof_pos

    def get_dof_vel(self, obj_id):
        self._flush()
        return self._dof_vel

    def get_dof_forces(self, obj_id):
        return self._dof_forces

    def get_body_pos(self, obj_id):
        self._flush()
        return self._body_pos

    def get_body_rot(self, obj_id):
        self._flush()
        return self._body_rot

    def get_body_vel(self, obj_id):
        self._flush()
        return self._body_vel

    def get_body_ang_vel(self, obj_id):
        self._flush()
        return self._body_ang_vel

    def get_contact_forces(self, obj_id):
        return self._contact_forces

    def get_ground_contact_forces(self, obj_id):
        return self._ground_contact_forces

    # ---- setters -----------------------------------------------------------------------

    def _set(self, buf, env_id, value):
        if env_id is None:
            buf[:] = value
        else:
            buf[env_id] = value
        self._dirty = True
        return

    def set_root_pos(self, env_id, obj_id, root_pos):
        self._set(self._root_pos, env_id, root_pos)

    def set_root_rot(self, env_id, obj_id, root_rot):
        self._set(self._root_rot, env_id, root_rot)

    def set_root_vel(self, env_id, obj_id, root_vel):
        self._set(self._root_vel, env_id, root_vel)

    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        self._set(self._root_ang_vel, env_id, root_ang_vel)

    def set_dof_pos(self, env_id, obj_id, dof_pos):
        self._set(self._dof_pos, env_id, dof_pos)

    def set_dof_vel(self, env_id, obj_id, dof_vel):
        self._set(self._dof_vel, env_id, dof_vel)

    def _assert_zero_write(self, value, what):
        """deepmimic's _ref_state_init clears newton's STALE body-velocity buffers with 0.0
        after setting root+dof state. Here body velocities are DERIVED (mj_forward from the
        scattered qvel at flush), so the clear is a semantic no-op -- and a NON-zero per-body
        write has no qvel counterpart and stays unsupported."""
        z = torch.as_tensor(value)
        assert torch.count_nonzero(z) == 0, "v1: only zero body-%s writes (buffer clears)" % what
        return

    def set_body_vel(self, env_id, obj_id, body_vel):
        self._assert_zero_write(body_vel, "vel")

    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
        self._assert_zero_write(body_ang_vel, "ang-vel")

    def set_body_pos(self, env_id, obj_id, body_pos):
        # char_env._reset_char_rigid_body_state seeds newton's body_q buffers with the KIN
        # model's FK of the root+dof state it just wrote (newton would serve stale buffers
        # otherwise). Here body poses are DERIVED at flush (mj_forward on the same root+dof),
        # so the seed is redundant -- absorbed. (The visual ref-char write is unreachable:
        # v1 asserts no visual objects.)
        return

    def set_body_rot(self, env_id, obj_id, body_rot):
        return

    def set_body_forces(self, env_id, obj_id, body_id, forces):
        raise NotImplementedError("v1: external body forces unsupported")

    # ---- object metadata ---------------------------------------------------------------

    def get_objs_per_env(self):
        return 1

    def get_obj_type(self, obj_id):
        return engine.ObjType.articulated

    def get_obj_num_dofs(self, obj_id):
        return self._num_dofs

    def get_obj_num_bodies(self, obj_id):
        return self._num_bodies

    def get_obj_body_names(self, obj_id):
        m = self._model
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, m.nbody)]

    def find_obj_body_id(self, obj_id, body_name):
        return self.get_obj_body_names(obj_id).index(body_name)

    def get_obj_torque_limits(self, env_id, obj_id):
        m = self._model
        lim = np.abs(m.actuator_forcerange[:, 1])
        if not np.any(lim):
            lim = np.abs(m.actuator_gear[:, 0])
        return lim.astype(np.float32)

    def get_obj_dof_limits(self, env_id, obj_id):
        m = self._model
        return (m.jnt_range[1:, 0].astype(np.float32),
                m.jnt_range[1:, 1].astype(np.float32))

    def get_obj_pd_gains(self, env_id, obj_id):
        m = self._model
        kp = m.actuator_gainprm[:, 0].copy()
        kd = -m.actuator_biasprm[:, 2].copy()
        return kp.astype(np.float32), kd.astype(np.float32)

    def calc_obj_mass(self, env_id, obj_id):
        return float(self._model.body_mass[1:].sum())

    def get_control_mode(self):
        return self._control_mode

    # ---- misc --------------------------------------------------------------------------

    def render(self):
        return

    def set_camera_pose(self, pos, look_at):
        return

    def get_camera_pos(self):
        return np.zeros(3)

    def get_camera_dir(self):
        return np.array([1.0, 0.0, 0.0])
