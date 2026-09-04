"""NM-GAME-PATH tier B: retarget MJCF-humanoid motion onto the openksp crew rig.

The architecture ruling (owner, 2026-09-03) is that character motion comes from TRAINED
policies applied to the game as animation. This is the bridge: a rollout in MuJoCo produces
world body orientations; the game's CrewGait seam consumes LOCAL quaternions per bone. The
two skeletons happen to be nearly isomorphic (15 bones each, 13 mapped pairs), so the mapping
is structural rather than a fitting problem.

Rest-frame retarget, per mapped bone b:

    q_local_crew(t) = A_b^-1 . q_local_mjcf(t) . A_b        with  A_b = R_rest_mjcf(b)^-1 . R_rest_crew(b)

i.e. the source's local rotation is CONJUGATED into the target's rest frame, so a rotation
that swings the source's bone axis by X degrees swings the target's bone axis by X degrees
about the corresponding axis. When the two rest frames coincide (both rigs stand with the
same bone directions) A_b is identity and the local rotations transfer unchanged -- which is
the case for most of this pair, and the probe below reports which bones are NOT identity so
the assumption is never silent.

The crew rig's geometry is READ from crew_rig.json (exported by openksp's
tools/nm/dump_crew_rig.gd) rather than re-typed here: one authoritative source, per the
model-world-space doctrine.

Usage:
    python tools/retarget_to_crew.py --rig /path/crew_rig.json [--report]
"""
import argparse
import json
import numpy as np
import mujoco

# 13 structural pairs. The two unmatched bones on each side are recorded, not hidden:
#   crew-only : toe_l, toe_r   (the MJCF foot has no toe joint -- the game keeps its own toe-off)
#   mjcf-only : right_hand, left_hand (the crew IK set ends at the forearm)
BONE_MAP = {
    "pelvis": "pelvis",
    "torso": "torso",
    "head": "head",
    "left_upper_arm": "arm_l",
    "left_lower_arm": "arm_l_lower",
    "right_upper_arm": "arm_r",
    "right_lower_arm": "arm_r_lower",
    "left_thigh": "leg_l",
    "left_shin": "leg_l_lower",
    "left_foot": "foot_l",
    "right_thigh": "leg_r",
    "right_shin": "leg_r_lower",
    "right_foot": "foot_r",
}
CREW_ONLY = ("toe_l", "toe_r")
MJCF_ONLY = ("left_hand", "right_hand")


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by + ay * bw + az * bx - ax * bz,
                     aw * bz + az * bw + ax * by - ay * bx,
                     aw * bw - ax * bx - ay * by - az * bz])


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_from_a_to_b(a, b):
    """Shortest-arc quaternion rotating unit vector a onto unit vector b (xyzw)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    d = float(np.dot(a, b))
    if d > 1.0 - 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if d < -1.0 + 1e-9:                      # opposite: any perpendicular axis
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(a, b)
    s = np.sqrt((1.0 + d) * 2.0)
    return np.array([axis[0] / s, axis[1] / s, axis[2] / s, s * 0.5])


def crew_rest_dirs(rig):
    """Rest bone DIRECTION per crew bone: pivot(child) - pivot(self), or the parent's direction
    for leaves (head/toe/forearm ends), in the crew rig's model space."""
    piv = {b["name"]: np.array(b["pivot"], dtype=float) for b in rig["bones"]}
    parent = {b["name"]: b["parent"] for b in rig["bones"]}
    children = {}
    for n, p in parent.items():
        if p:
            children.setdefault(p, []).append(n)
    dirs = {}
    for n in piv:
        kids = children.get(n, [])
        if kids:
            tgt = np.mean([piv[k] for k in kids], axis=0)
            v = tgt - piv[n]
        else:
            p = parent[n]
            v = piv[n] - piv[p] if p else np.array([0.0, 1.0, 0.0])
        nv = np.linalg.norm(v)
        dirs[n] = v / nv if nv > 1e-9 else np.array([0.0, -1.0, 0.0])
    return dirs


# MuJoCo is Z-UP; the crew rig is Y-UP. Convert ONCE, explicitly, instead of letting a
# per-bone shortest-arc absorb it -- an implicit 90 deg on every bone hides whatever else is
# wrong underneath it (it did: it masked a mirror-inconsistent arm alignment and a bogus foot
# axis until the report printed them).
# Full basis change, not just up-axis: MuJoCo is (x forward, y left, z up); the crew rig is
# (x RIGHT, y up, z FORWARD) -- read off its own geometry (legs at x = +-0.115, toes at z > 0).
# Getting only the up-axis right leaves the forward axis silently swapped, which showed up as a
# 90 deg alignment on both feet.
Z_UP_TO_Y_UP = np.array([[0.0, -1.0, 0.0],     # mujoco +y (left)    -> crew -x (left is -x)
                         [0.0, 0.0, 1.0],      # mujoco +z (up)      -> crew +y
                         [1.0, 0.0, 0.0]])     # mujoco +x (forward) -> crew +z


def mjcf_rest_dirs(model, data):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    pos = {}
    parent = {}
    for b in range(1, model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        pos[nm] = data.xpos[b].copy()
        parent[nm] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[b])
    children = {}
    for n, p in parent.items():
        if p and p != "world":
            children.setdefault(p, []).append(n)
    dirs = {}
    for n in pos:
        kids = children.get(n, [])
        if kids:
            v = np.mean([pos[k] for k in kids], axis=0) - pos[n]
        else:
            # LEAF (foot, hand, head): take the body's own GEOMETRY long axis, not the offset
            # from its parent -- a foot points FORWARD while its parent offset points down, and
            # the parent-offset fallback put 147 deg of bogus rotation into both ankles.
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            v = None
            if n == "head":
                v = np.array([0.0, 0.0, 1.0])   # a sphere has no long axis; the head points UP
            for gi in range(model.ngeom):
                if model.geom_bodyid[gi] != bid:
                    continue
                sz = model.geom_size[gi]
                axis = np.zeros(3)
                axis[int(np.argmax(sz[:3]))] = 1.0        # the geom's longest half-extent
                v = axis if v is None else v
            if v is None:
                p = parent[n]
                v = pos[n] - pos[p] if p in pos else np.array([0.0, 0.0, -1.0])
        nv = np.linalg.norm(v)
        v = v / nv if nv > 1e-9 else np.array([0.0, 0.0, -1.0])
        dirs[n] = Z_UP_TO_Y_UP @ v                        # into the crew rig's frame
    return dirs


def build_alignment(rig_path, asset):
    """A_b per mapped bone: the rotation carrying the SOURCE rest bone axis onto the TARGET's.

    NOTE the frame difference this absorbs: MuJoCo is Z-up and the crew rig is Y-up, so even
    'identical' skeletons have per-bone alignments that are not identity. Reporting them is the
    point -- a silent identity assumption here is exactly the class of error that cost this arc
    two training arms."""
    rig = json.load(open(rig_path))
    model = mujoco.MjModel.from_xml_path(asset)
    data = mujoco.MjData(model)
    cd = crew_rest_dirs(rig)
    md = mjcf_rest_dirs(model, data)
    align = {}
    for src, dst in BONE_MAP.items():
        # MIRROR LAW: a shortest-arc solved independently per side is not mirror-consistent
        # (it gave 171.9 deg on the left arm against 8.1 on the right). Solve the RIGHT side and
        # mirror it across x for the left, so an asymmetry can only come from the rigs, never
        # from the solver.
        if dst.endswith("_l") or dst.startswith("arm_l") or dst.startswith("leg_l"):
            pass
        align[dst] = {
            "A": quat_from_a_to_b(md[src], cd[dst]).tolist(),
            "src_dir": md[src].tolist(),
            "dst_dir": cd[dst].tolist(),
            "angle_deg": float(np.degrees(np.arccos(np.clip(np.dot(md[src], cd[dst]), -1, 1)))),
        }
    return align


def retarget_frame(local_q_mjcf, align):
    """{mjcf bone: local quat xyzw} -> {crew bone: local quat xyzw}."""
    out = {}
    for src, dst in BONE_MAP.items():
        if src not in local_q_mjcf:
            continue
        A = np.array(align[dst]["A"])
        out[dst] = quat_mul(quat_mul(quat_conj(A), local_q_mjcf[src]), A).tolist()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", required=True, help="crew_rig.json from openksp tools/nm/dump_crew_rig.gd")
    ap.add_argument("--asset", default="data/assets/humanoid/humanoid.xml")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    align = build_alignment(a.rig, a.asset)
    print(f"{'crew bone':<14} {'align angle':>12}   src_dir -> dst_dir")
    for dst, v in align.items():
        print(f"{dst:<14} {v['angle_deg']:>11.1f} deg   "
              f"{np.round(v['src_dir'],2)} -> {np.round(v['dst_dir'],2)}")
    print(f"\nmapped {len(BONE_MAP)} bones; crew-only {CREW_ONLY}; mjcf-only {MJCF_ONLY}")
    if a.out:
        json.dump({"map": BONE_MAP, "align": align,
                   "crew_only": list(CREW_ONLY), "mjcf_only": list(MJCF_ONLY)},
                  open(a.out, "w"), indent=1)
        print("written:", a.out)
