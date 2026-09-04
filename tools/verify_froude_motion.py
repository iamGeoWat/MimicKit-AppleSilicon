"""Certify that MotionLib's Froude time-scale obeys the similarity law it claims.

A reference motion library is captured at ONE gravity. Transplanting it to gravity g' under
Froude similarity means replaying the SAME joint trajectory over a time stretched by
sqrt(g/g'). The library implements that as a divide on fps; this tool proves the three
consequences that makes it a Froude scaling rather than just "slower playback":

    lengths    scale by  s          (times stretch)
    velocities scale by  1/s        (v ~ sqrt(g) -- the Froude velocity law)
    positions  UNCHANGED            (same body, same limb lengths)

and it checks the third explicitly, because a scaling that quietly moved the root positions
would still pass the first two while no longer describing the same motion.

    python tools/verify_froude_motion.py [--g_from 9.81 --g_to 1.62]
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))
import anim.motion_lib as motion_lib
import anim.mjcf_char_model as mjcf_char_model

ROOT = os.path.join(os.path.dirname(__file__), "..")
TOL = 1e-4


def load(motion_file, char_file, scale):
    km = mjcf_char_model.MJCFCharModel("cpu")
    km.load(os.path.join(ROOT, char_file))
    return motion_lib.MotionLib(motion_file=os.path.join(ROOT, motion_file),
                                kin_char_model=km, device="cpu", time_scale=scale)


def mean_root_speed(lib):
    v = lib._frame_root_vel[:, :2]
    return float(torch.linalg.norm(v, dim=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_file", default="data/datasets/dataset_humanoid_locomotion.yaml")
    ap.add_argument("--char_file", default="data/assets/humanoid/humanoid.xml")
    ap.add_argument("--g_from", type=float, default=9.81)
    ap.add_argument("--g_to", type=float, default=1.62)
    a = ap.parse_args()

    s = math.sqrt(a.g_from / a.g_to)
    print(f"Froude time stretch for {a.g_from} -> {a.g_to} m/s^2 : s = {s:.4f}\n")

    base = load(a.motion_file, a.char_file, 1.0)
    scaled = load(a.motion_file, a.char_file, s)

    checks = []

    # 1. times stretch by exactly s
    l0, l1 = base.get_total_length(), scaled.get_total_length()
    checks.append(("motion length x s", l1 / l0, s))

    # 2. velocities scale by 1/s -- this is the Froude velocity law v ~ sqrt(g)
    v0, v1 = mean_root_speed(base), mean_root_speed(scaled)
    checks.append(("root speed x 1/s", v1 / v0, 1.0 / s))
    checks.append(("root speed x sqrt(g'/g)", v1 / v0, math.sqrt(a.g_to / a.g_from)))

    # 3. angular velocities scale by 1/s too (a rotation per unit of stretched time)
    w0 = float(torch.linalg.norm(base._frame_root_ang_vel, dim=-1).mean())
    w1 = float(torch.linalg.norm(scaled._frame_root_ang_vel, dim=-1).mean())
    checks.append(("root ang vel x 1/s", w1 / w0, 1.0 / s))

    ok = True
    for name, got, want in checks:
        good = abs(got - want) < TOL * max(1.0, abs(want))
        ok &= good
        print(f"  [{'OK ' if good else 'BAD'}] {name:<26} got {got:.6f}  want {want:.6f}")

    # 4. POSITIONS UNCHANGED -- the check that separates a Froude scaling from a rescaled body
    dp = float((base._frame_root_pos - scaled._frame_root_pos).abs().max())
    dq = float((base._frame_joint_rot - scaled._frame_joint_rot).abs().max())
    good = dp == 0.0 and dq == 0.0
    ok &= good
    print(f"  [{'OK ' if good else 'BAD'}] positions/rotations untouched   max dpos {dp:.2e}  "
          f"max drot {dq:.2e}")

    # 5. PLANTED BAD: a scale that is NOT the Froude one must fail the velocity law, or the
    #    checks above are vacuous (they would pass for any monotone playback change).
    wrong = load(a.motion_file, a.char_file, s * 1.5)
    vw = mean_root_speed(wrong) / v0
    caught = abs(vw - math.sqrt(a.g_to / a.g_from)) > TOL
    ok &= caught
    print(f"  [{'OK ' if caught else 'BAD'}] planted-bad s*1.5 is REJECTED  got {vw:.6f}  "
          f"(Froude wants {math.sqrt(a.g_to / a.g_from):.6f})")

    print(f"\n{'FROUDE MOTION SCALING CERTIFIED' if ok else 'FAILED'}")
    print(f"reference at {a.g_from}: mean root speed {v0:.3f} m/s")
    print(f"reference at {a.g_to}:  mean root speed {v1:.3f} m/s  "
          f"(this is what the discriminator now asks for)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
