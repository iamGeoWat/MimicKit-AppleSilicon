# Changes in this fork

Apache-2.0 §4(b) requires a fork to state what it changed. This file is that statement, and
doubles as the map for syncing with upstream: everything here is ADDITIVE except the three
marked **[modifies upstream]**, which is what keeps merges cheap.

Upstream: https://github.com/xbpeng/MimicKit (remote `upstream`).

## Added — the Apple Silicon / no-CUDA path

- `mimickit/engines/mujoco_cpu_engine.py` — a native C MuJoCo engine (the `mujoco` Python
  bindings, one `MjData` per env, physics stepped through `mujoco.rollout`'s C-side thread
  pool). Engine name `mujoco_cpu`. **Measured 365x the warp-CPU fallback** on the same model
  and machine (368 → 134,376 raw control-samples/s; ~18.6k through the full engine interface
  at 64 envs on an M-series laptop). Semantics mirror the newton engine choice by choice:
  world-frame velocities, XYZW quaternions, `ControlMode.pos` implemented by rewriting the
  MJCF's `<motor>` actuators as `<position kp kv forcerange>` servos with the joint's passive
  stiffness/damping zeroed — the same repurposing newton does in `_build_controls`.
  Gravity is configurable per run (`gravity: [x, y, z]` in the engine yaml).
- `data/engines/mujoco_cpu_engine.yaml` — its config.
- `tools/probe_mujoco_cpu_engine.py` — five physical probes that must pass before any
  training on a changed engine: settled standing force balance equals m·g exactly, PD
  tracking, byte determinism, a kinematic angular-velocity frame proof, and a throughput
  report (informational — it is machine-load sensitive).
- `tools/probe_cross_arm.py` — newton vs mujoco_cpu on identical written state, with a
  kin-model finite-difference ORACLE that adjudicates disagreements. This is what caught the
  dof-convention mismatch below, and it also measured a **0.5 m/s deviation of the newton
  arm's spherical dof velocities from `kin_char_model`'s own convention** (upstream issue,
  not introduced here).
- `tools/probe_dof_codec.py` — calibration for the hinge↔spherical exp-map codec: composition
  order proven against MuJoCo FK to machine precision, decomposition roundtrip 1e-15, numeric
  angular-velocity Jacobian, and the contract check `kin-FK == mujoco-FK` to 3e-7 m.
- `tools/probe_env_gravity.py` — per-env gravity: a single batch holding 9.81 and 1.62 drops
  at ratio 6.06, exactly g1/g2.
- `tools/eval_gait_metrics.py` — gait evaluation for a checkpoint (speed, cadence, duty
  factor, aerial fraction, Froude, episode length), engine-agnostic, with the reference clip
  pushed through the identical code path (`--demo`) so the bands are cut from the same stock.
- `tools/film_checkpoint.py` — renders a checkpoint rollout to mp4 (retina-aware).
- `data/envs/amp_humanoid_walk_env.yaml`, `data/envs/deepmimic_humanoid_walk_env.yaml` — walk
  variants of the shipped env configs.

## Modified upstream files

- **[modifies upstream]** `mimickit/engines/engine_builder.py` — one `elif` for `mujoco_cpu`.
- **[modifies upstream]** `mimickit/engines/newton_recorder.py` — **bug fix**:
  `_set_camera_pose` did `get_root_pos(...)[env].cpu().numpy()` and then wrote the camera's
  target height into element 2. `get_root_pos` returns a torch VIEW into the sim's `joint_q`,
  and on a CPU device `.cpu()` is a no-op, so that line teleported the character every frame.
  A filmed policy fell every ~2 s while the identical unfilmed rollout never fell in 20 s.
  Invisible on CUDA, where `.cpu()` copies. Fix: `.copy()`. Localized by a three-arm bisection
  (no recorder / recorder idle / recorder capturing → 0, 0, 6 falls per 300 frames), a
  per-field state diff (`joint_q` moved 0.21 rad while `body_q`, `body_qd`, `joint_qd`,
  `body_f` stayed byte-identical) and a five-call bisection inside `_record_frame`.
- **[modifies upstream]** `mimickit/engines/newton_engine.py` — adds `set_env_gravity` /
  `get_env_gravity` (newton already carries gravity per world; this exposes it through the
  engine interface so one batch can span gravities).

## Naming

Upstream PR #110 adds an engine called `mujoco` backed by **mujoco_warp** (GPU/CUDA). This
fork's engine is deliberately named `mujoco_cpu`: the two are complementary, not competing —
#110 serves machines with a CUDA GPU, this one serves machines without.
