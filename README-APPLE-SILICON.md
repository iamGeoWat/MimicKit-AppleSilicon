# MimicKit on Apple Silicon

Upstream MimicKit's three engines — Isaac Gym, Isaac Lab, Newton — all reach the GPU through
NVIDIA's Warp, so on a Mac the only thing that runs is Warp's CPU fallback: GPU-shaped kernels
executed one thread at a time. This fork adds a **native C MuJoCo engine** and the tooling
around it, so motion-imitation research actually runs on an Apple laptop.

Everything below is measured on this machine, not estimated. Numbers were taken on an
M-series laptop (15 cores) with the humanoid from `data/assets/humanoid/humanoid.xml`
(28 dof), 240 Hz sim / 30 Hz control.

## What you get

| path | control-samples/s | verdict |
|---|---:|---|
| upstream newton engine, Warp CPU fallback, 64 envs | 368 | unusable |
| **this fork's `mujoco_cpu`, 64 envs, through the full engine interface** | **~18,600** | usable |
| the same policy trained here, all-in with learning | ~3,600 | a walk in ~4 h |
| (for scale) newton on an RTX 3080 Ti, 4096 envs, all-in | ~14,000 | a walk in ~1 h |

So: an Apple laptop is roughly **4x slower than a mid-range NVIDIA desktop** for this
workload, rather than 40x slower — the difference between "prototype on the machine you have"
and "don't bother".

## Install

```bash
python3.12 -m venv ~/.venvs/mimickit      # 3.12: no mujoco wheel for 3.14 yet
source ~/.venvs/mimickit/bin/activate
pip install -r requirements.txt "mujoco==3.5.0"
# only if you also want the newton arm for viewing/recording:
pip install "newton==1.0.0" "mujoco-warp==3.5.0.2" "warp-lang==1.14.0"
```

**`warp-lang==1.14.0` is a hard pin if you install the newton arm.** Warp 1.17 fails to
compile mujoco_warp's solver (`solver.py`, `Referencing undefined symbol: J_kj`) — a scoping
bug in mujoco_warp that older Warp's laxer codegen tolerated. Declared ranges say `>=1.11`;
they lie.

## Run

Physics on the CPU, **networks on the Apple GPU** (`--agent_device mps`) — the hybrid split
this fork adds. Measured end-to-end, steady state (`tools/phase_timing.py`):

| envs | agent device | s/iteration | of which UPDATE | all-in samples/s |
|---:|---|---:|---:|---:|
| 256 | cpu | 4.88 | 2.81 | 1,678 |
| 256 | **mps** | **3.06** | **0.53** | **2,681** (1.60x) |
| 512 | cpu | 9.04 | 5.37 | 1,813 |
| 512 | **mps** | **4.65** | **0.89** | **3,520** (1.94x) |

The update phase alone is 5-6x faster on the GPU; the rollout is slightly slower (per-step
CPU-GPU crossings), and the win grows with batch size. (Numbers taken while another job had
the CPU busy, so treat the absolute rates as a floor; the ratio is back-to-back.)

```bash
python mimickit/run.py --mode train --num_envs 512 --devices cpu --agent_device mps \
  --engine_config data/engines/mujoco_cpu_engine.yaml \
  --env_config data/envs/amp_humanoid_walk_env.yaml \
  --agent_config data/agents/amp_humanoid_agent.yaml \
  --visualize false --logger txt --out_dir output/walk

# or CPU-only:
python mimickit/run.py --mode train --num_envs 64 --devices cpu \
  --engine_config data/engines/mujoco_cpu_engine.yaml \
  --env_config data/envs/amp_humanoid_walk_env.yaml \
  --agent_config data/agents/amp_humanoid_agent.yaml \
  --visualize false --logger txt --out_dir output/walk

python tools/probe_mujoco_cpu_engine.py     # five physics probes, run these after any change
python tools/eval_gait_metrics.py --demo    # the reference clip, through the judging code
python tools/eval_gait_metrics.py --model output/walk/model.pt --steps 900 --envs 8
```

## Does it give the same answers as the CUDA path?

Two checks say yes, at two different levels:

- **State semantics.** `tools/probe_cross_arm.py` writes identical state into both engines and
  compares: forward kinematics agree to 2.7e-7 m, and a finite-difference oracle built from
  `kin_char_model` matches this engine's body velocities to 9e-4 m/s.
- **Behaviour.** A policy trained entirely in `mujoco_cpu` was evaluated zero-shot in the
  newton engine: 1.296 m/s vs 1.282 at home, cadence 2.20 vs 2.12, aerial fraction 0.002 vs
  0.000 — 5/5 gait bands in both. The policy is not exploiting one solver's artifacts.

A side effect worth stating: native MuJoCo is the *reference* implementation, so this engine
also works as a correctness oracle. Used that way it found a 0.5 m/s inconsistency between
the newton arm's spherical dof velocities and `kin_char_model`'s own convention.

## Why there is no Metal backend (measured, 2026-09-03)

The obvious question is why not use the Apple GPU. Three routes, all closed today:

1. **A Metal target inside Warp** — a full MSL codegen and runtime inside NVIDIA's framework,
   which NVIDIA would not carry; a permanent compiler fork.
2. **A CUDA-compatibility layer under Warp** (e.g. CuMetal) — Warp JIT-compiles through
   NVRTC, which those layers do not provide, and mujoco_warp leans on exactly the tile and
   cooperative primitives they bound most tightly.
3. **MJX on `jax-metal`** — the live-looking one, so it was tested rather than assumed:
   `jax.devices()` reports `METAL(id=0)`, a 512×512 matmul returns the exact answer,
   `mjx.put_model` succeeds on the humanoid, and then `mjx.step` dies at
   `mjx/_src/smooth.py:304` with **`failed to legalize operation 'mhlo.cholesky'`** —
   `jax-metal` does not implement the Cholesky factorization MJX needs for the mass matrix on
   every step. Apple's plugin is `0.1.1`, **last uploaded 2024-10-08**, and newer JAX emits
   StableHLO it cannot parse at all (`unknown attribute code: 22`).

That wall was then attacked properly rather than accepted: two of the cholesky walls come
down with configuration (`jacobian=sparse` routes to MJX's own hand-written LDL,
`solver=CG` avoids the Newton Hessian), a third with a one-line MJX patch (guard the tendon
spring-damper when `ntendon == 0`; the zero-width slice breaks Metal's shape inference). The
fourth is an assertion **inside Apple's closed MPSGraph binary** — it reproduces on a
two-link pendulum, padding empty arrays turns it into a segfault, and closing the model over
as constants only moves the parameter index. `jax-metal` is unmaintained since 2024-10 and
the JAX tracker carries years of the same class of failure, closed without fixes.
**Full evidence, wall by wall, with the reproduction: [docs/METAL-STATUS.md](docs/METAL-STATUS.md).**

Until an actively-maintained bridge exists, the CPU engine is not a fallback; it is the only
thing that runs.

## Roadmap (honest)

- **Now:** CPU engine, probes, evaluation, per-env gravity.
- **Watching:** `jax-metal` gaining `cholesky` (would unlock MJX on the Apple GPU);
  mujoco_warp or MJX growing a non-CUDA GPU target.
- **Would be nice:** MPS for the learning half — measured 4.45x over CPU for the update step
  at batch 2048, roughly break-even for small rollout inference, so it is worth doing once
  physics stops being the bottleneck.

Bug reports and numbers from other Apple machines are welcome; the point of the fork is to be
the place where the Apple-side facts live.
