# Running physics on the Apple GPU: measured status (2026-09-03)

The obvious question about this fork is why the engine is CPU-based when the machine has a
capable GPU. This page is the evidence, so nobody has to repeat the day. Everything here was
run on an M5 Pro, macOS 26, against MimicKit's own humanoid (`nq=35, nv=34, nu=28`) and a
two-link pendulum.

## Summary

| route | status | where it stops |
|---|---|---|
| Metal target inside NVIDIA Warp | not attempted | a full MSL codegen + runtime in a vendor framework that would not carry it |
| CUDA-compat layer (e.g. CuMetal) under Warp | closed | Warp JIT-compiles via NVRTC, which those layers do not provide |
| **MJX on `jax-metal`** | **closed, four walls deep** | an assertion inside Apple's closed MPSGraph binary |
| PyTorch MPS for the LEARNING half | **open and useful** | nothing — measured 4.45x over CPU at batch 2048 |

## The MJX attempt, wall by wall

Three of the four walls came down. The fourth is not ours to move.

1. **`mhlo.cholesky` (mass matrix).** `smooth.factor_m` factorizes M densely via
   `jax.scipy.linalg.cho_factor`; `jax-metal` cannot legalize it. **Crossed by configuration:**
   MJX's `is_sparse` picks dense below `nv=60`, so forcing
   `m.opt.jacobian = mjJAC_SPARSE` routes to MJX's own hand-written LDL, which is plain JAX ops.
2. **`mhlo.cholesky` again (constraint solver).** `solver.py:385` factorizes the Newton
   solver's Hessian. **Crossed by configuration:** `m.opt.solver = mjSOL_CG` — conjugate
   gradient needs no factorization.
3. **`mps.strided_slice` shape inference on an EMPTY tensor.** `passive._spring_damper`
   slices tendon arrays; with `ntendon == 0` the slice is zero-width and the Metal backend
   infers `1x1` where the type is `1x0`. **Crossed by patching MJX** (guard the tendon block
   with `if m.ntendon:` — strictly less work on every backend, since the term is identically
   zero):
   ```python
   # mjx/_src/passive.py, in _spring_damper
   if m.ntendon:
       below, above = m.tendon_lengthspring.T - d.ten_length
       ...
   ```
4. **`MPSGraphExecutable.mm:3467: failed assertion 'Incompatible shape for parameter at
   index N'`.** This is inside Apple's binary plugin, and it is where the road ends:
   - it reproduces on a **two-link pendulum**, so it is not about model size or our humanoid;
   - padding every zero-sized leaf of the model and data to size 1 turns the assertion into a
     **SIGSEGV** (exit 139);
   - closing the model over as compile-time constants (so the executable takes only the data)
     merely moves the index (7 instead of 36) — the marshalling itself is what fails.

## Why this will not be fixed by trying harder

`jax-metal` is a closed-source PJRT plugin. Its last release is **0.1.1, uploaded
2024-10-08**, and it requires `jax>=0.4.34`; anything newer emits StableHLO it cannot parse
(`unknown attribute code: 22`). The JAX tracker shows the same class of failure reported for
years and closed without a fix: `mhlo.cholesky` (#16321), `mhlo.triangular_solve` (#17490),
`mhlo.custom_call` (#16287), `dot_general` for some einsums (#20114), multi-operand `reduce`
(#21384), `mhlo.return` / StableHLO drift (#32800).

So the honest statement is not "MJX is slow on Metal" but **"the only XLA-to-Metal bridge is
an unmaintained binary that cannot compile MJX's graph, and no amount of patching MJX changes
that"**.

## What IS worth doing on the Apple GPU

- **The learning half, via PyTorch MPS.** Measured on the AMP-shaped network: rollout
  inference at batch 64 is a wash (154 µs CPU vs 130 µs MPS), but the update step at batch
  2048 with backward is **4.45x** faster (8.07 ms → 1.81 ms). Physics is the bottleneck
  today, so this is queued rather than shipped, but it needs no new dependency.
- **Watch for an actively-maintained bridge.** The live candidates are Apple's own **MLX**
  (open source, actively developed, has a kernel API) and **Taichi** (mature open-source
  Metal backend). Either could host a physics step; neither gives MuJoCo semantics for free,
  so that is a project, not a patch.
- **If `jax-metal` ever ships the missing ops**, walls 1 and 2 stop needing workarounds and
  wall 4 becomes the only question — the config flags and the `passive.py` guard above are
  already the recipe to retry with.

## Reproducing

```bash
python3.11 -m venv ~/.venvs/mjx-metal && source ~/.venvs/mjx-metal/bin/activate
pip install "jax==0.4.34" "jaxlib==0.4.34" jax-metal==0.1.1 "mujoco-mjx==3.2.4" "mujoco==3.2.4"
python - <<'PY'
import jax, mujoco
from mujoco import mjx
print(jax.devices())                       # [METAL(id=0)]
m = mujoco.MjModel.from_xml_string(open('pendulum.xml').read())
m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE   # wall 1
m.opt.solver   = mujoco.mjtSolver.mjSOL_CG         # wall 2
mx = mjx.put_model(m); dx = mjx.make_data(mx)
jax.jit(mjx.step)(mx, dx)                  # wall 4: MPSGraph assertion
PY
```

## Ecosystem outlook (checked 2026-09-03) — is anything coming that would unlock this?

Short answer: **no, and the current is running the other way.**

| project | signal | direction |
|---|---|---|
| `jax-metal` (the only XLA→Metal bridge) | last release **2024-10-08**; JAX merged *"Remove mentions of jax-metal"* (jax#34485, Jan 2026); a deadlock at `block_until_ready` is still **open** (jax#37374, May 2026) | being retired |
| `mujoco_warp` (MuJoCo's GPU future) | actively developed (pushed today), description reads *"designed for NVIDIA hardware"* | consolidating on CUDA |
| NVIDIA Warp | no Metal work; the Apple-side issues are CPU precision (#1035) and dylib symbol leakage (#1758) | not coming |
| Apple **MLX** | 28k stars, releases every 2-4 weeks, **has `cholesky` and a real linalg surface** | alive — but zero physics ecosystem: every MLX project is LLM/inference |
| PyTorch **MPS** | maintained, and the lever we already measured (4.45x on the update step) | usable today |

So the unlock will not arrive as a vendor update. The two paths that could produce one are
both projects, not patches: a physics step written against **MLX** (the linear algebra is
there; the rigid-body dynamics are not), or an actively-maintained XLA→Metal bridge that
nobody is currently building.

### The adjacent opportunity worth more than the Metal chase

MuJoCo issue **#2813, "An MJX-style JAX FFI for CPU-based MuJoCo"** (open since 2025-08, a
maintainer said he would look at it, stalled since 2026-02) asks for exactly the premise this
fork demonstrates: that **batched, threaded, native CPU MuJoCo is fast enough to be a
first-class path**, and pleasanter than a GPU pipeline for iteration and debugging. The
requester's motivation — MJX's poor scaling on contact-rich scenes and its slow compiles —
is the same wall from the other side.

This fork does not implement that FFI, but it does carry the numbers that argue for it:
134,376 physics steps/s threaded on a laptop, 365x the Warp CPU fallback, and a full
motion-imitation training loop at ~3,600 all-in samples/s. If the CPU-MuJoCo direction gets
picked up upstream, those measurements are the evidence it needs.

## Would a `mujoco-mlx` port pay off? Measured, 2026-09-03

MLX is the one live Apple-side framework, so the natural question is whether porting MuJoCo's
step to it (the way MJX ported it to JAX) would beat this fork's native C engine. Two gates
were run before writing any port.

**Gate 1 — the kernel shapes.** MLX GPU vs 15 CPU threads on the pieces a step is made of:

| batch | op | MLX GPU | torch CPU (15 thr) | GPU |
|---:|---|---:|---:|---:|
| 4096 | 20x elementwise on (N,34,34) | 4.95 ms | 27.8 ms | **5.6x** |
| 4096 | batched 34x34 matvec | 0.51 ms | 1.36 ms | **2.7x** |
| 256 | batched 34x34 matvec | 0.20 ms | 0.09 ms | 0.44x (GPU loses small) |
| any | `mx.linalg.cholesky` | **unsupported on GPU** | — | — |

(The missing GPU cholesky is survivable: MJX's sparse path uses a hand-written LDL in plain
array ops, the same route that crossed wall 1 above.)

**Gate 2 — a step-shaped workload at realistic shapes** (180 elementwise ops over `(N,34)`,
`(N,15,6)`, `(N,15,3)` plus a mass-matrix apply and a Jacobian apply — the mix a real step
actually has, not one big (N,34,34) array):

| envs | MLX GPU ms/step | CPU ms/step | GPU speedup | GPU steps/s | vs this fork's native C (1.07M/s) |
|---:|---:|---:|---:|---:|---:|
| 512 | 2.33 | 18.92 | 8.1x | 219,708 | 0.21x |
| 2048 | 2.04 | 35.96 | 17.7x | 1,005,995 | 0.94x |
| 4096 | 3.53 | 40.97 | 11.6x | 1,160,482 | **1.08x** |
| 8192 | 6.92 | 53.14 | 7.7x | 1,182,980 | **1.11x** |

**Verdict: parity, not a multiple.** A months-long port would buy roughly 1.1x over what the
native C engine already delivers on this machine, and the GPU curve is already flat by 4096
envs (launch- and bandwidth-bound at these per-env sizes). The proxy is also OPTIMISTIC: it
has no collision broadphase and no iterative constraint solver, both of which are
data-dependent and branchy — the parts GPUs handle worst at small scene sizes.

Two things would change this verdict, and they are worth restating because they are specific:

1. **A different Mac.** These numbers are an M5 Pro: a very strong 15-core CPU against a
   mid-sized GPU. On a base M-series (fewer CPU cores, similar GPU) or an Ultra (roughly the
   same CPU, 2-3x the GPU), the ratio moves — plausibly decisively — in the GPU's favour.
   Anyone with an Ultra running these two benchmarks would settle it.
2. **Differentiability and a single-device pipeline.** MLX physics would be autodiff-capable
   and share unified memory with the policy network. That is a *qualitative* argument for a
   port — a different project with a different justification than "make Mac training faster",
   which the native C engine already solved.

The benchmarks live in this repo's history and are reproducible in ten minutes; if you have
an Ultra, the interesting number is Gate 2's last column.
