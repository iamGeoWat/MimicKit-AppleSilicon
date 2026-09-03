import sys, time
sys.path.insert(0, "mimickit")
import torch
import util.arg_parser as arg_parser, util.mp_util as mp_util
import envs.env_builder as env_builder, learning.agent_builder as agent_builder
dev, N = sys.argv[1], int(sys.argv[2])
args = arg_parser.ArgParser()
args.load_args(["--env_config","data/envs/amp_humanoid_walk_env.yaml",
                "--agent_config","data/agents/amp_humanoid_agent.yaml"])
mp_util.init(0,1,"cpu","6800")
env = env_builder.build_env(args.parse_string("env_config"), "data/engines/mujoco_cpu_engine.yaml", N, "cpu", visualize=False)
ag = agent_builder.build_agent(args.parse_string("agent_config"), env, dev)
T = {"roll": 0.0, "data": 0.0, "upd": 0.0, "ITER": 0.0, "norm": 0.0}
import learning.base_agent as ba
def patch(cls, name, key):
    orig = getattr(cls, name)
    def w(self, *a, **k):
        t0 = time.perf_counter(); r = orig(self, *a, **k)
        if dev == "mps": torch.mps.synchronize()
        T[key] += time.perf_counter() - t0; return r
    setattr(cls, name, w)
patch(ba.BaseAgent, "_rollout_train", "roll")
patch(type(ag), "_build_train_data", "data")
patch(type(ag), "_update_model", "upd")
patch(ba.BaseAgent, "_train_iter", "ITER")
patch(ba.BaseAgent, "_update_normalizers", "norm") if hasattr(ba.BaseAgent, "_update_normalizers") else None
ag._init_train()
ag._curr_obs, ag._curr_info = ag._reset_envs()
for _ in range(3):
    ag._train_iter()
it = T["ITER"]/3
parts = (T['roll']+T['data']+T['upd']+T['norm'])/3
print(f"device={dev:<4} envs={N:<5} iter {it:6.2f}s = roll {T['roll']/3:5.2f} + data {T['data']/3:5.2f} "
      f"+ upd {T['upd']/3:5.2f} + norm {T['norm']/3:5.2f} + OTHER {it-parts:5.2f}  "
      f"| samples/s {N*ag._steps_per_iter/it:7.0f}")
