try:
    import isaacgym.gymapi as gymapi
except ImportError:
    pass

def build_engine(config, num_envs, device, visualize, record_video=False):
    eng_name = config["engine_name"]

    if (eng_name == "isaac_gym"):
        import engines.isaac_gym_engine as isaac_gym_engine
        engine = isaac_gym_engine.IsaacGymEngine(config, num_envs, device, visualize, record_video=record_video)
    elif (eng_name == "isaac_lab"):
        import engines.isaac_lab_engine as isaac_lab_engine
        engine = isaac_lab_engine.IsaacLabEngine(config, num_envs, device, visualize, record_video=record_video)
    elif (eng_name == "newton"):
        import engines.newton_engine as newton_engine
        engine = newton_engine.NewtonEngine(config, num_envs, device, visualize, record_video=record_video)
    elif (eng_name == "mujoco_cpu"):
        # native C MuJoCo (this fork) -- distinct from PR #110's mujoco_warp GPU backend
        import engines.mujoco_cpu_engine as mujoco_cpu_engine
        engine = mujoco_cpu_engine.MujocoCPUEngine(config, num_envs, device, visualize, record_video=record_video)
    else:
        assert False, print("Unsupported engine: {:s}".format(eng_name))

    return engine