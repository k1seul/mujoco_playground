import os
os.environ["MUJOCO_GL"] = "egl"
import mujoco_playground


import functools
from datetime import datetime
from IPython.display import clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import imageio
import hydra
from omegaconf import DictConfig, OmegaConf

# wandb 로깅 추가
import wandb

from mujoco_playground import registry, wrapper
from mujoco_playground import dm_control_suite
from mujoco_playground.config import dm_control_suite_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo

@hydra.main(config_path="conf", config_name="config", version_base=None)

def main(cfg: DictConfig):
    print("Hydra config:\n", OmegaConf.to_yaml(cfg))
    # headless 옵션에 따라 MUJOCO_GL 설정
    if cfg.get("headless", True):
        os.environ["MUJOCO_GL"] = "egl"
    else:
        if "MUJOCO_GL" in os.environ:
            del os.environ["MUJOCO_GL"]

    # wandb 초기화
    wandb.init(
        project="mujoco_playground",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"{cfg.env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    env_cfg = registry.get_default_config(cfg.env_name)

    # 환경 및 설정 불러오기
    if cfg.get("vision", False):
        num_envs = cfg.ppo.num_envs
        ctrl_dt = cfg.vision_config.ctrl_dt
        episode_length = int(3 / ctrl_dt)

        config_overrides = {
            "vision": True,
            "vision_config.render_batch_size": num_envs,
            "action_repeat": 1,
            "ctrl_dt": ctrl_dt,
            "episode_length": episode_length,
        }

        env = dm_control_suite.load(cfg.env_name, config_overrides=config_overrides)
        env = wrapper.wrap_for_brax_training(
            env,
            vision=True,
            num_vision_envs=num_envs,
            action_repeat=1,
            episode_length=episode_length,
        )
        env_cfg["episode_length"] = episode_length
    else:
        env = registry.load(cfg.env_name)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # PPO 파라미터
    if cfg.get("vision", False):
        # Load vision-specific PPO configuration tuned for CartpoleBalance
        ppo_params = dm_control_suite_params.brax_vision_ppo_config(cfg.env_name)
        ppo_params.episode_length = episode_length
        ppo_params.network_factory = ppo_networks_vision.make_ppo_networks_vision
        ppo_params['batch_size'] = 128
        ppo_params['num_envs'] = ppo_params['num_eval_envs'] = num_envs
    else:
        ppo_params = dm_control_suite_params.brax_ppo_config(cfg.env_name)
    # config.yaml의 ppo 파라미터 덮어쓰기
    if "ppo" in cfg:
        for k, v in cfg.ppo.items():
            ppo_params[k] = v
    print("PPO params:", ppo_params)

    # 학습 진행 상황 시각화용 변수
    x_data, y_data, y_dataerr = [], [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        clear_output(wait=True)
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        y_dataerr.append(metrics["eval/episode_reward_std"])
        plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
        plt.ylim([0, 1100])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={y_data[-1]:.3f}")
        plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
        plt.savefig("test.png")
        # wandb metric 로깅
        wandb.log({
            "num_steps": num_steps,
            **{k: float(v) for k, v in metrics.items() if hasattr(v, '__float__') or isinstance(v, (int, float))}
        })

    # 체크포인트 저장 경로 지정 (절대경로로 변경)
    checkpoint_path = os.path.abspath(cfg.checkpoint_path)

    # 네트워크 및 학습 함수 설정
    ppo_training_params = dict(ppo_params)
    if cfg.get("vision", False):
        train_fn = functools.partial(
            ppo.train, **dict(ppo_params), progress_fn=progress
        )
    else:
        network_factory = ppo_networks.make_ppo_networks
        if "network_factory" in ppo_params:
            del ppo_training_params["network_factory"]
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks,
                **ppo_params.network_factory
            )

        train_fn = functools.partial(
            ppo.train, **ppo_training_params,
            network_factory=network_factory,
            progress_fn=progress,
            save_checkpoint_path=checkpoint_path,
        )

    # 학습 실행
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    # 추론 함수(jit)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(42)
    rollout = []
    n_episodes = 1
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    for _ in range(n_episodes):
        state = jit_reset(rng)
        rollout.append(state)
        for i in range(env_cfg.episode_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state)

    render_every = 1
    frames = env.render(rollout[::render_every])
    rewards = [s.reward for s in rollout]

    # media.show_video 대신 동영상 파일로 저장
    output_video_path = "ppo_rollout.mp4"
    imageio.mimsave(output_video_path, frames, fps=int(1.0 / env.dt / render_every))

    print(f"동영상이 {output_video_path}로 저장되었습니다.")
    # wandb에 동영상 업로드
    wandb.log({"rollout_video": wandb.Video(output_video_path, fps=int(1.0 / env.dt / render_every), format="mp4")})
    wandb.finish()

if __name__ == "__main__":
    main()