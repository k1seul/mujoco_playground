# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
from IPython.display import clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import os
import imageio  # 추가

from mujoco_playground import registry, wrapper
from mujoco_playground.config import dm_control_suite_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

# MuJoCo 렌더러 설정
os.environ["MUJOCO_GL"] = "egl"

# 환경 및 설정 불러오기
env = registry.load('CartpoleBalance')
env_cfg = registry.get_default_config('CartpoleBalance')

# PPO 파라미터
ppo_params = dm_control_suite_params.brax_ppo_config('CartpoleBalance')
print(ppo_params)

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

# 체크포인트 저장 경로 지정 (절대경로로 변경)
checkpoint_path = os.path.abspath("./ppo_ckpt")

# 네트워크 및 학습 함수 설정
ppo_training_params = dict(ppo_params)
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
    save_checkpoint_path=checkpoint_path,  # 절대경로 사용
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