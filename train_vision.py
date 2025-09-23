# @title Import packages for plotting and creating graphics
import os
os.environ["MUJOCO_GL"] = "egl"
import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt
# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools

import time

from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from IPython.display import clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
import numpy as np

from mujoco_playground import wrapper

np.set_printoptions(precision=3, suppress=True, linewidth=100)

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from mujoco_playground import dm_control_suite


num_envs = 16
ctrl_dt = 0.04
episode_length = int(3 / ctrl_dt)

config_overrides = {
    "vision": True,
    "vision_config.render_batch_size": num_envs,
    "action_repeat": 1,
    "ctrl_dt": ctrl_dt,
    "episode_length": episode_length,
}

env_name = "CartpoleBalance"
env = dm_control_suite.load(
    env_name, config_overrides=config_overrides
)

env = wrapper.wrap_for_brax_training(
    env,
    vision=True,
    num_vision_envs=num_envs,
    action_repeat=1,
    episode_length=episode_length,
)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

def unvmap(x):
  return jax.tree.map(lambda y: y[0], x)


state = jit_reset(jax.random.split(jax.random.PRNGKey(0), num_envs))
rollout = [unvmap(state)]

f = 0.2
for i in range(episode_length):
  action = []
  for j in range(env.action_size):
    action.append(
        jp.sin(
            unvmap(state).data.time * 2 * jp.pi * f
            + j * 2 * jp.pi / env.action_size
        )
    )
  action = jp.tile(jp.array(action), (num_envs, 1))
  state = jit_step(state, action)
  rollout.append(unvmap(state))

frames = env.render(rollout, camera="fixed", width=256, height=256)
k = next(iter(rollout[0].obs.items()), None)[0]  # ex: pixels/view_0
obs = [r.obs[k][..., 0] for r in rollout]  # visualise first channel

media.show_videos([frames, obs], fps=1.0 / env.dt, height=256)