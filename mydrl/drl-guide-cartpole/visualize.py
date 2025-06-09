import gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pygame
from transformers.models.pix2struct.image_processing_pix2struct import render_text

np.bool8 = np.bool_

# 创建 CartPole 环境
env = gym.make('CartPole-v1')

# 定义和训练 PPO 模型
model = PPO('MlpPolicy', env, verbose=1,)
model = PPO.load("ppo_cartpole")
# 初始化用于存储每一帧图像的列表
frames = []

# 重置环境
obs, _ = env.reset()
for _ in range(200):  # 模拟 200 步
    action, _states = model.predict(obs)
    # 解包 env.step(action) 的返回值，兼容 gym 0.26 及更高版本
    obs, rewards, done, truncated, info = env.step(action)
    # 通常将 done 和 truncated 结合判断回合是否结束
    done = done or truncated
    # 渲染环境并保存当前帧，指定 mode 为 'rgb_array'
    frames.append(env.render())
    if done:
        obs, _ = env.reset()

# 关闭环境
env.close()
