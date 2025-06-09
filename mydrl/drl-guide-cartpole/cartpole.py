import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import numpy as np
np.bool8 = np.bool_


# 创建 CartPole 环境
env = gym.make('CartPole-v1')
# 向量化环境
env = DummyVecEnv([lambda: env])

# 定义 PPO 模型
model = PPO('MlpPolicy', env, verbose=1)

# 训练模型
num_timesteps = 10000
model.learn(total_timesteps=num_timesteps)

# 保存模型
model.save("ppo_cartpole")

# 加载模型
loaded_model = PPO.load("ppo_cartpole")

# 重置环境
obs = env.reset()
total_rewards = []
total_reward = 0

for _ in range(200):  # 进行 200 步的测试
    action, _states = loaded_model.predict(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    total_rewards.append(total_reward)
    if done:
        obs = env.reset()
        total_reward = 0

# 绘制奖励曲线
plt.plot(total_rewards)
plt.title('Total Rewards over Time')
plt.xlabel('Steps')
plt.ylabel('Total Reward')
plt.show()