from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environment import BlackJackEnv


def train():
    env = BlackJackEnv()
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_blackjack")


if __name__ == "__main__":
    train()