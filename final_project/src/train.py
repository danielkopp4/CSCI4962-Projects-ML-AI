from model import PPOModel
from environment import BlackJackEnv

def get_winrate(model, trials=1000):
    env = BlackJackEnv()
    obs = env.reset()
    t = 0
    wins = 0
    losses = 0
    while t < trials:
        t += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        if reward == -1:
            losses += 1

        if reward == 1:
            wins += 1

        if done:
            obs = env.reset()
    
    return wins / (wins + losses)

hours = 0.5
fps = 730
timesteps = int(hours * 60 * 60 * fps)

def train():
    env = BlackJackEnv(True)
    model = PPOModel(env, 0.1, 0.5, 0.3, timesteps, 64, 0.2, 0,0, "ppo_blackjack_infinite")
    model.train()
    
    wr = get_winrate(model)
    print(f"win rate {100*wr:0.04f}")


if __name__ == "__main__":
    train()