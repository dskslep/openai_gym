import gym

from classic_control.mountain_car_v0.bots import QBot
from trainer import train


if __name__ == '__main__':
    env = gym.make('MountainCar-v0').env
    bot = QBot(env, grid_size=10, epsilon=0.1, gamma=1, step_size=0.1, start_values=0, step_planning=100,
               episode_planning=1000)
    bot = train(bot=bot, env=env, num_episodes=10, rand_start=True, render=True, reward_f=lambda r, d: r)
    # bot.save('data/w.json')
