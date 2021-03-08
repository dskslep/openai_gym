def train(bot, env, num_episodes, rand_start, render, reward_f=lambda r, d: r):
    for i_episode in range(num_episodes):
        observation = env.reset()
        action = bot.start(observation, rand_start)
        observation, reward, done, info = env.step(action)
        bot.update(observation, reward, done)
        t = 1
        while not done:
            if render:
                env.render()
            action = bot.policy_action(observation)
            observation, reward, done, info = env.step(action)
            t += 1
            bot.update(observation, reward_f(reward, done), done)
            bot.plan_step()
            if done:
                print(f"Episode {i_episode+1} finished after {t} steps")
        bot.plan_episode()

    return bot
