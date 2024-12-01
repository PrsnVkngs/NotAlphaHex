from ourhexenv import OurHexGame
from g01agent import G01Agent
from g99agent import G99Agent
import random

env = OurHexGame(board_size=11)
env.reset()

# player 1
g01agent = G01Agent(env)
# player 2
g99agent = G99Agent(env)

smart_agent_player_id = random.choice(env.agents)

done = False
while not done:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            break

        
        if agent == 'player_1':
            action = g01agent.select_action(observation, reward, termination, truncation, info)
        else:
            action = g99agent.select_action(observation, reward, termination, truncation, info)

        env.step(action)
        env.render()
