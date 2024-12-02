import copy
import math
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from ourhexenv import OurHexGame

# MCTS HYPER PARAMETERS
EXPLORE_CONSTANT = 1.41
SEARCH_TIME = 1

# CNN HYPER PARAMETERS
PADDING = 1
STRIDE = 1
KERNEL_SIZE = 3


def is_terminal(env: OurHexGame):
    return any(env.terminations.values())

def get_valid_actions(env: OurHexGame):
    action_mask = env.generate_info(env.agent_selection)['action_mask']
    valid_actions = []
    for i in range(len(action_mask)):
        if action_mask[i] == 1:
            valid_actions.append(i)

    return valid_actions

dc = copy.deepcopy

def clone_env(env: OurHexGame):
    new_env = OurHexGame(env.board_size, env.sparse_flag, "dont_render")
    new_env.board = dc(env.board)
    new_env.agents = dc(env.agents)
    new_env.agent_selection = dc(env.agent_selection)
    new_env.agent_selector = dc(env.agent_selector)
    new_env.is_first = dc(env.is_first)
    new_env.is_pie_rule_usable = dc(env.is_pie_rule_usable)
    new_env.is_pie_rule_used = dc(env.is_pie_rule_used)
    new_env.dones = dc(env.dones)
    new_env.infos = dc(env.infos)
    new_env._cumulative_rewards = dc(env.cumulative_rewards)
    new_env.terminations = dc(env.terminations)
    new_env.truncations = dc(env.truncations)
    new_env.rewards = dc(env.rewards)

    return new_env


class HexCNN(nn.Module):
    def __init__(self):
        super(HexCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=KERNEL_SIZE, padding=PADDING, stride=STRIDE)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=KERNEL_SIZE, padding=PADDING, stride=STRIDE)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=KERNEL_SIZE, padding=PADDING, stride=STRIDE)
        self.fc1 = nn.Linear(128 * 11 * 11 + 1, 512)
        self.fc2 = nn.Linear(512, 122)  # 11x11 board

    def forward(self, x, pie_rule_flag):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 11 * 11)
        x = torch.cat([x, pie_rule_flag.unsqueeze(1)], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class MCTSNode:
    def __init__(self, board_state, env: OurHexGame, parent=None, action=None):
        self.env = env
        self.board_state = board_state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_actions = get_valid_actions(self.env)

    def select_child(self):
        return max(self.children.values(), key=lambda c: c.ucb_score())

    def ucb_score(self):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + (EXPLORE_CONSTANT * math.sqrt(math.log(self.parent.visits) / self.visits))

    def expand(self):
        if len(self.untried_actions) > 0:
            action = self.untried_actions.pop()
        else:
            return None
        action_mask = self.env.generate_info(self.env.agent_selection)['action_mask']
        try:
            while action_mask[action] == 0:
                action = self.untried_actions.pop()
        except IndexError:
            return None

        next_state = clone_env(self.env)
        next_state.step(action)
        child_node = MCTSNode(next_state, next_state, parent=self, action=action)
        self.children[action] = child_node
        return child_node

    def simulate(self, model):
        current_env_clone = clone_env(self.env)
        old_reward = current_env_clone.rewards[current_env_clone.agent_selection]
        while not is_terminal(current_env_clone):
            board = torch.tensor(current_env_clone.board).unsqueeze(0).unsqueeze(0).float().cuda()
            pie_rule = torch.tensor([current_env_clone.is_pie_rule_used]).float().cuda()
            action_probs = model(board, pie_rule).squeeze().detach().cpu().numpy()
            valid_actions = get_valid_actions(current_env_clone)

            if not valid_actions:
                # No valid actions left, end the simulation
                break

            valid_probs = action_probs[valid_actions]
            valid_probs /= valid_probs.sum()
            action = np.random.choice(valid_actions, p=valid_probs)
            current_env_clone.step(action)
        return current_env_clone.rewards[current_env_clone.agent_selection] - old_reward

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)


class G01Agent:
    def __init__(self, model, env, time_limit=SEARCH_TIME):
        self.root = MCTSNode(env.observe(env.agent_selection)['observation'], env)
        self.model = model
        self.env = env
        self.time_limit = time_limit

    def search(self):
        end_time = time.time() + self.time_limit
        result = 0
        while time.time() < end_time:
            leaf = self.select()
            if not is_terminal(leaf.env):
                child = leaf.expand()
                if not child:
                    continue
                result += child.simulate(self.model)
            else:
                win_loss_reward = leaf.env.board_size ** 2 // 2
                result += win_loss_reward if leaf.env.check_winner(leaf.env.agent_selection) else -win_loss_reward
            leaf.backpropagate(result)

        return {action: child.visits / self.root.visits for action, child in self.root.children.items()}

    def select(self):
        node = self.root
        while node.untried_actions == [] and node.children != {}:
            node = node.select_child()
        return node

    def best_action(self):
        return max(self.root.children.items(), key=lambda x: x[1].visits)[0]

    def get_action(self):
        pass


def self_play_game(model, env):
    agent = G01Agent(model, env)
    states, policies, values = [], [], []

    while not any(env.terminations.values()):
        state = env.observe(env.agent_selection)['observation']
        policy = agent.search()
        action = max(policy, key=policy.get)
        states.append(state)
        policies.append(policy)
        print(f'chose action {action}', f'board state {state}')
        env.step(action)

    if env.check_winner(1):
        winner = 1
    elif env.check_winner(2):
        winner = -1
    else:
        winner = 0

    values = [winner if i % 2 == 0 else -winner for i in range(len(states))]

    return states, policies, values


def train_network(model, optimizer, states, policies, values):
    criterion = nn.MSELoss()

    for state, policy, value in zip(states, policies, values):
        board = torch.tensor(state).unsqueeze(0).unsqueeze(0).float().cuda()
        pie_rule = torch.tensor([0]).float().cuda()  # Assuming pie rule is not used, adjust if necessary

        predicted_policy = model(board, pie_rule)

        policy_loss = criterion(predicted_policy, torch.tensor(list(policy.values())).cuda())
        value_loss = criterion(predicted_policy.sum(), torch.tensor(value).float().cuda())

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def self_play_training(num_iterations=1000, num_games_per_iteration=100, render='human'):
    game = OurHexGame(board_size=11, render_mode=render)
    model = HexCNN().cuda()
    optimizer = optim.Adam(model.parameters())

    for iteration in range(num_iterations):
        game_data = []

        for _ in range(num_games_per_iteration):
            game.reset()
            states, policies, values = self_play_game(model, game)
            game_data.extend(zip(states, policies, values))

        random.shuffle(game_data)
        train_network(model, optimizer, *zip(*game_data))

        if iteration % 10 == 0:
            torch.save(model.state_dict(), f'hex_model_iteration_{iteration}.pth')
            torch.save(optimizer.state_dict(), f'hex_optimizer_iteration_{iteration}.pth')

    torch.save(model.state_dict(), 'hex_model_final.pth')
    torch.save(optimizer.state_dict(), 'hex_optimizer_final.pth')


# Run the training
self_play_training(2, 5, 'human')
