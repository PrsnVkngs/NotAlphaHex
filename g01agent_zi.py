import random
import math
import warnings
import functools
import pygame
import json
import numpy as np
from pathlib import Path
from typing import Dict

from PyQt5.sip import array
from pettingzoo.utils.env import AECEnv
from gymnasium import spaces
from pettingzoo.utils import agent_selector
from tensorflow.python.keras.metrics import TruePositives

from ourhexenv import OurHexGame

class G01Agent():
    def __init__(self, game_env, agent_selector=1):
        super().__init__()
        self._env = game_env
        self._board_size = 11 * 11
        self._Q_value_table = dict()
        self._epsilon = 0.1
        self._agent_selector = agent_selector

    def set_set_epsilon(self, epsilon: float):
        self._epsilon = epsilon
        pass

    def select_action(self, env_observation, env_reward, env_termination, env_truncation, env_info):
        if self._agent_selector == 0:
            action_chosen = self.select_action_dumb_agent(env_observation, env_reward, env_termination, env_truncation, env_info)
        else:
            action_chosen = self.select_action_basic_agent(env_observation, env_reward, env_termination, env_truncation, env_info)

        return action_chosen

    def set_epsilon(self, epsilon: float):
        self._epsilon = epsilon
        pass

    def update_Q_value_table(self, S, P, A, value):
        Q_key = self.convert_board_array_to_key(S, P)
        if Q_key in self._Q_value_table.keys():
            self._Q_value_table[Q_key][A] = value
        else:
            self._Q_value_table[Q_key] = {A: value}
        pass

    def read_Q_value_table(self, S, P, A):
        Q_key = self.convert_board_array_to_key(S, P)
        if Q_key in self._Q_value_table.keys():
            if A in self._Q_value_table[Q_key].keys():
                ret_val = self._Q_value_table[Q_key][A]
            else:
                ret_val = 0
        else:
            ret_val = 0
        return ret_val

    def convert_board_array_to_key(self, S: list[list[int]], P: int) -> str:
        key_str = ""
        i = 0
        while i < len(S):
            j = 0
            while j < len(S[i]):
                key_str += str(S[i][j])
                j += 1
            i += 1
        key_str += str(P)
        return key_str

    def get_all_possible_actions(self, S: list[list[int]], P: int) -> list[int]:
        possible_actions_list = []
        action_value = 0
        i = 0
        while i < len(S):
            j = 0
            while j < len(S[i]):
                if S[i][j] == 0:
                    possible_actions_list.append(action_value)
                j += 1
                action_value += 1
            i += 1
        if P == 0:
            possible_actions_list.append(action_value)
        return possible_actions_list

    def get_all_Q_values_of_actions(self, S: list[list[int]], P: int,  AP_list: list[int]) -> list[int]:
        Q_values_list = []
        for a in AP_list:
            Q_values_list.append(self.read_Q_value_table(S, P, a))
        return Q_values_list

    def get_epsilon_greedy_action(self, S: list[list[int]], P: int) -> int:
        possible_action_list = self.get_all_possible_actions(S, P)
        possible_action_Q_values_list = self.get_all_Q_values_of_actions(S, P,  possible_action_list)

        max_Q_value = max(possible_action_Q_values_list)
        max_Q_value_actions = list()
        i = 0
        for x in possible_action_Q_values_list:
            if x == max_Q_value:
                max_Q_value_actions.append(possible_action_list[i])
            i += 1

        num_max_Q_value_actions = len(max_Q_value_actions)
        num_total_actions = len(possible_action_list)
        unit_rand = random.uniform(0, 1)
        if unit_rand > 1-self._epsilon:
            action_idx = random.randint(0, num_total_actions-1)
            action_chosen = possible_action_list[action_idx]
        else:
            action_idx = random.randint(0, num_max_Q_value_actions-1)
            action_chosen = max_Q_value_actions[action_idx]

        return(action_chosen)

    def select_action_dumb_agent(self, env_observation, env_reward, env_termination, env_truncation, env_info) -> int:
        number_of_valid_moves = np.count_nonzero(env_info["action_mask"] == 1)
        if number_of_valid_moves > 0:
            move_choice = random.randint(0, self._board_size)
            empty_space_not_found = True
            while empty_space_not_found:
                if env_info["action_mask"][move_choice] == 1:
                    empty_space_not_found = False
                else:
                    move_choice = random.randint(0, self._board_size)
        else:
            move_choice = None
        return move_choice

    def select_action_basic_agent(self, env_observation, env_reward, env_termination, env_truncation, env_info) -> int:

        return(0)

    def load_past_experience(self) -> bool:
        path = './basic_agent_experience_01.json'
        path_obj = Path(path)
        ret_val = False
        if path_obj.exists():
            experience_file_json = open('./basic_agent_experience_01.json')
            self._Q_value_table = json.load(experience_file_json)
            experience_file_json.close()
            ret_val = True
        else:
            ret_val = False

        return ret_val
