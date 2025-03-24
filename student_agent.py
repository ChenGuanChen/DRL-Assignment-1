# Remember to adjust your student ID in meta.xml
from math import inf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import importlib.util
import pickle
import time
from IPython.display import clear_output
import random

def get_action(obs):
    def get_agent_state(obs, preact = -1, preobs = None, yet_visited = None, dest = 0):
        taxi_row, taxi_col, station1y, station1x, station2y, station2x, station3y, station3x, station4y, station4x, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
        station1 = (station1y, station1x)
        station2 = (station2y, station2x)
        station3 = (station3y, station3x)
        station4 = (station4y, station4x)
        stations = [station1, station2, station3, station4]
        taxi = (taxi_row, taxi_col)

        at_station = 1 if(taxi in stations) else 0
        at_dest = 1 if(taxi in stations  and destination_look) else 0
        at_people = 1 if(taxi in stations and passenger_look) else 0

        pre_taxi_row, pre_taxi_col, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = preobs if preobs else (-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0)
        pre_taxi = (pre_taxi_row, pre_taxi_col)

        if yet_visited:
            for x in range(len(stations)):
                if pre_taxi == stations[x] and yet_visited & (1 << x):
                    yet_visited = yet_visited ^ (1 << x)
        else:
            yet_visited = 15

        if at_dest:
            for x in range(len(stations)):
                if taxi == stations[x]:
                    dest = 1 << x
                    break

        shortesty, shortestx, total = -1, -1, np.inf
        if dest and at_people and preact == 4:
            for x in range(len(stations)):
                if dest == 1 << x:
                    shortesty = stations[x][0] - taxi_row
                    shortestx = stations[x][1] - taxi_col
                    break
        else:
            for z in range(len(stations)):
                checking = 1 << z
                if yet_visited & checking:
                    y, x = stations[z]
                    if abs(y - taxi_row) + abs(x - taxi_col) < total:
                        shortesty = stations[z][0] - taxi_row
                        shortestx = stations[z][1] - taxi_col
                        total = abs(shortesty) + abs(shortestx)
        if not shortesty == 0:
            shortesty //= abs(shortesty)
        if not shortestx == 0:
            shortestx //= abs(shortestx)

        # self.state_size = (6, 2, 2, 2, 3, 3, 2, 2, 2, 2, 16, 16, 6)
        # self.action_size = 6
        return (preact, at_station, at_dest, at_people, shortesty + 1, shortestx + 1, obstacle_north, obstacle_south, obstacle_east, obstacle_west), yet_visited, dest
    
    cnt = 0
    with open("param.pkl", "rb") as f:
        preact, preobs, pre_yet_visited, pre_dest, cnt = pickle.load(f)
    f.close()

    with open("table_without_drop_faster.pkl", "rb") as f1:
        q_table = pickle.load(f1)
    f1.close()

    state, yet_visited, dest = get_agent_state(obs, preact, preobs, pre_yet_visited, pre_dest)
    if state not in q_table:
        q_table[state] = np.zeros(6)
        action = np.random.choice([x for x in range(6)])
    else:
        action = np.argmax(q_table[state]) #if np.random.rand() >= 0.15 else np.random.choice([x for x in range(5)])

    _, _, at_dest, at_people, _, _, _, _, _, _ = state

    ans = (action, obs, yet_visited, dest, cnt + 1)
    if (at_dest and at_people and action == 5) or cnt + 1 == 5000:
        ans = (-1, {}, 15, 0, 0)
    ans = (-1, {}, 15, 0, 0)
    with open('param.pkl', 'wb') as f:
        pickle.dump(ans, f)
    return  action# Choose a random action, 4, 5
    # You can submit this random agent to evaluate the performance of a purely random strategy.