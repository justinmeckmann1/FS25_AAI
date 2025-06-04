# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:10:37 2025

@author: JumpStart
"""

import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from sty import bg,fg, rs

# Grid world parameters
grid_rows = 8
grid_cols = 8
gamma = 1.0        # Discount factor
theta = 1e-6       # Convergence threshold

# Terminal states 
terminals = [
    (6, 7) 
]

# Obstacles (walls)
obstacles = [
    (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),
    (1,0),(1,7),
    (2,2),(2,3),(2,5),(2,7),
    (3,0),(3,3),(3,4),(3,7),
    (4,0),(4,1),(4,4),(4,6),(4,7),
    (5,0),(5,2),(5,4),(5,7),
    (6,0),(6,5),
    (7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),
]

# Possible actions
actions = ['L', 'U', 'D', 'R']

# Action vectors for movement
action_vectors = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

# All states in the grid world
states = []
for i in range(grid_rows):
    for j in range(grid_cols):
        s = (i, j)
        if s not in obstacles and s not in terminals:
            states.append(s)

# Initialize the policy with equal probability for all actions
policy = {}
for s in states:
    policy[s] = {}
    for a in actions:
        policy[s][a] = 1/len(actions)


# Initialize the state-value function V(s) to zero, including terminal states
V = {}
for s in states:
    V[s] = 0.0
for s in terminals:
    V[s] = -1.0 # Terminal state
       

def get_transition(s, a):
    """Get the transition from state s when action a is taken, returns (next_state, reward)"""

    # Check if the state is within the grid and not an obstacle.
    def is_valid_state(s):
        i, j = s
        return 0 <= i < grid_rows and 0 <= j < grid_cols and s not in obstacles
 
    if s in terminals:  # No transition from terminal states
        return ()
    
    di, dj = action_vectors[a]
    next_state = (s[0] + di, s[1] + dj)
    
    # Check for collisions with walls or obstacles
    if not is_valid_state(next_state): 
        next_state = s     # Agent stays in the same state

    reward = -1.0

    return next_state, reward


def print_policy(policy):
    """Print the policy grid."""

    action_symbols = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}
    print("\nPolicy:")
    for i in range(grid_rows):
        print(end=' ')
        for j in range(grid_cols):
            s = (i, j)
            if s in policy:
                print(end=' ')
                for a in actions:
                    print(fg(int(255*policy[s][a]),0,0)+f"{action_symbols[a]}"+rs.all, end='')
                print(end=' ')
            else:
                if s in terminals:
                    print(bg.yellow+"      "+rs.all, end='')
                else:
                    print(bg.grey+"      "+rs.all, end='')                    
            print(end=' ')
        print('\n')
        
        
def print_state_value_function(V):
    """Print the state-value function grid."""
    
    # Colormap
    cm = plt.get_cmap('coolwarm')
    mi, ma = (min(V.values()), max(V.values()))
    def to_color(v):
        return [int(255*x) for x in clrs.to_rgb(cm((v-mi)/(ma-mi)))]

    print("\nState-Value Function:")
    for i in range(grid_rows):
        for j in range(grid_cols):
            s = (i, j)
            if s in V:
                print(fg.black+bg(*to_color(V[s])) + f"{V[s]:6.1f} " + rs.all, end='')
            else:
                print(bg.grey+"       "+rs.all, end='')               
        print()



# Value Iteration Algorithm
iteration = 0
while True:
    iteration += 1
    # Value Iteration
    delta = 0
    V_prev = V.copy()
    for s in states:
        v = V[s]
        action_values = {}
        for a in actions:
            next_state, reward = get_transition(s, a)
            action_values[a] = reward + gamma * V_prev[next_state]
        V[s] = max(action_values.values())
        delta = max(delta, abs(v - V[s]))
    if delta < theta:
        break
    if iteration > 1:
        print_state_value_function(V)
        print("Iteration : " + f"{iteration:6}")

# Determine a greedy policy
p = {} 
for s in states:
    p[s] = {}
    action_values = {}
    for a in actions:
        next_state, reward = get_transition(s, a)
        action_values[a] = reward + gamma * V[next_state]
    a_star = max(action_values, key=action_values.get)
    for a in actions:
        p[s][a] = 1.0 if a == a_star else 0.0
print_policy(p)
        


