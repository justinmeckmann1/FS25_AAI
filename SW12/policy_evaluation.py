# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:10:37 2025

@author: JumpStart
"""

import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from sty import bg,fg, rs

# Grid world parameters
grid_rows = 4
grid_cols = 4
gamma = 1.0        # Discount factor

# Terminal states 
terminals = [
    (0, 0),
    (3, 3) 
]

# Obstacles (walls)
obstacles = [
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
    V[s] = 0.0  # Terminal states have no future value
       

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


# Policy Evaluation Algorithm (Fig. 4.2)
iteration = 0
while iteration <= 1000:
    iteration += 1
    # Policy Evaluation
    delta = 0
    V_prev = V.copy()
    for s in states:
        v = V[s]
        V[s] = 0
        for a in actions:
            next_state, reward = get_transition(s, a)
            V[s] += policy[s][a] * (reward + gamma * V_prev[next_state])

    # Print the value function after certain iterations
    if iteration in [1,2,3,10,1000]:
        print_state_value_function(V)
        print("Iteration : " + f"{iteration:6}")


