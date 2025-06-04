# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:10:37 2025

@author: JumpStart
"""

import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import random 
from sty import bg,fg, rs

# Grid world parameters
grid_rows = 6
grid_cols = 6
gamma = 1.0        # Discount factor

# Terminal states and their rewards
terminals = [
    (0, 0),
    (5, 5) 
]

# Obstacles (walls)
obstacles = [
    (2,2),
    (3,3)
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


# Select action epsilon greedy
def select_action(s, Q):
    a_star = max(Q[s], key=Q[s].get)
    [a] = random.choices(actions,[epsilon/(len(actions)-1) if not a == a_star else (1-epsilon) for a in actions], k=1)
    return a

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
 
        
def print_action_value_function(Q):
    """Print the action-value function grid."""
    
    # Colormap
    cm = plt.get_cmap('coolwarm')
#    mi, ma = (min(Q.values()), max(Q.values()))
    mi, ma = (-20,0)
    def to_color(v):
        return [int(255*x) for x in clrs.to_rgb(cm((v-mi)/(ma-mi)))]

    print("\nAction-Value Function:")
    for i in range(grid_rows):
        for j in range(grid_cols):
            s = (i, j)
            if s in Q:
                a_max = max(Q[s], key=Q[s].get)
                print(fg.black+bg(*to_color(Q[s][a_max])) + f"{Q[s][a_max]:6.1f} " + rs.all, end='')
            else:
                print(bg.grey+"       "+rs.all, end='')               
        print()


# Q-Learning (off policy TD control), estimates pi = pi_*

alpha = 0.7
epsilon = 0.5

# Initialize the action-value function Q(s) to zero, including terminal states
Q = {}
for s in states:
    Q[s] = {}
    for a in actions:
        Q[s][a] = random.uniform(-10, 10)
for s in terminals:
    Q[s] = {}
    for a in actions:
        Q[s][a] = 0.0            
 

for i in range(5000):
    S = random.choice(states)
    if i%100 == 0:
        print("Iteration : " + f"{i:6}")
    while True:
        A = select_action(S,Q)
        S_, R = get_transition(S, A)
        Q[S][A] += alpha*(R + gamma*max(Q[S_].values()) - Q[S][A])
        S = S_
        if S in terminals:
            break
       

# Final output
print_action_value_function(Q)
