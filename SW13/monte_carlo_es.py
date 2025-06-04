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

# Initialize the policy
policy = {}
for s in states:
    policy[s] = {}
    for a in actions:
        policy[s][a] = 1/len(actions)

       
# Select action randomly according to pi 
def select_action(s):
    a = policy[s]
    return random.choices(list(a.keys()),list(a.values()), k=1)


def generate_episode(s0, a0):
    trajectory = []
    terminal = False
    first = True
    while not terminal:
        if first:
            a, s = (a0, s0)
            first = False
        else:
            [a] = select_action(s)
        s_, reward = get_transition(s, a)
        trajectory.append((s, a, reward))
        s = s_
        if s in terminals:
            terminal = True
    return trajectory, len(trajectory)


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
                if s in terminals:
                    print(bg.yellow+"       "+rs.all, end='')
                else:
                    print(bg.grey+"       "+rs.all, end='')                    
        print()


# MC Control ES control according to Figure 5.4


# Initialize the action-value function Q(s) to zero, including terminal states
Q = {}
for s in states:
    Q[s] = {}
    for a in actions:
        Q[s][a] = 0.0

Returns = {}
for s in states:
    for a in actions:
        Returns[s,a]  = []

for i in range(10000):
    traj, T = generate_episode(random.choice(states), random.choice(actions))
    G = 0
    if i%100 == 0:
        print("Iteration : " + f"{i:6}")
    for t in range(T-1,0,-1):
        S, A, R = traj[t]
        G = gamma*G + R
        if not (S, A) in [ (s,a) for (s,a,_) in [traj[i] for i in range(0,t)]]:
            Returns[S, A].append(G)
            Q[S][A] = sum(Returns[S, A])/len(Returns[S, A])
            A_star = max(Q[S], key=Q[S].get)
            eps = 0 if i > 1000 else 0.2
            for a in actions:
                policy[S][a] = 1-eps if a == A_star else eps


# Final output
print_policy(policy)
print_action_value_function(Q)
