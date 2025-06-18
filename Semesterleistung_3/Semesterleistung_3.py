import numpy as np
import random
states = [(f, fa)for f in range(3)for fa in range(3)]
actions = [0 , 1 , 2] # study, short break, long break
q_table = np.zeros((len(states),len(actions)))
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.2 # Exploration rate
random.seed(42) 

def state_index(state):
    return states.index(state)

def step(state, action):
    focus, fatigue = state
    if action == 0: # Study --> increase fatigue and decrease focus by 1
        fatigue = min(fatigue + 1 , 2)
        focus = max(focus - 1 , 0)
    elif action == 1: # Short break --> reduce fatigue and increase focus by 1
        fatigue = max(fatigue - 1 , 0)
        focus = min(focus + 1 , 2)
    elif action == 2: # Long break --> maximize focus and reset fatigue
        fatigue = 0
        focus = 2
    
    new_state =(focus , fatigue)
    
    if focus == 2 and fatigue == 0: # Max focus and no fatigue --> optimal state
        reward = 5
    elif action == 0 and fatigue == 2 or action == 0 and focus == 0: # Studying with high fatigue or low focus --> not optimal
        reward = -3
    elif action in [1 , 2] and fatigue == 0 or action in [1 , 2] and focus == 2: # Taking a break with no fatigue or high concentration --> not optimal
        reward = -1
    else:
        reward = 1 # all other actions (e.g. good transition states)
        
    return new_state , reward
for episode in range(1000):
    state = random.choice(states)
    for _ in range(10):
        if random.uniform(0 , 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(q_table[state_index(state)])
        next_state , reward = step(state, action)
        old_q = q_table[state_index(state), action ]
        next_max = np.max(q_table[state_index(next_state)])
        new_q = old_q + learning_rate*(reward + discount_factor * next_max - old_q)
        q_table[state_index(state), action] = new_q
        state = next_state

# Final policy output
levels = ['Low', 'Medium', 'High']
actions = ['Study', 'Short Break', 'Long Break']

print(f"{'Focus':<12} {'Fatigue':<12} {'Action':<6} ({'Description'})")
print("-" * 46)

for state in states:
    best_action = np.argmax(q_table[state_index(state)])
    focus = levels[state[0]]
    fatigue = levels[state[1]]
    action_desc = actions[best_action]
    print(f"{focus:<12} {fatigue:<12} {best_action:<6} ({action_desc})")