import gymnasium as gym
import cv2
import pickle as pkl
import numpy as np

cliffEnv = gym.make("CliffWalking-v0", render_mode="ansi")

q_table=pkl.load(open("sarsa_q_table.pkl","rb"))

def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2

    # Vertical Lines
    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), color=(0, 0, 0), thickness=1)

    # Horizontal Lines
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), color=(0, 0, 0), thickness=1)

    # Cliff Box
    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), color=(200, 0, 255),
                        thickness=-1)
    img = cv2.putText(img, text="Cliff", org=(49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # Goal
    frame = cv2.putText(img, text="G", org=(49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    
    return frame


def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    state_idx = state[0] if isinstance(state, tuple) else state
    row, column = np.unravel_index(indices=state_idx, shape=(4, 12))
    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img


def policy(state,explore=0.0):
    state_idx = state[0] if isinstance(state, tuple) else state
    action=int(np.argmax(q_table[state_idx]))
    if np.random.random() <= explore:
        action=int(np.random.randint(low=0,high=4,size=1))
    return action

NUM_EPSIODES=5

for episode in range(NUM_EPSIODES):
    done=False
    frame=initialize_frame()
    state=cliffEnv.reset()
    total_reward=0
    epsiode_length=0


    while not done:
        frame2=put_agent(frame.copy(),state)
        cv2.imshow("Cliff Walking",frame2)
        cv2.waitKey(250)
        action=policy(state)
        state,reward,terminated,truncated,info=cliffEnv.step(action)
        done = terminated or truncated
        epsiode_length+=1
        total_reward+=reward


cliffEnv.close()