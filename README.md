# 🧠 Cliff Walking - Reinforcement Learning (SARSA & Q-Learning)

This project implements the **Cliff Walking** problem using two popular reinforcement learning algorithms — **SARSA** (on-policy TD control) and **Q-Learning** (off-policy TD control). The environment used is from **OpenAI Gym**.

## 📌 Overview

In the Cliff Walking problem (a classic example from Sutton & Barto), the agent must learn to reach a goal state without falling off a cliff. The challenge is to balance exploration and exploitation while avoiding large negative rewards.

### 📍 Environment: `CliffWalking-v0` from `gymnasium`

- Gridworld of size `4 x 12`
- Start state: `(3, 0)`
- Goal state: `(3, 11)`
- Cliff region: positions between `(3, 1)` and `(3, 10)`
- Falling into the cliff gives a reward of `-100` and ends the episode
- Step penalty: `-1`

---

## 🚀 Algorithms Implemented

### ✅ SARSA (State-Action-Reward-State-**Action**)
- **On-policy** learning
- Updates Q-values using the action **actually taken**

### ✅ Q-Learning
- **Off-policy** learning
- Updates Q-values using the **maximum estimated future reward** regardless of the agent’s actual action

---

## 🛠️ Requirements

- Python 3.8+
- `gymnasium`
- `numpy`
- `matplotlib` (optional for plotting)
- `cv2` (if visualization is enabled)
- `pickle` (for saving/loading Q-tables)

Install dependencies:
```bash
pip install gymnasium numpy matplotlib opencv-python
