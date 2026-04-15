# Snake AI — Deep Q-Network

A Snake game where an AI agent learns to play from scratch using **Deep Q-Learning (DQN)** and PyTorch.

---

## Demo

| Training (live visualization) | Play mode (trained model) |
|---|---|
| Score chart + stats sidebar update in real-time | Greedy inference at human-watchable speed |

Launch everything from a single entry point:

```bash
python view_ai.py
```

---

## How It Works

The agent observes an **11-element state vector** and outputs **Q-values** for three actions.

### State Vector (11 values)

| Group | Features |
|---|---|
| Danger | Straight ahead, Right, Left (boolean) |
| Direction | Moving Left, Right, Up, Down (one-hot) |
| Food | Food Left, Right, Up, Down (boolean) |

### Actions

```
[1, 0, 0]  →  Go straight
[0, 1, 0]  →  Turn right
[0, 0, 1]  →  Turn left
```

### Network Architecture

```
Input (11)
    │
Linear(11 → 256)
    │
  ReLU
    │
Linear(256 → 3)
    │
Q-values (3)
```

### Training Algorithm

The agent uses **two-stage Q-learning**:

1. **Short memory** — single Bellman update after every game step
2. **Long memory** — random batch sampled from a 100 000-experience replay buffer at game-over

Bellman target:
```
Q_new = reward + γ · max Q(next_state)   (non-terminal)
Q_new = reward                            (terminal)
```

Key hyperparameters:

| Parameter | Value |
|---|---|
| Replay buffer size | 100 000 |
| Batch size | 1 000 |
| Learning rate | 0.001 (Adam) |
| Discount factor γ | 0.9 |
| Epsilon decay | `max(0, 80 − n_games)` |

Exploration drops to zero after **80 games**, after which the agent acts purely on learned Q-values.

---

## Project Structure

```
dl_snake_game/
├── view_ai.py       # Entry point — menu, live training UI, play mode
├── train.py         # Headless training loop (no pygame)
├── agent.py         # DQN agent: replay buffer, epsilon-greedy, Bellman updates
├── model.py         # LinearQNet — feed-forward Q-network with save/load
├── snake_ai.py      # Snake game engine + InputLayer (state extraction)
├── helper.py        # Matplotlib score plotting (for notebook use)
├── sample_run.py    # Minimal usage example
├── model.pth        # Saved model weights (updated on each new record)
└── old_ml_training/ # Earlier experiments
```

---

## Getting Started

### Prerequisites

```bash
pip install torch pygame matplotlib
```

### Run the interactive app

```bash
python view_ai.py
```

- Press **T** to start training — watch the agent learn in real time
- Press **P** to watch the trained model play (available once `model.pth` exists)
- Press **Escape** from either mode to return to the menu

### Run headless training (terminal only)

```bash
python train.py
```

Prints `Game N | Score: X | Record: Y` each episode. Saves `model.pth` whenever a new record is set.

---

## Architecture Details

### `view_ai.py` — Pygame Application

- **Menu** — detects `model.pth`; Play option is grayed out if no model is saved yet
- **Training mode** — game at 60 fps, bottom bar chart of last 60 episode scores, right-side stats sidebar (game count / score / record)
- **Play mode** — loads saved weights, runs greedy inference at 12 fps

### `agent.py` — DQN Agent

- Holds `LinearQNet`, Adam optimizer, MSELoss, and a `deque` replay buffer
- `get_state(game)` — calls `InputLayer` to produce the 11-element state vector
- `get_action(state)` — epsilon-greedy selection; epsilon decays linearly over first 80 games
- `train_short_memory` / `train_long_memory` — delegates to `_train_step`, which handles both single-step and batched inputs

### `model.py` — LinearQNet

- Simple two-layer network: `Linear(11→256) → ReLU → Linear(256→3)`
- `model.save(path)` / `LinearQNet.load(path)` for checkpoint management

### `snake_ai.py` — Game Engine + InputLayer

- `Game_AI.turn(action)` — steps the game, returns `(reward, done, score)`
- `InputLayer.get_state(game)` — computes the 11-element boolean/one-hot vector

---

## License

MIT
