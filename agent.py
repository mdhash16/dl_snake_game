import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from model import LinearQNet

# ── Hyperparameters ────────────────────────────────────────────────────────────
MAX_MEMORY = 100_000   # how many (s, a, r, s', done) tuples we keep
BATCH_SIZE = 1_000     # sample size for long-memory replay
LR         = 0.001     # Adam learning rate
GAMMA      = 0.9       # discount factor for future rewards


class Agent:
    """
    DQN agent.

    Short-memory  : train on the single most recent step.
    Long-memory   : train on a random batch from the replay buffer
                    (called once per game-over).
    Exploration   : epsilon-greedy — random early, greedy as n_games grows.
    """

    def __init__(self):
        self.n_games  = 0
        self.memory   = deque(maxlen=MAX_MEMORY)
        self.model    = LinearQNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn  = nn.MSELoss()

    # ── Memory ─────────────────────────────────────────────────────────────────

    def get_state(self, game):
        """Ask the game's InputLayer to produce the 11-neuron state vector."""
        return game.input_layer.get_state(game.snake, game.direction, game.apple)

    def remember(self, state, action, reward, next_state, done):
        """Store one experience tuple in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    # ── Action selection ───────────────────────────────────────────────────────

    def get_action(self, state):
        """
        Epsilon-greedy policy.
        Early games → explore (random moves).
        Later games → exploit (model prediction).
        Returns a one-hot list, e.g. [1,0,0] = straight.
        """
        epsilon = max(0, 80 - self.n_games)   # decays to 0 after 80 games
        action  = [0, 0, 0]

        if random.randint(0, 200) < epsilon:
            move = random.randint(0, 2)        # random exploration
        else:
            state_t = torch.tensor(state, dtype=torch.float)
            q_vals  = self.model(state_t)
            move    = torch.argmax(q_vals).item()  # greedy exploitation

        action[move] = 1
        return action

    # ── Training ───────────────────────────────────────────────────────────────

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train on the single most recent step."""
        self._train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        """Sample a random batch from replay memory and train on it."""
        batch = (
            random.sample(self.memory, BATCH_SIZE)
            if len(self.memory) >= BATCH_SIZE
            else list(self.memory)
        )
        states, actions, rewards, next_states, dones = zip(*batch)
        self._train_step(states, actions, rewards, next_states, dones)

    def _train_step(self, state, action, reward, next_state, done):
        """
        One gradient update using the Bellman equation:
            Q_new = r                          (if terminal)
            Q_new = r + gamma * max Q(s')      (otherwise)
        We only update the Q-value for the action that was taken.
        """
        # Convert everything to tensors; handle both single and batch inputs.
        state      = torch.tensor(np.array(state),      dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action     = torch.tensor(np.array(action),     dtype=torch.long)
        reward     = torch.tensor(np.array(reward),     dtype=torch.float)

        if state.dim() == 1:   # single step → add batch dimension
            state      = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action     = action.unsqueeze(0)
            reward     = reward.unsqueeze(0)
            done       = (done,)

        pred   = self.model(state)        # current Q-values for all actions
        target = pred.clone()             # copy; we'll only modify the chosen action

        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + GAMMA * torch.max(self.model(next_state[i]))
            taken_action = torch.argmax(action[i]).item()
            target[i][taken_action] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_fn(target, pred)
        loss.backward()
        self.optimizer.step()
