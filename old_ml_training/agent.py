import torch 
import random 
import numpy as np
from collections import deque
from snake_ai import Game_AI, Direction, P

from model import linear_qnet, QTrainer
from helper import plot
from view_ai import redner

MAX_MEMORY = 100_000_0
BATCH_SIZE = 10000
LR = 0.001
render = redner()


class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = linear_qnet(11,256,3) 
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)

    def get_state(self,game):
        head = game.snake[-1]
        point_l = P(head.x - 1, head.y)
        point_r = P(head.x + 1, head.y)
        point_u = P(head.x, head.y - 1)
        point_d = P(head.x, head.y + 1)

        l = game.direction == Direction.LEFT
        r = game.direction == Direction.RIGHT
        u = game.direction == Direction.UP
        d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (u and game.is_collision(point_u))or
            (d and game.is_collision(point_d))or
            (l and game.is_collision(point_l))or
            (r and game.is_collision(point_r)),

            (u and game.is_collision(point_r))or
            (d and game.is_collision(point_l))or
            (u and game.is_collision(point_u))or
            (d and game.is_collision(point_d)),

            (u and game.is_collision(point_r))or
            (d and game.is_collision(point_l))or
            (r and game.is_collision(point_u))or
            (l and game.is_collision(point_d)),

            l,
            r,
            u,
            d,

            game.apple.x < head.x,
            game.apple.x > head.x,
            game.apple.y < head.y,
            game.apple.y > head.y
        ]
        return np.array(state,dtype=int)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0]
        if(random.randint(0, 100) < self.epsilon):
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # prediction by model 
            move = torch.argmax(prediction).item()
            final_move[move]=1 
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game_AI()
    while True:

        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.turn(final_move)
        render.render(game)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old,final_move,reward,state_new,done)
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()
            if(score > reward): # new High score 
                reward = score
                agent.model.save()
            print('Game:',agent.n_game,'Score:',score,'Record:',record, 'Reward:', reward)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if(__name__=="__main__"):
    train()