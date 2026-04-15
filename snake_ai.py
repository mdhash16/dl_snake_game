import collections
import random
from enum import Enum
import numpy as np

from collections import namedtuple
import math

class P:
    def __init__(self, x_init, y_init):
        self.x = x_init
        self.y = y_init

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(('x', self.x, 'y', self.y))

class InputLayer:
    '''
    11 input neurons (indices match InputLayerState constants - 1):
      [0] danger straight
      [1] danger right
      [2] danger left
      [3] moving left
      [4] moving right
      [5] moving up
      [6] moving down
      [7] food left
      [8] food right
      [9] food up
      [10] food down
    '''

    def __init__(self, W=30, H=30):
        self.W = W
        self.H = H
        self.neurons = np.zeros(11, dtype=int)
        self._head = None  # cached by is_danger_around, used by is_food_around

    def _next_point(self, origin, direction):
        return P(origin.x + direction.value.x, origin.y + direction.value.y)

    def _is_collision(self, pt, snake):
        # Wall collision
        if pt.x >= self.W or pt.x < 0 or pt.y >= self.H or pt.y < 0:
            return True
        # Body collision (tail vacates its cell each step, so exclude it)
        snake_body = list(snake)[:-1]
        return pt in snake_body

    def is_danger_around(self, snake, direction):
        # Get head, then probe one step ahead in the straight, right, and left
        # directions relative to current travel; set neurons[0..2] to 1 if danger.
        head = snake[0]
        self._head = head

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(direction)

        dir_straight = direction
        dir_right    = clock_wise[(idx + 1) % 4]
        dir_left     = clock_wise[(idx - 1) % 4]

        self.neurons[0] = int(self._is_collision(self._next_point(head, dir_straight), snake))
        self.neurons[1] = int(self._is_collision(self._next_point(head, dir_right),    snake))
        self.neurons[2] = int(self._is_collision(self._next_point(head, dir_left),     snake))

    def get_moving_direction(self, direction):
        # One-hot encode the current travel direction into neurons[3..6].
        self.neurons[3] = int(direction == Direction.LEFT)
        self.neurons[4] = int(direction == Direction.RIGHT)
        self.neurons[5] = int(direction == Direction.UP)
        self.neurons[6] = int(direction == Direction.DOWN)

    def is_food_around(self, p_apple):
        # Compare apple coordinates to the snake head (cached by is_danger_around)
        # and set neurons[7..10] to indicate which side the food is on.
        head = self._head
        self.neurons[7]  = int(p_apple.x < head.x)  # food left
        self.neurons[8]  = int(p_apple.x > head.x)  # food right
        self.neurons[9]  = int(p_apple.y > head.y)  # food up
        self.neurons[10] = int(p_apple.y < head.y)  # food down

    def get_state(self, snake, direction, apple):
        # Fill all 11 neurons in the correct order and return a copy of the array.
        # This is the single entry point used by the agent at each game step.
        self.is_danger_around(snake, direction)
        self.get_moving_direction(direction)
        self.is_food_around(apple)
        return self.neurons.copy()

class InputLayerState:
    DANGER_STAIGHT = 1
    DANGER_RIGHT = 2
    DANGER_LEFT = 3
    MOVING_LEFT = 4
    MOVING_RIGHT = 5 
    MOVING_UP = 6
    MOVING_DOWN = 7
    FOOD_LEFT = 8
    FOOD_RIGHT = 9
    FOOD_UP = 10
    FOOD_DOWN = 11

class Direction(Enum):
    RIGHT   = P( 1,  0)
    DOWN    = P( 0,  -1)
    LEFT    = P(-1,  0)
    UP      = P( 0,  1)
    
class Game_AI:
    W = 30
    H = 30
    snake = collections.deque()
    direction = Direction.DOWN
    game_over = False
    score = 0
    turns = 0
    input_layer = InputLayer()

    def __random_with_exlcuded_list(self, max_x, min_x, max_y, min_y, snake):
        p = P(random.randint(min_x, max_x), random.randint(min_y, max_y))
        
        while p in snake:
            p = P(random.randint(min_x, max_x), random.randint(min_y, max_y))
        return p

    def __spawn_apple(self):
        return self.__random_with_exlcuded_list(self.W - 1, 1, self.H - 1, 1, self.snake)

    def __get_snake_start_points(self):
        start_p = P(self.W / 2, self.H / 2)
        return collections.deque([start_p, P(start_p.x + 1, start_p.y), P(start_p.x + 2, start_p.y)])

    def __init__(self):
        self.snake = self.__get_snake_start_points()
        self.apple = self.__spawn_apple()

    def __move_snake(self):
        head = self.snake[0]
        self.snake.appendleft(P(head.x + self.direction.value.x, head.y + self.direction.value.y))
        self.snake.pop()

    def __is_out_of_border(self):
        point = self.snake[-1]
        if point.x >= self.W or point.x < 0 or point.y >= self.H or point.y < 0:
            return True
        return False

    def __is_apple_eaten(self):
        head = self.snake[0]
        return self.apple.x == head.x and self.apple.y == head.y

    def __is_ate_himself(self):
        snake_set = set(self.snake)
        if len(snake_set) != len(self.snake):
            return True
        else:
            return False
    
    def __grow_snake(self):
        last = self.snake[-1]
        self.snake.append(P(last.x + self.direction.value.x, last.y + self.direction.value.y))

    def is_collision(self, pt = None):
        if pt is None:
            pt = self.snake[-1]
        if pt.x > self.W or pt.x < 0 or pt.y > self.H or pt.y < 0:
            return True
        copy_snake = self.snake.copy()
        copy_snake.popleft
        if pt in copy_snake:
            return True
        return False

    def reset(self):
        self.snake.clear()
        self.snake = self.__get_snake_start_points()
        self.apple = self.__spawn_apple()
        self.score = 0
        self.turns = 0
        self.direction = Direction.DOWN
        self.game_over = False
        self.reward = 0

    def turn(self, action):

        clock_wise = [Direction.RIGHT, 
                      Direction.DOWN,
                      Direction.LEFT,
                      Direction.UP]

        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action,[1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action,[0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right Turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left Turn

        self.direction = new_dir
        self.__move_snake()

        is_out = self.__is_out_of_border()
        apple_eaten = self.__is_apple_eaten()
        ate_himself = self.__is_ate_himself()
        reward = 0

        self.turns += 1

        if is_out or ate_himself or self.turns > 100*len(self.snake):
            self.game_over = True
            reward = -10
            return reward, self.game_over, self.score
        
        if apple_eaten:
            self.score += 1
            reward = 10
            self.apple = self.__spawn_apple()
            self.__grow_snake()
        
        return reward, self.game_over, self.score
    
