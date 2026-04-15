import collections
import random
from enum import Enum

class P:
    def __init__(self, x_init, y_init):
        self.x = x_init
        self.y = y_init

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(('x', self.x, 'y', self.y))

class Direction(Enum):
    UP      = P( 0,  -1)
    DOWN    = P( 0,  1)
    RIGHT   = P( 1,  0)
    LEFT    = P(-1,  0)

class Game:
    W = 60
    H = 60
    snake = collections.deque()
    direction = Direction.DOWN
    game_over = False
    score = 0

    @staticmethod
    def __random_with_exlcuded_list(max_x, min_x, max_y, min_y, snake):
        p = P(random.randint(min_x, max_x), random.randint(min_y, max_y))
        
        while p in snake:
            p = P(random.randint(min_x, max_x), random.randint(min_y, max_y))
        return p

    apple = __random_with_exlcuded_list(W, 0, H, 0, snake)

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
        for point in self.snake:
            if point.x >= self.W or point.x < 0 or point.y >= self.H or point.y < 0:
                print(point.x, point.y)
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

    def turn(self):
        self.__move_snake()

        is_out = self.__is_out_of_border()
        apple_eaten = self.__is_apple_eaten()
        ate_himself = self.__is_ate_himself()

        if is_out or ate_himself:
            self.game_over = True
            return
        
        if apple_eaten:
            self.score += 1
            self.apple = self.__spawn_apple()
            self.__grow_snake()
    

