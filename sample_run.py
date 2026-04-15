from snake import Game
from snake import Direction

import pygame

pygame.init()

white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

dis_width = 600
dis_height = 600


dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake Game Kurwa')
game_instance = Game()

block_size = (dis_width / game_instance.W)
snake_speed = 15

clock = pygame.time.Clock()

font_style = pygame.font.SysFont("bahnschrift", 15)
score_font = pygame.font.SysFont("comicsansms", 15)

def draw_score(score):
    value = score_font.render("Your Score: " + str(score), True, yellow)
    dis.blit(value, [0, 0])

def draw_snake(snake):
    for p in snake:
        pygame.draw.rect(dis, white, 
                        [p.x * block_size, 
                        p.y * block_size, 
                        block_size, 
                        block_size])

def game_loop():
    while game_instance.game_over != True:
        game_instance.turn()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_instance.game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if game_instance.direction == Direction.RIGHT:
                        continue
                    game_instance.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    if game_instance.direction == Direction.LEFT:
                        continue
                    game_instance.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    if game_instance.direction == Direction.DOWN:
                        continue
                    game_instance.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    if game_instance.direction == Direction.UP:
                        continue
                    game_instance.direction = Direction.DOWN
        dis.fill(black)
        draw_snake(game_instance.snake)
        pygame.draw.rect(dis, red, 
                        [game_instance.apple.x * block_size, 
                        game_instance.apple.y * block_size, 
                        block_size, 
                        block_size])
        draw_score(game_instance.score)
        pygame.display.update()
        clock.tick(snake_speed)
    print("Game over")
    pygame.quit()
    quit()    
game_loop()