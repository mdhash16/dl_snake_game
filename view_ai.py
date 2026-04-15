import os
import sys
import pygame
import torch

from snake_ai import Game_AI
from model    import LinearQNet
from agent    import Agent

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = 'model.pth'
GRID       = 30                  # game grid is 30×30
WIN_W      = 700                 # extra 100 px on right for stats panel
WIN_H      = 600
BLOCK      = WIN_H // GRID       # 20 px per cell
FPS_TRAIN  = 60                  # fast during training
FPS_PLAY   = 12                  # slow enough to watch

# ── Colors ─────────────────────────────────────────────────────────────────────
BLACK  = (  0,   0,   0)
WHITE  = (255, 255, 255)
RED    = (213,  50,  80)
YELLOW = (255, 255, 102)
GREEN  = (  0, 200,   0)
DARK   = ( 20,  20,  20)
GRAY   = (120, 120, 120)


# ── Renderer ───────────────────────────────────────────────────────────────────

class Renderer:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock  = pygame.time.Clock()
        self.big    = pygame.font.SysFont('bahnschrift', 32)
        self.sm     = pygame.font.SysFont('bahnschrift', 18)
        pygame.display.set_caption('Snake AI')

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _text(self, font, text, color, center=None, topleft=None):
        surf = font.render(text, True, color)
        pos  = surf.get_rect(center=center) if center else surf.get_rect(topleft=topleft)
        self.screen.blit(surf, pos)

    def _draw_game(self, game):
        for p in game.snake:
            pygame.draw.rect(self.screen, WHITE,
                (p.x * BLOCK, p.y * BLOCK, BLOCK - 1, BLOCK - 1))
        pygame.draw.rect(self.screen, RED,
            (game.apple.x * BLOCK, game.apple.y * BLOCK, BLOCK - 1, BLOCK - 1))

    def _sidebar(self, lines):
        """Draw a dark panel on the right with lines of text."""
        panel_x = WIN_H          # game area is WIN_H × WIN_H (600×600)
        pygame.draw.rect(self.screen, DARK, (panel_x, 0, WIN_W - panel_x, WIN_H))
        for i, (label, value, color) in enumerate(lines):
            y = 30 + i * 60
            self._text(self.sm,  label, GRAY,   topleft=(panel_x + 10, y))
            self._text(self.big, value, color,   topleft=(panel_x + 10, y + 18))

    def _score_bars(self, scores):
        """Mini bar chart at the bottom of the game area (last 60 scores)."""
        if not scores:
            return
        recent  = scores[-60:]
        max_s   = max(recent) or 1
        bar_w   = WIN_H // 60          # 10 px wide per bar
        strip_h = 35
        y_base  = WIN_H - 1
        for i, s in enumerate(recent):
            h = max(1, int(s / max_s * strip_h))
            pygame.draw.rect(self.screen, GREEN, (i * bar_w, y_base - h, bar_w - 1, h))

    # ── Public screens ─────────────────────────────────────────────────────────

    def show_menu(self):
        """Draw menu and return 'train' or 'play' based on key press."""
        has_model = os.path.exists(MODEL_PATH)
        cx = WIN_W // 2

        while True:
            self.screen.fill(BLACK)
            self._text(self.big, 'Snake AI',              YELLOW, center=(cx, 160))
            self._text(self.sm,  '[T]  Train new model',  WHITE,  center=(cx, 280))
            self._text(self.sm,  '[P]  Play saved model',
                       WHITE if has_model else GRAY,              center=(cx, 330))
            if not has_model:
                self._text(self.sm, 'no model.pth found', RED,   center=(cx, 370))
            self._text(self.sm, '[Esc] quit',             GRAY,  center=(cx, 500))
            pygame.display.flip()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._exit()
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        self._exit()
                    if ev.key == pygame.K_t:
                        return 'train'
                    if ev.key == pygame.K_p and has_model:
                        return 'play'

    def draw_training(self, game, n_games, score, record, scores):
        self.screen.fill(BLACK)
        self._draw_game(game)
        self._score_bars(scores)
        self._sidebar([
            ('GAME',   str(n_games), WHITE),
            ('SCORE',  str(score),   YELLOW),
            ('RECORD', str(record),  GREEN),
        ])
        pygame.display.flip()

    def draw_play(self, game, record):
        self.screen.fill(BLACK)
        self._draw_game(game)
        self._sidebar([
            ('SCORE',  str(game.score), YELLOW),
            ('RECORD', str(record),     GREEN),
        ])
        pygame.display.flip()

    def tick(self, fps):
        self.clock.tick(fps)

    def handle_events(self):
        """Process window events. Returns True if Escape was pressed."""
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self._exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return True
        return False

    def _exit(self):
        pygame.quit()
        sys.exit()


# ── Modes ──────────────────────────────────────────────────────────────────────

def run_training(renderer):
    game   = Game_AI()
    agent  = Agent()
    record = 0
    scores = []            # history of per-game scores for the bar chart

    while True:
        if renderer.handle_events():
            return         # Escape → back to menu

        state  = agent.get_state(game)
        action = agent.get_action(state)

        reward, done, score = game.turn(action)

        next_state = agent.get_state(game)
        agent.train_short_memory(state, action, reward, next_state, done)
        agent.remember(state, action, reward, next_state, done)

        renderer.draw_training(game, agent.n_games, score, record, scores)
        renderer.tick(FPS_TRAIN)

        if done:
            agent.train_long_memory()
            agent.n_games += 1
            scores.append(score)

            if score > record:
                record = score
                agent.model.save(MODEL_PATH)   # save only when a new record is set

            game.reset()


def run_play(renderer):
    model  = LinearQNet.load(MODEL_PATH)
    game   = Game_AI()
    record = 0

    while True:
        if renderer.handle_events():
            return         # Escape → back to menu

        state   = game.input_layer.get_state(game.snake, game.direction, game.apple)
        q_vals  = model(torch.tensor(state, dtype=torch.float))
        action  = [0, 0, 0]
        action[torch.argmax(q_vals).item()] = 1

        _, done, score = game.turn(action)

        renderer.draw_play(game, record)
        renderer.tick(FPS_PLAY)

        if done:
            if score > record:
                record = score
            game.reset()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    renderer = Renderer()
    while True:
        choice = renderer.show_menu()
        if choice == 'train':
            run_training(renderer)
        elif choice == 'play':
            run_play(renderer)


if __name__ == '__main__':
    main()
