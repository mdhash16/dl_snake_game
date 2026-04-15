from snake_ai import Game_AI
from agent    import Agent

MODEL_PATH = 'model.pth'


def train():
    game   = Game_AI()
    agent  = Agent()
    record = 0

    while True:
        # 1. Current state (via InputLayer)
        state = agent.get_state(game)

        # 2. Choose action  [straight, right, left]
        action = agent.get_action(state)

        # 3. Step the game
        reward, done, score = game.turn(action)

        # 4. State after the step
        next_state = agent.get_state(game)

        # 5. Train on this single transition
        agent.train_short_memory(state, action, reward, next_state, done)

        # 6. Store in replay buffer
        agent.remember(state, action, reward, next_state, done)

        if done:
            agent.train_long_memory()
            agent.n_games += 1
            game.reset()

            if score > record:
                record = score
                agent.model.save(MODEL_PATH)

            print(f"Game {agent.n_games:4d} | Score: {score:3d} | Record: {record:3d}")


if __name__ == "__main__":
    train()
