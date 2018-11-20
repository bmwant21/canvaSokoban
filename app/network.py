"""
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp36-cp36m-linux_x86_64.whl
pip install $TF_BINARY_URL
pip install tflearn
"""
import math
import numpy as np
from random import randint
from statistics import mean
from collections import Counter

import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

from app.utils import logger
from app.game import Game, State
from app.field import Position, DIRECTIONS


class GameNetwork(object):
    def __init__(
        self,
        game_cls,
        initial_games=20000,
        test_games=1000,
        lr=1e-2,
        filename='game_nn_1.tflearn',
    ):
        self.game_cls = game_cls
        self.initial_games = initial_games
        self.test_games = test_games
        self.lr = lr
        self.filename = filename
        self.features_len = 13
        self.n_epochs = 10

    def initial_population(self):
        """
        Generate training data with random games
        """
        training_data = []
        logger.debug('Playing %s random games '
                     'to create initial training data', self.initial_games)
        for _ in range(self.initial_games):
            game = self.game_cls.create_game()
            state = game.start()
            prev_observation = self.generate_observation(state)
            prev_score = state.score
            # values that should influence rewarding score
            prev_distance = state.distance
            # max steps should be MxN, corresponding to the size of the board
            for _ in range(game.max_steps):
                # generate random move
                action_value, action = self.generate_action(state)
                new_state = game.step(action)
                distance = state.distance

                data = self.add_action_to_observation(
                    action_value, prev_observation)

                if new_state.moves_left == 0 and not new_state.failed:
                    logger.info('At least one correct game!')
                    training_data.append([data, 1])

                if new_state.failed:
                    # punish
                    training_data.append([data, -1])
                    break
                elif new_state.score > prev_score:
                    # still can make it, but score may vary
                    # reward
                    if distance < prev_distance:
                        training_data.append([data, 1])
                    else:
                        training_data.append([data, 0])
                else:
                    # bumped into a wall
                    training_data.append([data, -1])

                prev_observation = self.generate_observation(state)
                prev_score = new_state.score

        return training_data

    def generate_action(self, state: State) -> (int, Position):
        """
        Get new random position on the board from the available
        directions where we want move to
        """
        index = randint(0, 3)
        return index, self.get_game_action(state, action_index=index)

    def get_game_action(self, state: State, action_index: int) -> Position:
        direction = list(DIRECTIONS.keys())[action_index]
        return state.current_position + DIRECTIONS[direction]

    def generate_observation(self, state):
        # Get all the available features we can extract from the game
        # barrier left, score left +
        # barrier right, score right +
        # barrier top, score top +
        # barrier bottom score bottom +
        # distance x +
        # distance y +
        # distance (it's actually an x+y sum), redundant? +
        # moves left +
        return np.array([
            state.distance_x,
            state.distance_y,
            state.distance,
            state.moves_left,
            state.barrier_up,
            state.barrier_down,
            state.barrier_left,
            state.barrier_right,
            state.value_up,
            state.value_down,
            state.value_left,
            state.value_right,
        ])

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def model(self):
        network = input_data(shape=[None, self.features_len, 1], name='input')
        network = fully_connected(
            network,
            self.features_len**2,
            activation='relu'
        )
        network = fully_connected(network, 1, activation='linear')
        network = regression(
            network,
            optimizer='adam',
            learning_rate=self.lr,
            loss='mean_square',
            name='target',
        )
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data, model):
        X = np.array([row[0] for row in training_data]) \
            .reshape(-1, self.features_len, 1)
        y = np.array([row[1] for row in training_data]).reshape(-1, 1)
        logger.debug('Training network with %s epochs', self.n_epochs)
        model.fit(X,y, n_epoch=self.n_epochs, shuffle=True, run_id=self.filename)
        model.save(self.filename)
        return model

    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            game = SnakeGame()
            _, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                    predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(snake, action - 1)
                done, score, snake, food  = game.step(game_action)
                game_memory.append([prev_observation, action])
                if done:
                    print('-----')
                    print(steps)
                    print(snake)
                    print(food)
                    print(prev_observation)
                    print(predictions)
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        print('Average steps:', mean(steps_arr))
        print(Counter(steps_arr))
        print('Average score:', mean(scores_arr))
        print(Counter(scores_arr))

    def play_game(self, model, game):
        _path = []
        state = game.start()
        prev_observation = self.generate_observation(state)
        while not state.failed and state.moves_left:
            predictions = []
            # which move is the best? move index lives on [0, 3] segment
            for action in range(4):
                data = self.add_action_to_observation(
                    prev_observation, action).reshape(-1, self.features_len, 1)
                prediction = model.predict(data)
                predictions.append(prediction)
            action = np.argmax(np.array(predictions))
            direction = list(DIRECTIONS.keys())[action]
            game_action = self.get_game_action(state, action)
            state = game.step(game_action)
            _path.append(direction)
            prev_observation = self.generate_observation(state)

        if not state.failed:
            logger.info('You win the game!')
        logger.info('End the game with a score: %s', state.score)
        logger.info('Path: %s', ','.join(_path))
        return state

    def train(self):
        training_data = self.initial_population()
        model = self.train_model(training_data, self.model())
        # self.test_model(nn_model)
        return model

    def load_trained_model(self):
        model = self.model()
        model.load(self.filename)
        return model
        # self.visualise_game(nn_model)

    def test(self):
        model = self.model()
        model.load(self.filename)
        self.test_model(model)


if __name__ == '__main__':
    # create random game
    game_instance = Game.create_game()
    network = GameNetwork(
        game_cls=Game,
    )
    model = network.train()
    # model = network.load_trained_model()
    print(game_instance)
    state = network.play_game(model=model, game=game_instance)
    print('Win ?', not state.failed)
    print('Finish position', state.current_position)
    print('Moves left', state.moves_left)
