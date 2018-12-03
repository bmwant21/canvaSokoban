"""
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp36-cp36m-linux_x86_64.whl
pip install $TF_BINARY_URL
pip install tflearn
"""
import math
from random import randint
from statistics import mean
from collections import Counter

import tflearn
import numpy as np
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

from app.utils import logger
from app.game import Game, State
from app.field import Position, DIRECTIONS


class GameNetwork(object):
    def __init__(
        self,
        game_cls,
        initial_games=10000,
        test_games=1000,
        lr=1e-2,
        filename='game_nn_1.tflearn',
    ):
        self.game_cls = game_cls
        self.initial_games = initial_games
        self.test_games = test_games
        self.lr = lr
        self.filename = filename
        self.features_len = 8
        self.n_epochs = 4

    def initial_population(self):
        """
        Generate training data with random games
        """
        training_data = []
        logger.debug('Playing %s random games '
                     'to create initial training data', self.initial_games)
        for _ in range(self.initial_games):
        # for _ in range(1):
            game = self.game_cls.create_game_v2()
            prev_state = game.start()
            prev_observation = self.generate_observation_v2(prev_state, None)
            # values that should influence rewarding score
            prev_distance = prev_state.distance
            # max steps should be MxN, corresponding to the size of the board
            for _ in range(game.max_steps):
                # generate random move
                action_value, action = self.generate_action(prev_state)
                new_state = game.step(action)
                distance = new_state.distance

                data = self.add_action_to_observation(
                    action_value, prev_observation)
                # print(data)
                if new_state.failed:
                    # punish
                    training_data.append([data, -1])
                    break
                elif distance < prev_distance:
                    training_data.append([data, 1])
                else:
                    # bumped into a wall or wrong direction
                    training_data.append([data, 0])

                direction = list(DIRECTIONS.keys())[action_value]
                move_vector = DIRECTIONS[direction]
                prev_observation = self.generate_observation_v2(
                    prev_state, move_vector
                )
                prev_state = new_state
        # for o in training_data:
        #     print(' '.join([str(i) for i in o]))
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

    def generate_observation_v1(self, state):
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

    def generate_observation_v2(self, state, move_vector=None):
        angle = self.get_angle(state, move_vector)
        evaluate_directions = [
            self._die_from_here(state.current_position+mv, state)
            for mv in DIRECTIONS.values()
        ]
        return np.array([
            *evaluate_directions,
            state.distance_x,
            state.distance_y,
            angle
        ])

    def _die_from_here(self, position, state) -> bool:
        return position.steps_to(state.finish_position) <= state.moves_left

    def normalize_vector(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / np.linalg.norm(v)

    def get_angle(self, state, move_vector):
        if move_vector is None:
            move_vector = state.current_position
        direction_vector = (state.current_position + move_vector).to_vector()
        finish_vector = (state.finish_position -
                         state.current_position).to_vector()
        a = self.normalize_vector(direction_vector)
        b = self.normalize_vector(finish_vector)
        return math.atan2(a[0]*b[1] - a[1]*b[0],
                          a[0]*b[0] + a[1]*b[1]) / math.pi

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def model(self):
        # input
        network = input_data(shape=[None, self.features_len, 1], name='input')
        # hidden layer
        network = fully_connected(
            network,
            self.features_len*2,  # somewhere between size of the input and output layer
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
        model = tflearn.DNN(
            network,
            tensorboard_dir='log',
            tensorboard_verbose=3,
        )
        return model

    def train_model(self, training_data, model):
        X = np.array([row[0] for row in training_data]) \
            .reshape(-1, self.features_len, 1)
        y = np.array([row[1] for row in training_data]).reshape(-1, 1)
        logger.debug('Training network with %s epochs', self.n_epochs)
        model.fit(X, y, n_epoch=self.n_epochs, shuffle=True, run_id=self.filename)
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
        prev_observation = self.generate_observation_v2(state)
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
            move_vector = DIRECTIONS[direction]
            game_action = self.get_game_action(state, action)
            state = game.step(game_action)
            _path.append(direction)
            prev_observation = self.generate_observation_v2(state, move_vector)

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
    game_instance = Game.create_game_v2()
    network = GameNetwork(
        game_cls=Game,
    )
    model = network.train()
    # model = network.load_trained_model()
    print(game_instance)
    # Get resulting state
    st = network.play_game(model=model, game=game_instance)
    print('Win ?', not st.failed)
    print('Finish position', st.current_position)
    print('Moves left', st.moves_left)

    # is dying when going to the left/right/up/down
    # normalized angle between start and end
    # output - suggested direction

    # is the step was successful (distance becomes shorter)
    # -1 didn't survive
    # 0 survived direction is wrong
    # 1 survived direction is right

    # Min loss for now 0.35963
