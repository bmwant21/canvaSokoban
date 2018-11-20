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
        initial_games=10000,
        test_games=1000,
        goal_steps=2000,
        lr=1e-2,
        filename='game_nn_1.tflearn',
    ):
        self.game_cls = game_cls
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.vectors_and_keys = [
            [[-1, 0], 0],
            [[0, 1], 1],
            [[1, 0], 2],
            [[0, -1], 3]
        ]

    def initial_population(self):
        """
        Generate training data with random games
        """
        training_data = []
        logger.debug('Playing some random games '
                     'to create initial training data')
        for _ in range(self.initial_games):
            game = self.game_cls.create_game()
            state = game.start()
            prev_observation = self.generate_observation(state)
            prev_score = state.score
            # values that should influence rewarding score
            # distance = self.get_distance(state)
            # max steps should be MxN, corresponding to the size of the board
            for _ in range(game.max_steps):
                # generate random move
                action = self.generate_action(state)
                # done, score, snake, food = self.game.step(action)
                new_state = game.step(action)
                if new_state.failed:
                    # punish
                    training_data.append([self.add_action_to_observation(), -1])
                    break
                elif new_state.score > prev_score:
                    # still can make it, but score may vary
                    # reward
                    training_data.append([self.add_action_to_observation(), 0])
                else:
                    # bumped into a wall
                    training_data.append([self.add_action_to_observation(), 0])

                prev_observation = self.generate_observation(state)
                prev_score = new_state.score

        return training_data

    def generate_action(self, state: State) -> Position:
        """
        Get new random position on the board from the available
        directions where we want move to
        """
        index = randint(0, 3)
        direction = list(DIRECTIONS.keys())[index]
        return state.current_position + DIRECTIONS[direction]

    def get_game_action(self, snake, action):
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction
        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)
        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
                return game_action

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

        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        angle = self.get_angle(snake_direction, food_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right), angle])

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def get_food_direction_vector(self, snake, food):
        return np.array(food) - np.array(snake[0])

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X,y, n_epoch = 3, shuffle = True, run_id = self.filename)
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

    def visualise_game(self, model):
        game = SnakeGame(gui=True)
        _, _, snake, food = game.start()
        prev_observation = self.generate_observation(snake, food)
        for _ in range(self.goal_steps):
            precictions = []
            for action in range(-1, 2):
                precictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
            action = np.argmax(np.array(precictions))
            game_action = self.get_game_action(snake, action - 1)
            done, _, snake, food  = game.step(game_action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(snake, food)

    def train(self):
        training_data = self.initial_population()
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        self.test_model(nn_model)

    def visualise(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.visualise_game(nn_model)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)


if __name__ == '__main__':
    # game_instance = Game()
    network = GameNetwork()
    network.train()
