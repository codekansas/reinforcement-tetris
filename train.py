#!/usr/bin/env python3
# Run this file to train a model.

import argparse
import math
import random
import time
import warnings
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

from engine import TetrisEngine


def sample_action(q_dist, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randrange(0, len(q_dist))
    else:
        return q_dist.argmax()


def build_models(board_width, board_length, num_shapes, num_actions):
    '''Builds the predictive model.
    Args:
        board_width (int): the width of the playing board.
        board_lenght (int): the length of the playing board.
        num_shapes (int): the number of shapes available.
        num_actions (int): the number of actions available.
    Returns:
        a keras model for predicting the reward over each action from the
        current state.
    '''

    BOARD_SHAPE = (board_width, board_length, 1)

    # Defines the input tensors; action, reward, the board (state), and the
    # board after taking the action.
    action = ks.layers.Input(shape=(1,), dtype='int32')
    s = ks.layers.Input(shape=BOARD_SHAPE)

    q_value_model = ks.models.Sequential([
        ks.layers.Reshape((board_width, board_length),
                          input_shape=BOARD_SHAPE),
        ks.layers.Permute((2, 1)),
        ks.layers.Bidirectional(ks.layers.LSTM(64, return_sequences=True)),
        ks.layers.Bidirectional(ks.layers.LSTM(64, return_sequences=False)),
        # ks.layers.LSTM(128, return_sequences=True),
        # ks.layers.LSTM(128, return_sequences=False),
        ks.layers.Dense(num_actions),
    ])

    # q_value_model = ks.models.Sequential([
    #     ks.layers.Conv2D(128, (4, 4), (2, 2), padding='valid',
    #                      input_shape=BOARD_SHAPE),
    #     ks.layers.Conv2D(128, (4, 4), (2, 2), padding='valid'),
    #     ks.layers.Conv2D(128, (1, 3), (1, 1), padding='valid'),
    #     ks.layers.LeakyReLU(),
    #     ks.layers.BatchNormalization(),
    #     ks.layers.Flatten(),
    #     ks.layers.Dense(num_actions),
    # ])

    # q_value_model = ks.models.Sequential([
    #     ks.layers.Flatten(input_shape=BOARD_SHAPE),
    #     ks.layers.Dense(512),
    #     ks.layers.LeakyReLU(),
    #     ks.layers.BatchNormalization(),
    #     ks.layers.Dense(512),
    #     ks.layers.LeakyReLU(),
    #     ks.layers.BatchNormalization(),
    #     ks.layers.Dense(32),
    #     ks.layers.LeakyReLU(),
    #     ks.layers.BatchNormalization(),
    #     ks.layers.Dense(num_actions),
    # ])

    print(q_value_model.summary())

    # Applies the Q value model to the state.
    q_values = q_value_model(s)

    def _get_index(x):
        '''Selects the index associated with a particular action.'''
        b, a = x
        a = tf.squeeze(a, 1)
        a.set_shape(b.get_shape()[:1])
        inds = tf.stack([tf.range(tf.shape(a)[0]), a], 1)
        pols = tf.reshape(tf.gather_nd(b, inds), (-1, 1))
        return pols

    # Gets the Q value associated with the action.
    action_q = ks.layers.Lambda(_get_index)([q_values, action])

    # Builds the training model.
    train_model = ks.models.Model([s, action], action_q, name='train_model')

    # Adds a regularizing penalty to make actions equally likely.
    q_mean_diff = q_values - tf.reduce_mean(q_values, axis=1, keepdims=True)
    q_reg_loss = ks.regularizers.l1()(q_mean_diff)
    train_model.add_loss(q_reg_loss * 1e-3)

    # Compiles the training model; reduce error between Q values.
    train_model.compile(
        # optimizer=ks.optimizers.SGD(lr=1e-2),
        optimizer=ks.optimizers.Nadam(),
        loss='mae',
    )

    # Builds the sampling model.
    sample_model = ks.models.Model(s, q_values, name='sample_model')

    return train_model, sample_model


def collect_samples(model, engines, nsamples, gamma=0.95):
    '''Collects samples by running the model.
    Args:
        model (keras model): the model that takes as input the current state
            of the board and predicts a distribution over the expected TD
            rewards for each action.
        engines (list of tetris engines): the tetris engine to use, which the
            model operates on.
        nsamples (int): number of samples to draw.
    Returns:
        samples consisting of state-reward pairs, where the state is the board
        and the rewards are the TD rewards.
    '''

    all_boards, all_actions, all_rewards = [], [], []

    for i in range(nsamples):
        boards = np.concatenate(tuple(map(
            lambda e: e.get_board(),
            engines,
        )))
        q_dist = model.predict(boards)  # Prediction over Q values.

        # Samples the next action.
        epsilon = 0.9 ** (engines[0].time / 100)
        actions = np.array(tuple(map(
            lambda d: sample_action(d, epsilon),
            q_dist,
        )))

        # Steps the engine with the new action and sees what happens.
        rewards = np.array(tuple(map(
            lambda engine, action: engine.step(action),
            engines,
            actions,
        )))

        all_boards.append(boards)
        all_actions.append(actions)
        all_rewards.append(rewards)

    # Gets the model's maximum Q value for the subsequent board.
    boards = np.concatenate(list(map(lambda e: e.get_board(), engines)))
    q_dist = model.predict(boards)
    actions = q_dist.argmax(axis=1)
    rewards = np.array(list(map(
        lambda engine, action: engine.step(action),
        engines,
        actions,
    )))

    # Accumulates the rewards backwards.
    for i in range(len(all_rewards) - 1, -1, -1):
        rewards = all_rewards[i] + (gamma * rewards)
        all_rewards[i] = rewards

    # Converts the various lists to arrays with the correct dimensions.
    all_rewards = np.array(all_rewards).reshape(-1, 1)
    all_boards = np.array(all_boards).reshape(-1, *all_boards[0].shape[1:])
    all_actions = np.array(all_actions).reshape(-1, 1)

    # for i in range(10):
    #     board = all_boards[i].squeeze().transpose(1, 0)
    #     s = '\n'.join(''.join('X' if i else '.' for i in b) for b in board)
    #     print(s)
    #     print('Reward: {:.3f}'.format(all_rewards[i, 0]))

    return all_boards, all_actions, all_rewards


def train(train_model, sample_model, engines, sample_len, n_epochs,
          model_save_loc):
    '''Trains the model by iteratively sampling and predicting TD targets.
    Args:
        todo
    '''

    mean_rewards = []
    boards, actions, targets = None, None, None

    for epoch in range(1, n_epochs + 1):
        boards, actions, rewards = collect_samples(
            sample_model, engines, sample_len,
        )

        mean_rewards.append(rewards.mean())
        print('Rewards: ' + ' -- '.join(
            'Epoch {}: {:.3f}'.format(i + 1, mean_rewards[i])
            for i in range(max(len(mean_rewards) - 5, 0), len(mean_rewards))))

        # Fits the states and actions to their TD targets.
        train_model.fit(
            [boards, actions],
            rewards,
            batch_size=100,
            initial_epoch=epoch - 1,
            epochs=epoch + 20,
            validation_split=0.05,
            callbacks=[ks.callbacks.EarlyStopping('val_loss')],
        )

        # Evaluates the model after it's done.
        if epoch % 10 == 0:
            for _ in range(200):
                board = engines[0].get_board()
                q_dist = sample_model.predict(board)
                action = q_dist.squeeze(0).argmax()
                print(engines[0])
                print('Score: {}'.format(engines[0].score))
                print('Action: {}'.format(action))
                print('Q Values: {:.3f}'.format(q_dist.squeeze(0)[action]))
                reward = engines[0].step(action)
                print('Reward: {}'.format(reward))
                time.sleep(0.05)
        print('Total lines cleared: {}'.format(engines[0].cleared_lines))
        print('Total deaths: {}'.format(engines[0].deaths))

        train_model.save_weights(model_save_loc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Tetris RL agent!')
    parser.add_argument('-m', '--model-save-loc',
                        help='Where to save the model',
                        metavar='LOC', type=str, default='model.h5')
    parser.add_argument('-w', '--width',
                        help='Horizontal width of the board',
                        metavar='W', type=int, default=10)
    parser.add_argument('-l', '--length',
                        help='Vertical length of the board',
                        metavar='L', type=int, default=20)
    parser.add_argument('-s', '--sample-len',
                        help='Length of each sampling cycle',
                        metavar='S', type=int, default=100)
    parser.add_argument('-e', '--epoch-len',
                        help='Number of training epochs',
                        metavar='E', type=int, default=1000)
    parser.add_argument('-n', '--num-engines',
                        help='Number of simultaneous training engines',
                        metavar='N', type=int, default=100)
    args = parser.parse_args()

    # Initializes the engines.
    engines = [TetrisEngine(args.width, args.length)
               for _ in range(args.num_engines)]

    # Builds the various models.
    train_model, sample_model = build_models(
        args.width, args.length,
        len(engines[0].shapes), len(engines[0].actions),
    )

    # Attempts to load existing weights, if they exist.
    if os.path.exists(args.model_save_loc):
        try:
            train_model.load_weights(args.model_save_loc)
        except:
            warnings.warn('Could not load weights from {}; presuming new run.'
                          .format(args.model_save_loc))

    train(train_model, sample_model, engines,
          args.sample_len, args.epoch_len, args.model_save_loc)
