#!/usr/bin/env python3
# Run this file to train a model.

import argparse
import math
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

from engine import TetrisEngine


def get_as_batch(arr):
    w, l = arr.shape
    return np.copy(arr.reshape(1, w, l, 1))


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
    b = ks.layers.Input(shape=(board_width, board_length, 1))
    d = ks.layers.Input(shape=(board_width, board_length, 1))

    x = ks.layers.Conv2D(32, (board_width, 2))(b)
    x = ks.layers.LeakyReLU()(x)
    x = ks.layers.Dropout(0.3)(x)
    x = ks.layers.Conv2D(32, (1, 2))(x)
    x = ks.layers.LeakyReLU()(x)
    x = ks.layers.Dropout(0.3)(x)

    y = ks.layers.Conv2D(32, (board_width, 2))(d)
    y = ks.layers.LeakyReLU()(y)
    y = ks.layers.Dropout(0.3)(y)
    y = ks.layers.Conv2D(32, (1, 2))(y)
    y = ks.layers.LeakyReLU()(y)
    y = ks.layers.Dropout(0.3)(y)

    # Combines and flattens the layer.
    x = ks.layers.concatenate([x, y])
    x = ks.layers.Flatten()(x)

    x = ks.layers.Dense(512)(x)
    x = ks.layers.LeakyReLU()(x)
    x = ks.layers.Dropout(0.3)(x)
    x = ks.layers.Dense(num_actions)(x)
    sample_model = ks.models.Model([b, d], x)

    # Compiles a separate model for training.
    def _get_index(x):
        b, a = x
        a = tf.reshape(a, (-1,))
        inds = tf.stack([tf.range(tf.shape(a)[0]), a], 1)
        return tf.reshape(tf.gather_nd(b, inds), (-1, 1))

    def maxr(y_true, _):
        return tf.reduce_max(y_true)

    def minr(y_true, _):
        return tf.reduce_min(y_true)

    a = ks.layers.Input(shape=(1,), dtype='int32')
    x = ks.layers.Lambda(_get_index)([x, a])
    train_model = ks.models.Model([b, d, a], x)
    train_model.compile(optimizer='adam', loss='mae', metrics=[maxr, minr])

    return train_model, sample_model


def sample_action(q_dist, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randrange(0, len(q_dist))
    else:
        return q_dist.argmax()


def collect_samples(model, engine, nsamples, epsilon=0.9, gamma=0.95):
    '''Collects samples by running the model.
    Args:
        model (keras model): the model that takes as input the current state
            of the board and predicts a distribution over the expected TD
            rewards for each action.
        engine (tetris engine): the tetris engine to use, which the model
            operates on.
        nsamples (int): number of samples to draw.
        epsilon (float): the epsilon to use for epsilon-greedy sampling.
        gamma (float): the discount rate for future rewards.
    Returns:
        samples consisting of state-reward pairs, where the state is the board
        and the rewards are the TD rewards.
    '''

    boards, dummies, actions, rewards, next_qs = [], [], [], [], []
    engine.clear()

    prev_score = 0
    for i in range(nsamples):
        engine.set_piece(on=True, on_dummy=True)
        board = get_as_batch(engine.board)
        dummy = get_as_batch(engine.dummy)
        q_dist = model.predict([board, dummy]).squeeze(0)
        action = sample_action(q_dist, epsilon)
        engine.set_piece(on=False, on_dummy=True)

        # Steps the engine with the new action and sees what happens.
        engine.step(action)
        if engine.dead:
            reward = -3
        else:
            reward = (engine.score - prev_score) / 100  # Normalization factor.
        prev_score = engine.score

        boards.append(board)
        dummies.append(dummy)
        actions.append(action)
        rewards.append(reward)

    # Gets the "final" q-value.
    board = get_as_batch(engine.board)
    dummy = get_as_batch(engine.dummy)
    rewards.append(model.predict([board, dummy]).max())

    # Computes the targets by reverse-accumulating.
    for i in range(len(rewards) - 2, -1, -1):
        rewards[i] += rewards[i+1] * gamma

    rewards = np.expand_dims(np.array(rewards[:-1]), -1)
    boards = np.concatenate(boards, 0)
    dummies = np.concatenate(dummies, 0)
    actions = np.expand_dims(np.array(actions), -1)

    return boards, dummies, actions, rewards


def train(train_model, sample_model, engine, sample_len, n_epochs,
          initial_eps=0.9, eps_decay=10):
    '''Trains the model by iteratively sampling and predicting TD targets.
    Args:
        todo
    '''
    sample_idx = 0
    boards, dummies, actions, targets = None, None, None, None
    for epoch in range(1, n_epochs + 1):
        epsilon = initial_eps * math.exp(-sample_idx / eps_decay)
        sample_idx += 1

        # Collects samples following the epsilon-greedy policy.
        boards, dummies, actions, targets = collect_samples(
            sample_model, engine, sample_len,
            epsilon=epsilon,
        )

        # Fits the states and actions to their TD targets.
        train_model.fit(
            [boards, dummies, actions],
            targets,
            batch_size=100,
            initial_epoch=epoch - 1,
            epochs=epoch + 5,
            validation_split=0.05,
            callbacks=[ks.callbacks.EarlyStopping()],
        )

        # Evaluates the model after it's done.
        if epoch % 10 == 0:
            start_score = engine.score
            for _ in range(100):
                board = get_as_batch(engine.board)
                dummy = get_as_batch(engine.dummy)
                q_dist = sample_model.predict([board, dummy])
                action = sample_action(q_dist, epsilon)
                engine.step(action)
                print(engine)
                time.sleep(0.01)
            print('Score gain: {}'.format(engine.score - start_score))
        print('Total lines cleared: {}'.format(engine.cleared_lines))
        print('Total deaths: {}'.format(engine.deaths))


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
                        metavar='S', type=int, default=10000)
    parser.add_argument('-e', '--epoch-len',
                        help='Number of training epochs',
                        metavar='E', type=int, default=1000)
    args = parser.parse_args()

    engine = TetrisEngine(args.width, args.length)
    train_model, sample_model = build_models(
        args.width, args.length, len(engine.shapes), len(engine.actions))
    train(train_model, sample_model, engine,
          args.sample_len, args.epoch_len)
