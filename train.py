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


def build_models(board_width, board_length, num_shapes, num_actions,
                 gamma=0.9):
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
    reward = ks.layers.Input(shape=(1,))
    s = ks.layers.Input(shape=BOARD_SHAPE)
    sp = ks.layers.Input(shape=BOARD_SHAPE)

    def _build_model(num_outputs, activation):
        return ks.models.Sequential([
            ks.layers.Conv2D(64, (2, 2), (2, 2), input_shape=BOARD_SHAPE),
            ks.layers.LeakyReLU(),
            ks.layers.BatchNormalization(),
            ks.layers.Conv2D(64, (2, 2), (2, 2)),
            ks.layers.LeakyReLU(),
            ks.layers.BatchNormalization(),
            ks.layers.Flatten(),
            ks.layers.Dense(num_outputs, activation=activation),
        ])

    # Predicts the policy space.
    policy_model = _build_model(num_actions, 'softmax')
    policy = policy_model(s)

    # Predicts the value function.
    value_model = _build_model(1, None)

    # Computes the advantage function.
    sy, spy = value_model(s), value_model(sp)

    # Makes a layer to predict the discounted future reward.
    def _compute_dfr(x):
        reward, spy = x
        return tf.stop_gradient(reward + gamma * spy)
    dfr = ks.layers.Lambda(_compute_dfr)([reward, spy])

    # Gets the value loss as a Keras tensor.
    def _value_loss(x):
        l = ks.losses.get('mse')(*x) * 0.5
        return tf.reshape(l, (-1, 1))
    value_loss = ks.layers.Lambda(_value_loss, name='value')([dfr, sy])

    def _get_index(x):
        '''Selects the index associated with a particular action.'''
        b, a = x
        a = tf.squeeze(a, 1)
        a.set_shape(b.get_shape()[:1])
        inds = tf.stack([tf.range(tf.shape(a)[0]), a], 1)
        pols = tf.reshape(tf.gather_nd(b, inds), (-1, 1))
        return pols

    # Gets the policy probability associated with the action.
    action_prob = ks.layers.Lambda(_get_index)([policy, action])

    # Outputs a layer to compute the policy loss.
    def _policy_loss(x):
        dfr, sy, action_prob = x
        advantage = dfr - sy
        policy_loss = ks.losses.get('binary_crossentropy')
        policy_loss = -tf.stop_gradient(advantage) * tf.log(action_prob + 1e-9)
        return policy_loss
    policy_loss = ks.layers.Lambda(_policy_loss, name='policy')([
        dfr, sy, action_prob
    ])

    # Adds a layer to compute the entropy loss..
    def _entropy_loss(policy):
        ent = -tf.log(policy + 1e-9) * policy
        return -tf.reduce_mean(ent, keep_dims=True) * 0.05
    entropy_loss = ks.layers.Lambda(_entropy_loss, name='entropy')(action_prob)

    # Builds the training model.
    train_model = ks.models.Model(
        [s, sp, action, reward],
        [policy_loss, entropy_loss, value_loss],
    )

    train_model.compile(optimizer='adam', loss=lambda _, i: i)

    # Builds the sampling model.
    sample_model = ks.models.Model(s, policy)

    return train_model, sample_model


def sample_action(q_dist, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randrange(0, len(q_dist))
    else:
        return q_dist.argmax()


def collect_samples(model, engine, nsamples, epsilon=0.9):
    '''Collects samples by running the model.
    Args:
        model (keras model): the model that takes as input the current state
            of the board and predicts a distribution over the expected TD
            rewards for each action.
        engine (tetris engine): the tetris engine to use, which the model
            operates on.
        nsamples (int): number of samples to draw.
        epsilon (float): the epsilon to use for epsilon-greedy sampling.
    Returns:
        samples consisting of state-reward pairs, where the state is the board
        and the rewards are the TD rewards.
    '''

    boards, actions, rewards, next_qs = [], [], [], []
    engine.clear()

    prev_score = engine.score
    for i in range(nsamples + 1):
        engine.set_piece()
        board = get_as_batch(engine.board)
        engine.clear_piece()
        q_dist = model.predict(board).squeeze(0)
        action = sample_action(q_dist, epsilon)

        # Steps the engine with the new action and sees what happens.
        engine.step(action)
        if engine.dead:
            reward = -100.
        else:
            reward = float(engine.score - prev_score)
        prev_score = engine.score

        boards.append(board)
        actions.append(action)
        rewards.append(reward)

    # Gets the "final" q-value.
    board = get_as_batch(engine.board)

    rewards = np.expand_dims(np.array(rewards), -1)
    boards = np.concatenate(boards, 0)
    actions = np.expand_dims(np.array(actions), -1)

    # Normalizes rewards.
    # rewards /= 100.

    return boards, actions, rewards


def train(train_model, sample_model, engine, sample_len, n_epochs,
          initial_eps=0.9, eps_decay=10):
    '''Trains the model by iteratively sampling and predicting TD targets.
    Args:
        todo
    '''
    mean_rewards = []
    sample_idx = 0
    boards, actions, targets = None, None, None
    for epoch in range(1, n_epochs + 1):
        epsilon = initial_eps * math.exp(-sample_idx / eps_decay)
        sample_idx += 1

        # Collects samples following the epsilon-greedy policy.
        boards, actions, rewards = collect_samples(
            sample_model, engine, sample_len,
            epsilon=epsilon,
        )

        mean_rewards.append(rewards.mean())
        print('Rewards:')
        print('\n'.join('  Epoch {}: {:.3f}'.format(i, r)
                        for i, r in enumerate(mean_rewards, 1)))

        # Off-sets by one time step.
        s, sp = boards[:-1], boards[1:]
        actions = actions[:-1]
        rewards = rewards[:-1]

        # Fits the states and actions to their TD targets.
        batch_size = 100
        train_model.fit(
            [s, sp, actions, rewards],
            [np.zeros((s.shape[0],) + i[1:]) for i in train_model.output_shape],
            batch_size=100,
            initial_epoch=epoch - 1,
            epochs=epoch + 5,
            validation_split=0.05,
            callbacks=[ks.callbacks.EarlyStopping()],
        )

        # Evaluates the model after it's done.
        if epoch % 1 == 0:
            start_score = engine.score
            for _ in range(20):
                board = get_as_batch(engine.board)
                q_dist = sample_model.predict(board)
                action = sample_action(q_dist, epsilon)
                engine.step(action)
                print(engine)
                time.sleep(0.05)
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
