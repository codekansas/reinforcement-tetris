#!/usr/bin/env python3
# Run this file to run the engine according to an analytic policy.

import argparse
import math
import random
import time
import warnings
import os

import numpy as np

from engine import TetrisEngine


def compute_dropped_score_height(engine, good_ijs, cleared):
    '''Computes a score based on the resulting height.'''
    score = 0
    for i, c in enumerate(engine.board):
        buf = 0
        for j, v in enumerate(c):
            if j in cleared:
                buf += 1
            elif v or (i, j) in good_ijs:
                score += engine.height - j - buf
    return score


def compute_dropped_score_left(engine, good_ijs, cleared):
    '''Computes a score based on how far left the dropped pieces are.'''
    score = 0
    board = engine.board.T
    for j, c in enumerate(board):
        if j in cleared:
            continue
        flag = None
        for i, v in enumerate(c):
            v = v or (i, j) in good_ijs
            if flag is not None and flag != v:
                score += 1
            flag = v
    return score


def compute_dropped_score_holes(engine, good_ijs, cleared):
    '''Computes a score based on the number of "holes" that are created.'''
    score = 0
    for i, c in enumerate(engine.board):
        flag = 0
        for j, v in enumerate(c):
            if j in cleared:
                continue
            v = v or (i, j) in good_ijs
            if flag == 0 and v:
                flag = 1
            elif flag == 1 and not v:
                flag = 2
            if flag == 2:
                score += 1
    return score


def compute_combined_heuristic(engine, good_ijs, cleared):
    height_score = compute_dropped_score_height(engine, good_ijs, cleared)
    holes_score = compute_dropped_score_holes(engine, good_ijs, cleared)
    left_score = compute_dropped_score_left(engine, good_ijs, cleared)
    return (
        holes_score +
        height_score * 1e-4 +
        left_score
    )


def compute_dropped_score(engine, board, shape, anchor):
    shape, anchor = engine.actions.hard_drop(shape, anchor, board)
    good_ijs = set((anchor[0] + s[0], anchor[1] + s[1]) for s in shape)
    score = 0

    # Finds lines that are cleared.
    board_t = board.T
    cleared = set(
        j
        for j, c in enumerate(board_t)
        if all(v or (i, j) in good_ijs for i, v in enumerate(c))
    )

    # return compute_dropped_score_height(engine, good_ijs, cleared)
    # return compute_dropped_score_holes(engine, good_ijs, cleared)
    return compute_combined_heuristic(engine, good_ijs, cleared)


def compute_helper(engine, shape, anchor, action):
    board = engine.board
    actions = []
    min_score = 10000000
    for i in range(engine.width):
        new_score = compute_dropped_score(engine, board, shape, anchor)
        if new_score < min_score:
            actions = [action] * i + [engine.actions.HARD_DROP]
            min_score = new_score
        if engine.has_dropped(shape, anchor, board):
            break
        shape, new_anchor = engine.actions[action](shape, anchor, board)
        if new_anchor == anchor:
            break
        shape, anchor = engine.actions.soft_drop(shape, new_anchor, board)

    return actions, min_score


def compute_optimal_steps(engine):
    actions = []
    min_score = 10000000  # Very large number.
    board = engine.board
    possible_pre_actions = [
        [],
        [engine.actions.ROTATE_LEFT],
        [engine.actions.ROTATE_LEFT, engine.actions.ROTATE_LEFT],
        [engine.actions.ROTATE_RIGHT],
    ]

    for pre_actions in possible_pre_actions:
        shape, anchor = engine.shape, engine.anchor

        # Applies the pre-actions.
        for a in pre_actions:
            shape, anchor = engine.actions[a](shape, anchor, board)
            shape, anchor = engine.actions.soft_drop(shape, anchor, board)

        # Tests the best sequence of post-actions.
        for action in [engine.actions.LEFT, engine.actions.RIGHT]:
            new_actions, new_score = compute_helper(engine, shape, anchor, action)
            if new_score < min_score:
                actions = pre_actions + new_actions
                min_score = new_score

    return actions


if __name__ == '__main__':
    engine = TetrisEngine(width=10, height=20)
    steps = compute_optimal_steps(engine)

    while True:
        steps = compute_optimal_steps(engine)
        for step in steps:
            engine.step(step)
            print(engine)
            time.sleep(0.05)
