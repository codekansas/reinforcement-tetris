#!/usr/bin/env python3
# Run this file to interact with the engine.

import numpy as np
import random


class Shapes(object):
    T = 0
    J = 1
    L = 2
    Z = 3
    S = 4
    I = 5
    O = 6

    ORIENTATIONS = {
        T: [(0, 0), (-1, 0), (1, 0), (0, -1)],
        J: [(0, 0), (-1, 0), (0, -1), (0, -2)],
        L: [(0, 0), (1, 0), (0, -1), (0, -2)],
        Z: [(0, 0), (-1, 0), (0, -1), (1, -1)],
        S: [(0, 0), (-1, -1), (0, -1), (1, 0)],
        I: [(0, 0), (0, -1), (0, -2), (0, -3)],
        O: [(0, 0), (0, -1), (-1, 0), (-1, -1)],
    }

    def __contains__(self, i):
        return isinstance(i, int) and 0 <= i <= 6

    def __getitem__(self, i):
        return self.ORIENTATIONS[i]

    def __len__(self):
        return 7


class Actions(object):
    LEFT = 0
    RIGHT = 1
    SHIFTS = (0, 1)
    HARD_DROP = 2
    SOFT_DROP = 3
    DROPS = (2, 3)
    ROTATE_LEFT = 4
    ROTATE_RIGHT = 5
    ROTATES = (4, 5)

    def __init__(self):
        self.FUNCTIONS = {
            self.LEFT: self.left,
            self.RIGHT: self.right,
            self.HARD_DROP: self.hard_drop,
            self.SOFT_DROP: self.soft_drop,
            self.ROTATE_LEFT: self.rotate_left,
            self.ROTATE_RIGHT: self.rotate_right,
        }
        self.drop_distance = 0

    def rotated(self, shape, cclk=False):
        if cclk:
            return [(-j, i) for i, j in shape]
        else:
            return [(j, -i) for i, j in shape]

    def is_occupied(self, shape, anchor, board):
        for i, j in shape:
            x, y = int(anchor[0] + i), int(anchor[1] + j)
            if y < 0:
                continue
            if x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
                return True
        return False

    def left(self, shape, anchor, board):
        new_anchor = (anchor[0] - 1, anchor[1])
        if self.is_occupied(shape, new_anchor, board):
            return (shape, anchor)
        else:
            return (shape, new_anchor)

    def right(self, shape, anchor, board):
        new_anchor = (anchor[0] + 1, anchor[1])
        if self.is_occupied(shape, new_anchor, board):
            return (shape, anchor)
        else:
            return (shape, new_anchor)

    def soft_drop(self, shape, anchor, board):
        new_anchor = (anchor[0], anchor[1] + 1)
        if self.is_occupied(shape, new_anchor, board):
            return (shape, anchor)
        else:
            return (shape, new_anchor)

    def hard_drop(self, shape, anchor, board):
        while True:
            shape, anchor_new = self.soft_drop(shape, anchor, board)
            if anchor_new == anchor:
                return shape, anchor_new
            anchor = anchor_new

    def rotate_left(self, shape, anchor, board):
        new_shape = self.rotated(shape, cclk=False)
        if self.is_occupied(new_shape, anchor, board):
            return (shape, anchor)
        else:
            return (new_shape, anchor)

    def rotate_right(self, shape, anchor, board):
        new_shape = self.rotated(shape, cclk=True)
        if self.is_occupied(new_shape, anchor, board):
            return (shape, anchor)
        else:
            return (new_shape, anchor)

    def __contains__(self, i):
        return isinstance(i, int) and 0 <= i <= 5

    def __getitem__(self, i):
        return self.FUNCTIONS[i]

    def __len__(self):
        return 6

class TetrisEngine(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=np.bool)

        self.shapes = Shapes()
        self.actions = Actions()

        # Variables for running the engine.
        self.time = -1
        self.score = -1
        self.combo_counter = 0
        self.cleared_lines = 0
        self.tetris_flag = False
        self.anchor = None
        self.shape_idx = None
        self.shape = None
        self.deaths = 0
        self.dead = False

        # Used for generating shapes in a non-iid way.
        self._shape_counts = [0] * len(self.shapes)

        # Clear after initialization.
        self.clear()

    def _choose_shape(self):
        maxm = max(self._shape_counts)
        m = [5 + maxm - x for x in self._shape_counts]
        r = random.randint(1, sum(m))
        for i, n in enumerate(m):
            r -= n
            if r <= 0:
                self._shape_counts[i] += 1
                return i
        # return 6

    def _new_piece(self):
        self.anchor = (self.width / 2, 0)
        self.shape_idx = self._choose_shape()
        self.shape = self.shapes[self.shape_idx]

    def has_dropped(self, shape, anchor, board):
        return self.actions.is_occupied(
            shape,
            (anchor[0], anchor[1] + 1),
            board,
        )

    def _has_dropped(self):
        return self.has_dropped(self.shape, self.anchor, self.board)

    def _update_score(self, cleared_lines=None):
        if cleared_lines is not None:
            if cleared_lines == 0:
                self.combo_counter = 0
            else:
                self.score += 50 * self.combo_counter
                self.combo_counter += 1

            if cleared_lines == 1:
                self.score += 100
            elif cleared_lines == 2:
                self.score += 300
            elif cleared_lines == 3:
                self.score += 500

            if cleared_lines == 4:
                if self.tetris_flag:
                    self.score += 1200
                else:
                    self.tetris_flag = True
                    self.score += 800
            else:
                self.tetris_flag = False

    def _clear_lines(self):
        cleared = np.all(self.board, axis=0)
        num_cleared = np.sum(cleared)
        if not num_cleared:
            return
        keep_lines, = np.where(~cleared)
        self.cleared_lines += num_cleared
        self.board = np.concatenate([
            np.zeros(shape=(self.width, num_cleared), dtype=np.bool),
            self.board[:, keep_lines],
        ], axis=1)
        self._update_score(cleared_lines=num_cleared)

    def get_board(self, include_dropped=True):
        '''Returns a copy of the current board.'''

        # Adds the dropped piece to the board.
        if include_dropped:
            s, a = self.actions.hard_drop(self.shape, self.anchor, self.board)
            self.toggle_piece(True, shape=s, anchor=a)
            self.set_piece()
            w, l = self.board.shape
            board_copy = np.copy(self.board).reshape(1, w, l, 1)
            self.toggle_piece(False, shape=s, anchor=a)
        else:
            self.set_piece()
            w, l = self.board.shape
            board_copy = np.copy(self.board).reshape(1, w, l, 1)

        self.clear_piece()
        return board_copy

    def step(self, action):
        prev_score = self.score
        self.dead = False
        act_params = (self.shape, self.anchor, self.board)
        self.shape, self.anchor = self.actions[action](*act_params)
        self.time += 1

        # Drops once every 5 steps, unless it was a hard drop.
        if action not in self.actions.DROPS:
            act_params = (self.shape, self.anchor, self.board)
            self.shape, self.anchor = self.actions.soft_drop(*act_params)

        if self._has_dropped():
            self.set_piece()
            self._clear_lines()
            if np.any(self.board[:, 0]):
                self.clear()
                self.deaths += 1
                self.dead = True
            else:
                self._new_piece()

        # Returns the computed score.
        if self.dead:
            return -1.
        else:
            return float(self.score - prev_score) * 0.01

    def clear(self):
        self.time = 0
        self.score = 0
        self._new_piece()
        self.board[:] = 0

    def clear_piece(self):
        self.toggle_piece(False)

    def set_piece(self):
        self.toggle_piece(True)

    def toggle_piece(self, on, shape=None, anchor=None):
        anchor = anchor or self.anchor
        shape = shape or self.shape
        for i, j in shape:
            x, y = int(i + anchor[0]), int(j + anchor[1])
            if x >= 0 and y >= 0:
                self.board[x, y] = on

    def serialize_board(self, board, include_score=True):
        board = np.squeeze(board)
        s = ['o' + '-' * board.shape[0] + 'o']
        f = lambda i: '|' + ''.join('X' if j else ' ' for j in i) + '|'
        s += [f(i) for i in board.T]
        s += ['o' + '-' * board.shape[0] + 'o']
        return '\n'.join(s)

    def __repr__(self):
        return self.serialize_board(self.get_board(include_dropped=False))


if __name__ == '__main__':
    from utils import Getch
    getch = Getch()

    engine = TetrisEngine(10, 20)
    actions = Actions()

    # Maps keys to their corresponding actions.
    key_map = {
        'a': actions.LEFT,
        'd': actions.RIGHT,
        'w': actions.ROTATE_LEFT,
        's': actions.ROTATE_RIGHT,
        'z': actions.HARD_DROP,
        'x': actions.SOFT_DROP,
    }

    print(engine)
    while True:
        action = getch()
        if action not in key_map:
            if action == 'q':
                break
            print('Invalid move: {}'.format(action))
            print('Possible: {}'.format(set(key_map.keys())))
            print('Press <q> to quit')
        else:
            engine.step(key_map[action])
            print(engine)
            print('Score: {}'.format(engine.score))
