# This file provides an implementation of the alpha-beta pruning algorithm for the student player
# Inspired by https://github.com/KeithGalli/Connect4-Python/blob/503c0b4807001e7ea43a039cb234a4e55c4b226c/connect4_with_ai.py#L168

from shutil import move
import numpy as np
import math

NUM_ROWS = 6
NUM_COLS = 7

OPPONENT = -1
STUDENT = 1

def _utility_window(window):
    """
    Given a window of four cells, computes a utility score for the student
    """
    score = 0
    

def _utility_board(board):
    """
    Given the current board, computes an utility score for the student by evaluating 
    all possible windows of four horizontally/vertically/diagonally connected cells
    """
    return 0

def _is_terminal_node(board):
    """
    Returns if the board is in a terminal state, i.e. if the game has ended
    """

    for player in [OPPONENT, STUDENT]:
        # Check for horizontal wins
        for row in range(NUM_ROWS):
            for start_col in range(NUM_COLS - 3):
                if (board[row, start_col : start_col + 4] == player).all():
                    return True

        # Check for vertical wins
        for start_row in range(NUM_ROWS - 3):
            for col in range(NUM_COLS):
                if (board[start_row : start_row + 4, col] == player).all():
                    return True

        # Check for diagonal wins
        for start_row in range(NUM_ROWS - 3):
            for start_col in range(NUM_COLS - 3):
                end_row = start_row + 4
                end_col = start_col + 4
                # positively slopped
                if (board[range(start_row, end_row), range(start_col, end_col)] == player).all():
                    return True
                # negatively slopped
                if (board[range(end_row - 1, start_row - 1, -1), range(start_col, end_col)] == player).all():
                    return True
                
    # Check for a draw
    num_zeros = np.count_nonzero(board == 0)
    if num_zeros == 0:
        return True
                
    return False

def _max_for_student(board, depth, alpha, beta):
    if _is_terminal_node(board) or depth == 0:
        return _utility_board(board)
    
    reward = alpha
    available_moves = np.where(board[0, :] == 0)[0].tolist()
    
    for next_move_x in available_moves:
        next_move_y = np.where(board[:, next_move_x] == 0)[0].max()
        next_board = board.copy()
        next_board[next_move_y, next_move_x] = STUDENT 
        reward = max(reward, _min_for_opponent(next_board, depth-1, alpha, beta))
        if reward >= beta:
            return reward
        alpha = max(alpha, reward)
        
    return reward

def _min_for_opponent(board, depth, alpha, beta):
    if _is_terminal_node(board) or depth == 0:
        return _utility_board(board)
    
    reward = beta
    available_moves = np.where(board[0, :] == 0)[0].tolist()
    
    for next_move_x in available_moves:
        next_move_y = np.where(board[:, next_move_x] == 0)[0].max()
        next_board = board.copy()
        next_board[next_move_y, next_move_x] = OPPONENT 
        reward = min(reward, _max_for_student(next_board, depth-1, alpha, beta))
        if reward <= alpha:
            return reward
        beta = min(beta, reward)
        
    return reward

def alpha_beta_decision(board, depth=5):
    available_moves = np.where(board[0, :] == 0)[0].tolist()
    
    best_move_x = -1
    best_reward = -math.inf
    
    for move_x in available_moves:
        move_y = np.where(board[:, move_x] == 0)[0].max()
        next_board = board.copy()
        next_board[move_y, move_x] = STUDENT  
        reward = _min_for_opponent(next_board, depth, alpha=-math.inf, beta=math.inf)
        if reward > best_reward:
            best_reward = reward
            best_move_x = move_x
        
    return best_move_x
