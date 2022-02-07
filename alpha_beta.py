# This file provides an implementation of the alpha-beta pruning algorithm for the student player
# Inspired by https://github.com/KeithGalli/Connect4-Python/blob/503c0b4807001e7ea43a039cb234a4e55c4b226c/connect4_with_ai.py#L168

import numpy as np
import math

NUM_ROWS = 6
NUM_COLS = 7

OPPONENT = -1
STUDENT = 1
EMPTY = 0

def _utility_window(window):
    """
    Given a window of four cells, computes a utility score for the student
    """  
    # Acquire win
    if (window == STUDENT).all():
        return 100
    # Three connected cells
    elif (window[:3] == STUDENT).all() and window[3] == EMPTY:
        return 20
    elif (window[1:] == STUDENT).all() and window[0] == EMPTY:
        return 20
    # Two connected cells
    elif (window[:2] == STUDENT).all() and (window[2:] == EMPTY).all():
        return 10
    elif (window[2:] == STUDENT).all() and (window[:2] == EMPTY).all():
        return 10
    # Allow opponent to connect three cells that can be used for a win
    elif (window[:2] == OPPONENT).all() and (window[2:] == EMPTY).all():
        return -10
    elif (window[2:] == OPPONENT).all() and (window[:2] == EMPTY).all():
        return -10
    # Allow opponent to win
    elif (window[:3] == OPPONENT).all() and (window[3] == EMPTY).all():
        return -75
    elif (window[1:] == OPPONENT).all() and (window[0] == EMPTY).all():
        return -75
    # Any other case
    else:
        return 0
    
def _utility_board(board):
    """
    Given the current board, computes an utility score for the student by evaluating 
    all possible windows of four horizontally/vertically/diagonally connected cells
    """
    score = 0
    
    # Incentivize placements in the center to block diagonal lines of opponents
    middle_column = board[:, NUM_COLS // 2]
    middle_entries = (middle_column == STUDENT).sum()
    score += middle_entries * 3
    
    # Horizontal lines
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS - 3):
            window = board[row, col : col + 4]
            score += _utility_window(window)
            
    # Vertical lines
    for row in range(NUM_ROWS - 3):
        for col in range(NUM_COLS):
            window = board[row : row + 4, col]
            score += _utility_window(window)
     
    # Diagonal lines
    for start_row in range(NUM_ROWS - 3):
        for start_col in range(NUM_COLS - 3):
            end_row = start_row + 4
            end_col = start_col + 4
            # positively slopped
            window = board[range(start_row, end_row), range(start_col, end_col)]
            score += _utility_window(window)
            # negatively slopped
            window = board[range(end_row - 1, start_row - 1, -1), range(start_col, end_col)]
            score += _utility_window(window)    
                
    return score

def _is_terminal_node(board):
    """
    Returns if the board is in a terminal state, i.e. if the game has ended
    """
    for player in [OPPONENT, STUDENT]:
        # Check for horizontal wins
        for row in range(NUM_ROWS):
            for start_col in range(NUM_COLS - 3):
                end_col = start_col + 4
                if (board[row, start_col : end_col] == player).all():
                    return True

        # Check for vertical wins
        for start_row in range(NUM_ROWS - 3):
            for col in range(NUM_COLS):
                end_row = start_row + 4
                if (board[start_row : end_row, col] == player).all():
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

def _get_updated_board(board, move_x, player):
    move_y = np.where(board[:, move_x] == 0)[0].max()
    new_board = board.copy()
    new_board[move_y, move_x] = player
    return new_board

def _max_for_student(board, depth, alpha, beta):
    if _is_terminal_node(board) or depth == 0:
        return _utility_board(board)
    
    reward = alpha
    available_moves = np.where(board[0, :] == 0)[0]
    
    for next_move in available_moves:
        next_board = _get_updated_board(board, next_move, STUDENT) 
        reward = max(reward, _min_for_opponent(next_board, depth-1, alpha, beta))
        if reward >= beta:
            return reward
        alpha = max(alpha, reward)
        
    return reward

def _min_for_opponent(board, depth, alpha, beta):
    if _is_terminal_node(board) or depth == 0:
        return _utility_board(board)
    
    reward = beta
    available_moves = np.where(board[0, :] == 0)[0]
    
    for next_move in available_moves:
        next_board = _get_updated_board(board, next_move, OPPONENT)
        reward = min(reward, _max_for_student(next_board, depth-1, alpha, beta))
        if reward <= alpha:
            return reward
        beta = min(beta, reward)
        
    return reward

def alpha_beta_decision(board, depth=3):
    available_moves = np.where(board[0, :] == 0)[0]
    best_move = -1
    best_reward = -math.inf
    
    for next_move in available_moves:
        next_board = _get_updated_board(board, next_move, STUDENT) 
        reward = _min_for_opponent(next_board, depth, alpha=-math.inf, beta=math.inf)
        if reward > best_reward:
            best_reward = reward
            best_move = next_move
        
    return best_move