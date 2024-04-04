from random import random
import copy
from exceptions import AgentException


class AlphaBetaAgent:
    def __init__(self, my_token, depth=3):
        self.my_token = my_token
        self.depth = depth

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        moves = connect4.possible_drops()
        best_result = -1
        best_move = -1
        alpha = float('-inf')
        beta = float('inf')
        for move in moves:
            temp_connect = copy.deepcopy(connect4)
            temp_connect.drop_token(move)
            temp_result = self.mh(temp_connect, self.other_player(self.my_token), self.depth, alpha, beta)
            if temp_result >= best_result:
                best_result = temp_result
                best_move = move
                if best_result == 1:
                    print("Moving ", self.my_token, " with result: ", best_result)
                    return best_move
        print("Moving ", self.my_token, " with result: ", best_result)
        return best_move

    def other_player(self, player):
        return 'o' if player == 'x' else 'x'

    def mh(self, connect4, current_player, d, alfa, beta):
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return 1
            elif connect4.wins is not None:
                return -1
            else:
                return 0
        elif d == 0:
            return self.h(connect4)
        elif current_player == self.my_token:
            moves = connect4.possible_drops()
            v = float('-inf')
            for move in moves:
                temp_connect = copy.deepcopy(connect4)
                temp_connect.drop_token(move)
                v = max(self.mh(temp_connect, self.other_player(current_player), d - 1, alfa, beta), v)
                alfa = max(alfa, v)
                if v >= beta:
                    break
            return v
        else:
            moves = connect4.possible_drops()
            v = float('inf')
            for move in moves:
                temp_connect = copy.deepcopy(connect4)
                temp_connect.drop_token(move)
                v = min(self.mh(temp_connect, self.other_player(current_player), d - 1, alfa, beta), v)
                beta = min(beta, v)
                if v <= alfa:
                    break
            return v

    def h(self, connect4):
        if not connect4.possible_drops():
            return 0

        board_cells = connect4.width * connect4.height
        x_score = 0
        o_score = 0
        for four in connect4.iter_fours():
            if four.count('_') == 1:
                if four.count('x') == 3:
                    x_score += 3
                elif four.count('o') == 3:
                    o_score += 3
            elif four.count('_') == 0:
                if four.count('x') == 4:
                    x_score += 10
                elif four.count('o') == 4:
                    o_score += 10
            elif four.count('_') == 2 and (four.count('x') == 2 or four.count('o') == 2):
                if four.count('x') == 2:
                    x_score += 1
                elif four.count('o') == 2:
                    o_score += 1

        center_column = connect4.center_column()
        x_score += center_column.count('x')*2
        o_score += center_column.count('o')*2

        if self.my_token == 'x':
            temp = (x_score - o_score) / board_cells
            return (x_score - o_score) / board_cells
        else:
            temp = (o_score - x_score) / board_cells
            return (o_score - x_score) / board_cells
