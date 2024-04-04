from exceptions import GameplayException
from connect4 import Connect4
from randomagent import RandomAgent
from minmaxagent import MinMaxAgent
from alphabetaagent import AlphaBetaAgent
import time

connect4 = Connect4(width=7, height=6)
agent = MinMaxAgent('x', 3, False)
agent1 = MinMaxAgent('o', 3, True)
agent2 = AlphaBetaAgent('x', 5)
connect4.who_moves = 'o'

x_player_time = []
o_player_time = []
while not connect4.game_over:
    connect4.draw()
    try:
        who_is_moving = connect4.who_moves
        start_time = time.time()
        if connect4.who_moves == agent2.my_token:
            n_column = agent2.decide(connect4)
        else:
            n_column = agent1.decide(connect4)
            # n_column = int(input(':'))
        end_time = time.time()
        connect4.drop_token(n_column)
        if who_is_moving == 'o':
            o_player_time.append(end_time - start_time)
        else:
            x_player_time.append(end_time - start_time)
        print("Moving [", who_is_moving, "] with time: ", end_time - start_time)
    except (ValueError, GameplayException):
        print('invalid move')

connect4.draw()

print("O player time ", o_player_time)
print("X player time ", x_player_time)
