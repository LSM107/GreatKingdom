# game.py

import numpy as np
import sys
from collections import deque
from copy import deepcopy

EMPTY = 0

UPBOUND = 1
RIGHTBOUND = 2
DOWNBOUND = 3
LEFTBOUND = 4

NEUTRAL = 5

PLAYER1 = 6  # 파란색 성
PLAYER2 = 7  # 주황색 성

CELL_SYMBOLS = {
    EMPTY: '.',
    NEUTRAL: 'N',
    PLAYER1: 'B',
    PLAYER2: 'O',
}

TERRITORY_SYMBOLS = {
    PLAYER1: '*',
    PLAYER2: '+',
}

class GameState:
    def __init__(self, board, current_player, consecutive_passes, game_over=False, winner=None):
        self.board = board
        self.current_player = current_player
        self.consecutive_passes = consecutive_passes
        self.game_over = game_over
        self.winner = winner

class GreatKingdomGame:
    def __init__(self):
        self.board_size = 9
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        center = self.board_size // 2
        self.board[center, center] = NEUTRAL

        self.current_player = PLAYER1
        self.game_over = False
        self.winner = None
        self.consecutive_passes = 0
        self.player1_territories = set()
        self.player2_territories = set()

        self.player1_castles = set()
        self.player2_castles = set()

        self.empty_indices = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.empty_indices.append((i, j))
        self.empty_indices.remove((center, center))

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        center = self.board_size // 2
        self.board[center, center] = NEUTRAL
        self.current_player = PLAYER1
        self.game_over = False
        self.winner = None
        self.consecutive_passes = 0
        self.player1_territories.clear()
        self.player2_territories.clear()
        self.player1_castles.clear()
        self.player2_castles.clear()
        
        self.empty_indices = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i, j) != (center, center):
                    self.empty_indices.append((i, j))

    def search_blob_board(self, x, y, board):
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return []
        if board[x, y] != EMPTY:
            return []

        visited = np.zeros((self.board_size, self.board_size), dtype=bool)
        queue = deque()
        queue.append((x, y))
        blob = []

        while queue:
            cx, cy = queue.popleft()
            if visited[cx, cy]:
                continue
            visited[cx, cy] = True
            blob.append((cx, cy))
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if board[nx, ny] == EMPTY and not visited[nx, ny]:
                        queue.append((nx, ny))
        return blob

    def compute_confirmed_territories_board(self, board):
        player1_territories = set()
        player2_territories = set()

        empty_indices = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == EMPTY:
                    empty_indices.append((i, j))

        while empty_indices:
            indice = empty_indices[0]
            blob = self.search_blob_board(*indice, board)
            for x, y in blob:
                empty_indices.remove((x, y))

            adjacent_information = []
            for x, y in blob:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if nx < 0:
                        adjacent_information.append(UPBOUND)
                    elif nx >= self.board_size:
                        adjacent_information.append(DOWNBOUND)
                    elif ny < 0:
                        adjacent_information.append(LEFTBOUND)
                    elif ny >= self.board_size:
                        adjacent_information.append(RIGHTBOUND)
                    else:
                        adjacent_information.append(board[nx, ny])

            if (UPBOUND in adjacent_information) and (DOWNBOUND in adjacent_information) and (LEFTBOUND in adjacent_information) and (RIGHTBOUND in adjacent_information):
                continue

            if (PLAYER1 in adjacent_information) and (PLAYER2 in adjacent_information):
                continue

            if PLAYER1 in adjacent_information:
                player1_territories.update(blob)
            if PLAYER2 in adjacent_information:
                player2_territories.update(blob)

        return player1_territories, player2_territories

    def place_castle(self, x, y, player=None):
        if player is None:
            player = self.current_player
        if self.game_over:
            return False
        if self.board[x, y] != EMPTY:
            return False
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return False
        if player == PLAYER1 and (x, y) in self.player2_territories:
            return False
        if player == PLAYER2 and (x, y) in self.player1_territories:
            return False

        self.board[x, y] = player
        self.empty_indices.remove((x, y))
        if player == PLAYER1:
            self.player1_castles.add((x, y))
        else:
            self.player2_castles.add((x, y))

        self.seige()
        self.compute_confirmed_territories()
        self.consecutive_passes = 0
        self.current_player = PLAYER1 if self.current_player == PLAYER2 else PLAYER2
        return True

    def compute_confirmed_territories(self):
        # 기존 self.player1_territories, self.player2_territories 계산을 위해
        self.player1_territories, self.player2_territories = self.compute_confirmed_territories_board(self.board)

    def pass_turn(self):
        if self.game_over:
            return False

        self.consecutive_passes += 1
        if self.consecutive_passes >= 2:
            self.game_over = True
            p1_territory_count = len(self.player1_territories)
            p2_territory_count = len(self.player2_territories)
            if p1_territory_count >= p2_territory_count + 3:
                self.winner = PLAYER1
            else:
                self.winner = PLAYER2
            return True

        self.current_player = PLAYER1 if self.current_player == PLAYER2 else PLAYER2
        return True

    def search_castle_blob_board(self, x, y, board):
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return []
        player = board[x, y]
        if player not in [PLAYER1, PLAYER2]:
            return []
        visited = np.zeros((self.board_size, self.board_size), dtype=bool)
        queue = deque()
        queue.append((x, y))
        blob = []

        while queue:
            cx, cy = queue.popleft()
            if visited[cx, cy]:
                continue
            visited[cx, cy] = True
            blob.append((cx, cy))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if board[nx, ny] == player and not visited[nx, ny]:
                        queue.append((nx, ny))
        return blob

    def search_castles_board(self, board):
        player1_castles = set()
        player2_castles = set()

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == PLAYER1:
                    player1_castles.add((i, j))
                elif board[i, j] == PLAYER2:
                    player2_castles.add((i, j))

        return player1_castles, player2_castles

    def seige_board(self, board, player):
        player1_castles, player2_castles = self.search_castles_board(board)

        if player == PLAYER1:
            opponent_castles = list(deepcopy(player2_castles))
        else:
            opponent_castles = list(deepcopy(player1_castles))

        game_over = False
        winner = None

        while not len(opponent_castles) == 0:
            castle = opponent_castles[0]
            blob = self.search_castle_blob_board(*castle, board)

            for x, y in blob:
                opponent_castles.remove((x, y))

            adjacent_information = []
            for x, y in blob:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy

                    if nx < 0:
                        adjacent_information.append(UPBOUND)
                    elif nx >= self.board_size:
                        adjacent_information.append(DOWNBOUND)
                    elif ny < 0:
                        adjacent_information.append(LEFTBOUND)
                    elif ny >= self.board_size:
                        adjacent_information.append(RIGHTBOUND)
                    else:
                        adjacent_information.append(board[nx, ny])

            if player not in adjacent_information:
                continue

            if EMPTY in adjacent_information:
                continue

            game_over = True
            winner = player

        return game_over, winner

    def get_action_size(self):
        return self.board_size * self.board_size + 1

    def get_valid_moves_board(self, board, player):
        valid_moves = [0] * self.get_action_size()

        empty_cells = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == EMPTY:
                    empty_cells.append((i, j))

        player1_territories, player2_territories = self.compute_confirmed_territories_board(board)

        if player == PLAYER1:
            forbidden = player2_territories
        else:
            forbidden = player1_territories

        for (x, y) in empty_cells:
            if (x,y) not in forbidden:
                valid_moves[x * self.board_size + y] = 1
        valid_moves[-1] = 1
        return valid_moves

    def get_next_state(self, state: GameState, action):
        game_over = False
        winner = None
        next_board = state.board.copy()

        if action == self.get_action_size() - 1:
            consecutive_passes = state.consecutive_passes + 1
            if consecutive_passes >= 2:
                # 게임 종료
                temp_game = GreatKingdomGame()
                temp_game.board = next_board
                p1_terr, p2_terr = temp_game.compute_confirmed_territories_board(next_board)
                p1_count = len(p1_terr)
                p2_count = len(p2_terr)
                if p1_count >= p2_count + 3:
                    winner = PLAYER1
                else:
                    winner = PLAYER2
                game_over = True
                return GameState(next_board, state.current_player, consecutive_passes, game_over, winner)
            else:
                next_player = PLAYER1 if state.current_player == PLAYER2 else PLAYER2
                return GameState(next_board, next_player, consecutive_passes, game_over, winner)

        else:
            x = action // self.board_size
            y = action % self.board_size
            next_board[x, y] = state.current_player
            temp_game = GreatKingdomGame()
            temp_game.board = next_board.copy()
            game_over, winner = temp_game.seige_board(next_board, state.current_player)
            if game_over:
                return GameState(next_board, state.current_player, 0, game_over, winner)
            else:
                next_player = PLAYER1 if state.current_player == PLAYER2 else PLAYER2
                return GameState(next_board, next_player, 0, game_over, winner)

    def seige(self):
        # 실제 게임 진행 중 place_castle 후 공성 로직
        if self.current_player == PLAYER1:
            opponent_castles = list(deepcopy(self.player2_castles))
        else:
            opponent_castles = list(deepcopy(self.player1_castles))

        while not len(opponent_castles) == 0:
            castle = opponent_castles[0]
            blob = self.search_castle_blob(*castle)
            for x, y in blob:
                opponent_castles.remove((x, y))

            adjacent_information = []
            for x, y in blob:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if nx < 0:
                        adjacent_information.append(UPBOUND)
                    elif nx >= self.board_size:
                        adjacent_information.append(DOWNBOUND)
                    elif ny < 0:
                        adjacent_information.append(LEFTBOUND)
                    elif ny >= self.board_size:
                        adjacent_information.append(RIGHTBOUND)
                    else:
                        adjacent_information.append(self.board[nx, ny])

            if self.current_player not in adjacent_information:
                continue
            if EMPTY in adjacent_information:
                continue

            self.game_over = True
            self.winner = self.current_player

    def search_castle_blob(self, x, y):
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return []
        player = self.board[x, y]
        if player not in [PLAYER1, PLAYER2]:
            return []
        visited = np.zeros((self.board_size, self.board_size), dtype=bool)
        queue = deque()
        queue.append((x, y))
        blob = []

        while queue:
            cx, cy = queue.popleft()
            if visited[cx, cy]:
                continue
            visited[cx, cy] = True
            blob.append((cx, cy))
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if self.board[nx, ny] == player and not visited[nx, ny]:
                        queue.append((nx, ny))
        return blob

    def print_board(self):
        header = '    ' + ''.join(['{:2d}'.format(i) for i in range(self.board_size)])
        print(header)

        for i in range(self.board_size):
            row = ' {:2d} '.format(i)
            for j in range(self.board_size):
                cell = self.board[i, j]
                if (i, j) in self.player1_territories:
                    symbol = TERRITORY_SYMBOLS[PLAYER1]
                elif (i, j) in self.player2_territories:
                    symbol = TERRITORY_SYMBOLS[PLAYER2]
                else:
                    symbol = CELL_SYMBOLS.get(cell, '.')
                row += ' ' + symbol
            print(row)
        print()
