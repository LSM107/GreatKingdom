# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from game import GreatKingdomGame, GameState, PLAYER1, PLAYER2, NEUTRAL, EMPTY
import math
import random

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=9, num_actions=82, num_res_blocks=20, channels=256):  
        ### 변경: num_res_blocks=20, channels=256 으로 증가
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size
        self.num_actions = num_actions
        self.game = GreatKingdomGame()
        self.input_channels = 5
        
        # 인풋 레이어
        self.conv_init = nn.Conv2d(self.input_channels, channels, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_res_blocks)]) ### 변경

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, num_actions)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.res_blocks:
            x = block(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * self.board_size * self.board_size)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, self.board_size * self.board_size)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def save_model(self, path='trained_alpha_zero_net.pth'):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path='trained_alpha_zero_net.pth'):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.eval()

    def preprocess_board(self, board):
        # 5채널: PLAYER1, PLAYER2, NEUTRAL, PLAYER1_TERR, PLAYER2_TERR
        p1_terr, p2_terr = self.game.compute_confirmed_territories_board(board)
        board_tensor = np.zeros((5, self.game.board_size, self.game.board_size), dtype=np.float32)
        for i in range(self.game.board_size):
            for j in range(self.game.board_size):
                cell = board[i, j]
                if cell == PLAYER1:
                    board_tensor[0, i, j] = 1.0
                elif cell == PLAYER2:
                    board_tensor[1, i, j] = 1.0
                elif cell == NEUTRAL:
                    board_tensor[2, i, j] = 1.0
                if (i, j) in p1_terr:
                    board_tensor[3, i, j] = 1.0
                if (i, j) in p2_terr:
                    board_tensor[4, i, j] = 1.0
        return torch.from_numpy(board_tensor).unsqueeze(0)

class MCTSNode:
    def __init__(self, state: GameState, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None

    def is_fully_expanded(self, valid_actions):
        if self.untried_actions is None:
            self.untried_actions = [a for a, valid in enumerate(valid_actions) if valid]
        return len(self.untried_actions) == 0

    def best_child(self, c_param=math.sqrt(2)):
        choices_weights = [
            (child.value / child.visits) 
            + c_param * math.sqrt((2 * math.log(self.visits)) / child.visits)
            for action, child in self.children.items()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]

    def expand(self, action, next_state: GameState):
        child_node = MCTSNode(next_state, parent=self, parent_action=action)
        self.children[action] = child_node
        if self.untried_actions is not None and action in self.untried_actions:
            self.untried_actions.remove(action)
        return child_node

    def update(self, value):
        self.visits += 1
        self.value += value

class MCTS:
    def __init__(self, game: GreatKingdomGame, network: AlphaZeroNet, iterations=1000, c_param=math.sqrt(2)):
        self.game = game
        self.network = network
        self.iterations = iterations
        self.c_param = c_param
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.network.eval()

    def run_mcts(self, initial_state: GameState):
        root = MCTSNode(initial_state)

        for _ in range(self.iterations):
            node = root
            state = deepcopy(initial_state)

            # Selection
            while not state.game_over and node.is_fully_expanded(self.game.get_valid_moves_board(state.board, state.current_player)):
                node = node.best_child(self.c_param)
                action = node.parent_action
                state = self.game.get_next_state(state, action)

            # Expansion
            if not state.game_over:
                valid_moves = self.game.get_valid_moves_board(state.board, state.current_player)
                if node.untried_actions is None:
                    node.untried_actions = [a for a, valid in enumerate(valid_moves) if valid]
                if node.untried_actions:
                    action = random.choice(node.untried_actions)
                    state = self.game.get_next_state(state, action)
                    node = node.expand(action, state)

            # Evaluation
            if state.game_over:
                if state.winner == PLAYER1:
                    value = 1
                elif state.winner == PLAYER2:
                    value = -1
                else:
                    value = 0
            else:
                board_tensor = self.network.preprocess_board(state.board).to(self.device)
                with torch.no_grad():
                    policy, value = self.network(board_tensor)
                value = value.cpu().numpy()[0][0]

            # Backpropagation
            while node is not None:
                node.update(value)
                value = -value
                node = node.parent

        return root

    def search(self, initial_state: GameState):
        root = self.run_mcts(initial_state)
        if not root.children:
            return self.game.get_action_size() -1
        actions, visits = zip(*[(action, child.visits) for action, child in root.children.items()])
        best_action = actions[np.argmax(visits)]
        return best_action
