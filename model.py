# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from game import GreatKingdomGame, GameState, PLAYER1, PLAYER2, NEUTRAL, EMPTY
import math
import random

# ===== ResidualBlock, AlphaZeroNet 부분은 기존과 동일 ===== #
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
    def __init__(self, board_size=9, num_actions=82, num_res_blocks=5, channels=64):  
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size
        self.num_actions = num_actions
        self.game = GreatKingdomGame()
        self.input_channels = 5
        
        self.conv_init = nn.Conv2d(self.input_channels, channels, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_res_blocks)])

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


# ======================================================
# [CHANGED] 아래부터 MCTS 부분 (PUCT 반영) 
# ======================================================
def add_dirichlet_noise(probabilities, alpha=0.03, frac=0.25):
    """
    루트 노드 다양성 확보용 Dirichlet 노이즈
    probabilities: shape [num_actions], 각 action 사전확률
    alpha: Dirichlet 파라미터
    frac: 노이즈 혼합 비율
    """
    num_actions = len(probabilities)
    dirichlet_sample = np.random.dirichlet([alpha]*num_actions)
    new_prob = (1 - frac) * probabilities + frac * dirichlet_sample
    # 정규화
    new_prob = new_prob / np.sum(new_prob)
    return new_prob

class MCTSNode:
    # [CHANGED] PUCT 사용 위해 prior, value_sum 등을 새로 정의
    def __init__(self, state: GameState, parent=None, parent_action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        
        self.children = {}
        
        self.visits = 0
        self.value_sum = 0.0  # 누적 value
        self.prior = prior    # 자식 노드로 이동하는 사전확률 (P)

        self.untried_actions = None

    @property
    def q_value(self):
        # 방문 전에는 0 리턴
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def is_fully_expanded(self, valid_actions):
        if self.untried_actions is None:
            self.untried_actions = [a for a, valid in enumerate(valid_actions) if valid]
        return len(self.untried_actions) == 0

    # [CHANGED] PUCT 적용
    def best_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_move = None
        parent_visits_sum = sum(child.visits for child in self.children.values())

        for action, child in self.children.items():
            q = child.q_value
            # PUCT 공식: Q + c_puct * P * sqrt(ΣN) / (1 + N)
            u = c_puct * child.prior * math.sqrt(parent_visits_sum) / (1 + child.visits)
            score = q + u

            if score > best_score:
                best_score = score
                best_move = child

        return best_move

    # [CHANGED] expand 시 prior를 함께 전달
    def expand(self, action, next_state: GameState, prior):
        child_node = MCTSNode(next_state, parent=self, parent_action=action, prior=prior)
        self.children[action] = child_node
        if self.untried_actions is not None and action in self.untried_actions:
            self.untried_actions.remove(action)
        return child_node

    # [CHANGED] 누적 value 업데이트
    def update(self, value):
        self.visits += 1
        self.value_sum += value


class MCTS:
    def __init__(self, game: GreatKingdomGame, network: AlphaZeroNet, iterations=1000, c_param=1.0):
        self.game = game
        self.network = network
        self.iterations = iterations
        self.c_param = c_param
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.network.eval()

    def run_mcts(self, initial_state: GameState):
        # [CHANGED] 루트 노드 prior=1.0 으로 생성
        root = MCTSNode(initial_state, prior=1.0)

        # 먼저 루트 state에서 policy, value를 한번 구해본 후, Dirichlet 노이즈 적용 (AlphaZero 스타일)
        board_tensor = self.network.preprocess_board(initial_state.board).to(self.device)
        with torch.no_grad():
            policy_logits, root_value = self.network(board_tensor)
        root_value = root_value.cpu().numpy()[0][0]
        policy_probs = torch.exp(policy_logits).cpu().numpy()[0]  # log-softmax -> exp

        # 루트에서 valid move만 추출
        valid_moves = self.game.get_valid_moves_board(initial_state.board, initial_state.current_player)
        # valid move에 대해서만 사전확률 적용, 그 외는 0
        for i, valid in enumerate(valid_moves):
            if not valid:
                policy_probs[i] = 0.0
        sum_probs = np.sum(policy_probs)
        if sum_probs > 0:
            policy_probs /= sum_probs
        else:
            # 모두 0이면 균등분포 처리
            policy_probs = np.array(valid_moves, dtype=np.float32)
            policy_probs /= np.sum(policy_probs)

        # [CHANGED] 루트 노드에 Dirichlet 노이즈 섞기
        policy_probs = add_dirichlet_noise(policy_probs, alpha=0.03, frac=0.25)

        # 루트 자식 노드 미리 만들어둘 수도 있고, (단일 expand로 처리해도 무관)
        # 여기서는 그냥 루트에서 children만 설정(선택적으로)
        # -> single expand 방식을 쓰면, 시뮬레이션 한 번마다 action을 하나씩 펼쳐도 됨

        # 루트 노드의 첫 번째 value 업데이트(백프로파게이션)할 필요는 없음(AlphaZero에서는 보통 루트 value는 사용x)
        # 아래에서는 iteration마다 selection->expansion->evaluation->backprop 순서를 진행

        for _ in range(self.iterations):
            node = root
            state = deepcopy(initial_state)

            # 1) Selection
            while not state.game_over and node.is_fully_expanded(self.game.get_valid_moves_board(state.board, state.current_player)):
                node = node.best_child(self.c_param)
                action = node.parent_action
                state = self.game.get_next_state(state, action)

            # 2) Expansion + Evaluation
            if not state.game_over:
                valid_moves = self.game.get_valid_moves_board(state.board, state.current_player)
                if node.untried_actions is None:
                    node.untried_actions = [a for a, v in enumerate(valid_moves) if v]

                if node.untried_actions:
                    action = random.choice(node.untried_actions)
                    next_state = self.game.get_next_state(state, action)

                    # 자식 노드의 policy, value 계산
                    board_tensor = self.network.preprocess_board(next_state.board).to(self.device)
                    with torch.no_grad():
                        policy_logits, value_pred = self.network(board_tensor)
                    value_pred = value_pred.cpu().numpy()[0][0]
                    child_policy = torch.exp(policy_logits).cpu().numpy()[0]

                    # valid move 아닌 것 0 처리
                    valid_moves_child = self.game.get_valid_moves_board(next_state.board, next_state.current_player)
                    for i, v in enumerate(valid_moves_child):
                        if not v:
                            child_policy[i] = 0.0
                    sum_child = np.sum(child_policy)
                    if sum_child > 0:
                        child_policy /= sum_child
                    else:
                        child_policy = np.array(valid_moves_child, dtype=np.float32)
                        child_policy /= np.sum(child_policy)

                    # 자식 노드 expand
                    prior_for_action = child_policy[action]  # 단일 action expand 시에는 이 값 사용
                    child_node = node.expand(action, next_state, prior_for_action)

                    # 3) Backprop
                    # 현재 확장된 node의 value= value_pred
                    rollout_value = value_pred
                else:
                    # 확장할 액션이 없으면 leaf
                    rollout_value = 0
            else:
                # 게임 종료
                if state.winner == PLAYER1:
                    rollout_value = 1
                elif state.winner == PLAYER2:
                    rollout_value = -1
                else:
                    rollout_value = 0

            # Backprop
            while node is not None:
                node.update(rollout_value)
                rollout_value = -rollout_value
                node = node.parent

        return root

    def search(self, initial_state: GameState):
        root = self.run_mcts(initial_state)
        if not root.children:
            return self.game.get_action_size() - 1
        actions, visits = zip(*[(action, child.visits) for action, child in root.children.items()])
        best_action = actions[np.argmax(visits)]
        return best_action
