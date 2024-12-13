# train_self_play.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from collections import deque
import random
import matplotlib.pyplot as plt

from game import GreatKingdomGame, GameState, PLAYER1, PLAYER2, NEUTRAL, EMPTY
from model import AlphaZeroNet, MCTS

import os
import time

time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def preprocess_board_for_training(board, board_size=9):
    temp_game = GreatKingdomGame()
    temp_game.board = board.copy()
    p1_terr, p2_terr = temp_game.compute_confirmed_territories_board(board)

    board_tensor = np.zeros((5, board_size, board_size), dtype=np.float32)
    for i in range(board_size):
        for j in range(board_size):
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
    return board_tensor

def self_play_episode(game: GreatKingdomGame, network: AlphaZeroNet, mcts_iterations=100, temperature=1):
    states = []
    mcts_policies = []
    current_players = []
    moves = []
    state = GameState(deepcopy(game.board), game.current_player, game.consecutive_passes, game.game_over, game.winner)
    mcts = MCTS(game, network, iterations=mcts_iterations)

    while not state.game_over:
        board_tensor = preprocess_board_for_training(state.board, board_size=game.board_size)
        valid_moves = game.get_valid_moves_board(state.board, state.current_player)

        root = mcts.run_mcts(state)
        visit_counts = []
        for a in range(game.get_action_size()):
            child = root.children.get(a)
            if child is not None:
                visit_counts.append(child.visits)
            else:
                visit_counts.append(0)

        visit_counts = np.array(visit_counts, dtype=np.float32)
        if temperature == 0:
            best_action = np.argmax(visit_counts)
            pi = np.zeros_like(visit_counts)
            pi[best_action] = 1.0
        else:
            if visit_counts.sum() > 0:
                visit_counts = visit_counts ** (1.0 / temperature)
                pi = visit_counts / visit_counts.sum()
            else:
                pi = np.zeros_like(visit_counts)
                pi[-1] = 1.0

        states.append(board_tensor)
        mcts_policies.append(pi)
        current_players.append(state.current_player)

        action = np.random.choice(len(pi), p=pi)
        if action == game.get_action_size() - 1:
            moves.append((state.current_player, 'pass'))
        else:
            x = action // game.board_size
            y = action % game.board_size
            moves.append((state.current_player, x, y))

        next_state = game.get_next_state(state, action)
        state = next_state

    if state.winner == PLAYER1:
        z = 1.0
    elif state.winner == PLAYER2:
        z = -1.0
    else:
        z = 0.0

    training_data = []
    for s, pi, p in zip(states, mcts_policies, current_players):
        if p == PLAYER1:
            training_data.append((s, pi, z))
        else:
            training_data.append((s, pi, -z))

    return training_data, moves, state.winner, state

def flip_up_down(state):
    return np.flipud(state)

def flip_left_right(state):
    return np.fliplr(state)

def rotate_180(state):
    return np.rot90(state, 2, axes=(1, 2))

def flip_pi_up_down(pi, board_size):
    pass_index = board_size*board_size
    pi_2d = pi[:pass_index].reshape(board_size, board_size)
    pi_2d = np.flipud(pi_2d)
    pi_new = pi_2d.flatten()
    pi_new = np.append(pi_new, pi[pass_index])
    return pi_new

def flip_pi_left_right(pi, board_size):
    pass_index = board_size*board_size
    pi_2d = pi[:pass_index].reshape(board_size, board_size)
    pi_2d = np.fliplr(pi_2d)
    pi_new = pi_2d.flatten()
    pi_new = np.append(pi_new, pi[pass_index])
    return pi_new

def flip_pi_180(pi, board_size):
    pass_index = board_size*board_size
    pi_2d = pi[:pass_index].reshape(board_size, board_size)
    pi_2d = np.rot90(pi_2d, 2)
    pi_new = pi_2d.flatten()
    pi_new = np.append(pi_new, pi[pass_index])
    return pi_new

def augment_data(states, pis, vs, board_size=9):
    augmented_states = []
    augmented_pis = []
    augmented_vs = []

    for s, pi, v in zip(states, pis, vs):
        transform_type = random.choice(['ud', 'lr', '180', 'none'])
        if transform_type == 'ud':
            s_aug = flip_up_down(s)
            pi_aug = flip_pi_up_down(pi, board_size)
        elif transform_type == 'lr':
            s_aug = flip_left_right(s)
            pi_aug = flip_pi_left_right(pi, board_size)
        elif transform_type == '180':
            s_aug = rotate_180(s)
            pi_aug = flip_pi_180(pi, board_size)
        else:
            s_aug = s
            pi_aug = pi

        augmented_states.append(s_aug)
        augmented_pis.append(pi_aug)
        augmented_vs.append(v)

    return np.array(augmented_states), np.array(augmented_pis), np.array(augmented_vs)

def get_final_game_state_str(final_state: GameState):
    temp_game = GreatKingdomGame()
    temp_game.board = final_state.board.copy()
    p1_terr, p2_terr = temp_game.compute_confirmed_territories_board(final_state.board)

    CELL_SYMBOLS = {
        EMPTY: '.',
        NEUTRAL: 'N',
        PLAYER1: 'B',
        PLAYER2: 'O',
    }

    TERRITORY_SYMBOLS_P1 = '*'
    TERRITORY_SYMBOLS_P2 = '+'

    board_size = temp_game.board_size
    lines = []
    header = '    ' + ''.join(['{:2d}'.format(i) for i in range(board_size)])
    lines.append(header)
    for i in range(board_size):
        row_str = ' {:2d} '.format(i)
        for j in range(board_size):
            cell = final_state.board[i, j]
            if (i, j) in p1_terr:
                symbol = TERRITORY_SYMBOLS_P1
            elif (i, j) in p2_terr:
                symbol = TERRITORY_SYMBOLS_P2
            else:
                symbol = CELL_SYMBOLS.get(cell, '.')
            row_str += ' ' + symbol
        lines.append(row_str)
    lines.append('')
    return '\n'.join(lines)

def print_final_game_state(final_state: GameState):
    board_str = get_final_game_state_str(final_state)
    print(board_str)

def run_training_loop(num_iterations=100,
                      games_per_iteration=50,
                      mcts_iterations=800,
                      batch_size=64,
                      epochs=10,
                      lr=0.001,
                      pretrained_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = GreatKingdomGame()

    if pretrained_model is not None:
        print("Pretrained model loaded")
        # ### 변경: 네트워크 생성 시 새로운 파라미터 적용
        network = AlphaZeroNet(board_size=9, num_actions=82, num_res_blocks=20, channels=256) 
        network.load_model(pretrained_model)
        network.to(device)
    else:
        # ### 변경: 네트워크 생성 시 새로운 파라미터 적용
        network = AlphaZeroNet(board_size=9, num_actions=82, num_res_blocks=20, channels=256).to(device)

    optimizer = optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    replay_buffer = deque(maxlen=50000)

    total_losses = []
    policy_losses = []
    value_losses = []

    for it in range(num_iterations):
        print(f"=== Iteration {it+1}/{num_iterations} ===")
        iteration_data = []
        all_moves = []
        final_states = []

        if it < 10:
            temperature = 1.0
        elif it < 20:
            temperature = 0.5
        elif it < 50:
            temperature = 0.25
        elif it < 90:
            temperature = 0.1
        else:
            temperature = 0.0

        for g in range(games_per_iteration):
            game.reset()
            episode_data, moves, winner, final_state = self_play_episode(game, network, mcts_iterations=mcts_iterations, temperature=temperature)
            iteration_data.extend(episode_data)
            all_moves.append((moves, winner))
            final_states.append(final_state)

        replay_buffer.extend(iteration_data)

        if not os.path.exists("play_note"):
            os.makedirs("play_note")

        if not os.path.exists(f"play_note/{time_now}"):
            os.makedirs(f"play_note/{time_now}")

        note_path = f"play_note/{time_now}/iteration_{it+1}_games.txt"
        with open(note_path, 'w') as f:
            for idx, (moves, winner) in enumerate(all_moves):
                f.write(f"Game {idx+1}:\n")
                for mv in moves:
                    if mv[1] == 'pass':
                        f.write(f"Player {mv[0]}: pass\n")
                    else:
                        f.write(f"Player {mv[0]}: {mv[1]}, {mv[2]}\n")
                f.write(f"Winner: {winner}\n\n")

            f.write("Last Game Final Board State:\n")
            final_board_str = get_final_game_state_str(final_states[-1])
            f.write(final_board_str)

        print("Iteration 마지막 게임의 최종 보드 상태:")
        print_final_game_state(final_states[-1])

        rb_list = list(replay_buffer)
        it_policy_loss = []
        it_value_loss = []
        it_total_loss = []

        for e in range(epochs):
            random.shuffle(rb_list)
            batches = [rb_list[i:i+batch_size] for i in range(0, len(rb_list), batch_size)]

            for batch in batches:
                states, target_pis, target_vs = zip(*batch)
                states = np.array(states, dtype=np.float32)
                target_pis = np.array(target_pis, dtype=np.float32)
                target_vs = np.array(target_vs, dtype=np.float32)

                states, target_pis, target_vs = augment_data(states, target_pis, target_vs, board_size=9)

                states = torch.tensor(states, dtype=torch.float32, device=device)
                target_pis = torch.tensor(target_pis, dtype=torch.float32, device=device)
                target_vs = torch.tensor(target_vs, dtype=torch.float32, device=device).unsqueeze(-1)

                p, v = network(states)
                policy_loss = -(target_pis * p).sum(dim=1).mean()
                value_loss = loss_fn(v, target_vs)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                it_policy_loss.append(policy_loss.item())
                it_value_loss.append(value_loss.item())
                it_total_loss.append(loss.item())

        mean_policy_loss = np.mean(it_policy_loss) if it_policy_loss else 0
        mean_value_loss = np.mean(it_value_loss) if it_value_loss else 0
        mean_total_loss = np.mean(it_total_loss) if it_total_loss else 0

        print(f"Iteration {it+1}: Total Loss={mean_total_loss:.4f}, Policy Loss={mean_policy_loss:.4f}, Value Loss={mean_value_loss:.4f}")

        if not os.path.exists("checkpoint"):
            os.makedirs("checkpoint")

        if not os.path.exists(f"checkpoint/{time_now}"):
            os.makedirs(f"checkpoint/{time_now}")

        network.save_model(f"checkpoint/{time_now}/alpha_zero_checkpoint_{it+1}.pth")

    network.save_model("alpha_zero_final.pth")
    print("학습 완료!")

    plt.figure(figsize=(10,6))
    plt.plot(total_losses, label='Total Loss')
    plt.plot(policy_losses, label='Policy Loss')
    plt.plot(value_losses, label='Value Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()

if __name__ == "__main__":
    # 변경된 네트워크 파라미터 적용한 상태에서 학습 실행
    run_training_loop(
        num_iterations=100,
        games_per_iteration=50,
        mcts_iterations=800,
        batch_size=64,
        epochs=10,
        lr=0.001,
        pretrained_model=None
    )
