# play.py

import torch
from copy import deepcopy

from game import GreatKingdomGame, GameState, PLAYER1, PLAYER2, EMPTY
from model import AlphaZeroNet, MCTS

def human_turn(game: GreatKingdomGame):
    while True:
        game.print_board()
        print('당신의 턴입니다. 좌표(x y)를 입력하거나 "pass"를 입력:')
        command = input().strip()
        if command == 'pass':
            if game.pass_turn():
                return
            else:
                print("패스할 수 없습니다.")
                continue

        try:
            x, y = map(int, command.split())
            if not (0 <= x < game.board_size and 0 <= y < game.board_size):
                print("올바른 좌표를 입력하세요.")
                continue
            if game.place_castle(x, y, game.current_player) is False:
                continue
            else:
                return
        except ValueError:
            print("올바른 입력 형식을 사용하세요.")
            continue

def ai_turn(game: GreatKingdomGame, network: AlphaZeroNet, mcts_iterations=1000):
    state = GameState(deepcopy(game.board), game.current_player, game.consecutive_passes, game.game_over, game.winner)
    mcts = MCTS(game, network, iterations=mcts_iterations, c_param=1.0)  # c_param 적절히
    best_action = mcts.search(state)

    if best_action == game.get_action_size() - 1:
        game.pass_turn()
    else:
        x = best_action // game.board_size
        y = best_action % game.board_size
        game.place_castle(x, y, game.current_player)

def main():
    game = GreatKingdomGame()

    network = AlphaZeroNet(board_size=9, num_actions=82)
    network.load_model('alpha_zero_final.pth')
    network.eval()

    print("게임 시작!")
    print("당신은 파란색 (B) 성입니다. AI는 주황색 (O) 성입니다.")
    print("중립 성(N)은 보드 중앙에 있습니다.")

    while not game.game_over:
        if game.current_player == PLAYER1:
            human_turn(game)
        else:
            ai_turn(game, network, mcts_iterations=1000)

    game.print_board()
    if game.winner == PLAYER1:
        print("당신(파란색)이 승리했습니다!")
    elif game.winner == PLAYER2:
        print("AI(주황색)가 승리했습니다.")
    else:
        print("무승부입니다.")

if __name__ == "__main__":
    main()
