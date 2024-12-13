import torch
from copy import deepcopy

from game import GreatKingdomGame, GameState, PLAYER1, PLAYER2, EMPTY
from model import AlphaZeroNet, MCTS

def human_turn(game: GreatKingdomGame):
    while True:
        game.print_board()
        print('당신의 턴입니다. 좌표(x y)를 입력하거나 "pass" 입력:')
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
            # 해당 위치에 놓을 수 있는지 확인
            state_before = GameState(deepcopy(game.board), game.current_player, game.consecutive_passes, game.game_over, game.winner)
            if game.place_castle(x, y, game.current_player) is False:
                # place_castle이 실패하면 False 반환. 실패 시 다시 입력 받기
                continue
            else:
                return
        except ValueError:
            print("올바른 입력 형식을 사용하세요.")
            continue

def ai_turn(game: GreatKingdomGame, network: AlphaZeroNet, mcts_iterations=100):
    # AI가 MCTS를 통해 수를 선택
    state = GameState(deepcopy(game.board), game.current_player, game.consecutive_passes, game.game_over, game.winner)
    mcts = MCTS(game, network, iterations=mcts_iterations)
    best_action = mcts.search(state)

    if best_action == game.get_action_size() - 1:
        # pass
        game.pass_turn()
    else:
        x = best_action // game.board_size
        y = best_action % game.board_size
        game.place_castle(x, y, game.current_player)

def main():
    # 게임 초기화
    game = GreatKingdomGame()

    # 학습된 모델 로드
    network = AlphaZeroNet(board_size=9, num_actions=82)
    network.load_model('alpha_zero_final.pth')
    network.eval()

    print("게임 시작!")
    print("당신은 파란색 (B) 성입니다. AI는 주황색 (O) 성입니다.")
    print("중립 성(N)은 보드 중앙에 있습니다.")

    while not game.game_over:
        if game.current_player == PLAYER1:
            # 인간 플레이어 턴
            human_turn(game)
        else:
            # AI 플레이어 턴
            ai_turn(game, network, mcts_iterations=1000)

    # 게임 종료 후 결과 출력
    game.print_board()
    if game.winner == PLAYER1:
        print("당신(파란색)이 승리했습니다!")
    elif game.winner == PLAYER2:
        print("AI(주황색)가 승리했습니다.")
    else:
        print("무승부입니다.")

if __name__ == "__main__":
    main()
