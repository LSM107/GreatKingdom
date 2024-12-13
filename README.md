# GreatKingdom
Great Kingdom 보드게임을 MCTS 기반 알파제로 알고리즘으로 학습합니다.

`model.py`는 에이전트가 사용하는 모델을 정의한 파이썬 파일입니다. 알파제로에서와 같이 residual connenction을 포함하는 CNN 구조로 되어있습니다. 그리고 MCTS 탐색을 구현하기 위한 로직들을 포함합니다.
`game.py`는 Gread Kingdom 게임을 정의한 파이썬 파일입니다. 공성, 영토 세기와 같은 메서드들이 내부에 정의돼 있습니다.
`train_self_play.py`는 자가 학습을 수행하는 파이썬 파일입니다.
'play.py'는 `train_self_play.py`에서 학습한 에이전트와 경기를 할 수 있는 파이썬 파일입니다.
