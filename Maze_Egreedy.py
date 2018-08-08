# 使用するパッケージの宣言
import numpy as np
import matplotlib.pyplot as plt



# 方策パラメータthetaを行動方策piに変換する関数の定義
def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

def softmax_convert_into_pi_from_theta(theta):
    beta = 1.
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    exp_theta = np.exp(beta * theta)

    for i in range(m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
    pi = np.nan_to_num(pi)
    return pi


# 行動aと1step移動後の状態sを求める関数を定義
def get_action(s, Q, epsilon, pi_0):
    direction = ['up', 'right', 'down', 'left']

    # 行動を決める
    if np.random.rand() < epsilon:
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:
        next_direction = direction[np.nanargmax(Q[s, :])]

    if next_direction == 'up':
        action = 0
    elif next_direction == 'right':
        action = 1
    elif next_direction == 'down':
        action = 2
    elif next_direction == 'left':
        action = 3

    return action


def get_s_next(s, a, Q, epsilon, pi_0):
    direction = ['up', 'right', 'down', 'left']
    next_direction = direction[a]

    if next_direction == 'up':
        s_next = s - 3
    elif next_direction == 'right':
        s_next = s + 1
    elif next_direction == 'down':
        s_next = s + 3
    elif next_direction == 'left':
        s_next = s - 1

    return s_next


# Sarsaによる行動価値関数Qの更新
def sarsa(s, a, r, s_next, a_next, Q, eta, gamma):
    if s_next == 8:
        Q[s,a] = Q[s,a] + eta*(r - Q[s,a])
    else:
        Q[s,a] = Q[s,a] + eta*(r + gamma*Q[s_next,a_next] - Q[s,a])

    return Q


# 迷路を解く関数の定義。状態と行動の履歴を出力
def goal_maze_ret_s_a(pi):
    s = 0
    s_a_history = [[0, np.nan]]

    while (1):
        [action, next_s] = get_action_and_next_s(pi, s)
        s_a_history[-1][1] = action
        s_a_history.append([next_s, np.nan])

        if next_s == 8:
            break
        else:
            s = next_s

    return s_a_history


# thetaの更新関数を定義
def update_theta(theta, pi, s_a_history):
    eta = 0.1
    T = len(s_a_history) - 1

    [m, n] = theta.shape
    delta_theta = theta.copy()

    for i in range(m):
        for j in range(n):
            if not (np.isnan(theta[i, j])):
                SA_i = [SA for SA in s_a_history if SA[0] == i]
                SA_ij = [SA for SA in s_a_history if SA == [i, j]]

                N_i = len(SA_i)
                N_ij = len(SA_ij)
                delta_theta[i, j] = (N_ij + pi[i, j] * N_i) / T

    new_theta = theta + eta * delta_theta
    return new_theta




# 初期位置での迷路の様子

# 図を描く大きさと、図の変数名を宣言
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()

# 赤い壁を描く
ax.plot([1, 1], [0, 1], color='red', linewidth=2)
ax.plot([1, 2], [2, 2], color='red', linewidth=2)
ax.plot([2, 2], [2, 1], color='red', linewidth=2)
ax.plot([2, 3], [1, 1], color='red', linewidth=2)

# 状態を示す文字S0〜S8を描く
ax.text(0.5, 2.5, 'S0', size=14, ha='center')
ax.text(1.5, 2.5, 'S1', size=14, ha='center')
ax.text(2.5, 2.5, 'S2', size=14, ha='center')
ax.text(0.5, 1.5, 'S3', size=14, ha='center')
ax.text(1.5, 1.5, 'S4', size=14, ha='center')
ax.text(2.5, 1.5, 'S5', size=14, ha='center')
ax.text(0.5, 0.5, 'S6', size=14, ha='center')
ax.text(1.5, 0.5, 'S7', size=14, ha='center')
ax.text(2.5, 0.5, 'S8', size=14, ha='center')
ax.text(0.5, 2.3, 'START', size=14, ha='center')
ax.text(2.5, 0.3, 'GOAL', size=14, ha='center')

# 描画範囲の設定と目盛りを消す設定
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
               labelleft=False)

# 現在地S0に緑丸を描画する
line, = ax.plot([0.5], [2.5], marker='o', color='g', markersize=60)

# 初期の方策を決定するパラメータtheta_0を設定
# 行は状態、列は移動方向で上、右、下、左を表す
theta_0 = np.array([
    [np.nan, 1, 1, np.nan],  # S0
    [np.nan, 1, np.nan, 1],  # S1
    [np.nan, np.nan, 1, 1],  # S2
    [1, 1, 1, np.nan],  # S3
    [np.nan, np.nan, 1, 1],  # S4
    [1, np.nan, np.nan, np.nan],  # S5
    [1, np.nan, np.nan, np.nan],  # S6
    [1, 1, np.nan, np.nan]  # S7
])

# 初期の行動価値関数Qを設定
[a, b] = theta_0.shape
Q = np.random.rand(a, b) * theta_0


# 初期の方策pi_0を求める
pi_0 = simple_convert_into_pi_from_theta(theta_0)

#
# pi = pi_0
# theta = theta_0
# is_continue = True
# stop_epsilon = 10 ** -6
# while is_continue:
#     # 迷路内をゴール目指して移動
#     s_a_history = goal_maze_ret_s_a(pi)
#
#     # 方策の更新
#     new_theta = update_theta(theta, pi, s_a_history)
#     new_pi = softmax_convert_into_pi_from_theta(new_theta)
#
#     if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
#         is_continue = False
#         print("かかったステップ数は{}", str(len(s_a_history) - 1))
#         print(s_a_history)
#         np.set_printoptions(precision=3, suppress=True)
#         print(new_pi)
#     else:
#         theta = new_theta
#         pi = new_pi


