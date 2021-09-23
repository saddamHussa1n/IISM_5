import random
import numpy as np
import typing as ty
import matplotlib.pyplot as mpp
from mpl_toolkits.mplot3d import Axes3D


def monte_carlo(a: ty.List[ty.List[float]], f: ty.List[float], m: int, N: int):
    res = list()
    pi = [1 / len(a)] * len(a)
    p = [pi] * 3

    for x_num in range(len(a)):
        h = [0] * x_num + [1] + [0] * (len(a) - x_num - 1)
        ksi = [0] * m

        # Моделирование m цепей Маркова длины N
        for j in range(m):

            i = list()  # цепь Маркова
            for _ in range(N + 1):
                alpha = random.random()

                for number in range(len(pi)):
                    if alpha < sum(pi[:number + 1]):
                        i.append(number)
                        break

            # Вычисление весов цепи Маркова
            Q = [h[i[0]]/pi[i[0]] if pi[i[0]] > 0 else 0]
            for k in range(1, N + 1):
                q = Q[k-1] * a[i[k-1]][i[k]] / p[i[k-1]][i[k]] if p[i[k-1]][i[k]] > 0 else 0
                Q.append(q)

            for k in range(N + 1):
                ksi[j] += Q[k] * f[i[k]]

        x = sum(ksi) / m
        res.append(x)

    return res


if __name__ == '__main__':
    M3 = np.array([
        [0.7, -0.3, -0.2],
        [-0.2, 0.2, -0.7],
        [-0.5, 0.1, 2]
    ])
    v3 = np.array([-1, -3, 8])
    res_true = np.linalg.solve(M3, v3)
    print(res_true)

    a_ = [
        [0.3, 0.3, 0.2],
        [0.1, 0.5, -0.3],
        [0, 0, -0.1]
    ]
    f_ = [-1, -1, 4]

    X, Y, Z = list(), list(), list()
    vars_ = [100 * i_ for i_ in range(1, 5)]
    for m_ in vars_:
        fs = list()
        for N_ in vars_:
            f_val = abs(np.array(res_true) - np.array(monte_carlo(a_, f_, m_, N_)))[0]
            print(np.array(monte_carlo(a_, f_, m_, N_)))
            # print(m_, N_, f_val)
            fs.append(f_val)

        X.append([m_] * len(vars_))
        Y.append(vars_)
        Z.append(fs)

    fig = mpp.figure()
    axes = Axes3D(fig)
    axes.set_xlabel('Кол-во цепей')
    axes.set_ylabel('Длина цепи')
    axes.set_zlabel('Разница')

    axes.plot_surface(np.array(X), np.array(Y), np.array(Z))

    mpp.show()
