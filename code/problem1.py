#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Created On  : 2023-04-03 00:24
# Last Modified : 2023-04-03 17:18
# Copyright © 2023 myron <yh131996@mail.ustc.edu.cn>
#
# Distributed under terms of the MIT license.


import numpy as np
from numpy import arange, cos, sqrt, sin, abs
from numpy import pi as π
from matplotlib import pyplot as plt

# wave's shape1
def func1(x, C=1, t=0):
    conds = [x < -0.4,\
             np.logical_and(x < -0.2, x >= -0.4),\
             np.logical_and(x >= -0.2, x < -0.1),\
             np.logical_and(x >= -0.1, x < 0),\
             x >= 0]
    funcs = [0,\
             lambda x:1.0-abs(x+0.3)/0.1,\
             0,\
             1,\
             0]
    return np.roll(np.piecewise(x, conds, funcs), int(t*C))

def Upwind(x, C=1, t=1):
    N = x.size
    tmp = np.zeros((2, N), dtype=x.dtype)
    tmp[0] = x.copy()
    tmp[1] = x.copy()
    result = tmp[0]
    for n in range(t):
        cur = n%2
        next = (n%2 + 1)%2
        for i in range(N-1):
            tmp[next,i+1] =  tmp[cur,i+1] - C*(tmp[cur, i+1] - tmp[cur, i])
            result = tmp[next]
    return result

if  __name__ == '__main__':
    res = 0.01
    # Δx: args.resolution
    x = np.arange(-1, 2, res)
    # C is Δt/Δx
    C = 0.7
    # Δt
    t = C * res

    T = 0.5
    n_t = int(T/t)

    print(t, n_t)
    # math output
    M0 = func1(x, C)
    M1 = func1(x, C, n_t)

    # simu output
    S1 = Upwind(M0, C, n_t)
    fig, axs = plt.subplots(2,
                            1,
                            figsize=(8, 6))
    axs[0].plot(x, M1, alpha=0.5)
    axs[0].plot(x, S1)
    plt.show()


