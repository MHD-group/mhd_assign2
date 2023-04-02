#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Created On  : 2023-04-03 00:24
# Last Modified : 2023-04-03 02:35
# Copyright © 2023 myron <yh131996@mail.ustc.edu.cn>
#
# Distributed under terms of the MIT license.


import numpy as np
from numpy import arange, cos, sqrt, sin, abs
from numpy import pi as π
from matplotlib import pyplot as plt
import argparse

# wave's shape1
def func1(x, t=0):
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
    return np.roll(np.piecewise(x, conds, funcs), t)

# wave's shape2
def func2(x, t=0):
    conds = [x < -0.8,\
             np.logical_and(x < -0.3, x >= -0.8),\
             np.logical_and(x < 0, x >= -0.3),\
             x >= 0]
    funcs = [1.8,\
             lambda x : 1.4 + 0.4 * cos(2*π * (x+0.8) ),\
             1.0,\
             1.8]
    return np.roll(np.piecewise(x, conds, funcs), t)

# Upwind (unfinished)
def Upwind(x, t=1):
    return func1(x, t)

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description="calculate X to the power of Y")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("-x", "--resolution", default=0.1, type=float, help="length of Δx")
    parser.add_argument("-C", "--ratio", default=2, type=int, help="Δt/Δx")
    parser.add_argument("-i", "--input", default=1, type=int, help="f(x) when t=0")
    parser.add_argument("-m", "--method", default="Upwind", type=str, help="methods")
    args = parser.parse_args()

    # Δx: args.resolution
    x = np.arange(-1, 2, args.resolution)
    # C is Δt/Δx
    C = args.ratio
    # Δt
    t = C * args.resolution

    T = 0.5
    n_t = int(T/t)

    # math output
    if args.input == 1:
        M0 = func1(x)
        M1 = func1(x, n_t)
    elif args.input == 2:
        M0 = func2(x)
        M1 = func2(x, n_t)
    else:
        print("error input function")

    # simu output
    if args.method == "Upwind":
        S1 = Upwind(x, n_t)
    else:
        print("error input function")
    fig, axs = plt.subplots(2,
                            1,
                            figsize=(8, 6))
    axs[0].plot(x, M0)
    axs[1].plot(x, M1)
    axs[1].plot(x, S1)
    plt.show()


