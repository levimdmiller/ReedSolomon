from interpolate import *
from model.poly import Polynomial
import numpy as np
from ff_factory import B4
from functools import reduce
import operator

FFT = Psi(B4, 4, B4(0))
Polynomial.set_psi(FFT)

f = Polynomial(np.array([B4(15), B4(12), B4(3), B4(6)]))

evals = FFT.transform(f.coeffs, f.msb, normalized=True)
print('FFT', evals)
print('IFFT', FFT.inverse(evals, f.msb, normalized=True))

g = np.zeros(5, dtype=B4)
g[4] = B4(15)
g = Polynomial(g)

print((f * g).coeffs)
print(~B4(6))

print(f, g)
print(f + g)

def conv(omega):
    return omega if isinstance(omega, B4) else B4(omega)

def brute_X(i, omega):
    omega = conv(omega)
    num = i
    bit = 0 # current position in binary expansion
    result = B4(1)
    while num > 0:
        if num % 2 == 1: # if current bit has a 1
            result *= FFT.eval_s(bit, omega) # s_bit(omega)
        bit += 1
        num //= 2
    return result

def eval(omega, *args):
    omega = conv(omega)
    result = B4(0)
    for i in range(len(args)):
        result += args[i] * brute_X(i, omega)
    return result
        

def add(*args):
    return sum([conv(x) for x in args])

def mul(*args):
    return reduce(operator.mul, [conv(x) for x in args], 1)

print('expected', [eval(i, *[x / p for x,p in zip(f.coeffs, FFT.norm_consts[:len(f.coeffs)])]) for i in range(len(f.coeffs))])
print('expected', [eval(i, *f.coeffs) for i in range(8)])
