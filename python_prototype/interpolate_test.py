import pytest
import numpy as np
import interpolate
from  ff_factory import B4
from  model.poly import Polynomial

FFT = interpolate.Psi(B4, 4, B4(0))
Polynomial.set_psi(FFT)
f = Polynomial([B4(15), B4(12), B4(3), B4(6)])

#### Interpolation
def interpolation_helper(shift, normalized):
    FFT = interpolate.Psi(B4, 4, B4(shift))
    Polynomial.set_psi(FFT)
    f = Polynomial(convert([15, 12, 3, 6]))
    return list(FFT.transform(f.coeffs, f.msb, normalized))

def convert(items):
    return [B4(x) for x in items]

def test_normalized_interpolation_beta_0():
    # should be f(0), f(1), f(2), f(3) (f normalized)
    assert(interpolation_helper(0, True) == convert([15, 3, 11, 1]))

def test_normalized_interpolation_beta_7():
    # should be f(7), f(6), f(5), f(4) (f normalized)
    assert(interpolation_helper(7, True) == convert([3, 14, 4, 15]))

def test_interpolation_beta_0():
    assert(interpolation_helper(0, False) == convert([15, 3, 0, 11]))

def test_interpolation_beta_7():
    assert(interpolation_helper(7, False) == convert([15, 5, 10, 7]))


#### Inverse Interpolation
def inverse_helper(shift, normalized):
    FFT = interpolate.Psi(B4, 4, B4(shift))
    Polynomial.set_psi(FFT)
    f = Polynomial(convert([15, 12, 3, 6]))
    return Polynomial(FFT.inverse(FFT.transform(f.coeffs, f.msb, normalized), f.msb, normalized)), f

def test_normalized_interpolation_beta_0():
    result, f = inverse_helper(0, True)
    assert(result == f)

def test_normalized_interpolation_beta_7():
    result, f = inverse_helper(7, True)
    assert(result == f)

def test_interpolation_beta_0():
    result, f = inverse_helper(0, False)
    assert(result == f)
    
def test_interpolation_beta_0():
    result, f = inverse_helper(7, False)
    assert(result == f)
    
###### Misc.
def test_eval_p():
    expected = [B4(1),B4(1),B4(6),B4(6),B4(7),B4(7),B4(1),B4(1),B4(1),B4(1),B4(6),B4(6),B4(7),B4(7),B4(1),B4(1)]
    assert([FFT.eval_p(i) for i in range(16)] == expected)

def test_add():
    f = Polynomial([B4(15), B4(12), B4(3), B4(6)])
    g = Polynomial([B4(15), B4(12), B4(3), B4(6)])
    assert(f + g == Polynomial([B4(0), B4(0), B4(0), B4(0)]))

def test_mul():
    f = Polynomial(convert([15, 12, 3, 6]))
    g = Polynomial(convert([0, 0, 0, 0, 15]))
    assert(f*g == Polynomial(convert([0, 0, 0, 0, 10, 8, 2, 4])))
