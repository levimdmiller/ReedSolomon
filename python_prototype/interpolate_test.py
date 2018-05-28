import pytest
import numpy as np
import interpolate
from  ff_factory import B4
from  model.poly import Polynomial

FFT = interpolate.Psi(B4, 4, B4(0))
Polynomial.set_psi(FFT)
f = Polynomial([B4(15), B4(12), B4(3), B4(6)])

def test_normalized_interpolation_beta_0():
    FFT = interpolate.Psi(B4, 4, B4(0))
    Polynomial.set_psi(FFT)
    f = Polynomial([B4(15), B4(12), B4(3), B4(6)])
    # should be f(0), f(1), f(2), f(3) (f normalized)
    assert(list(FFT.transform(f.coeffs, f.msb, normalized=True)) == [B4(15), B4(3), B4(11), B4(1)])

def test_normalized_interpolation_beta_7():
    FFT = interpolate.Psi(B4, 4, B4(7))
    Polynomial.set_psi(FFT)
    f = Polynomial([B4(15), B4(12), B4(3), B4(6)])
    # should be f(7), f(6), f(5), f(4) (f normalized)
    assert(list(FFT.transform(f.coeffs, f.msb, normalized=True)) == [B4(3), B4(14), B4(4), B4(15)])

def test_interpolation_beta_0():
    FFT = interpolate.Psi(B4, 4, B4(0))
    Polynomial.set_psi(FFT)
    f = Polynomial([B4(15), B4(12), B4(3), B4(6)])
    assert(list(FFT.transform(f.coeffs, f.msb)) == [B4(15), B4(3), B4(0), B4(11)])

def test_interpolation_beta_7():
    FFT = interpolate.Psi(B4, 4, B4(7))
    Polynomial.set_psi(FFT)
    f = Polynomial([B4(15), B4(12), B4(3), B4(6)])
    assert(list(FFT.transform(f.coeffs, f.msb)) == [B4(15), B4(5), B4(10), B4(7)])
    
def test_eval_p():
    expected = [B4(1),B4(1),B4(6),B4(6),B4(7),B4(7),B4(1),B4(1),B4(1),B4(1),B4(6),B4(6),B4(7),B4(7),B4(1),B4(1)]
    assert([FFT.eval_p(i) for i in range(16)] == expected)

def test_add():
    f = Polynomial([B4(15), B4(12), B4(3), B4(6)])
    g = Polynomial([B4(15), B4(12), B4(3), B4(6)])
    assert(f + g == Polynomial([B4(0), B4(0), B4(0), B4(0)]))

def test_mul():
    f = Polynomial([B4(15), B4(12), B4(3), B4(6)])
    g = Polynomial([B4(2), B4(7)])
    assert(f*g == Polynomial([B4(13), B4(2), B4(2), B4(4), B4(1)]))
