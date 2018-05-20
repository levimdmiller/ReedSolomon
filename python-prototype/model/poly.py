import math
import numpy as np

class Polynomial:
    def __init__(self, coeffs, psi):
        self.psi = psi
        self.coeffs = coeffs
        self.deg = len(coeffs) - 1 # 1 + x + ... + x^n, len = n + 1
        self.msb = math.ceil(math.log(self.deg + 1, 2)) # min{j : degf < 2^j}

    def __add__(self, other):
        if not isinstance(other, Polynomial):
            raise TypeError('Can only add other polynomials.')
        return Polynomial([], self.psi)
        

    def __mul__(self, other):
        """
        IFFT(FFT(self) * FFTX(other))
        """
        res_msb = math.ceil(math.log(self.deg + other.deg + 1, 2)) # msb of self*other
        # check if less than m? (degree of finite field)
        # pad lists so they have correct size.
        self_evals = self.psi.transform(np.append(self.coeffs, [0] * (2**res_msb - self.deg)), res_msb)
        other_evals = other.psi.transform(np.append(other.coeffs, [0] * (2**res_msb - other.deg)), res_msb)
        return Polynomial(self.psi.inverse(np.multiply(self_evals, other_evals), res_msb), self.psi)

    def __div__(self, other):
        pass
