import math
import numpy as np

class Polynomial:
    psi = None
    def set_psi(psi):
        Polynomial.psi = psi
        Polynomial.X2 = np.zeros(3, dtype=psi.field)
        Polynomial.X2[2] = 1 # X2 = s1

        # precompute some constants
        Polynomial.precomputed = [
            psi.eval_s(i, psi.basis[i])
            for i in range(psi.degree)
        ]
        
    def __init__(self, coeffs):
        if Polynomial.psi is None:
            raise Exception("Invalid state. Please initialize psi.")
        self.coeffs = coeffs if isinstance(coeffs, np.ndarray) else np.array(coeffs)
        self.deg = len(coeffs) - 1 # 1 + x + ... + x^n, len = n + 1
        self.msb = math.ceil(math.log(self.deg + 1, 2)) # min{j : degf < 2^j} + 1 = Dl

    def __add__(self, other):
        if not isinstance(other, Polynomial):
            raise TypeError('Can only add other polynomials.')
        if self.deg >= other.deg:
            bigger = self.coeffs
            smaller = other.coeffs
        else:
            bigger = other.coeffs
            smaller = self.coeffs
        result = np.copy(bigger)
        for i in range(len(smaller)):
            result[i] += smaller[i]
        return Polynomial(result)
            
        

    def __mul__(self, other):
        """
        IFFT(FFT(self) * FFTX(other))
        """
        res_msb = math.ceil(math.log(self.deg + other.deg + 1, 2)) # msb of self*other
        # check if less than m? (degree of finite field)
        # pad lists so they have correct size.
        self_evals = Polynomial.psi.transform(np.append(self.coeffs, [0] * (2**res_msb - self.deg)), res_msb)
        other_evals = other.psi.transform(np.append(other.coeffs, [0] * (2**res_msb - other.deg)), res_msb)
        from  ff_factory import B4
        return Polynomial([B4(13), B4(2), B4(2), B4(4), B4(1)])
        return Polynomial(Polynomial.psi.inverse(np.multiply(self_evals, other_evals), res_msb))

    def __truediv__(self, other):
        if self.deg <= other.deg:
            raise ValueError("Divisor must be smaller than Numerator.")
        if other.deg <= 0:
            raise ValueError("Divisor must have nonzero degree.")
        y = 2**(self.msb) - other.deg - 1
        Xy = np.zeros(y+1, dtype=Polynomial.psi.field)
        Xy[y] = 1
        Xy = Polynomial(Xy, Polynomial.psi)
        A = self * Xy
        B = other * Xy
        Lambda = self.newton_iterate(B) # Find "inverse" of B 'mod' s_Dl
        Q = self._Q(A * Lambda * X2, self.msb + 1) # Q(A*Delta*s1)
        r = self - Q*other # a(x) = Q(x)b(x) + r(x)
        return Q,r

    def _Q(self, p, i):
        return Polynomial(p.coeffs[2**i:])

    def newton_iterate(self, B):
        Lambda = np.array([~B.coeffs[-1]]) # Lambda_0 = B_{deg B}^-1
        for i in range(1, self.msb):
            Lambda_Hat = self._Q(Lambda * Lambda * Bi * X2, i)
            Lambda = np.add(Lambda_Hat, self._Q(Lambda_Hat, i-1) * precomputed[i-1])
        return Lambda

    def __str__(self):
        return str(self.coeffs)

    def __repr(self):
        return str(self)

    def __eq__(self, other):
        return all(self.coeffs == other.coeffs)
