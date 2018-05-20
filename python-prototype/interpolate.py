

class Psi:
    """
    Operator for the FFT algorithm
    """
    def __init__(self, field, degree, shift):
        self.field = field
        self.shift = shift
        self.basis = self.find_basis(degree) # v_0, ..., v_{m-1}
        self.precomputed = [
            self.eval_s(j, shift) / self.eval_s(j, self.basis[j])
            for j in range(degree)
        ]

    def find_basis(self, degree):
        # 1, x, x^2, ... is a basis.
        # representation of elements is based on binary number corresponding to the polynomial.
        # so 1 = 001 = 1, 2 = 010 = x, 4 = 100 = x^2
        basis = [self.field(2**i) for i in range(4)] # 1, x, x^2, ... is a basis.
        
        # Check that it is a basis.
        r = range(2)
        coeffs = [[a, b, c, d] for a in r for b in r for c in r for d in r][1:]
        for vec in coeffs:
            result = sum([x*y for x,y in zip(basis, vec)])
            if(result == self.field(0)):
                raise ValueError("Not a basis: {}{}".format(result, vec))
        return basis

    def omega(self, i):
        # works because of how the basis is chosen, and because
        # of how the table is indexed.
        # e.g., elem 3 = 011 = 1 + x = b0 + b1
        return self.field(i)

    def eval_s(self, k, omega):
        """
        uses the recursive definition
        """
        if k == 0:
            return omega # s_0(x) = x
        # s_k(x) = s_{k-1}(x) * s_{k-1}(x - v_{k-1})
        return self.eval_s(k-1, omega) * self.eval_s(k-1, omega - self.basis[k-1])
        
    def transform(self, poly):
        """
        \psi_{\beta}(f)
        """
        return self._transform_helper(poly.coeffs, poly.msb, self.shift)

    def _transform_helper(self, coeffs, k, beta):
        """
        Input: f(x) = (f_0, ..., f_{2^k - 1}), k = bin log of size, \beta = shift
        Output: 2^k evaluations, d_i = f(\omega + \beta)
        """
        if k == 0:
            return [coeffs[0]] # return f_0
        exp = 2**(k-1)
        D0 = [] # input for first recursive call
        D1 = [] # input for second recursive call
        for i in range(exp):
            g0 = coeffs[i] + self.precomputed[k-1] * coeffs[i + exp] # 'even' subset calculations
            g1 = g0 + coeffs[i + exp] # 'odd' subset calculations
            D0.append(g0)
            D1.append(g1)

        # call recursively on subproblems
        result = self._transform_helper(D0, k-1, beta) # d'_0, ..., d'_{2^{k-1}-1}
        result.extend(self._transform_helper(D1, k-1, beta)) # d'_{2^{k-1}}, ..., d'_{2^k}
        return result # d'_0, ..., d'_{2^k}

    def inverse(self, evals, k):
        """
        \psi^{-1}_{\beta}(0,0) \leftarrow FFT_h(\Delta^0_0,\beta,0,0)
        k = binary log of degree of polynomial.
        """
        return self.inverse_helper(evals, k, self.shift)

    def inverse_helper(self, evals, k, beta):
        """
        Input: (f(\omega_0), ..., f(\omega_{2^k-1}), k = bin log of size, \beta = shift
        Output: 2^k evaluations, d_i = f(\omega + \beta)
        """
        if k == 0:
            return [evals[0]] # return f_0
        exp = 2**(k-1)
        D0 = self.inverse_helper(evals[:exp], k-1, beta) # g0_0, ..., g0_{2^{k-1}-1}
        D1 = self.inverse_helper(evals[exp:], k-1, self.basis[k-1] + beta) # g1_0, ..., g1_{2^{k-1}-1}
        first_half = [] # can replace with an array in non-python impl.
        second_half = []
        for i in range(exp):
            d_odd = D0[i] + D1[i] # f_{i + 2^{k-1}}
            d_even = D0[i] + self.precomputed[k-1] * d_odd # f_i
            first_half.append(d_odd)
            second_half.append(d_even)
        second_half.extend(first_half)
        return second_half
