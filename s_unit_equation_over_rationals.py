r"""
SageMath implementation of the resolution of the S-unit equation over the
rationals.

We solve the S-unit equation

        X + Y = 1,        X, Y  S-units of Q,

i.e. X, Y are non-zero rationals whose numerator and denominator have no
prime divisors outside the finite set S.  Equivalently, writing
X = x/z, Y = y/z with x + y = z and x, y, z pairwise coprime positive
S-integers, we solve the exponential equation

        x + y = z,        x, y, z  S-integers.           (0.1)

The resolution follows B. M. M. de Weger, "Solving exponential Diophantine
equations using lattice basis reduction algorithms", J. Number Theory 26
(1987), 325-367, and his PhD thesis "Algorithms for Diophantine Equations"
(Leiden, 1988), Chapter 6.  It consists of three steps:

1. An initial upper bound for the solutions is derived from lower bounds
   for linear forms in logarithms (Waldschmidt in the real case, van der
   Poorten in the p-adic case).  This is Theorem 5.1 of the paper, i.e.
   Theorem 6.1 of the thesis; it is implemented in
   ``SunitEquationOverQ.initial_bound`` and gives an explicit integer
   bound for ``m(x*y*z) = max_i ord_{p_i}(x*y*z)``.

2. The bound is reduced with real and p-adic approximation lattices and
   the L^3-basis reduction algorithm (Sections 5.B-5.D of the paper,
   Sections 6.3-6.5 of the thesis).

3. The remaining finitely many exponent vectors are enumerated.

Currently only step 1 is implemented.
"""

from sage.all import ZZ, prod, log, exp, RealField, ceil


class SunitEquationOverQ:
    r"""
    Solve the S-unit equation  X + Y = 1  over Q.

    INPUT:

    - ``S`` -- a finite set (or iterable) of rational primes.
    - ``prec`` -- working precision (number of bits) used for the real
      computations (default: 200).
    - ``verbose`` -- if True, print progress information (default: False).

    EXAMPLES:

    Reproduce the bound of de Weger [1987], p. 350, for S = {2, 3, 5}::

        sage: import os, sys
        sage: sys.path.insert(0, os.getcwd())
        sage: from s_unit_equation_over_rationals import SunitEquationOverQ
        sage: eq = SunitEquationOverQ([2, 3, 5])
        sage: float(eq._constants(1, 3/8)["C10"]) < 6.76 * 10**41
        True
        sage: B = eq.initial_bound(mu=1, kappa=3/8)
        sage: B < 675 * 10**39
        True
    """

    def __init__(self, S, prec=200, verbose=False):
        self.S = sorted(set(ZZ(p) for p in S))
        for p in self.S:
            if not p.is_prime():
                raise ValueError("S must consist of primes only")
        self.t = len(self.S)
        self.prec = prec
        self.verbose = verbose
        self.RR = RealField(prec)
        self._e = exp(self.RR(1))

        # Data of Theorem 5.1 that is independent of the parameters mu, kappa.
        self._s = (2 * self.t) // 3                    # s = [2t/3]
        self._P = prod(self.S)                         # P = p_1 * ... * p_t
        self._V = [max(self._e, log(self.RR(p)))       # V_i = max(e, log p_i)
                   for p in self.S]
        self._Omega = prod(self._V[self.t - self._s:])  # V_{t-s+1} ... V_t
        self._G = self._compute_G()

        self._bound_data = None

    def _compute_G(self):
        r"""
        G = max_i G_{p_i} / log p_i, where G_2 = 2, G_3 = 6 and
        G_p = p - 1 for p >= 5.
        """
        RR = self.RR

        def Gp(p):
            return RR(p * (p - 1)) if p in (2, 3) else RR(p - 1)

        return max(Gp(p) / log(RR(p)) for p in self.S)

    # ------------------------------------------------------------------ #
    # Theorem 5.1 (de Weger [1987]) / Theorem 6.1 (thesis)
    # ------------------------------------------------------------------ #
    def _constants(self, mu, kappa):
        r"""
        Return the constants of Theorem 5.1 for the chosen parameters.

        INPUT:

        - ``mu``, ``kappa`` -- real parameters with
          ``2/(s+1) <= mu <= 2`` and ``0 < kappa < mu/2``, where
          ``s = [2t/3]``.

        OUTPUT:

        A dictionary with the constants ``s, P, V, Omega, G, mu, kappa,
        eps, k, C6, C7, C8, C9, C10`` of the theorem, for which all
        primitive solutions of (0.1) satisfy ``m(x*y*z) < C10``.

        This is an internal helper; the public entry point is
        :meth:`initial_bound`.
        """
        RR = self.RR
        if self.t < 3:
            raise ValueError("Theorem 5.1 requires at least 3 primes")

        s = self._s
        mu = RR(mu)
        kappa = RR(kappa)
        if not (RR(2) / (s + 1) <= mu <= RR(2)):
            raise ValueError("mu must satisfy 2/(s+1) <= mu <= 2")
        if not (RR(0) < kappa < mu / 2):
            raise ValueError("kappa must satisfy 0 < kappa < mu/2")

        p1 = self.S[0]
        V = self._V
        Om = self._Omega
        G = self._G

        # C_6: Waldschmidt constant (Lemma 2.1 with n = s)
        C6 = RR(2) ** (9 * s + 26) * RR(s) ** (s + 4) \
            * Om * log(self._e * V[self.t - 2])

        # eps, k as in Theorem 5.1
        eps = (mu - kappa) / ((1 + kappa) * (1 + mu) * (s + 1))
        k = max(RR(16 * s) ** ((1 + 1 / kappa) * (s + 1)),
                (RR(8) / eps) ** ((1 + mu) * (s + 1)),
                RR(16) ** (1 / eps))

        # C_7: van der Poorten constant (Lemma 2.2 with n = s)
        C7 = RR(4 * (s + 1) ** (s + 1)) * k ** (1 + mu) * G * Om

        # C_8, C_9, C_10
        C8 = RR(4) * (C6 + C7 * log(RR(self._P) / RR(p1))) / log(RR(p1))
        C9 = C7 * log(C7) ** 2
        C10 = max(C9, C8 * log(C8) ** 2)

        return dict(s=s, P=self._P, V=V, Omega=Om, G=G, mu=mu, kappa=kappa,
                    eps=eps, k=k, C6=C6, C7=C7, C8=C8, C9=C9, C10=C10)

    def _minimize_c10(self):
        r"""
        Numerically minimize C_10 over the feasible parameters.

        The feasible set is ``2/(s+1) <= mu <= 2`` and
        ``0 < kappa < mu/2``.  This is only used to make the initial bound
        as small as possible; any feasible pair of parameters gives a
        rigorous bound.
        """
        RR = self.RR
        s = self._s
        mu_lo = RR(2) / (s + 1)
        mu_hi = RR(2)
        N = 25

        def scan(mu_lo, mu_hi):
            """Best (C10, mu, kappa) on an N x (N-1) grid."""
            best = None
            for i in range(N + 1):
                mu = mu_lo + (mu_hi - mu_lo) * RR(i) / N
                for j in range(1, N):
                    kappa = (mu / 2) * RR(j) / N
                    C10 = self._constants(mu, kappa)["C10"]
                    if best is None or C10 < best[0]:
                        best = (C10, mu, kappa)
            return best

        # coarse scan over the whole feasible set, then refine around mu
        _, mu, kappa = scan(mu_lo, mu_hi)
        for _ in range(3):
            step = (mu_hi - mu_lo) / N
            mu_lo = max(RR(2) / (s + 1), mu - step)
            mu_hi = min(RR(2), mu + step)
            _, mu, kappa = scan(mu_lo, mu_hi)
        return mu, kappa

    def initial_bound(self, mu=None, kappa=None):
        r"""
        Return an integer ``B`` such that ``m(x*y*z) <= B`` for every
        primitive solution (x, y, z) of (0.1).

        This is Theorem 5.1 of de Weger [1987] (Theorem 6.1 of the
        thesis), giving ``m(x*y*z) < C_10``; the returned value is
        ``ceil(C_10) + 1`` (the ``+1`` absorbs the rounding error of the
        real computations).

        INPUT:

        - ``mu``, ``kappa`` -- parameters with ``2/(s+1) <= mu <= 2`` and
          ``0 < kappa < mu/2``.  If omitted, they are chosen numerically
          to minimize ``C_10`` (any feasible choice yields a rigorous
          bound).

        OUTPUT:

        An integer ``B`` with ``m(x*y*z) <= B`` for all primitive
        solutions, where ``m(a) = max_i ord_{p_i}(a)``.  In particular
        ``x*y*z <= P**B``, so ``log(x*y*z) <= B*log(P)``.
        """
        if mu is None or kappa is None:
            mu, kappa = self._minimize_c10()
        data = self._constants(mu, kappa)
        self._bound_data = data
        if self.verbose:
            print("C6 = %s" % data["C6"])
            print("C7 = %s" % data["C7"])
            print("C8 = %s" % data["C8"])
            print("C9 = %s" % data["C9"])
            print("C10 = %s" % data["C10"])
        return ZZ(ceil(data["C10"])) + 1
