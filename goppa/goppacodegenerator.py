from mceliece.mathutils import *

from sympy import GF, Poly
from sympy.abc import x, alpha

import logging
import numpy as np


log = logging.getLogger("goppacodegenerator")


class GoppaCodeGenerator:
    """
    Generator for a binary Goppa code Gamma(L, g).

    Current support convention
    --------------------------
    The support consists of nonzero powers

        L = (1, alpha, alpha^2, ..., alpha^(n-1))

    inside GF(2^m).

    Consequently this implementation currently requires

        n <= 2^m - 1.

    Support including the zero field element will be added in the
    next scalability stage.
    """

    def __init__(self, m, n, t):
        self.m = int(m)
        self.n = int(n)
        self.t = int(t)

        self.q = 2

        self.field_size = self.q ** self.m
        self.multiplicative_order = self.field_size - 1

        if self.m <= 0:
            raise ValueError("m must be positive.")

        if self.t <= 0:
            raise ValueError("t must be positive.")

        if self.n <= 0:
            raise ValueError("n must be positive.")

        # For this stage the support contains only nonzero elements.
        if self.n > self.multiplicative_order:
            raise ValueError(
                f"Current nonzero-power support requires "
                f"n <= 2^m - 1 = {self.multiplicative_order}, "
                f"but n={self.n}."
            )

        log.info(
            "GoppaCodeGenerator("
            f"m={self.m}, "
            f"n={self.n}, "
            f"t={self.t}, "
            f"q={self.q}, "
            f"q^m={self.field_size}"
            ") initiated"
        )

        # Diagnostic/support information.
        self.support_exponents = None
        self.support = None

    # ============================================================
    # Extension-field construction
    # ============================================================

    def _generate_primitive_field_polynomial(self):
        """
        Find an irreducible polynomial defining GF(2^m) for which
        alpha generates the multiplicative group.

        power_dict() has size 2^m - 1 exactly when alpha is primitive.
        """

        candidate = Poly(
            alpha ** self.m + alpha + 1,
            alpha,
            domain=GF(self.q),
        )

        if is_irreducible_poly(candidate, self.q):
            ring = power_dict(
                self.field_size,
                candidate,
                self.q,
            )
        else:
            ring = {}

        log.info(
            f"candidate field polynomial: "
            f"ring size={len(ring)}, "
            f"irr={candidate}"
        )

        while len(ring) < self.multiplicative_order:
            candidate = irreducible_poly(
                self.m,
                self.q,
                alpha,
            ).set_domain(GF(self.q))

            ring = power_dict(
                self.field_size,
                candidate,
                self.q,
            )

            log.info(
                f"candidate field polynomial: "
                f"ring size={len(ring)}, "
                f"irr={candidate}"
            )

        return candidate, ring

    # ============================================================
    # Goppa polynomial
    # ============================================================

    def _generate_goppa_polynomial(
        self,
        irr_poly,
        ring,
        support_exponents,
    ):
        """
        Generate a degree-t Goppa polynomial with no zeros on the
        selected support.
        """

        g_poly = Poly(
            1,
            x,
        )

        # --------------------------------------------------------
        # The original implementation contained machinery for
        # prescribed roots outside the support.  At the present
        # stage we use no prescribed roots and generate the entire
        # degree-t polynomial as an irreducible extension-field
        # polynomial.
        # --------------------------------------------------------

        g_roots = set()

        all_nonzero_exponents = set(
            range(self.multiplicative_order)
        )

        g_non_roots = list(
            sorted(
                all_nonzero_exponents - g_roots
            )
        )

        log.debug(
            f"g_roots({len(g_roots)}) = {g_roots}"
        )

        log.debug(
            f"g_non_roots({len(g_non_roots)}) = "
            f"{g_non_roots}"
        )

        for exponent in g_roots:
            g_poly = (
                g_poly
                * Poly(
                    x + alpha ** exponent,
                    x,
                )
            ).trunc(self.q)

        # --------------------------------------------------------
        # Complete g(x) to degree t.
        # --------------------------------------------------------

        if g_poly.degree() < self.t:
            required_degree = (
                self.t - g_poly.degree()
            )

            small_irr = None

            for _ in range(100):
                small_irr = (
                    irreducible_poly_ext_candidate(
                        required_degree,
                        irr_poly,
                        self.q,
                        x,
                        non_roots=g_non_roots,
                    )
                )

                log.debug(
                    f"candidate g-component = {small_irr}"
                )

                # Support contains 1 = alpha^0.
                if (
                    small_irr.eval(0).is_zero
                    or
                    small_irr.eval(1).is_zero
                ):
                    continue

                first_root = first_alpha_power_root(
                    small_irr,
                    irr_poly,
                    self.q,
                )

                if first_root > 0:
                    continue

                break

            else:
                raise RuntimeError(
                    "Unable to generate a suitable "
                    "degree-t Goppa polynomial."
                )

            g_poly = (
                g_poly
                * small_irr
            ).trunc(self.q)

        # Convert all nonzero extension-field coefficients into
        # alpha-power representation.
        g_poly = reduce_to_alpha_power(
            g_poly,
            irr_poly,
            ring,
            self.q,
        )

        if g_poly.degree() != self.t:
            raise RuntimeError(
                f"Generated Goppa polynomial has degree "
                f"{g_poly.degree()}, expected {self.t}."
            )

        # --------------------------------------------------------
        # Explicitly verify that g(L_j) != 0 for every support
        # element L_j.
        # --------------------------------------------------------

        irr_field = irr_poly.set_domain(
            GF(self.q)
        )

        for exponent in support_exponents:
            support_element = (
                alpha ** exponent
            )

            value_expr = g_poly.as_expr().subs(
                x,
                support_element,
            )

            value = Poly(
                value_expr,
                alpha,
                domain=GF(self.q),
            ).rem(irr_field)

            if value.is_zero:
                raise RuntimeError(
                    f"Goppa polynomial vanishes on support: "
                    f"g(alpha^{exponent}) = 0."
                )

        log.info(
            f"g(x) = {g_poly}"
        )

        return g_poly

    # ============================================================
    # Direct extension-field parity-check construction
    # ============================================================

    def _build_extension_parity_check(
        self,
        g_poly,
        irr_poly,
        ring,
        support_exponents,
    ):
        """
        Construct the t x n parity-check matrix directly.

        The previous implementation formed

            H = C X Y,

        where Y was an explicit n x n diagonal matrix.

        For column j, Y only multiplies that column by

            1 / g(L_j).

        We therefore compute each column directly.

        The numerator recurrence is algebraically equivalent to
        multiplication by C X:

            N_{t-1}(L) = g_t,

            N_i(L)
                = L N_{i+1}(L)
                  + g_{t-i-1},

        and

            H_{i,j}
                = N_i(L_j) / g(L_j).

        All intermediate field expressions are reduced immediately,
        preventing symbolic expression growth.
        """

        coeffs = g_poly.all_coeffs()

        if len(coeffs) != self.t + 1:
            raise RuntimeError(
                "Unexpected number of Goppa-polynomial "
                "coefficients."
            )

        H_ext = [
            [0 for _ in range(self.n)]
            for _ in range(self.t)
        ]

        for j, exponent in enumerate(
            support_exponents
        ):
            L_j = alpha ** exponent

            # ----------------------------------------------------
            # Compute g(L_j)^(-1).
            # ----------------------------------------------------

            g_value = g_poly.eval(
                L_j
            )

            g_inverse = get_alpha_power(
                g_value,
                irr_poly,
                ring,
                self.q,
                neg=True,
            )

            # ----------------------------------------------------
            # Compute the C X column using a recurrence.
            #
            # Bottom row:
            #
            #     N_{t-1} = g_t.
            #
            # Here coeffs[0] is the leading coefficient g_t.
            # ----------------------------------------------------

            numerators = [
                0 for _ in range(self.t)
            ]

            numerators[self.t - 1] = (
                get_alpha_power(
                    coeffs[0],
                    irr_poly,
                    ring,
                    self.q,
                )
            )

            # Work upward.
            for i in range(
                self.t - 2,
                -1,
                -1,
            ):
                coefficient_index = (
                    self.t - 1 - i
                )

                recurrence_value = (
                    L_j
                    * numerators[i + 1]
                    + coeffs[
                        coefficient_index
                    ]
                )

                numerators[i] = (
                    get_alpha_power(
                        recurrence_value,
                        irr_poly,
                        ring,
                        self.q,
                    )
                )

            # ----------------------------------------------------
            # Scale the complete column by 1/g(L_j).
            # ----------------------------------------------------

            for i in range(self.t):
                H_ext[i][j] = (
                    get_alpha_power(
                        numerators[i]
                        * g_inverse,
                        irr_poly,
                        ring,
                        self.q,
                    )
                )

        return H_ext

    # ============================================================
    # Binary expansion
    # ============================================================

    def _binary_expand_parity_check(
        self,
        H_ext,
        irr_poly,
    ):
        """
        Expand each GF(2^m) entry into m binary coordinates.

        The resulting matrix has dimensions

            (m t) x n.
        """

        H_binary = np.zeros(
            (
                self.m * self.t,
                self.n,
            ),
            dtype=np.uint8,
        )

        for i in range(self.t):
            for j in range(self.n):
                binary = (
                    get_binary_from_alpha(
                        H_ext[i][j],
                        irr_poly,
                        self.q,
                    )
                )

                if len(binary) != self.m:
                    raise RuntimeError(
                        "Unexpected extension-field "
                        "binary representation length."
                    )

                for b in range(self.m):
                    H_binary[
                        i * self.m + b,
                        j,
                    ] = int(binary[b].n)

        return GF2Matrix.from_list(
            H_binary
        )

    # ============================================================
    # Main generation routine
    # ============================================================

    def gen(self):
        """
        Generate

            G,
            H_bin,
            g(x),
            irr_poly.

        The public interface is kept compatible with the existing
        McElieceCipher implementation.
        """

        # --------------------------------------------------------
        # 1. Construct GF(2^m).
        # --------------------------------------------------------

        irr_poly, ring = (
            self._generate_primitive_field_polynomial()
        )

        log.debug(
            f"field power table = {ring}"
        )

        # --------------------------------------------------------
        # 2. Explicit support.
        #
        # For this stage:
        #
        #     L_j = alpha^j,
        #
        # j = 0,...,n-1.
        # --------------------------------------------------------

        support_exponents = list(
            range(self.n)
        )

        support = [
            alpha ** exponent
            for exponent in support_exponents
        ]

        self.support_exponents = (
            support_exponents
        )

        self.support = support

        log.info(
            f"support length = {len(support)}"
        )

        # --------------------------------------------------------
        # 3. Generate degree-t Goppa polynomial.
        # --------------------------------------------------------

        g_poly = (
            self._generate_goppa_polynomial(
                irr_poly,
                ring,
                support_exponents,
            )
        )

        # --------------------------------------------------------
        # 4. Direct extension-field parity-check matrix.
        #
        # No n x n matrix is constructed.
        # --------------------------------------------------------

        H_ext = (
            self._build_extension_parity_check(
                g_poly,
                irr_poly,
                ring,
                support_exponents,
            )
        )

        # --------------------------------------------------------
        # 5. Binary expansion.
        # --------------------------------------------------------

        H_bin = (
            self._binary_expand_parity_check(
                H_ext,
                irr_poly,
            )
        )

        expected_shape = (
            self.m * self.t,
            self.n,
        )

        if H_bin.arr.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected H shape "
                f"{H_bin.arr.shape}; "
                f"expected {expected_shape}."
            )

        log.info(
            f"H_bin shape = {H_bin.arr.shape}"
        )

        log.debug(
            f"H_bin =\n{H_bin}"
        )

        # --------------------------------------------------------
        # 6. Generator matrix from right nullspace of H.
        #
        #             H G^T = 0
        #
        # equivalently
        #
        #             G H^T = 0.
        # --------------------------------------------------------

        H_nullspace, nullity = (
            H_bin.nullspace()
        )

        log.info(
            f"binary parity-check rank = "
            f"{self.n - nullity}"
        )

        log.info(
            f"code dimension k = {nullity}"
        )

        G = GF2Matrix(
            H_nullspace.T()[
                :nullity
            ]
        )

        # --------------------------------------------------------
        # 7. Internal invariant.
        # --------------------------------------------------------

        orthogonality = (
            G
            * H_bin.T()
        )

        is_zero = all(
            int(e.n) == 0
            for e in orthogonality.arr.flat
        )

        if not is_zero:
            raise RuntimeError(
                "Internal Goppa-code construction error: "
                "G H^T != 0."
            )

        log.info(
            f"G shape = {G.arr.shape}"
        )

        log.debug(
            f"G =\n{G}"
        )

        return (
            G,
            H_bin,
            g_poly,
            irr_poly,
        )