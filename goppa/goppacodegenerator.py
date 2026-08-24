from mceliece.mathutils import *

from sympy import GF, Poly
from sympy.abc import x, alpha

import logging
import time

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

    Consequently the current implementation requires

        n <= 2^m - 1.

    Phase 2E.1
    ----------

    Phase 2E.1 adds detailed instrumentation to

        _generate_goppa_polynomial()

    without changing the mathematical construction.

    After that method runs, timing/counter information is available in

        self.goppa_profile.

    The profile separates:

        candidate generation,
        cheap candidate rejection,
        first_alpha_power_root(),
        reduce_to_alpha_power(),
        final support-wide verification,

    together with candidate counts and total timings.
    """

    # ============================================================
    # Construction
    # ============================================================

    def __init__(
        self,
        m,
        n,
        t,
    ):
        self.m = int(m)
        self.n = int(n)
        self.t = int(t)

        self.q = 2

        self.field_size = (
            self.q ** self.m
        )

        self.multiplicative_order = (
            self.field_size - 1
        )

        if self.m <= 0:
            raise ValueError(
                "m must be positive."
            )

        if self.t <= 0:
            raise ValueError(
                "t must be positive."
            )

        if self.n <= 0:
            raise ValueError(
                "n must be positive."
            )

        if self.n > self.multiplicative_order:
            raise ValueError(
                "Current nonzero-power support requires "
                f"n <= 2^m - 1 = "
                f"{self.multiplicative_order}, "
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

        # Support diagnostics.
        self.support_exponents = None
        self.support = None

        # Phase 2E.1 profiler.
        self.goppa_profile = (
            self._empty_goppa_profile()
        )

    # ============================================================
    # Phase 2E.1 profiling helpers
    # ============================================================

    @staticmethod
    def _empty_goppa_profile():
        """
        Return a fresh Goppa-polynomial profiling dictionary.

        All timings use time.perf_counter().
        """

        return {
            # Candidate-loop counters
            "candidate_attempts": 0,
            "candidate_accepted_attempt": 0,

            # Candidate generation
            "candidate_generation_sec": 0.0,

            # Cheap checks:
            #
            #     f(0) != 0
            #     f(1) != 0
            #
            "cheap_rejection_sec": 0.0,
            "cheap_rejections": 0,

            # first_alpha_power_root(...)
            "first_alpha_power_root_sec": 0.0,
            "first_alpha_power_root_calls": 0,
            "first_alpha_power_root_rejections": 0,
            "first_alpha_power_root_max_sec": 0.0,

            # Entire candidate loop
            "candidate_loop_sec": 0.0,

            # Representation conversion
            "reduce_to_alpha_power_sec": 0.0,

            # Degree verification
            "degree_check_sec": 0.0,

            # Final verification g(L_j) != 0
            "final_support_verification_sec": 0.0,
            "final_support_points": 0,

            # Entire method
            "total_sec": 0.0,
        }

    # ============================================================
    # Extension-field construction
    # ============================================================

    def _generate_primitive_field_polynomial(
        self,
    ):
        """
        Find an irreducible polynomial defining GF(2^m) for which
        alpha generates the multiplicative group.

        power_dict() has size 2^m - 1 exactly when alpha is
        primitive.
        """

        candidate = Poly(
            alpha ** self.m
            + alpha
            + 1,
            alpha,
            domain=GF(
                self.q
            ),
        )

        if is_irreducible_poly(
            candidate,
            self.q,
        ):
            ring = power_dict(
                self.field_size,
                candidate,
                self.q,
            )

        else:
            ring = {}

        log.info(
            "candidate field polynomial: "
            f"ring size={len(ring)}, "
            f"irr={candidate}"
        )

        while (
            len(ring)
            <
            self.multiplicative_order
        ):
            candidate = (
                irreducible_poly(
                    self.m,
                    self.q,
                    alpha,
                )
                .set_domain(
                    GF(
                        self.q
                    )
                )
            )

            ring = power_dict(
                self.field_size,
                candidate,
                self.q,
            )

            log.info(
                "candidate field polynomial: "
                f"ring size={len(ring)}, "
                f"irr={candidate}"
            )

        return (
            candidate,
            ring,
        )

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

        Phase 2E.1 instruments each expensive component while
        preserving the existing construction.
        """

        total_start = (
            time.perf_counter()
        )

        profile = (
            self._empty_goppa_profile()
        )

        self.goppa_profile = profile

        g_poly = Poly(
            1,
            x,
        )

        # --------------------------------------------------------
        # Current construction uses no prescribed roots.
        # --------------------------------------------------------

        g_roots = set()

        all_nonzero_exponents = set(
            range(
                self.multiplicative_order
            )
        )

        g_non_roots = list(
            sorted(
                all_nonzero_exponents
                -
                g_roots
            )
        )

        log.debug(
            f"g_roots({len(g_roots)}) = "
            f"{g_roots}"
        )

        log.debug(
            f"g_non_roots("
            f"{len(g_non_roots)}) = "
            f"{g_non_roots}"
        )

        # --------------------------------------------------------
        # Prescribed-root machinery retained for compatibility.
        # --------------------------------------------------------

        for exponent in g_roots:
            g_poly = (
                g_poly
                *
                Poly(
                    x
                    +
                    alpha ** exponent,
                    x,
                )
            ).trunc(
                self.q
            )

        # --------------------------------------------------------
        # Complete g(x) to degree t.
        # --------------------------------------------------------

        if (
            g_poly.degree()
            <
            self.t
        ):
            required_degree = (
                self.t
                -
                g_poly.degree()
            )

            small_irr = None

            candidate_loop_start = (
                time.perf_counter()
            )

            for attempt in range(
                1,
                101,
            ):
                profile[
                    "candidate_attempts"
                ] += 1

                # =================================================
                # 1. Candidate generation
                # =================================================

                candidate_start = (
                    time.perf_counter()
                )

                small_irr = (
                    irreducible_poly_ext_candidate(
                        required_degree,
                        irr_poly,
                        self.q,
                        x,
                        non_roots=g_non_roots,
                    )
                )

                candidate_elapsed = (
                    time.perf_counter()
                    -
                    candidate_start
                )

                profile[
                    "candidate_generation_sec"
                ] += candidate_elapsed

                log.debug(
                    "candidate g-component "
                    f"(attempt {attempt}) = "
                    f"{small_irr}"
                )

                # =================================================
                # 2. Cheap rejection at 0 and 1
                # =================================================

                cheap_start = (
                    time.perf_counter()
                )

                zero_is_root = (
                    small_irr
                    .eval(
                        0
                    )
                    .is_zero
                )

                one_is_root = (
                    small_irr
                    .eval(
                        1
                    )
                    .is_zero
                )

                cheap_elapsed = (
                    time.perf_counter()
                    -
                    cheap_start
                )

                profile[
                    "cheap_rejection_sec"
                ] += cheap_elapsed

                if (
                    zero_is_root
                    or
                    one_is_root
                ):
                    profile[
                        "cheap_rejections"
                    ] += 1

                    continue

                # =================================================
                # 3. Full alpha-power root search
                # =================================================

                root_start = (
                    time.perf_counter()
                )

                first_root = (
                    first_alpha_power_root(
                        small_irr,
                        irr_poly,
                        self.q,
                    )
                )

                root_elapsed = (
                    time.perf_counter()
                    -
                    root_start
                )

                profile[
                    "first_alpha_power_root_sec"
                ] += root_elapsed

                profile[
                    "first_alpha_power_root_calls"
                ] += 1

                profile[
                    "first_alpha_power_root_max_sec"
                ] = max(
                    profile[
                        "first_alpha_power_root_max_sec"
                    ],
                    root_elapsed,
                )

                if first_root > 0:
                    profile[
                        "first_alpha_power_root_rejections"
                    ] += 1

                    continue

                # Candidate accepted.
                profile[
                    "candidate_accepted_attempt"
                ] = attempt

                break

            else:
                profile[
                    "candidate_loop_sec"
                ] = (
                    time.perf_counter()
                    -
                    candidate_loop_start
                )

                profile[
                    "total_sec"
                ] = (
                    time.perf_counter()
                    -
                    total_start
                )

                self.goppa_profile = (
                    profile
                )

                raise RuntimeError(
                    "Unable to generate a suitable "
                    "degree-t Goppa polynomial."
                )

            profile[
                "candidate_loop_sec"
            ] = (
                time.perf_counter()
                -
                candidate_loop_start
            )

            g_poly = (
                g_poly
                *
                small_irr
            ).trunc(
                self.q
            )

        # ========================================================
        # 4. Convert coefficients to alpha-power representation
        # ========================================================

        reduce_start = (
            time.perf_counter()
        )

        g_poly = (
            reduce_to_alpha_power(
                g_poly,
                irr_poly,
                ring,
                self.q,
            )
        )

        profile[
            "reduce_to_alpha_power_sec"
        ] = (
            time.perf_counter()
            -
            reduce_start
        )

        # ========================================================
        # 5. Degree check
        # ========================================================

        degree_start = (
            time.perf_counter()
        )

        degree_ok = (
            g_poly.degree()
            ==
            self.t
        )

        profile[
            "degree_check_sec"
        ] = (
            time.perf_counter()
            -
            degree_start
        )

        if not degree_ok:
            profile[
                "total_sec"
            ] = (
                time.perf_counter()
                -
                total_start
            )

            self.goppa_profile = (
                profile
            )

            raise RuntimeError(
                "Generated Goppa polynomial "
                f"has degree "
                f"{g_poly.degree()}, "
                f"expected {self.t}."
            )

        # ========================================================
        # 6. Verify g(L_j) != 0 over complete support
        # ========================================================

        verification_start = (
            time.perf_counter()
        )

        irr_field = (
            irr_poly
            .set_domain(
                GF(
                    self.q
                )
            )
        )

        verified_points = 0

        for exponent in (
            support_exponents
        ):
            support_element = (
                alpha ** exponent
            )

            value_expr = (
                g_poly
                .as_expr()
                .subs(
                    x,
                    support_element,
                )
            )

            value = (
                Poly(
                    value_expr,
                    alpha,
                    domain=GF(
                        self.q
                    ),
                )
                .rem(
                    irr_field
                )
            )

            verified_points += 1

            if value.is_zero:
                profile[
                    "final_support_points"
                ] = (
                    verified_points
                )

                profile[
                    "final_support_verification_sec"
                ] = (
                    time.perf_counter()
                    -
                    verification_start
                )

                profile[
                    "total_sec"
                ] = (
                    time.perf_counter()
                    -
                    total_start
                )

                self.goppa_profile = (
                    profile
                )

                raise RuntimeError(
                    "Goppa polynomial "
                    "vanishes on support: "
                    f"g(alpha^{exponent}) = 0."
                )

        profile[
            "final_support_points"
        ] = verified_points

        profile[
            "final_support_verification_sec"
        ] = (
            time.perf_counter()
            -
            verification_start
        )

        # ========================================================
        # Total timing
        # ========================================================

        profile[
            "total_sec"
        ] = (
            time.perf_counter()
            -
            total_start
        )

        self.goppa_profile = (
            profile
        )

        log.info(
            f"g(x) = {g_poly}"
        )

        log.info(
            "Goppa polynomial profile: "
            f"attempts="
            f"{profile['candidate_attempts']}, "
            f"candidate_gen="
            f"{profile['candidate_generation_sec']:.6f}s, "
            f"first_root="
            f"{profile['first_alpha_power_root_sec']:.6f}s, "
            f"reduce="
            f"{profile['reduce_to_alpha_power_sec']:.6f}s, "
            f"support_verify="
            f"{profile['final_support_verification_sec']:.6f}s, "
            f"total="
            f"{profile['total_sec']:.6f}s"
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

        For column j,

            H_{i,j}
                =
            N_i(L_j) / g(L_j),

        with

            N_{t-1}(L)
                =
            g_t,

            N_i(L)
                =
            L N_{i+1}(L)
            +
            g_{t-i-1}.

        No explicit n x n diagonal matrix is constructed.
        """

        coeffs = (
            g_poly
            .all_coeffs()
        )

        if (
            len(coeffs)
            !=
            self.t + 1
        ):
            raise RuntimeError(
                "Unexpected number of "
                "Goppa-polynomial coefficients."
            )

        H_ext = [
            [
                0
                for _ in range(
                    self.n
                )
            ]
            for _ in range(
                self.t
            )
        ]

        for (
            j,
            exponent,
        ) in enumerate(
            support_exponents
        ):
            L_j = (
                alpha ** exponent
            )

            # ----------------------------------------------------
            # Compute 1/g(L_j).
            # ----------------------------------------------------

            g_value = (
                g_poly
                .eval(
                    L_j
                )
            )

            g_inverse = (
                get_alpha_power(
                    g_value,
                    irr_poly,
                    ring,
                    self.q,
                    neg=True,
                )
            )

            # ----------------------------------------------------
            # Numerator recurrence.
            # ----------------------------------------------------

            numerators = [
                0
                for _ in range(
                    self.t
                )
            ]

            numerators[
                self.t - 1
            ] = (
                get_alpha_power(
                    coeffs[0],
                    irr_poly,
                    ring,
                    self.q,
                )
            )

            for i in range(
                self.t - 2,
                -1,
                -1,
            ):
                coefficient_index = (
                    self.t
                    -
                    1
                    -
                    i
                )

                recurrence_value = (
                    L_j
                    *
                    numerators[
                        i + 1
                    ]
                    +
                    coeffs[
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
            # Scale column by 1/g(L_j).
            # ----------------------------------------------------

            for i in range(
                self.t
            ):
                H_ext[i][j] = (
                    get_alpha_power(
                        numerators[i]
                        *
                        g_inverse,
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

        Result:

            H_bin in GF(2)^(mt x n).
        """

        H_binary = np.zeros(
            (
                self.m
                *
                self.t,
                self.n,
            ),
            dtype=np.uint8,
        )

        for i in range(
            self.t
        ):
            for j in range(
                self.n
            ):
                binary = (
                    get_binary_from_alpha(
                        H_ext[i][j],
                        irr_poly,
                        self.q,
                    )
                )

                if (
                    len(binary)
                    !=
                    self.m
                ):
                    raise RuntimeError(
                        "Unexpected extension-field "
                        "binary representation length."
                    )

                for b in range(
                    self.m
                ):
                    H_binary[
                        i * self.m + b,
                        j,
                    ] = int(
                        binary[b].n
                    )

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
        """

        # --------------------------------------------------------
        # 1. Construct GF(2^m).
        # --------------------------------------------------------

        (
            irr_poly,
            ring,
        ) = (
            self
            ._generate_primitive_field_polynomial()
        )

        log.debug(
            f"field power table = "
            f"{ring}"
        )

        # --------------------------------------------------------
        # 2. Support:
        #
        #     L_j = alpha^j,
        #
        # j = 0,...,n-1.
        # --------------------------------------------------------

        support_exponents = list(
            range(
                self.n
            )
        )

        support = [
            alpha ** exponent
            for exponent
            in support_exponents
        ]

        self.support_exponents = (
            support_exponents
        )

        self.support = (
            support
        )

        log.info(
            "support length = "
            f"{len(support)}"
        )

        # --------------------------------------------------------
        # 3. Goppa polynomial.
        # --------------------------------------------------------

        g_poly = (
            self
            ._generate_goppa_polynomial(
                irr_poly,
                ring,
                support_exponents,
            )
        )

        # --------------------------------------------------------
        # 4. Extension-field parity-check matrix.
        # --------------------------------------------------------

        H_ext = (
            self
            ._build_extension_parity_check(
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
            self
            ._binary_expand_parity_check(
                H_ext,
                irr_poly,
            )
        )

        expected_shape = (
            self.m
            *
            self.t,
            self.n,
        )

        if (
            H_bin.arr.shape
            !=
            expected_shape
        ):
            raise RuntimeError(
                "Unexpected H shape "
                f"{H_bin.arr.shape}; "
                f"expected {expected_shape}."
            )

        log.info(
            "H_bin shape = "
            f"{H_bin.arr.shape}"
        )

        log.debug(
            f"H_bin =\n"
            f"{H_bin}"
        )

        # --------------------------------------------------------
        # 6. Generator from right nullspace:
        #
        #     H G^T = 0
        #
        # or equivalently
        #
        #     G H^T = 0.
        # --------------------------------------------------------

        (
            H_nullspace,
            nullity,
        ) = (
            H_bin
            .nullspace()
        )

        log.info(
            "binary parity-check rank = "
            f"{self.n - nullity}"
        )

        log.info(
            "code dimension k = "
            f"{nullity}"
        )

        G = GF2Matrix(
            H_nullspace
            .T()[
                :nullity
            ]
        )

        # --------------------------------------------------------
        # 7. Verify G H^T = 0.
        # --------------------------------------------------------

        orthogonality = (
            G
            *
            H_bin.T()
        )

        is_zero = all(
            int(e.n) == 0
            for e in (
                orthogonality
                .arr
                .flat
            )
        )

        if not is_zero:
            raise RuntimeError(
                "Internal Goppa-code "
                "construction error: "
                "G H^T != 0."
            )

        log.info(
            f"G shape = "
            f"{G.arr.shape}"
        )

        log.debug(
            f"G =\n"
            f"{G}"
        )

        return (
            G,
            H_bin,
            g_poly,
            irr_poly,
        )