import logging

import numpy as np

from goppa.goppacodegenerator import GoppaCodeGenerator
from mceliece.mathutils import (
    GF2Matrix,
    GF2Poly,
    GF2m,
    GF2mPoly,
    GF2mRing,
    ext_euclid_poly_alt,
    get_binary_from_alpha,
    random_inv_matrix,
    rref,
)


log = logging.getLogger("mceliececipher")


class McElieceCipher:
    """
    Binary Goppa-code McEliece implementation.

    Parameters
    ----------
    m : int
        Extension degree.  The Goppa code is defined using GF(2^m).

    n : int
        Code length.

    t : int
        Degree of the Goppa polynomial and the number of errors
        injected by the baseline McEliece encryption algorithm.

    Notes
    -----
    The current implementation uses:

        G       secret binary Goppa generator matrix
        H       secret binary parity-check matrix
        S       secret invertible scrambling matrix
        P       secret permutation vector
        P_inv   inverse permutation vector
        Gp      public generator matrix

    The public generator satisfies mathematically

        G_pub = S G P.

    The permutation P is represented by an integer vector instead
    of a dense n x n permutation matrix.

    For a row vector x, the action of P is

        x P := x[P].

    Therefore

        (x P) P^{-1}
        = x[P][P_inv]
        = x.

    This reduces permutation storage from O(n^2) entries to O(n).
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

        # --------------------------------------------------------
        # Basic parameter validation
        # --------------------------------------------------------

        if self.m <= 0:
            raise ValueError(
                "m must be positive."
            )

        if self.n <= 0:
            raise ValueError(
                "n must be positive."
            )

        if self.t <= 0:
            raise ValueError(
                "t must be positive."
            )

        if self.n > 2 ** self.m:
            raise ValueError(
                f"n={self.n} exceeds "
                f"field size 2^m={2 ** self.m}."
            )

        log.info(
            "McEliece("
            f"m={self.m}, "
            f"n={self.n}, "
            f"t={self.t}, "
            f"q={self.q}, "
            f"q^m={self.q ** self.m}"
            ") initiated"
        )

        # --------------------------------------------------------
        # Secret Goppa-code data
        # --------------------------------------------------------

        self.G = None
        self.H = None
        self.k = None

        # --------------------------------------------------------
        # McEliece permutation
        #
        # Phase 2B representation:
        #
        #     P     : ndarray shape (n,)
        #     P_inv : ndarray shape (n,)
        #
        # rather than dense n x n binary matrices.
        # --------------------------------------------------------

        self.P = None
        self.P_inv = None

        # --------------------------------------------------------
        # Scrambling matrix
        # --------------------------------------------------------

        self.S = None
        self.S_inv = None

        # --------------------------------------------------------
        # Public generator
        # --------------------------------------------------------

        self.Gp = None

        # --------------------------------------------------------
        # Goppa polynomial and extension-field modulus
        # --------------------------------------------------------

        self.g_poly = None
        self.irr_poly = None

    # ============================================================
    # Internal conversion helpers
    # ============================================================

    @staticmethod
    def _as_gf2_matrix(value):
        """
        Convert value into GF2Matrix if necessary.

        This allows the class to work with both objects generated
        directly in memory and arrays restored from serialized keys.
        """

        if isinstance(
            value,
            GF2Matrix,
        ):
            return value

        arr = np.asarray(
            value
        )

        # A NumPy object array can occasionally contain a single
        # GF2Matrix object after serialization.
        if arr.ndim == 0:
            item = arr.item()

            if isinstance(
                item,
                GF2Matrix,
            ):
                return item

            arr = np.asarray(
                item
            )

        return GF2Matrix.from_list(
            arr
        )

    @staticmethod
    def _as_permutation_vector(
        value,
        n,
        name="permutation",
    ):
        """
        Normalize and validate a permutation vector.

        Parameters
        ----------
        value
            Array-like object representing a permutation.

        n : int
            Required permutation length.

        name : str
            Name used in error messages.

        Returns
        -------
        numpy.ndarray
            Integer array of shape (n,).

        A valid permutation must contain each integer

            0, 1, ..., n-1

        exactly once.
        """

        arr = np.asarray(
            value,
            dtype=np.int64,
        ).reshape(-1)

        if arr.shape != (n,):
            raise ValueError(
                f"{name} must have shape ({n},), "
                f"received {arr.shape}."
            )

        expected = np.arange(
            n,
            dtype=np.int64,
        )

        if not np.array_equal(
            np.sort(arr),
            expected,
        ):
            raise ValueError(
                f"{name} is not a valid "
                f"permutation of 0,...,{n - 1}."
            )

        return arr

    # ============================================================
    # Permutation helpers
    # ============================================================

    @staticmethod
    def _permute_vector(
        vector,
        permutation,
    ):
        """
        Apply a permutation to a GF2 row vector.

        If

            y = x[P],

        this method returns y.
        """

        vector = McElieceCipher._as_gf2_matrix(
            vector
        )

        permutation = np.asarray(
            permutation,
            dtype=np.int64,
        ).reshape(-1)

        if vector.arr.ndim != 1:
            raise ValueError(
                "_permute_vector expects "
                "a one-dimensional GF2 vector."
            )

        if len(vector) != len(permutation):
            raise ValueError(
                "Vector and permutation lengths "
                "do not agree."
            )

        return GF2Matrix(
            vector.arr[
                permutation
            ]
        )

    @staticmethod
    def _permute_columns(
        matrix,
        permutation,
    ):
        """
        Apply a right-hand permutation to matrix columns.

        If A is k x n, mathematically this implements

            A P

        using

            A[:, P].
        """

        matrix = McElieceCipher._as_gf2_matrix(
            matrix
        )

        permutation = np.asarray(
            permutation,
            dtype=np.int64,
        ).reshape(-1)

        if matrix.arr.ndim != 2:
            raise ValueError(
                "_permute_columns expects "
                "a two-dimensional matrix."
            )

        if matrix.arr.shape[1] != len(
            permutation
        ):
            raise ValueError(
                "Matrix column count and "
                "permutation length do not agree."
            )

        return GF2Matrix(
            matrix.arr[
                :,
                permutation
            ]
        )

    # ============================================================
    # Key generation
    # ============================================================

    def generate_random_keys(self):
        """
        Generate a binary Goppa code and McEliece key pair.

        The secret objects are

            G,
            H,
            S,
            S^{-1},
            P,
            P^{-1}.

        The public generator is

            G_pub = S G P.

        In Phase 2B, P is represented by a permutation vector.
        """

        # --------------------------------------------------------
        # 1. Generate binary Goppa code
        # --------------------------------------------------------

        (
            self.G,
            self.H,
            self.g_poly,
            self.irr_poly,
        ) = GoppaCodeGenerator(
            self.m,
            self.n,
            self.t,
        ).gen()

        # --------------------------------------------------------
        # 2. Store Goppa polynomial coefficients
        #
        # Every coefficient lies in GF(2^m).  Store each
        # coefficient as its m-bit basis representation:
        #
        #     1, alpha, ..., alpha^(m-1).
        # --------------------------------------------------------

        g_coeffs = []

        for coefficient in (
            self.g_poly
            .all_coeffs()[::-1]
        ):
            bits = get_binary_from_alpha(
                coefficient,
                self.irr_poly,
                2,
            )

            g_coeffs.append(
                [
                    int(bit.n)
                    for bit in bits
                ]
            )

        self.g_poly = np.array(
            g_coeffs,
            dtype=np.uint8,
        )

        # --------------------------------------------------------
        # 3. Store irreducible extension-field polynomial
        #
        # Coefficients are stored in ascending order.
        # --------------------------------------------------------

        self.irr_poly = np.array(
            [
                int(c) % 2
                for c
                in self.irr_poly
                .all_coeffs()[::-1]
            ],
            dtype=np.uint8,
        )

        # --------------------------------------------------------
        # 4. Actual binary code dimension
        # --------------------------------------------------------

        self.k = (
            self.G.arr.shape[0]
        )

        log.info(
            "Generated binary Goppa code "
            f"[n={self.n}, k={self.k}]"
        )

        # --------------------------------------------------------
        # 5. Generate secret permutation
        #
        # P is a vector containing each coordinate exactly once.
        # --------------------------------------------------------

        self.P = np.random.permutation(
            self.n
        ).astype(
            np.int64
        )

        self.P_inv = np.argsort(
            self.P
        ).astype(
            np.int64
        )

        # --------------------------------------------------------
        # 6. Validate permutation and inverse
        # --------------------------------------------------------

        identity_positions = np.arange(
            self.n,
            dtype=np.int64,
        )

        if not np.array_equal(
            np.sort(self.P),
            identity_positions,
        ):
            raise RuntimeError(
                "Generated P is not a valid permutation."
            )

        if not np.array_equal(
            np.sort(self.P_inv),
            identity_positions,
        ):
            raise RuntimeError(
                "Generated P_inv is not a valid permutation."
            )

        if not np.array_equal(
            self.P[
                self.P_inv
            ],
            identity_positions,
        ):
            raise RuntimeError(
                "Invalid permutation inverse: "
                "P[P_inv] != identity."
            )

        if not np.array_equal(
            self.P_inv[
                self.P
            ],
            identity_positions,
        ):
            raise RuntimeError(
                "Invalid permutation inverse: "
                "P_inv[P] != identity."
            )

        # --------------------------------------------------------
        # 7. Generate secret scrambling matrix
        # --------------------------------------------------------

        self.S = GF2Matrix.from_list(
            random_inv_matrix(
                self.k
            )
        )

        self.S_inv = (
            self.S.inv()
        )

        # --------------------------------------------------------
        # 8. Construct public generator
        #
        # Mathematically:
        #
        #     G_pub = S G P.
        #
        # Compute SG first, then apply P as a column permutation.
        # --------------------------------------------------------

        SG = (
            self.S
            * self.G
        )

        self.Gp = (
            self._permute_columns(
                SG,
                self.P,
            )
        )

        # --------------------------------------------------------
        # 9. Sanity checks
        # --------------------------------------------------------

        if self.Gp.arr.shape != (
            self.k,
            self.n,
        ):
            raise RuntimeError(
                "Public generator has unexpected "
                f"shape {self.Gp.arr.shape}."
            )

        log.info(
            "McEliece key generation completed."
        )

    # ============================================================
    # Encryption
    # ============================================================

    def encrypt(
        self,
        msg_arr,
    ):
        """
        Encrypt a k-bit message using the public generator matrix.

        Baseline McEliece encryption computes

            c = m G_pub + e,

        where

            wt(e) = t.
        """

        if self.Gp is None:
            raise RuntimeError(
                "Public key has not been "
                "generated or loaded."
            )

        Gp = self._as_gf2_matrix(
            self.Gp
        )

        k = Gp.arr.shape[0]
        n = Gp.arr.shape[1]

        msg_arr = np.asarray(
            msg_arr,
            dtype=np.uint8,
        ).flatten()

        if len(msg_arr) != k:
            raise ValueError(
                "Wrong message length. "
                f"Expected {k} bits, "
                f"received {len(msg_arr)}."
            )

        if self.t > n:
            raise ValueError(
                f"Cannot insert {self.t} errors "
                f"into a ciphertext of length {n}."
            )

        log.debug(
            f"plaintext = {msg_arr}"
        )

        message = GF2Matrix.from_list(
            msg_arr
        )

        # --------------------------------------------------------
        # Clean public codeword
        #
        #     c_0 = m G_pub.
        # --------------------------------------------------------

        ciphertext = (
            message
            * Gp
        )

        log.debug(
            f"clean ciphertext = {ciphertext}"
        )

        # --------------------------------------------------------
        # Sample exactly t distinct error coordinates
        # --------------------------------------------------------

        bits_to_flip = np.random.choice(
            n,
            size=self.t,
            replace=False,
        )

        log.debug(
            "error positions = "
            f"{bits_to_flip}"
        )

        for position in (
            bits_to_flip
        ):
            ciphertext[
                position
            ] = ciphertext[
                position
            ].flip()

        log.debug(
            f"noisy ciphertext = {ciphertext}"
        )

        return ciphertext

    # ============================================================
    # Patterson error repair
    # ============================================================

    def repair_errors(
        self,
        msg_arr,
        syndrome,
    ):
        """
        Repair errors using the Patterson-style decoder from the
        original implementation.

        The current Goppa support is

            L_i = alpha^i,

        for

            i = 0,...,n-1.

        Consequently the locator polynomial is evaluated at

            1, alpha, alpha^2, ...
        """

        # --------------------------------------------------------
        # 1. Reconstruct binary extension-field modulus
        # --------------------------------------------------------

        if not isinstance(
            self.irr_poly,
            GF2Poly,
        ):
            self.irr_poly = (
                GF2Poly.from_numpy(
                    np.asarray(
                        self.irr_poly,
                        dtype=np.uint8,
                    )
                )
            )

        ring = GF2mRing(
            self.m,
            self.irr_poly,
        )

        # --------------------------------------------------------
        # 2. Reconstruct Goppa polynomial over GF(2^m)
        # --------------------------------------------------------

        if not isinstance(
            self.g_poly,
            GF2mPoly,
        ):
            coefficients = []

            for coefficient in (
                self.g_poly
            ):
                coefficient_poly = (
                    GF2Poly.from_list(
                        [
                            int(e)
                            for e
                            in coefficient
                        ]
                    )
                )

                coefficients.append(
                    GF2m(
                        coefficient_poly,
                        ring,
                    )
                )

            self.g_poly = (
                GF2mPoly.from_list(
                    coefficients
                )
            )

        log.debug(
            f"irr_poly = {self.irr_poly}"
        )

        log.debug(
            f"g_poly = {self.g_poly}"
        )

        # --------------------------------------------------------
        # 3. Convert binary syndrome to polynomial over GF(2^m)
        # --------------------------------------------------------

        syndrome_matrix = (
            self._as_gf2_matrix(
                syndrome
            )
        )

        syndrome_flat = list(
            syndrome_matrix
            .arr
            .flat
        )

        syndrome_coefficients = []

        number_of_coefficients = (
            len(syndrome_flat)
            // self.m
        )

        for i in range(
            number_of_coefficients
        ):
            block = syndrome_flat[
                i * self.m:
                (i + 1) * self.m
            ]

            binary_poly = (
                GF2Poly.from_list(
                    [
                        int(e)
                        for e
                        in block
                    ]
                )
            )

            syndrome_coefficients.append(
                GF2m(
                    binary_poly,
                    ring,
                )
            )

        S_poly = (
            GF2mPoly.from_list(
                syndrome_coefficients
            )
        )

        log.debug(
            f"S_poly = {S_poly}"
        )

        # --------------------------------------------------------
        # 4. Patterson decoding
        # --------------------------------------------------------

        S_inv_poly = (
            S_poly.inv_mod(
                self.g_poly
            )
        )

        log.debug(
            f"S_inv_poly = {S_inv_poly}"
        )

        log.debug(
            "S_poly * S_inv_poly mod g = "
            f"{(S_poly * S_inv_poly) % self.g_poly}"
        )

        # --------------------------------------------------------
        # Special low-degree case
        # --------------------------------------------------------

        if (
            S_inv_poly.degree() == 1
            and
            S_inv_poly[1]
            .n
            .degree() == 0
            and
            S_inv_poly[1]
            .n
            .poly
            .coeffs()[0] == 1
        ):
            tau_poly = (
                S_inv_poly
            )

        else:
            # ----------------------------------------------------
            # Split g into even/odd components
            # ----------------------------------------------------

            g0, g1 = (
                self.g_poly.split()
            )

            log.debug(
                f"g0 = {g0}"
            )

            log.debug(
                f"g1 = {g1}"
            )

            log.debug(
                "g0^2 + z g1^2 = "
                f"{g0 ** 2 + GF2mPoly.x(ring) * g1 ** 2}"
            )

            g1_inv = (
                g1.inv_mod(
                    self.g_poly
                )
            )

            log.debug(
                f"g1_inv = {g1_inv}"
            )

            w = (
                g0
                * g1_inv
            )

            log.debug(
                f"w = {w}"
            )

            # ----------------------------------------------------
            # H(z) = S^{-1}(z) + z
            # ----------------------------------------------------

            H_poly = (
                S_inv_poly
                +
                GF2mPoly.from_list(
                    [
                        GF2m(
                            GF2Poly.from_list(
                                [0]
                            ),
                            ring,
                        ),
                        GF2m(
                            GF2Poly.from_list(
                                [1]
                            ),
                            ring,
                        ),
                    ]
                )
            )

            log.debug(
                f"H_poly = {H_poly}"
            )

            H0, H1 = (
                H_poly.split()
            )

            log.debug(
                f"H0 = {H0}"
            )

            log.debug(
                f"H1 = {H1}"
            )

            R = (
                H0
                +
                w * H1
            )

            log.debug(
                f"R = {R}"
            )

            log.debug(
                "R^2 mod g = "
                f"{(R ** 2) % self.g_poly}"
            )

            b, _, a = (
                ext_euclid_poly_alt(
                    R,
                    self.g_poly,
                    ring,
                    self.t,
                )
            )

            log.debug(
                f"a = {a}"
            )

            log.debug(
                f"b = {b}"
            )

            log.debug(
                f"a^2 = {a ** 2}"
            )

            log.debug(
                f"b^2 = {b ** 2}"
            )

            log.debug(
                "z b^2 = "
                f"{GF2mPoly.x(ring) * b ** 2}"
            )

            # ----------------------------------------------------
            # Error-locator polynomial
            # ----------------------------------------------------

            tau_poly = (
                a ** 2
                +
                GF2mPoly.x(
                    ring
                )
                * b ** 2
            )

        log.debug(
            "error locator tau = "
            f"{tau_poly}"
        )

        # --------------------------------------------------------
        # 5. Evaluate locator polynomial on support
        #
        # Current support:
        #
        #     1, alpha, alpha^2, ...
        # --------------------------------------------------------

        test_elem = ring.one()

        for i in range(
            len(msg_arr)
        ):
            value = (
                tau_poly.eval(
                    test_elem
                )
            )

            log.debug(
                f"tau(alpha^{i}) = {value}"
            )

            if value == 0:
                msg_arr[
                    i
                ] = msg_arr[
                    i
                ].flip()

                log.info(
                    "REPAIRED ERROR "
                    f"ON POSITION {i}"
                )

            test_elem = (
                test_elem
                * ring.alpha()
            )

        return msg_arr

    # ============================================================
    # Goppa decoding
    # ============================================================

    def decode(
        self,
        msg_arr,
    ):
        """
        Decode a secret-coordinate noisy Goppa codeword.

        The decoder computes

            s = c H^T.

        If s != 0, Patterson error correction is applied.

        The corrected codeword then satisfies

            c_clean = a G,

        and a is recovered by solving the corresponding binary
        linear system.

        The RREF message-recovery step will be replaced by a
        precomputed right inverse in Phase 2C.
        """

        if self.G is None:
            raise RuntimeError(
                "Secret generator matrix G "
                "is unavailable."
            )

        if self.H is None:
            raise RuntimeError(
                "Parity-check matrix H "
                "is unavailable."
            )

        G = self._as_gf2_matrix(
            self.G
        )

        H = self._as_gf2_matrix(
            self.H
        )

        msg_arr = (
            self._as_gf2_matrix(
                msg_arr
            )
        )

        n = G.arr.shape[1]
        k = G.arr.shape[0]

        if len(msg_arr) != n:
            raise ValueError(
                "Received-word length does "
                "not match the Goppa code length."
            )

        log.debug(
            "received word length = "
            f"{len(msg_arr)}"
        )

        # --------------------------------------------------------
        # 1. Syndrome
        #
        #     s = c H^T
        # --------------------------------------------------------

        syndrome = (
            msg_arr
            * H.T()
        )

        log.info(
            f"syndrome =\n{syndrome}"
        )

        syndrome_is_zero = all(
            int(x.n) == 0
            for x
            in syndrome.arr.flat
        )

        # --------------------------------------------------------
        # 2. Patterson correction
        # --------------------------------------------------------

        if not syndrome_is_zero:
            msg_arr = (
                self.repair_errors(
                    msg_arr,
                    syndrome,
                )
            )

            # ----------------------------------------------------
            # Verify repaired word is in the code
            # ----------------------------------------------------

            repaired_syndrome = (
                msg_arr
                * H.T()
            )

            repaired_is_zero = all(
                int(x.n) == 0
                for x
                in repaired_syndrome
                .arr
                .flat
            )

            if not repaired_is_zero:
                raise ValueError(
                    "Patterson decoding did not "
                    "produce a valid codeword."
                )

        # --------------------------------------------------------
        # 3. Recover a from
        #
        #     a G = c_clean.
        #
        # Solve by RREF of
        #
        #     (G^T | c_clean^T).
        #
        # This remains the principal decoding scalability
        # bottleneck and will be removed in Phase 2C.
        # --------------------------------------------------------

        augmented_numpy = np.append(
            G.arr.T,
            msg_arr.arr.reshape(
                n,
                1,
            ),
            axis=1,
        )

        augmented = (
            GF2Matrix.from_list(
                augmented_numpy
            )
        )

        log.debug(
            "G^T | c^T =\n"
            f"{augmented}"
        )

        reduced = rref(
            augmented,
            steps=k,
        )

        log.debug(
            "RREF(G^T | c^T) =\n"
            f"{reduced}"
        )

        message_column = (
            reduced[
                :k,
                k:
            ]
            .flatten()
        )

        return GF2Matrix.from_list(
            message_column
        )

    # ============================================================
    # Decryption
    # ============================================================

    def decrypt(
        self,
        msg_arr,
    ):
        """
        Decrypt a McEliece ciphertext.

        Encryption has the form

            c_pub
                = m S G P + e_pub.

        With the vector-permutation convention

            x P = x[P],

        undoing the permutation gives

            c_sec
                = c_pub[P_inv]
                = m S G + e_sec.

        Patterson decoding recovers

            m S,

        after which multiplication by S^{-1} gives m.
        """

        if self.H is None:
            raise RuntimeError(
                "Private parity-check matrix H "
                "is unavailable."
            )

        if self.P_inv is None:
            raise RuntimeError(
                "Private inverse permutation "
                "P_inv is unavailable."
            )

        if self.S_inv is None:
            raise RuntimeError(
                "Private inverse scrambling "
                "matrix S_inv is unavailable."
            )

        H = self._as_gf2_matrix(
            self.H
        )

        S_inv = self._as_gf2_matrix(
            self.S_inv
        )

        ciphertext = (
            self._as_gf2_matrix(
                msg_arr
            )
        )

        n = H.arr.shape[1]

        if len(ciphertext) != n:
            raise ValueError(
                "Wrong ciphertext length. "
                f"Expected {n} bits, "
                f"received {len(ciphertext)}."
            )

        # --------------------------------------------------------
        # Normalize inverse permutation
        # --------------------------------------------------------

        P_inv = (
            self._as_permutation_vector(
                self.P_inv,
                n,
                name="P_inv",
            )
        )

        log.debug(
            f"ciphertext = {ciphertext}"
        )

        # --------------------------------------------------------
        # 1. Undo secret permutation
        #
        # If
        #
        #     y = x[P],
        #
        # then
        #
        #     x = y[P_inv].
        # --------------------------------------------------------

        secret_coordinate_word = (
            self._permute_vector(
                ciphertext,
                P_inv,
            )
        )

        log.debug(
            "after inverse permutation = "
            f"{secret_coordinate_word}"
        )

        # --------------------------------------------------------
        # 2. Patterson decode
        #
        # This produces the scrambled message mS.
        # --------------------------------------------------------

        scrambled_message = (
            self.decode(
                secret_coordinate_word
            )
        )

        log.debug(
            "decoded scrambled message = "
            f"{scrambled_message}"
        )

        # --------------------------------------------------------
        # 3. Undo scrambling
        #
        #     m = (mS) S^{-1}.
        # --------------------------------------------------------

        message = (
            scrambled_message
            * S_inv
        )

        log.debug(
            f"plaintext = {message}"
        )

        return message.to_numpy()