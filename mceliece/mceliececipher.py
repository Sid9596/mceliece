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
    random_perm_matrix,
    rref,
)


log = logging.getLogger("mceliececipher")


class McElieceCipher:
    """
    Binary Goppa-code McEliece implementation.

    Parameters
    ----------
    m : int
        Extension degree. The Goppa code is defined using GF(2^m).

    n : int
        Code length.

    t : int
        Degree of the Goppa polynomial and number of errors
        injected by the baseline McEliece encryption algorithm.

    Notes
    -----
    This class is currently used as the baseline implementation
    that will later be extended for the homomorphic-encryption
    construction.
    """

    def __init__(self, m, n, t):
        self.m = int(m)
        self.n = int(n)
        self.t = int(t)

        self.q = 2

        if self.m <= 0:
            raise ValueError("m must be positive.")

        if self.n <= 0:
            raise ValueError("n must be positive.")

        if self.t <= 0:
            raise ValueError("t must be positive.")

        if self.n > 2 ** self.m:
            raise ValueError(
                f"n={self.n} exceeds field size 2^m={2 ** self.m}."
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

        # Secret Goppa-code data.
        self.G = None
        self.H = None
        self.k = None

        # McEliece scrambling/permutation matrices.
        self.P = None
        self.P_inv = None

        self.S = None
        self.S_inv = None

        # Public generator matrix.
        self.Gp = None

        # Goppa polynomial and extension-field modulus.
        self.g_poly = None
        self.irr_poly = None

    # ============================================================
    # Internal conversion helpers
    # ============================================================

    @staticmethod
    def _as_gf2_matrix(value):
        """
        Convert value into GF2Matrix if necessary.

        This allows the class to work both with matrices generated
        directly in memory and matrices loaded from NumPy files.
        """

        if isinstance(value, GF2Matrix):
            return value

        arr = np.asarray(value)

        # Handle a possible zero-dimensional object array containing
        # a GF2Matrix object.
        if arr.ndim == 0:
            item = arr.item()

            if isinstance(item, GF2Matrix):
                return item

            arr = np.asarray(item)

        return GF2Matrix.from_list(arr)

    # ============================================================
    # Key generation
    # ============================================================

    def generate_random_keys(self):
        """
        Generate a binary Goppa code and the McEliece key pair.

        Produces

            G       secret generator matrix
            H       parity-check matrix
            S       invertible scrambling matrix
            P       permutation matrix
            Gp      public generator S G P

        satisfying

            G H^T = 0.
        """

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
        # Store the Goppa polynomial coefficients.
        #
        # Every coefficient lies in GF(2^m).  We store it as its
        # m-bit coefficient vector with respect to the basis
        #
        #     1, alpha, ..., alpha^(m-1).
        #
        # This avoids the old SymPy ZZ/GF(2) domain conflict.
        # --------------------------------------------------------

        g_coeffs = []

        for coefficient in self.g_poly.all_coeffs()[::-1]:
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
        # Store the irreducible extension-field polynomial in
        # ascending coefficient order.
        # --------------------------------------------------------

        self.irr_poly = np.array(
            [
                int(c) % 2
                for c in self.irr_poly.all_coeffs()[::-1]
            ],
            dtype=np.uint8,
        )

        # Actual code dimension determined by the null space.
        self.k = self.G.arr.shape[0]

        log.info(
            f"Generated binary Goppa code "
            f"[n={self.n}, k={self.k}]"
        )

        # --------------------------------------------------------
        # Generate McEliece permutation matrix.
        # --------------------------------------------------------

        self.P = GF2Matrix.from_list(
            random_perm_matrix(self.n)
        )

        self.P_inv = self.P.inv()

        # --------------------------------------------------------
        # Generate random invertible scrambling matrix.
        # --------------------------------------------------------

        self.S = GF2Matrix.from_list(
            random_inv_matrix(self.k)
        )

        self.S_inv = self.S.inv()

        # --------------------------------------------------------
        # Public generator
        #
        #             G_pub = S G P.
        # --------------------------------------------------------

        self.Gp = (
            self.S
            * self.G
            * self.P
        )

        log.info(
            "McEliece key generation completed."
        )

    # ============================================================
    # Encryption
    # ============================================================

    def encrypt(self, msg_arr):
        """
        Encrypt a k-bit message using the public generator matrix.

        Baseline McEliece encryption computes

            c = m G_pub + e,

        where e has Hamming weight exactly t.
        """

        if self.Gp is None:
            raise RuntimeError(
                "Public key has not been generated or loaded."
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
                f"Wrong message length. "
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

        # Clean public codeword.
        ciphertext = (
            message
            * Gp
        )

        log.debug(
            f"clean ciphertext = {ciphertext}"
        )

        # Select exactly t distinct error positions.
        bits_to_flip = np.random.choice(
            n,
            size=self.t,
            replace=False,
        )

        log.debug(
            f"error positions = {bits_to_flip}"
        )

        for position in bits_to_flip:
            ciphertext[position] = (
                ciphertext[position].flip()
            )

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
        """

        # --------------------------------------------------------
        # Reconstruct the binary extension-field modulus.
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
        # Reconstruct the Goppa polynomial over GF(2^m).
        # --------------------------------------------------------

        if not isinstance(
            self.g_poly,
            GF2mPoly,
        ):
            coefficients = []

            for coefficient in self.g_poly:
                coefficient_poly = (
                    GF2Poly.from_list(
                        [
                            int(e)
                            for e in coefficient
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
        # Convert the binary syndrome into a polynomial over
        # GF(2^m).
        # --------------------------------------------------------

        syndrome_matrix = (
            self._as_gf2_matrix(
                syndrome
            )
        )

        syndrome_flat = list(
            syndrome_matrix.arr.flat
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
                        for e in block
                    ]
                )
            )

            syndrome_coefficients.append(
                GF2m(
                    binary_poly,
                    ring,
                )
            )

        S_poly = GF2mPoly.from_list(
            syndrome_coefficients
        )

        log.debug(
            f"S_poly = {S_poly}"
        )

        # --------------------------------------------------------
        # Patterson decoding.
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

        # Special low-degree case.
        if (
            S_inv_poly.degree() == 1
            and
            S_inv_poly[1].n.degree() == 0
            and
            S_inv_poly[1]
            .n
            .poly
            .coeffs()[0] == 1
        ):
            tau_poly = S_inv_poly

        else:
            # Split g into even/odd components.
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

            # H(z) = S^{-1}(z) + z
            H_poly = (
                S_inv_poly
                +
                GF2mPoly.from_list(
                    [
                        GF2m(
                            GF2Poly.from_list([0]),
                            ring,
                        ),
                        GF2m(
                            GF2Poly.from_list([1]),
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

            tau_poly = (
                a ** 2
                +
                GF2mPoly.x(ring)
                * b ** 2
            )

        log.debug(
            f"error locator tau = {tau_poly}"
        )

        # --------------------------------------------------------
        # Evaluate the locator polynomial at all support elements.
        # A zero identifies an error coordinate.
        # --------------------------------------------------------

        test_elem = ring.one()

        for i in range(
            len(msg_arr)
        ):
            value = tau_poly.eval(
                test_elem
            )

            log.debug(
                f"tau(alpha^{i}) = {value}"
            )

            if value == 0:
                msg_arr[i] = (
                    msg_arr[i].flip()
                )

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

    def decode(self, msg_arr):
        """
        Decode a secret-coordinate noisy Goppa codeword.

        The decoder first computes the syndrome.  If the syndrome is
        nonzero, Patterson error correction is applied.  The message
        coefficients are then recovered from the corrected codeword.
        """

        if self.G is None:
            raise RuntimeError(
                "Secret generator matrix G is unavailable."
            )

        if self.H is None:
            raise RuntimeError(
                "Parity-check matrix H is unavailable."
            )

        G = self._as_gf2_matrix(
            self.G
        )

        H = self._as_gf2_matrix(
            self.H
        )

        msg_arr = self._as_gf2_matrix(
            msg_arr
        )

        log.debug(
            f"received word length = {len(msg_arr)}"
        )

        # --------------------------------------------------------
        # Syndrome
        #
        #          s = c H^T.
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
            for x in syndrome.arr.flat
        )

        if not syndrome_is_zero:
            msg_arr = self.repair_errors(
                msg_arr,
                syndrome,
            )

            # Optional diagnostic check after repair.
            repaired_syndrome = (
                msg_arr
                * H.T()
            )

            repaired_is_zero = all(
                int(x.n) == 0
                for x
                in repaired_syndrome.arr.flat
            )

            if not repaired_is_zero:
                raise ValueError(
                    "Patterson decoding did not "
                    "produce a valid codeword."
                )

        # --------------------------------------------------------
        # Recover m from
        #
        #                m G = c.
        #
        # We solve the linear system by row reducing
        #
        #              (G^T | c^T).
        #
        # This is suitable for the baseline implementation.
        # Later we will systematicize G so that this RREF step
        # can be removed.
        # --------------------------------------------------------

        k = G.arr.shape[0]
        n = G.arr.shape[1]

        if len(msg_arr) != n:
            raise ValueError(
                "Decoded-word length does not "
                "match generator matrix."
            )

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

        message_column = reduced[
            :k,
            k:
        ].flatten()

        return GF2Matrix.from_list(
            message_column
        )

    # ============================================================
    # Decryption
    # ============================================================

    def decrypt(self, msg_arr):
        """
        Decrypt a McEliece ciphertext.

        For

            c = m S G P + e,

        first compute

            c P^{-1},

        decode the Goppa codeword, and finally multiply by S^{-1}.
        """

        if self.H is None:
            raise RuntimeError(
                "Private parity-check matrix H is unavailable."
            )

        if self.P_inv is None:
            raise RuntimeError(
                "Private inverse permutation P_inv is unavailable."
            )

        if self.S_inv is None:
            raise RuntimeError(
                "Private inverse scrambling matrix S_inv is unavailable."
            )

        H = self._as_gf2_matrix(
            self.H
        )

        P_inv = self._as_gf2_matrix(
            self.P_inv
        )

        S_inv = self._as_gf2_matrix(
            self.S_inv
        )

        ciphertext = self._as_gf2_matrix(
            msg_arr
        )

        n = H.arr.shape[1]

        if len(ciphertext) != n:
            raise ValueError(
                f"Wrong ciphertext length. "
                f"Expected {n} bits, "
                f"received {len(ciphertext)}."
            )

        log.debug(
            f"ciphertext = {ciphertext}"
        )

        # Undo the secret permutation.
        secret_coordinate_word = (
            ciphertext
            * P_inv
        )

        log.debug(
            "after P^{-1} = "
            f"{secret_coordinate_word}"
        )

        # Decode to obtain mS.
        scrambled_message = self.decode(
            secret_coordinate_word
        )

        log.debug(
            "decoded scrambled message = "
            f"{scrambled_message}"
        )

        # Undo scrambling.
        message = (
            scrambled_message
            * S_inv
        )

        log.debug(
            f"plaintext = {message}"
        )

        return message.to_numpy()