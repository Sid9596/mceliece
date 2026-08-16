"""
Baseline correctness test for the modified jkrauze/mceliece implementation.

This test verifies:

1. Binary Goppa-code key generation.
2. Expected matrix dimensions.
3. Generator/parity-check orthogonality:

       G H^T = 0.

4. Invertibility of the scrambling matrix:

       S S^{-1} = I_k.

5. Invertibility of the permutation matrix:

       P P^{-1} = I_n.

6. Public-key consistency:

       G_pub = S G P.

7. McEliece encryption/decryption correctness over several
   independently sampled plaintexts and error patterns.

The current test parameters are deliberately small:

       m = 4,
       n = 15,
       t = 2.

They are intended only to establish a working baseline before
scalability modifications are introduced.
"""

import numpy as np

from mceliece.mceliececipher import McElieceCipher
from mceliece.mathutils import GF2Matrix


# ================================================================
# Configuration
# ================================================================

M = 4
N = 15
T = 2

NUM_TRIALS = 10
RANDOM_SEED = 1


# ================================================================
# Helper functions
# ================================================================

def gf2_matrix_is_zero(matrix):
    """
    Return True if every entry of a GF2Matrix is zero.
    """

    if not isinstance(matrix, GF2Matrix):
        raise TypeError(
            "gf2_matrix_is_zero() expects a GF2Matrix."
        )

    return all(
        int(entry.n) == 0
        for entry in matrix.arr.flat
    )


def gf2_matrix_equal(A, B):
    """
    Compare two GF2Matrix objects entry by entry.
    """

    if not isinstance(A, GF2Matrix):
        raise TypeError("A must be a GF2Matrix.")

    if not isinstance(B, GF2Matrix):
        raise TypeError("B must be a GF2Matrix.")

    if A.arr.shape != B.arr.shape:
        return False

    return all(
        int(a.n) == int(b.n)
        for a, b in zip(
            A.arr.flat,
            B.arr.flat,
        )
    )


def identity_gf2(size):
    """
    Construct the size x size identity matrix over GF(2).
    """

    return GF2Matrix.from_list(
        np.eye(
            size,
            dtype=np.uint8,
        )
    )


def hamming_weight(vector):
    """
    Return the Hamming weight of a binary vector.
    """

    return int(
        np.sum(
            np.asarray(
                vector,
                dtype=np.uint8,
            )
        )
    )


def print_header(title):
    """
    Print a formatted section header.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# Main baseline test
# ================================================================

def main():
    np.random.seed(RANDOM_SEED)

    print_header(
        "BASELINE McELIECE TEST"
    )

    print(
        f"Parameters: "
        f"m={M}, "
        f"n={N}, "
        f"t={T}"
    )

    print(
        f"Encryption/decryption trials: "
        f"{NUM_TRIALS}"
    )

    # ------------------------------------------------------------
    # 1. Construct McEliece instance
    # ------------------------------------------------------------

    print()
    print("[1] Creating McEliece instance...")

    mc = McElieceCipher(
        m=M,
        n=N,
        t=T,
    )

    # ------------------------------------------------------------
    # 2. Key generation
    # ------------------------------------------------------------

    print(
        "[2] Generating binary Goppa code "
        "and McEliece keys..."
    )

    mc.generate_random_keys()

    k = mc.k

    print()
    print("Generated parameters:")
    print(f"  m = {mc.m}")
    print(f"  n = {mc.n}")
    print(f"  k = {mc.k}")
    print(f"  t = {mc.t}")

    print()
    print("Matrix dimensions:")

    print(
        "  G     =",
        mc.G.arr.shape,
    )

    print(
        "  H     =",
        mc.H.arr.shape,
    )

    print(
        "  S     =",
        mc.S.arr.shape,
    )

    print(
        "  P     =",
        mc.P.arr.shape,
    )

    print(
        "  G_pub =",
        mc.Gp.arr.shape,
    )

    print(
        "  g_poly storage =",
        mc.g_poly.shape,
    )

    print(
        "  irr_poly storage =",
        mc.irr_poly.shape,
    )

    # ------------------------------------------------------------
    # 3. Dimension checks
    # ------------------------------------------------------------

    print()
    print("[3] Checking matrix dimensions...")

    expected_k_lower_bound = (
        N - M * T
    )

    print(
        "  n - m*t =",
        expected_k_lower_bound,
    )

    print(
        "  actual k =",
        k,
    )

    assert mc.G.arr.shape == (
        k,
        N,
    ), (
        "Generator matrix has incorrect dimensions."
    )

    assert mc.H.arr.shape[1] == N, (
        "Parity-check matrix has incorrect length."
    )

    assert mc.S.arr.shape == (
        k,
        k,
    ), (
        "Scrambling matrix has incorrect dimensions."
    )

    assert mc.P.arr.shape == (
        N,
        N,
    ), (
        "Permutation matrix has incorrect dimensions."
    )

    assert mc.Gp.arr.shape == (
        k,
        N,
    ), (
        "Public generator has incorrect dimensions."
    )

    print(
        "  Dimension checks: PASS"
    )

    # ------------------------------------------------------------
    # 4. Verify G H^T = 0
    # ------------------------------------------------------------

    print()
    print(
        "[4] Checking G H^T = 0..."
    )

    GHt = (
        mc.G
        * mc.H.T()
    )

    print(
        "  G H^T shape =",
        GHt.arr.shape,
    )

    gh_zero = gf2_matrix_is_zero(
        GHt
    )

    print(
        "  G H^T == 0 :",
        gh_zero,
    )

    assert gh_zero, (
        "Generator/parity-check "
        "orthogonality failed."
    )

    # ------------------------------------------------------------
    # 5. Verify S S^{-1} = I
    # ------------------------------------------------------------

    print()
    print(
        "[5] Checking S S^{-1} = I_k..."
    )

    SS_inv = (
        mc.S
        * mc.S_inv
    )

    I_k = identity_gf2(
        k
    )

    S_ok = gf2_matrix_equal(
        SS_inv,
        I_k,
    )

    print(
        "  S S^{-1} == I_k :",
        S_ok,
    )

    assert S_ok, (
        "Scrambling matrix inverse check failed."
    )

    # ------------------------------------------------------------
    # 6. Verify P P^{-1} = I
    # ------------------------------------------------------------

    print()
    print(
        "[6] Checking P P^{-1} = I_n..."
    )

    PP_inv = (
        mc.P
        * mc.P_inv
    )

    I_n = identity_gf2(
        N
    )

    P_ok = gf2_matrix_equal(
        PP_inv,
        I_n,
    )

    print(
        "  P P^{-1} == I_n :",
        P_ok,
    )

    assert P_ok, (
        "Permutation inverse check failed."
    )

    # ------------------------------------------------------------
    # 7. Verify public generator
    # ------------------------------------------------------------

    print()
    print(
        "[7] Checking G_pub = S G P..."
    )

    reconstructed_Gp = (
        mc.S
        * mc.G
        * mc.P
    )

    public_key_ok = gf2_matrix_equal(
        reconstructed_Gp,
        mc.Gp,
    )

    print(
        "  G_pub consistency :",
        public_key_ok,
    )

    assert public_key_ok, (
        "Public generator consistency failed."
    )

    # ------------------------------------------------------------
    # 8. Encryption/decryption trials
    # ------------------------------------------------------------

    print()
    print(
        "[8] Running encryption/decryption trials..."
    )

    successes = 0

    for trial in range(
        1,
        NUM_TRIALS + 1,
    ):
        # Random k-bit message.
        message = np.random.randint(
            0,
            2,
            size=k,
            dtype=np.uint8,
        )

        # Encrypt.
        ciphertext = mc.encrypt(
            message
        )

        ciphertext_numpy = (
            ciphertext.to_numpy()
            .astype(np.uint8)
        )

        # Decrypt.
        decoded = mc.decrypt(
            ciphertext
        )

        decoded = np.asarray(
            decoded,
            dtype=np.uint8,
        ).flatten()

        message = np.asarray(
            message,
            dtype=np.uint8,
        ).flatten()

        success = np.array_equal(
            message,
            decoded,
        )

        if success:
            successes += 1

        print()
        print(
            f"  Trial {trial:02d}:"
        )

        print(
            "    message    =",
            message,
        )

        print(
            "    ciphertext =",
            ciphertext_numpy,
        )

        print(
            "    ct weight  =",
            hamming_weight(
                ciphertext_numpy
            ),
        )

        print(
            "    decoded    =",
            decoded,
        )

        print(
            "    success    =",
            success,
        )

        if not success:
            raise AssertionError(
                f"Encryption/decryption failed "
                f"on trial {trial}."
            )

    # ------------------------------------------------------------
    # 9. Final report
    # ------------------------------------------------------------

    print_header(
        "BASELINE RESULT"
    )

    print(
        "Finite-field construction      : PASS"
    )

    print(
        "Goppa-code key generation      : PASS"
    )

    print(
        "Generator/parity-check test    : PASS"
    )

    print(
        "Scrambling inverse test        : PASS"
    )

    print(
        "Permutation inverse test       : PASS"
    )

    print(
        "Public-generator consistency   : PASS"
    )

    print(
        "Patterson decoding             : PASS"
    )

    print(
        "Encryption/decryption          : "
        f"{successes}/{NUM_TRIALS} PASS"
    )

    print()

    print(
        "BASELINE IMPLEMENTATION WORKS CORRECTLY."
    )


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    main()