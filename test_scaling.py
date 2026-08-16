import time
import numpy as np

from mceliece.mceliececipher import McElieceCipher
from mceliece.mathutils import GF2Matrix


# ================================================================
# Configuration
# ================================================================

PARAMETER_SETS = [
    # m, n, t, trials
    (4, 15, 2, 5),
    (5, 31, 2, 5),
    (6, 63, 3, 5),
]

BASE_SEED = 20260816


# ================================================================
# Helpers
# ================================================================

def gf2_numpy(M):
    """
    Convert GF2Matrix to an ordinary uint8 NumPy array
    while preserving its shape.
    """
    return np.array(
        [int(x.n) for x in M.arr.flat],
        dtype=np.uint8,
    ).reshape(M.arr.shape)


def matrix_is_zero(M):
    return not np.any(gf2_numpy(M))


def matrix_equal(A, B):
    return np.array_equal(
        gf2_numpy(A),
        gf2_numpy(B),
    )


def gf2_identity(size):
    return GF2Matrix.from_list(
        np.eye(size, dtype=np.uint8)
    )


def vector_weight(M):
    return sum(
        int(x.n)
        for x in M.arr.flat
    )


# ================================================================
# Single parameter-set test
# ================================================================

def test_parameter_set(
    m,
    n,
    t,
    trials,
    seed,
):
    print()
    print("=" * 78)
    print(
        f"SCALING TEST: "
        f"m={m}, n={n}, t={t}"
    )
    print("=" * 78)

    np.random.seed(seed)

    # ------------------------------------------------------------
    # Theoretical dimension lower bound
    # ------------------------------------------------------------

    dimension_lower_bound = n - m * t

    print()
    print("[1] Parameter checks")
    print(f"  field size             = {2 ** m}")
    print(f"  nonzero field elements = {2 ** m - 1}")
    print(f"  code length n          = {n}")
    print(f"  Goppa degree t         = {t}")
    print(
        f"  dimension lower bound  = "
        f"n - m*t = {dimension_lower_bound}"
    )

    if n > 2 ** m - 1:
        raise AssertionError(
            "Current implementation uses only nonzero "
            "support elements, so n must satisfy "
            "n <= 2^m - 1."
        )

    # ------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------

    print()
    print("[2] Generating Goppa code and McEliece keys...")

    mc = McElieceCipher(
        m=m,
        n=n,
        t=t,
    )

    start = time.perf_counter()

    mc.generate_random_keys()

    keygen_time = (
        time.perf_counter() - start
    )

    k = mc.k

    print()
    print("Generated parameters:")
    print(f"  m = {m}")
    print(f"  n = {n}")
    print(f"  k = {k}")
    print(f"  t = {t}")

    print()
    print("Matrix dimensions:")
    print(
        f"  G     = {mc.G.arr.shape}"
    )
    print(
        f"  H     = {mc.H.arr.shape}"
    )
    print(
        f"  S     = {mc.S.arr.shape}"
    )
    print(
        f"  P     = {mc.P.arr.shape}"
    )
    print(
        f"  G_pub = {mc.Gp.arr.shape}"
    )

    print()
    print(
        f"Key-generation time = "
        f"{keygen_time:.6f} seconds"
    )

    # ------------------------------------------------------------
    # Dimension/rank checks
    # ------------------------------------------------------------

    print()
    print("[3] Checking dimensions and rank...")

    expected_H_shape = (
        m * t,
        n,
    )

    if mc.H.arr.shape != expected_H_shape:
        raise AssertionError(
            f"H has shape {mc.H.arr.shape}, "
            f"expected {expected_H_shape}."
        )

    if mc.G.arr.shape != (k, n):
        raise AssertionError(
            "Unexpected generator matrix dimensions."
        )

    _, rank_H = mc.H.rref()

    print(
        f"  rank(H)                = {rank_H}"
    )

    print(
        f"  n - rank(H)            = "
        f"{n - rank_H}"
    )

    print(
        f"  actual k               = {k}"
    )

    print(
        f"  Goppa lower bound      = "
        f"{dimension_lower_bound}"
    )

    if k != n - rank_H:
        raise AssertionError(
            "Generator dimension does not equal "
            "the nullity of H."
        )

    # Important:
    #
    # For a binary Goppa code,
    #
    #     k >= n - m t.
    #
    # Equality need not be assumed in the test.

    if k < dimension_lower_bound:
        raise AssertionError(
            "Code dimension violates the standard "
            "binary-Goppa lower bound."
        )

    print("  Dimension/rank checks: PASS")

    # ------------------------------------------------------------
    # Orthogonality
    # ------------------------------------------------------------

    print()
    print("[4] Checking G H^T = 0...")

    GHt = (
        mc.G
        * mc.H.T()
    )

    print(
        f"  G H^T shape = "
        f"{GHt.arr.shape}"
    )

    orthogonal = matrix_is_zero(
        GHt
    )

    print(
        f"  G H^T == 0 : "
        f"{orthogonal}"
    )

    if not orthogonal:
        raise AssertionError(
            "G H^T != 0."
        )

    # ------------------------------------------------------------
    # Scrambling matrix
    # ------------------------------------------------------------

    print()
    print("[5] Checking S S^{-1} = I_k...")

    SSinv = (
        mc.S
        * mc.S_inv
    )

    scrambling_ok = matrix_equal(
        SSinv,
        gf2_identity(k),
    )

    print(
        f"  scrambling inverse : "
        f"{scrambling_ok}"
    )

    if not scrambling_ok:
        raise AssertionError(
            "S S^{-1} != I."
        )

    # ------------------------------------------------------------
    # Permutation
    # ------------------------------------------------------------

    print()
    print("[6] Checking P P^{-1} = I_n...")

    PPinv = (
        mc.P
        * mc.P_inv
    )

    permutation_ok = matrix_equal(
        PPinv,
        gf2_identity(n),
    )

    print(
        f"  permutation inverse : "
        f"{permutation_ok}"
    )

    if not permutation_ok:
        raise AssertionError(
            "P P^{-1} != I."
        )

    # ------------------------------------------------------------
    # Public generator
    # ------------------------------------------------------------

    print()
    print("[7] Checking G_pub = S G P...")

    expected_Gp = (
        mc.S
        * mc.G
        * mc.P
    )

    public_generator_ok = (
        matrix_equal(
            mc.Gp,
            expected_Gp,
        )
    )

    print(
        f"  public-generator consistency : "
        f"{public_generator_ok}"
    )

    if not public_generator_ok:
        raise AssertionError(
            "G_pub != S G P."
        )

    # ------------------------------------------------------------
    # Encryption/decryption
    # ------------------------------------------------------------

    print()
    print(
        f"[8] Running {trials} "
        f"encryption/decryption trials..."
    )

    successful_trials = 0

    encryption_times = []
    decryption_times = []

    for trial in range(
        1,
        trials + 1,
    ):
        message = np.random.randint(
            0,
            2,
            size=k,
            dtype=np.uint8,
        )

        # Clean public codeword.
        message_matrix = (
            GF2Matrix.from_list(
                message
            )
        )

        clean = (
            message_matrix
            * mc.Gp
        )

        # ----------------------------
        # Encrypt
        # ----------------------------

        start = time.perf_counter()

        ciphertext = mc.encrypt(
            message
        )

        encryption_time = (
            time.perf_counter()
            - start
        )

        encryption_times.append(
            encryption_time
        )

        # ----------------------------
        # Recover actual injected
        # error vector.
        #
        # In characteristic two:
        #
        #     e = c + m G_pub.
        # ----------------------------

        error = (
            ciphertext
            + clean
        )

        error_weight = vector_weight(
            error
        )

        if error_weight != t:
            raise AssertionError(
                f"Encryption inserted error "
                f"weight {error_weight}; "
                f"expected exactly {t}."
            )

        # ----------------------------
        # Decrypt
        # ----------------------------

        start = time.perf_counter()

        decoded = mc.decrypt(
            ciphertext
        )

        decryption_time = (
            time.perf_counter()
            - start
        )

        decryption_times.append(
            decryption_time
        )

        message_int = np.asarray(
            message,
            dtype=int,
        ).reshape(-1)

        decoded_int = np.asarray(
            decoded,
            dtype=int,
        ).reshape(-1)

        success = np.array_equal(
            message_int,
            decoded_int,
        )

        if success:
            successful_trials += 1

        print()
        print(
            f"  Trial {trial:02d}:"
        )

        print(
            f"    injected error weight = "
            f"{error_weight}"
        )

        print(
            f"    encryption time       = "
            f"{encryption_time:.6f} s"
        )

        print(
            f"    decryption time       = "
            f"{decryption_time:.6f} s"
        )

        print(
            f"    success               = "
            f"{success}"
        )

        if not success:
            print(
                f"    message = "
                f"{message_int}"
            )

            print(
                f"    decoded = "
                f"{decoded_int}"
            )

            raise AssertionError(
                f"Decryption failed for "
                f"(m,n,t)=({m},{n},{t})."
            )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    avg_enc = float(
        np.mean(encryption_times)
    )

    avg_dec = float(
        np.mean(decryption_times)
    )

    print()
    print("-" * 78)
    print("PARAMETER-SET RESULT")
    print("-" * 78)

    print(
        f"Parameters               : "
        f"({m}, {n}, {t})"
    )

    print(
        f"Dimension k              : "
        f"{k}"
    )

    print(
        f"rank(H)                  : "
        f"{rank_H}"
    )

    print(
        f"Key-generation time      : "
        f"{keygen_time:.6f} s"
    )

    print(
        f"Average encryption time  : "
        f"{avg_enc:.6f} s"
    )

    print(
        f"Average decryption time  : "
        f"{avg_dec:.6f} s"
    )

    print(
        f"Encryption/decryption    : "
        f"{successful_trials}/{trials} PASS"
    )

    return {
        "m": m,
        "n": n,
        "t": t,
        "k": k,
        "rank_H": rank_H,
        "keygen_time": keygen_time,
        "avg_enc": avg_enc,
        "avg_dec": avg_dec,
        "successes": successful_trials,
        "trials": trials,
    }


# ================================================================
# Main
# ================================================================

def main():
    print()
    print("=" * 78)
    print("BINARY GOPPA / McELIECE SCALING REGRESSION")
    print("=" * 78)

    results = []

    for index, parameters in enumerate(
        PARAMETER_SETS
    ):
        m, n, t, trials = parameters

        result = test_parameter_set(
            m=m,
            n=n,
            t=t,
            trials=trials,
            seed=BASE_SEED + index,
        )

        results.append(
            result
        )

    print()
    print("=" * 78)
    print("FINAL SCALING SUMMARY")
    print("=" * 78)

    for result in results:
        print(
            f"(m,n,t)="
            f"({result['m']},"
            f"{result['n']},"
            f"{result['t']})"
            f"  "
            f"k={result['k']}"
            f"  "
            f"rank(H)={result['rank_H']}"
            f"  "
            f"keygen={result['keygen_time']:.4f}s"
            f"  "
            f"dec={result['avg_dec']:.4f}s"
            f"  "
            f"{result['successes']}/"
            f"{result['trials']} PASS"
        )

    print()
    print(
        "ALL SCALING REGRESSION TESTS PASSED."
    )


if __name__ == "__main__":
    main()