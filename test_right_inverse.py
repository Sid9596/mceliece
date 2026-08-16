"""
Phase 2C right-inverse regression.

Tests:

    G R = I_k,

    (a G) R = a,

and verifies that Patterson decoding followed by the right-inverse
recovery still decrypts correctly.
"""

import time

import numpy as np

from mceliece.mceliececipher import McElieceCipher
from mceliece.mathutils import GF2Matrix


PARAMETER_SETS = [
    # m, n, t, trials
    (4, 15, 2, 5),
    (5, 31, 2, 5),
    (6, 63, 3, 5),
]

BASE_SEED = 20260816


def matrix_equal(A, B):
    if A.arr.shape != B.arr.shape:
        return False

    return all(
        int(a.n) == int(b.n)
        for a, b in zip(
            A.arr.flat,
            B.arr.flat,
        )
    )


def identity_gf2(k):
    return GF2Matrix.from_list(
        np.eye(
            k,
            dtype=np.uint8,
        )
    )


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
        f"RIGHT-INVERSE TEST: "
        f"m={m}, n={n}, t={t}"
    )
    print("=" * 78)

    np.random.seed(
        seed
    )

    mc = McElieceCipher(
        m,
        n,
        t,
    )

    start = time.perf_counter()

    mc.generate_random_keys()

    keygen_time = (
        time.perf_counter()
        - start
    )

    k = mc.k

    print()
    print(
        f"  G shape = {mc.G.arr.shape}"
    )

    print(
        f"  R shape = {mc.R.arr.shape}"
    )

    print(
        "  pivot count =",
        len(
            mc.right_inverse_pivots
        ),
    )

    assert mc.G.arr.shape == (
        k,
        n,
    )

    assert mc.R.arr.shape == (
        n,
        k,
    )

    assert len(
        mc.right_inverse_pivots
    ) == k

    # ------------------------------------------------------------
    # G R = I_k
    # ------------------------------------------------------------

    GR = (
        mc.G
        * mc.R
    )

    GR_ok = matrix_equal(
        GR,
        identity_gf2(k),
    )

    print()
    print(
        "  G R = I_k :",
        GR_ok,
    )

    assert GR_ok

    # ------------------------------------------------------------
    # Test clean message extraction
    #
    #     (a G) R = a.
    # ------------------------------------------------------------

    print()
    print(
        f"[Clean extraction: {trials} trials]"
    )

    for trial in range(
        1,
        trials + 1,
    ):
        a_numpy = np.random.randint(
            0,
            2,
            size=k,
            dtype=np.uint8,
        )

        a = GF2Matrix.from_list(
            a_numpy
        )

        codeword = (
            a
            * mc.G
        )

        recovered = (
            codeword
            * mc.R
        )

        recovered_numpy = (
            recovered
            .to_numpy()
            .astype(
                np.uint8
            )
        )

        success = np.array_equal(
            a_numpy,
            recovered_numpy,
        )

        print(
            f"  clean trial {trial:02d}: "
            f"{success}"
        )

        assert success

    # ------------------------------------------------------------
    # Test Patterson + right-inverse decoding.
    # ------------------------------------------------------------

    print()
    print(
        f"[Noisy encryption/decryption: "
        f"{trials} trials]"
    )

    successes = 0

    decode_times = []

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

        ciphertext = mc.encrypt(
            message
        )

        # Secret-coordinate ciphertext.
        secret_ct = GF2Matrix(
            ciphertext.arr[
                mc.P_inv
            ]
        )

        # Expected result of secret Goppa decoding is mS.
        message_matrix = GF2Matrix.from_list(
            message
        )

        expected_scrambled = (
            message_matrix
            * mc.S
        )

        start = time.perf_counter()

        decoded_scrambled = mc.decode(
            secret_ct
        )

        decode_time = (
            time.perf_counter()
            - start
        )

        decode_times.append(
            decode_time
        )

        scrambled_ok = matrix_equal(
            decoded_scrambled,
            expected_scrambled,
        )

        decoded_message = np.asarray(
            mc.decrypt(
                ciphertext
            ),
            dtype=np.uint8,
        ).reshape(-1)

        decrypt_ok = np.array_equal(
            decoded_message,
            message,
        )

        success = (
            scrambled_ok
            and decrypt_ok
        )

        if success:
            successes += 1

        print(
            f"  noisy trial {trial:02d}: "
            f"decode={scrambled_ok}, "
            f"decrypt={decrypt_ok}, "
            f"time={decode_time:.6f}s"
        )

        assert success

    avg_decode = float(
        np.mean(
            decode_times
        )
    )

    print()
    print("-" * 78)
    print("PARAMETER RESULT")
    print("-" * 78)

    print(
        f"Parameters       : ({m},{n},{t})"
    )

    print(
        f"k                : {k}"
    )

    print(
        f"R shape          : {mc.R.arr.shape}"
    )

    print(
        f"G R = I_k        : {GR_ok}"
    )

    print(
        f"Key generation   : {keygen_time:.6f}s"
    )

    print(
        f"Average decode   : {avg_decode:.6f}s"
    )

    print(
        f"Correct trials   : "
        f"{successes}/{trials}"
    )

    return {
        "m": m,
        "n": n,
        "t": t,
        "k": k,
        "keygen": keygen_time,
        "decode": avg_decode,
        "successes": successes,
        "trials": trials,
    }


def main():

    print()
    print("=" * 78)
    print(
        "PHASE 2C RIGHT-INVERSE REGRESSION"
    )
    print("=" * 78)

    results = []

    for index, params in enumerate(
        PARAMETER_SETS
    ):
        m, n, t, trials = params

        results.append(
            test_parameter_set(
                m,
                n,
                t,
                trials,
                BASE_SEED + index,
            )
        )

    print()
    print("=" * 78)
    print(
        "PHASE 2C FINAL SUMMARY"
    )
    print("=" * 78)

    for result in results:
        print(
            f"(m,n,t)="
            f"({result['m']},"
            f"{result['n']},"
            f"{result['t']}) "
            f"k={result['k']} "
            f"decode={result['decode']:.6f}s "
            f"{result['successes']}/"
            f"{result['trials']} PASS"
        )

    print()
    print(
        "G R = I_k                     : PASS"
    )

    print(
        "(a G) R = a                  : PASS"
    )

    print(
        "Patterson + right inverse     : PASS"
    )

    print(
        "Encryption/decryption         : PASS"
    )

    print()
    print(
        "PHASE 2C RIGHT-INVERSE "
        "REGRESSION PASSED."
    )


if __name__ == "__main__":
    main()