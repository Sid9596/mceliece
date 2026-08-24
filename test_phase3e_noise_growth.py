#!/usr/bin/env python3
"""
Phase 3E: noisy-input and fused-noise-growth validation.

Experiments
===========

E1. Nonzero admissible input noise
----------------------------------

Encrypt

    c_i = a_i G_pub + e_i

with

    e_i != 0

and

    e_i Lambda_msg = 0.

Use a clean fused evaluation key, tau_F = 0, and verify

    Dec(EvalMult(c_1,c_2))
        =
    mu_1 odot mu_2.

This tests the central plaintext-neutral-noise property.


E2. Fused evaluation-key noise
------------------------------

Generate

    Y_{u,v}
        =
    K_{u,v} + z_{u,v}

with

    wt(z_{u,v}) = tau_F

for a sweep of tau_F values.

For each multiplication record

    M
        =
    number of selected fused rows,

    W_M
        =
    actual output fused-noise weight,

and whether

    W_M <= t

and decryption succeeds.


E3. Independent-row theoretical comparison
-------------------------------------------

For M independently sampled fixed-weight tau_F rows,

    E[W_M]
      =
    n/2 * (1 - (1 - 2 tau_F/n)^M).

Let

    alpha = 1 - 2 tau_F/n

and

    beta
      =
    1
    -
    4 tau_F (n - tau_F) / (n(n-1)).

Then

    Var(W_M)
      =
    n/4 (1 - alpha^(2M))
    +
    n(n-1)/4
    (
        beta^M
        -
        alpha^(2M)
    ).

The script compares every observed multiplication against these
conditional moments.

Results are appended immediately to CSV.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from mceliece.mceliececipher import (
    McElieceCipher,
)

from mceliece.fused_he import (
    FusedMcElieceHE,
    gf2_matmul,
    gf2_vector,
    hamming_weight,
)


# ======================================================================
# Parameter sets
# ======================================================================

PARAMETER_SETS = [
    {
        "m": 4,
        "n": 15,
        "t": 2,
        "ell": 2,
        "seed": 401,
    },
    {
        "m": 10,
        "n": 1023,
        "t": 32,
        "ell": 2,
        "seed": 402,
    },
]


DEFAULT_CSV = (
    "phase3e_noise_growth_results.csv"
)


# ======================================================================
# CSV fields
# ======================================================================

CSV_FIELDS = [
    "experiment",
    "parameter_index",

    "m",
    "n",
    "t",
    "k",
    "ell",

    "trial",

    "tau_enc",
    "tau_F",

    "input_error_weight_left",
    "input_error_weight_right",

    "input_neutral_left",
    "input_neutral_right",

    "input_delta_tail_weight_left",
    "input_delta_tail_weight_right",

    "transport_identity_left",
    "transport_identity_right",

    "selected_rows",

    "iid_selector_reference",

    "observed_fused_noise_weight",

    "theory_expected_noise_weight",
    "theory_noise_variance",
    "theory_noise_std",
    "theory_z_score",

    "linear_noise_bound",

    "deterministic_decode_guarantee",

    "within_decoder_radius",

    "expected_message_matches_plaintext_product",

    "decode_success",

    "decoder_radius",
]


# ======================================================================
# Generic helpers
# ======================================================================

def bits_equal(
    left,
    right,
):
    return np.array_equal(
        np.asarray(
            left,
            dtype=np.uint8,
        ).reshape(-1),

        np.asarray(
            right,
            dtype=np.uint8,
        ).reshape(-1),
    )


def write_csv_row(
    filename,
    row,
):
    """
    Append one row immediately so partial experiments survive failures.
    """

    path = Path(
        filename
    )

    write_header = (
        not path.exists()
        or
        path.stat().st_size == 0
    )

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
        )

        if write_header:
            writer.writeheader()

        normalized = {
            field: row.get(
                field,
                "",
            )
            for field in CSV_FIELDS
        }

        writer.writerow(
            normalized
        )


# ======================================================================
# Theoretical noise model
# ======================================================================

def independent_noise_moments(
    n,
    tau,
    M,
):
    """
    Exact first two moments for XOR of M independent uniformly sampled
    fixed-weight-tau binary vectors of length n.

    Returns

        mean,
        variance,
        standard_deviation.
    """

    n = int(
        n
    )

    tau = int(
        tau
    )

    M = int(
        M
    )

    if M <= 0 or tau == 0:
        return (
            0.0,
            0.0,
            0.0,
        )

    if not (
        0
        <=
        tau
        <=
        n
    ):
        raise ValueError(
            "tau must satisfy 0 <= tau <= n."
        )

    alpha = (
        1.0
        -
        2.0
        *
        tau
        /
        n
    )

    mean = (
        n
        /
        2.0
        *
        (
            1.0
            -
            alpha ** M
        )
    )

    if n <= 1:
        variance = (
            mean
            *
            (
                1.0
                -
                mean
            )
        )

        variance = max(
            0.0,
            variance,
        )

        return (
            mean,
            variance,
            math.sqrt(
                variance
            ),
        )

    beta = (
        1.0
        -
        (
            4.0
            *
            tau
            *
            (
                n - tau
            )
            /
            (
                n
                *
                (
                    n - 1
                )
            )
        )
    )

    alpha_2M = (
        alpha
        **
        (
            2 * M
        )
    )

    variance = (
        n
        /
        4.0
        *
        (
            1.0
            -
            alpha_2M
        )
        +
        n
        *
        (
            n - 1
        )
        /
        4.0
        *
        (
            beta ** M
            -
            alpha_2M
        )
    )

    # Numerical roundoff can produce tiny negative values.
    variance = max(
        0.0,
        variance,
    )

    return (
        mean,
        variance,
        math.sqrt(
            variance
        ),
    )


def iid_selector_reference(
    n,
):
    """
    Reference selector count when ciphertext coordinates are modelled
    as independent unbiased bits.

    Diagonal selector probability:

        Pr[c1_u c2_u = 1] = 1/4.

    Off-diagonal compressed selector:

        c1_u c2_v + c1_v c2_u

    has probability 3/8.

    Therefore

        E[M]
          =
        n/4
        +
        3/8 * n(n-1)/2
          =
        (3n^2+n)/16.
    """

    n = int(
        n
    )

    return (
        3.0
        *
        n
        *
        n
        +
        n
    ) / 16.0


# ======================================================================
# Admissible-noise helpers
# ======================================================================

def find_nonzero_admissible_weight(
    he,
    max_attempts=200000,
):
    """
    Find the smallest w in {1,...,t} for which the current secret
    Lambda_msg admits a sampled error

        e != 0,
        wt(e) = w,
        e Lambda_msg = 0.

    This is diagnostic and secret-key dependent.
    """

    for weight in range(
        1,
        he.decoder_radius + 1,
    ):
        try:
            error = (
                he.sample_admissible_error(
                    weight=weight,
                    max_attempts=max_attempts,
                )
            )

        except RuntimeError:
            continue

        if hamming_weight(
            error
        ) != weight:
            continue

        if not np.any(
            error
        ):
            continue

        projected = (
            gf2_matmul(
                error,
                he.Lambda_msg,
            )
        )

        if not np.any(
            projected
        ):
            return weight

    raise RuntimeError(
        "No nonzero admissible error was found "
        "within the Goppa decoding radius."
    )


# ======================================================================
# Experiment E1
# ======================================================================

def run_noisy_input_experiment(
    parameter_index,
    he,
    clean_evk,
    trials,
    csv_filename,
):
    """
    Test multiplication with

        tau_enc > 0,
        tau_F = 0.

    Both input ciphertexts contain nonzero admissible errors.
    """

    print()
    print("=" * 88)
    print("E1. NONZERO ADMISSIBLE INPUT-NOISE MULTIPLICATION")
    print("=" * 88)

    admissible_weight = (
        find_nonzero_admissible_weight(
            he
        )
    )

    print(
        "Smallest sampled nonzero admissible "
        f"weight = {admissible_weight}"
    )

    old_tau_enc = (
        he.tau_enc
    )

    he.tau_enc = (
        admissible_weight
    )

    successes = 0

    tail_delta_weights_left = []
    tail_delta_weights_right = []

    selected_rows_values = []

    try:
        for trial in range(
            1,
            trials + 1,
        ):
            mu1 = (
                he.rng.integers(
                    0,
                    2,
                    size=he.ell,
                    dtype=np.uint8,
                )
            )

            mu2 = (
                he.rng.integers(
                    0,
                    2,
                    size=he.ell,
                    dtype=np.uint8,
                )
            )

            (
                c1,
                meta1,
            ) = (
                he.encrypt(
                    mu1,
                    return_metadata=True,
                )
            )

            (
                c2,
                meta2,
            ) = (
                he.encrypt(
                    mu2,
                    return_metadata=True,
                )
            )

            e1 = (
                meta1[
                    "error"
                ]
            )

            e2 = (
                meta2[
                    "error"
                ]
            )

            # ----------------------------------------------------------
            # Full projected input errors
            #
            #     delta_i = e_i Lambda.
            # ----------------------------------------------------------

            delta1 = (
                gf2_matmul(
                    e1,
                    he.Lambda,
                )
            )

            delta2 = (
                gf2_matmul(
                    e2,
                    he.Lambda,
                )
            )

            neutral1 = (
                not np.any(
                    delta1[
                        :he.ell
                    ]
                )
            )

            neutral2 = (
                not np.any(
                    delta2[
                        :he.ell
                    ]
                )
            )

            delta_tail_weight1 = int(
                np.count_nonzero(
                    delta1[
                        he.ell:
                    ]
                )
            )

            delta_tail_weight2 = int(
                np.count_nonzero(
                    delta2[
                        he.ell:
                    ]
                )
            )

            tail_delta_weights_left.append(
                delta_tail_weight1
            )

            tail_delta_weights_right.append(
                delta_tail_weight2
            )

            # ----------------------------------------------------------
            # Verify
            #
            #     c Lambda
            #       =
            #     a + e Lambda.
            # ----------------------------------------------------------

            c1_array = (
                gf2_vector(
                    c1,
                    expected_length=he.n,
                    name="c1",
                )
            )

            c2_array = (
                gf2_vector(
                    c2,
                    expected_length=he.n,
                    name="c2",
                )
            )

            projected_c1 = (
                gf2_matmul(
                    c1_array,
                    he.Lambda,
                )
            )

            projected_c2 = (
                gf2_matmul(
                    c2_array,
                    he.Lambda,
                )
            )

            expected_projected_c1 = (
                meta1[
                    "a"
                ]
                ^
                delta1
            )

            expected_projected_c2 = (
                meta2[
                    "a"
                ]
                ^
                delta2
            )

            transport_identity1 = (
                np.array_equal(
                    projected_c1,
                    expected_projected_c1,
                )
            )

            transport_identity2 = (
                np.array_equal(
                    projected_c2,
                    expected_projected_c2,
                )
            )

            # ----------------------------------------------------------
            # Fused multiplication with tau_F = 0.
            # ----------------------------------------------------------

            diagnostics = (
                he.multiplication_diagnostics(
                    c1,
                    c2,
                    evk=clean_evk,
                )
            )

            expected_plaintext = (
                he.plaintext_multiply(
                    mu1,
                    mu2,
                )
            )

            expected_message_matches = (
                bits_equal(
                    diagnostics[
                        "expected_message"
                    ],
                    expected_plaintext,
                )
            )

            decode_success = (
                bool(
                    diagnostics[
                        "decode_success"
                    ]
                )
                and
                expected_message_matches
            )

            selected_rows = int(
                diagnostics[
                    "selected_rows"
                ]
            )

            selected_rows_values.append(
                selected_rows
            )

            fused_noise_weight = int(
                diagnostics[
                    "fused_noise_weight"
                ]
            )

            trial_ok = (
                hamming_weight(
                    e1
                )
                ==
                admissible_weight
                and
                hamming_weight(
                    e2
                )
                ==
                admissible_weight
                and
                neutral1
                and
                neutral2
                and
                transport_identity1
                and
                transport_identity2
                and
                fused_noise_weight
                ==
                0
                and
                expected_message_matches
                and
                decode_success
            )

            if trial_ok:
                successes += 1

            write_csv_row(
                csv_filename,
                {
                    "experiment": (
                        "noisy_input_tauF0"
                    ),

                    "parameter_index": (
                        parameter_index
                    ),

                    "m": he.mc.m,
                    "n": he.n,
                    "t": he.mc.t,
                    "k": he.k,
                    "ell": he.ell,

                    "trial": trial,

                    "tau_enc": (
                        admissible_weight
                    ),

                    "tau_F": 0,

                    "input_error_weight_left": (
                        hamming_weight(
                            e1
                        )
                    ),

                    "input_error_weight_right": (
                        hamming_weight(
                            e2
                        )
                    ),

                    "input_neutral_left": (
                        neutral1
                    ),

                    "input_neutral_right": (
                        neutral2
                    ),

                    "input_delta_tail_weight_left": (
                        delta_tail_weight1
                    ),

                    "input_delta_tail_weight_right": (
                        delta_tail_weight2
                    ),

                    "transport_identity_left": (
                        transport_identity1
                    ),

                    "transport_identity_right": (
                        transport_identity2
                    ),

                    "selected_rows": (
                        selected_rows
                    ),

                    "iid_selector_reference": (
                        iid_selector_reference(
                            he.n
                        )
                    ),

                    "observed_fused_noise_weight": (
                        fused_noise_weight
                    ),

                    "theory_expected_noise_weight": (
                        0.0
                    ),

                    "theory_noise_variance": (
                        0.0
                    ),

                    "theory_noise_std": (
                        0.0
                    ),

                    "theory_z_score": (
                        ""
                    ),

                    "linear_noise_bound": (
                        0
                    ),

                    "deterministic_decode_guarantee": (
                        True
                    ),

                    "within_decoder_radius": (
                        diagnostics[
                            "within_decoder_radius"
                        ]
                    ),

                    "expected_message_matches_plaintext_product": (
                        expected_message_matches
                    ),

                    "decode_success": (
                        decode_success
                    ),

                    "decoder_radius": (
                        he.decoder_radius
                    ),
                },
            )

    finally:
        he.tau_enc = (
            old_tau_enc
        )

    print(
        "Trials                           : "
        f"{trials}"
    )

    print(
        "Successful noisy-input products  : "
        f"{successes}/{trials}"
    )

    print(
        "Average selected Y rows          : "
        f"{np.mean(selected_rows_values):.3f}"
    )

    print(
        "Mean wt(delta_1 tail)            : "
        f"{np.mean(tail_delta_weights_left):.3f}"
    )

    print(
        "Mean wt(delta_2 tail)            : "
        f"{np.mean(tail_delta_weights_right):.3f}"
    )

    if successes != trials:
        raise AssertionError(
            "Phase 3E E1 failed: "
            "nonzero admissible input-noise "
            "multiplication was not always correct."
        )

    print(
        "E1 RESULT                        : PASS"
    )

    return (
        admissible_weight
    )


# ======================================================================
# Experiment E2/E3
# ======================================================================

def run_fused_noise_sweep(
    parameter_index,
    he,
    tau_values,
    trials,
    csv_filename,
):
    """
    Sweep tau_F while setting tau_enc = 0.

    This isolates fused evaluation-key noise.
    """

    print()
    print("=" * 88)
    print("E2/E3. FUSED-NOISE GROWTH SWEEP")
    print("=" * 88)

    old_tau_enc = (
        he.tau_enc
    )

    he.tau_enc = 0

    reference_M = (
        iid_selector_reference(
            he.n
        )
    )

    print(
        "IID-bit selector reference E[M] : "
        f"{reference_M:.3f}"
    )

    print(
        "Decoder radius                  : "
        f"{he.decoder_radius}"
    )

    try:
        for tau_F in (
            tau_values
        ):
            tau_F = int(
                tau_F
            )

            print()
            print("-" * 88)

            print(
                f"tau_F = {tau_F}"
            )

            print("-" * 88)

            evk = (
                he.generate_evaluation_key(
                    tau_F=tau_F,
                    batch_size=64,
                    max_pairs=200000,
                    progress=False,
                    set_default=False,
                )
            )

            verification = (
                he.verify_evaluation_key(
                    evk,
                    samples=min(
                        32,
                        evk.pair_count,
                    ),
                )
            )

            if not verification[
                "all_rows_have_expected_noise_weight"
            ]:
                raise AssertionError(
                    "Generated Y table has incorrect "
                    "fused-row noise weights."
                )

            observed_weights = []
            expected_weights = []
            selected_counts = []

            within_radius_count = 0
            decode_success_count = 0
            deterministic_guarantee_count = 0

            z_scores = []

            for trial in range(
                1,
                trials + 1,
            ):
                mu1 = (
                    he.rng.integers(
                        0,
                        2,
                        size=he.ell,
                        dtype=np.uint8,
                    )
                )

                mu2 = (
                    he.rng.integers(
                        0,
                        2,
                        size=he.ell,
                        dtype=np.uint8,
                    )
                )

                c1 = (
                    he.encrypt(
                        mu1
                    )
                )

                c2 = (
                    he.encrypt(
                        mu2
                    )
                )

                diagnostics = (
                    he.multiplication_diagnostics(
                        c1,
                        c2,
                        evk=evk,
                    )
                )

                M = int(
                    diagnostics[
                        "selected_rows"
                    ]
                )

                observed = int(
                    diagnostics[
                        "fused_noise_weight"
                    ]
                )

                (
                    expected,
                    variance,
                    std,
                ) = (
                    independent_noise_moments(
                        he.n,
                        tau_F,
                        M,
                    )
                )

                if std > 0:
                    z_score = (
                        observed
                        -
                        expected
                    ) / std

                    z_scores.append(
                        z_score
                    )

                else:
                    z_score = ""

                linear_bound = min(
                    he.n,
                    M
                    *
                    tau_F,
                )

                deterministic_guarantee = (
                    M
                    *
                    tau_F
                    <=
                    he.decoder_radius
                )

                expected_plaintext = (
                    he.plaintext_multiply(
                        mu1,
                        mu2,
                    )
                )

                expected_message_matches = (
                    bits_equal(
                        diagnostics[
                            "expected_message"
                        ],
                        expected_plaintext,
                    )
                )

                decode_success = (
                    bool(
                        diagnostics[
                            "decode_success"
                        ]
                    )
                    and
                    expected_message_matches
                )

                within_radius = bool(
                    diagnostics[
                        "within_decoder_radius"
                    ]
                )

                selected_counts.append(
                    M
                )

                observed_weights.append(
                    observed
                )

                expected_weights.append(
                    expected
                )

                if within_radius:
                    within_radius_count += 1

                if decode_success:
                    decode_success_count += 1

                if deterministic_guarantee:
                    deterministic_guarantee_count += 1

                write_csv_row(
                    csv_filename,
                    {
                        "experiment": (
                            "fused_noise_sweep"
                        ),

                        "parameter_index": (
                            parameter_index
                        ),

                        "m": he.mc.m,
                        "n": he.n,
                        "t": he.mc.t,
                        "k": he.k,
                        "ell": he.ell,

                        "trial": trial,

                        "tau_enc": 0,
                        "tau_F": tau_F,

                        "input_error_weight_left": 0,
                        "input_error_weight_right": 0,

                        "input_neutral_left": True,
                        "input_neutral_right": True,

                        "input_delta_tail_weight_left": 0,
                        "input_delta_tail_weight_right": 0,

                        "transport_identity_left": True,
                        "transport_identity_right": True,

                        "selected_rows": M,

                        "iid_selector_reference": (
                            reference_M
                        ),

                        "observed_fused_noise_weight": (
                            observed
                        ),

                        "theory_expected_noise_weight": (
                            expected
                        ),

                        "theory_noise_variance": (
                            variance
                        ),

                        "theory_noise_std": (
                            std
                        ),

                        "theory_z_score": (
                            z_score
                        ),

                        "linear_noise_bound": (
                            linear_bound
                        ),

                        "deterministic_decode_guarantee": (
                            deterministic_guarantee
                        ),

                        "within_decoder_radius": (
                            within_radius
                        ),

                        "expected_message_matches_plaintext_product": (
                            expected_message_matches
                        ),

                        "decode_success": (
                            decode_success
                        ),

                        "decoder_radius": (
                            he.decoder_radius
                        ),
                    },
                )

            observed_mean = float(
                np.mean(
                    observed_weights
                )
            )

            predicted_mean = float(
                np.mean(
                    expected_weights
                )
            )

            mean_M = float(
                np.mean(
                    selected_counts
                )
            )

            min_M = int(
                np.min(
                    selected_counts
                )
            )

            max_M = int(
                np.max(
                    selected_counts
                )
            )

            mean_difference = (
                observed_mean
                -
                predicted_mean
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        (
                            np.asarray(
                                observed_weights,
                                dtype=float,
                            )
                            -
                            np.asarray(
                                expected_weights,
                                dtype=float,
                            )
                        )
                        **
                        2
                    )
                )
            )

            print(
                "EVK rows                       : "
                f"{evk.pair_count}"
            )

            print(
                "Packed EVK bytes               : "
                f"{evk.packed_bytes}"
            )

            print(
                "EVK generation time            : "
                f"{evk.generation_seconds:.6f} s"
            )

            print(
                "Average selected rows M        : "
                f"{mean_M:.3f}"
            )

            print(
                "Min/max selected rows          : "
                f"{min_M}/{max_M}"
            )

            print(
                "IID selector reference         : "
                f"{reference_M:.3f}"
            )

            print(
                "Observed mean noise weight      : "
                f"{observed_mean:.4f}"
            )

            print(
                "Predicted mean noise weight     : "
                f"{predicted_mean:.4f}"
            )

            print(
                "Observed - predicted mean       : "
                f"{mean_difference:+.4f}"
            )

            print(
                "Per-trial prediction RMSE       : "
                f"{rmse:.4f}"
            )

            print(
                "Within decoder radius           : "
                f"{within_radius_count}/{trials}"
            )

            print(
                "Successful decryptions          : "
                f"{decode_success_count}/{trials}"
            )

            print(
                "Worst-case guaranteed trials    : "
                f"{deterministic_guarantee_count}/{trials}"
            )

            if z_scores:
                print(
                    "Mean standardized deviation    : "
                    f"{np.mean(z_scores):+.4f}"
                )

            if tau_F > 0:
                safe_fan_in = (
                    he.decoder_radius
                    //
                    tau_F
                )

                print(
                    "Deterministic safe fan-in      : "
                    f"M <= {safe_fan_in}"
                )

            # ----------------------------------------------------------
            # tau_F = 0 is a mandatory correctness baseline.
            # ----------------------------------------------------------

            if tau_F == 0:
                if any(
                    weight != 0
                    for weight in observed_weights
                ):
                    raise AssertionError(
                        "tau_F=0 produced nonzero fused noise."
                    )

                if (
                    decode_success_count
                    !=
                    trials
                ):
                    raise AssertionError(
                        "tau_F=0 multiplication "
                        "did not decrypt correctly."
                    )

                print(
                    "tau_F=0 baseline               : PASS"
                )

    finally:
        he.tau_enc = (
            old_tau_enc
        )


# ======================================================================
# One complete parameter set
# ======================================================================

def run_parameter_set(
    parameter_index,
    params,
    input_trials,
    sweep_trials,
    tau_f_max,
    csv_filename,
):
    m = int(
        params[
            "m"
        ]
    )

    n = int(
        params[
            "n"
        ]
    )

    t = int(
        params[
            "t"
        ]
    )

    ell = int(
        params[
            "ell"
        ]
    )

    seed = int(
        params[
            "seed"
        ]
    )

    print()
    print("#" * 88)
    print("PHASE 3E PARAMETER SET")
    print("#" * 88)

    print(
        f"index = {parameter_index}"
    )

    print(
        f"(m,n,t,ell) = "
        f"({m},{n},{t},{ell})"
    )

    print(
        f"seed = {seed}"
    )

    # Existing McEliece key generation uses the legacy global
    # NumPy random state.
    np.random.seed(
        seed
    )

    mc = McElieceCipher(
        m,
        n,
        t,
    )

    mc.generate_random_keys()

    print(
        f"Derived k = {mc.k}"
    )

    he = FusedMcElieceHE(
        mc,
        ell=ell,
        tau_enc=0,
        tau_F=0,
        seed=seed + 10000,
    )

    he.construct_transport(
        randomization_trials=64,
        store_right_inverse=False,
    )

    print()
    print(
        "Transport identity "
        "G_pub Lambda = I_k       : "
        f"{he.transport_diagnostics['GpLambda_identity']}"
    )

    print(
        "Lambda zero rows                 : "
        f"{he.transport_diagnostics['lambda_zero_rows']}"
    )

    # ------------------------------------------------------------------
    # Clean EVK used by noisy-input experiment.
    # ------------------------------------------------------------------

    print()
    print(
        "Generating clean tau_F=0 "
        "evaluation key..."
    )

    clean_evk = (
        he.generate_evaluation_key(
            tau_F=0,
            batch_size=64,
            max_pairs=200000,
            progress=False,
            set_default=False,
        )
    )

    clean_verification = (
        he.verify_evaluation_key(
            clean_evk,
            samples=min(
                32,
                clean_evk.pair_count,
            ),
        )
    )

    if not clean_verification[
        "all_rows_have_expected_noise_weight"
    ]:
        raise AssertionError(
            "Clean evaluation key verification failed."
        )

    # ------------------------------------------------------------------
    # E1
    # ------------------------------------------------------------------

    admissible_weight = (
        run_noisy_input_experiment(
            parameter_index=parameter_index,
            he=he,
            clean_evk=clean_evk,
            trials=input_trials,
            csv_filename=csv_filename,
        )
    )

    print()
    print(
        "Confirmed admissible tau_enc     : "
        f"{admissible_weight}"
    )

    # ------------------------------------------------------------------
    # E2/E3
    # ------------------------------------------------------------------

    tau_values = list(
        range(
            0,
            tau_f_max + 1,
        )
    )

    run_fused_noise_sweep(
        parameter_index=parameter_index,
        he=he,
        tau_values=tau_values,
        trials=sweep_trials,
        csv_filename=csv_filename,
    )

    print()
    print(
        "PHASE 3E PARAMETER SET COMPLETE."
    )


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3E noisy-input and "
            "fused-noise-growth validation."
        )
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=(
            "Run one parameter-set index. "
            "Default: run all."
        ),
    )

    parser.add_argument(
        "--input-trials",
        type=int,
        default=20,
        help=(
            "Number of nonzero admissible-input "
            "multiplication trials."
        ),
    )

    parser.add_argument(
        "--sweep-trials",
        type=int,
        default=50,
        help=(
            "Number of multiplication trials "
            "for each tau_F."
        ),
    )

    parser.add_argument(
        "--tau-f-max",
        type=int,
        default=2,
        help=(
            "Sweep tau_F = 0,...,tau-f-max."
        ),
    )

    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=(
            "CSV output filename."
        ),
    )

    parser.add_argument(
        "--reset-csv",
        action="store_true",
        help=(
            "Delete the CSV before running."
        ),
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use five E1 trials and ten "
            "sweep trials."
        ),
    )

    args = parser.parse_args()

    if args.tau_f_max < 0:
        raise ValueError(
            "--tau-f-max must be nonnegative."
        )

    if args.quick:
        input_trials = 5
        sweep_trials = 10

    else:
        input_trials = int(
            args.input_trials
        )

        sweep_trials = int(
            args.sweep_trials
        )

    if input_trials <= 0:
        raise ValueError(
            "input trials must be positive."
        )

    if sweep_trials <= 0:
        raise ValueError(
            "sweep trials must be positive."
        )

    csv_path = Path(
        args.csv
    )

    if (
        args.reset_csv
        and
        csv_path.exists()
    ):
        csv_path.unlink()

    if args.index is None:
        indices = list(
            range(
                len(
                    PARAMETER_SETS
                )
            )
        )

    else:
        if not (
            0
            <=
            args.index
            <
            len(
                PARAMETER_SETS
            )
        ):
            raise ValueError(
                "Invalid parameter-set index."
            )

        indices = [
            args.index
        ]

    for parameter_index in (
        indices
    ):
        run_parameter_set(
            parameter_index=parameter_index,
            params=(
                PARAMETER_SETS[
                    parameter_index
                ]
            ),
            input_trials=input_trials,
            sweep_trials=sweep_trials,
            tau_f_max=args.tau_f_max,
            csv_filename=args.csv,
        )

    print()
    print("=" * 88)
    print("PHASE 3E COMPLETE")
    print("=" * 88)

    print(
        "CSV results: "
        f"{args.csv}"
    )


if __name__ == "__main__":
    main()