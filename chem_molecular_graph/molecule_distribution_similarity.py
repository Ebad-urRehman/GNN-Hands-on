"""
Distribution-similarity check for the molecule-size alignment experiment.

For every QM9 target property column, compare the value-distribution of the
downstream molecules (15-16 atoms) against each candidate size-group (labeled
by their atom ranges), sweeping around the downstream size:
    10_11 (10-11), 12_13 (12-13), 17_18 (17-18), 19_20 (19-20),
    21_22 (21-22), 22_23 (22-23), 23_24 (23-24), 25_26 (25-26)

All groups are downsampled to an equal size (the smallest pool size) so that
distribution comparisons are fair.

Metrics per (target, group):
    wasserstein   : Earth-mover distance between the two distributions (same units as property)
    ks_stat/ks_p  : Kolmogorov-Smirnov test for whether samples come from same distribution
    kl_divergence : KL divergence between shared-bin normalized histograms
    overlap       : histogram overlap coefficient (1.0 = identical histograms)

Figures:
    figures/molecule_graphs/target_{k}/dist_comparison_<group>.png  (overlaid histograms)
    figures/molecule_graphs/distribution_wasserstein_heatmap.png
    figures/molecule_graphs/distribution_similarity_summary.png     (labeled bar chart)
    figures/molecule_graphs/distribution_similarity_summary.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp, entropy

from torch_geometric.datasets import QM9

SEED = 42
BINS = 40

# standard QM9 property ordering for columns 0-18 of data.y
QM9_PROPERTIES = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u", "h",
    "g", "cv", "u0_atom", "u_atom", "h_atom", "g_atom", "a", "b", "c",
]

# size groups labeled by atom range, sweeping symmetrically around downstream
GROUPS = {
    "downstream": lambda n: 15 <= n <= 16,
    "10_11": lambda n: 10 <= n <= 11,
    "12_13": lambda n: 12 <= n <= 13,
    "17_18": lambda n: 17 <= n <= 18,
    "19_20": lambda n: 19 <= n <= 20,
    "21_22": lambda n: 21 <= n <= 22,
    "22_23": lambda n: 22 <= n <= 23,
    "23_24": lambda n: 23 <= n <= 24,
    "25_26": lambda n: 25 <= n <= 26,
}

# downstream is always the reference; compare against these groups
REFERENCE = "downstream"
COMPARED = ["10_11", "12_13", "17_18", "19_20", "21_22", "22_23", "23_24", "25_26"]
def main():
    np.random.seed(SEED)
    dataset = QM9(root="dataset/QM9")

    # build full candidate index lists per group (same ranges as main script)
    candidates = {name: [i for i, d in enumerate(dataset) if pred(d.num_nodes)]
                  for name, pred in GROUPS.items()}

    rng = np.random.default_rng(SEED)
    for name in candidates:
        rng.shuffle(candidates[name])

    # cap every group to the smallest pool size so all groups are equal-sized
    # (this keeps 10_11 in full and downsamples the bigger pools to match)
    N_PER_GROUP = min(len(idx) for idx in candidates.values())

    ref_indices = candidates[REFERENCE][:N_PER_GROUP]
    group_indices = {name: candidates[name][:N_PER_GROUP]
                     for name in COMPARED}

    print("N_PER_GROUP:", N_PER_GROUP)
    print("Group sizes:", {name: len(idx) for name, idx in
          {REFERENCE: ref_indices, **group_indices}.items()})

    # collect similarity rows per (target, group)
    rows = []
    for k in range(19):
        prop = QM9_PROPERTIES[k]
        out_dir = f"figures/molecule_graphs/target_{k}"
        os.makedirs(out_dir, exist_ok=True)

        ref_vals = np.array([dataset[i].y[0, k].item() for i in ref_indices])

        for name in COMPARED:
            other_vals = np.array([dataset[i].y[0, k].item() for i in group_indices[name]])

            dist = wasserstein_distance(ref_vals, other_vals)
            ks = ks_2samp(ref_vals, other_vals)

            # shared-bin normalized histograms for KL + overlap
            lo = min(ref_vals.min(), other_vals.min())
            hi = max(ref_vals.max(), other_vals.max())
            bins = np.linspace(lo, hi, BINS + 1)
            h1, _ = np.histogram(ref_vals, bins=bins, density=True)
            h2, _ = np.histogram(other_vals, bins=bins, density=True)
            p1 = h1 / (h1.sum() + 1e-12) + 1e-12
            p2 = h2 / (h2.sum() + 1e-12) + 1e-12
            kl = entropy(p1, p2)
            overlap = float(np.minimum(p1, p2).sum())

            rows.append({
                "target_index": k,
                "property": prop,
                "group": name,
                "ref_mean": float(ref_vals.mean()),
                "ref_std": float(ref_vals.std()),
                "group_mean": float(other_vals.mean()),
                "group_std": float(other_vals.std()),
                "wasserstein": float(dist),
                "kl_divergence": float(kl),
                "overlap": overlap,
                "ks_stat": float(ks.statistic),
                "ks_p": float(ks.pvalue),
            })

            # overlaid histogram saving into the per-target folder
            plt.figure(figsize=(8, 5))
            plt.hist(ref_vals, bins=30, alpha=0.6, density=True,
                     label=f"{REFERENCE} (15-16)")
            plt.hist(other_vals, bins=30, alpha=0.6, density=True, label=name)
            plt.xlabel(f"{prop} (target {k})")
            plt.ylabel("density")
            plt.title(f"Distribution comparison: {prop} - {REFERENCE} vs {name}\n"
                      f"Wasserstein={dist:.3g}  overlap={overlap:.3f}  KS p={ks.pvalue:.3g}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(f"{out_dir}/dist_comparison_{name}.png", dpi=120)
            plt.close()

            print(f"target {k:2d} ({prop:12s}) {REFERENCE} vs {name:20s} "
                  f"Wasserstein={dist:9.3g}  KL={kl:8.3g}  overlap={overlap:.3f}")

    #  summary outputs 
    df = pd.DataFrame(rows)

    # heatmap of log-transformed Wasserstein distance, rows=property, cols=group
    df["log_wasserstein"] = np.log1p(df["wasserstein"])
    heat = df.pivot(index="property", columns="group", values="log_wasserstein")
    heat = heat[COMPARED]

    plt.figure(figsize=(9, 8))
    im = plt.imshow(heat.values, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="log1p(Wasserstein distance)")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=45)
    plt.xlabel("Compared group")
    plt.ylabel("Target property")
    plt.title("Target-value distribution distance (downstream vs each group)")
    plt.tight_layout()
    plt.savefig("figures/molecule_graphs/distribution_wasserstein_heatmap.png", dpi=120)
    plt.show()

    # bar chart: wasserstein per group, one bar per property
    bar_df = df.pivot(index="property", columns="group",
                      values="wasserstein")[COMPARED]
    bar_df.plot(kind="bar", figsize=(13, 6), width=0.8)
    plt.ylabel("Wasserstein distance (property units)")
    plt.xlabel("Target property")
    plt.title("Target-value distribution distance by property and group")
    plt.legend(title="Group")
    plt.tight_layout()
    plt.savefig("figures/molecule_graphs/distribution_similarity_summary.png", dpi=120)
    plt.show()

    df.to_csv("figures/molecule_graphs/distribution_similarity_summary.csv", index=False)
    print("Saved summary figures and CSV to figures/molecule_graphs/")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()