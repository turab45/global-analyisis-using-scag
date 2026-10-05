# scag/ablation.py

from collections import Counter

import numpy as np
import pandas as pd
import torch


def community_ablation(
    model, hook_target, loader, communities, num_filters, device,
    val_segs=None, node2label=None, concept_names=None,
    n_random=1000, seed=2,
):
    rng = np.random.default_rng(seed)

    def predict(ablate=None):
        cache, handle = [], None

        if ablate is not None and len(ablate):
            channels = torch.zeros(num_filters, dtype=torch.bool, device=device)
            channels[list(ablate)] = True

            def zero_channels(module, inputs, output):
                output = output.clone()
                output[:, channels] = 0.0
                return output

            handle = hook_target.register_forward_hook(zero_channels)

        with torch.no_grad():
            for batch in loader:
                logits = model(batch[0].to(device))
                cache.append(torch.softmax(logits, dim=1).cpu())

        if handle:
            handle.remove()

        return torch.cat(cache, dim=0)

    clean = predict()

    def kl_values(q):
        return (
            clean * torch.log((clean + 1e-8) / (q + 1e-8))
        ).sum(dim=1).numpy()

    rows = []

    for ci, community in enumerate(communities):
        if len(community) < 3 or ci >= 8:
            continue

        ablated = predict(community)
        delta = kl_values(ablated)

        random_delta = np.stack([
            kl_values(
                predict(rng.choice(num_filters, len(community), replace=False))
            )
            for _ in range(n_random)
        ])

        row = {
            "comm": ci,
            "size": len(community),
            "kl": float(delta.mean()),
            "rnd": float(random_delta.mean()),
            "rnd_sd": float(random_delta.std()),
            "flip": float((clean.argmax(1) != ablated.argmax(1)).float().mean()),
        }

        if val_segs is not None and node2label is not None:
            labeled_nodes = [n for n in community if n in node2label]

            if labeled_nodes:
                dominant = int(
                    Counter(node2label[n] for n in labeled_nodes)
                    .most_common(1)[0][0]
                )

                burden = (val_segs == dominant).float().mean(dim=(1, 2, 3)).numpy()
                high = burden > np.median(burden)

                if high.sum() >= 3 and (~high).sum() >= 3:
                    row.update({
                        "dom_concept": dominant,
                        "n_high": int(high.sum()),
                        "n_low": int((~high).sum()),
                        "kl_high": float(delta[high].mean()),
                        "kl_low": float(delta[~high].mean()),
                        "rnd_high": float(random_delta[:, high].mean()),
                        "rnd_low": float(random_delta[:, ~high].mean()),
                    })

                    row["specificity"] = row["kl_high"] - row["kl_low"]
                    row["spec_ratio"] = row["kl_high"] / (row["kl_low"] + 1e-9)

        rows.append(row)

        message = (
            f"  comm {ci:02d}: KL={row['kl']:.4f} "
            f"(random {row['rnd']:.4f}±{row['rnd_sd']:.4f})"
        )

        if "specificity" in row and concept_names is not None:
            message += (
                f" | [{concept_names[row['dom_concept']]}] "
                f"high/low={row['n_high']}/{row['n_low']} "
                f"KL={row['kl_high']:.3f}/{row['kl_low']:.3f} "
                f"spec={row['specificity']:+.3f}"
            )

        print(message)

    return pd.DataFrame(rows)