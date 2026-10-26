# KCSE DiffusionRepair Plus

Research code and preserved results for the final KCSE experiment: **43 bugs × Base/Full × seeds 42, 43, 44 = 258 planned runs**. This public snapshot supports independent recomputation of the reported accounting from preserved candidates and evaluation records. Historical 51-bug experiments are not the final experiment denominator.

## Recompute the reported results

Python 3.10+ and its standard library are sufficient. No models, containers, GPU or private repository access are needed for these commands:

```bash
git clone https://github.com/lhs9275/KCSE_DiffusionRepair_Plus.git
cd KCSE_DiffusionRepair_Plus
python3 tools/verify_public.py
python3 tools/kcse.py aggregate --output outputs/accounting-k1.json
python3 tools/kcse.py aggregate --k 5 --output outputs/accounting-k5.json
python3 tools/kcse.py aggregate --k 10 --output outputs/accounting-k10.json
```

The k=1 recomputation checks identity, denominator, stage coverage, all rows and per-arm/per-group states against the preserved ACCOUNTING file. Success/failure/unresolved totals are **29/225/4**, with no missing evaluation among 258 planned runs. Unresolved outcomes are not converted to failures.

| Arm | prefix@1 | prefix@5 | prefix@10 |
| --- | ---: | ---: | ---: |
| Base | 0.1550 | 0.3721 | 0.4031 |
| Full | 0.0698 | 0.1395 | 0.1395 |

These are prefix repair rates; do not relabel them as unbiased pass@k estimates. The results do not demonstrate that Full outperforms Base.

## Included

- `generation/bundle/`: frozen generation code, exact task contract, metadata, retrieval corpus, type map and model identity manifest.
- `generation/results/`: candidate results and generation completion markers for the 258 planned runs.
- `evaluation/`: frozen evaluation code, candidate snapshots and preserved evaluation/measurement integrity records, including superseded attempts needed by the accounting.
- `training/repo/`: Planner preparation/training source; `training/assets/random_split.json` preserves the exact recovered split.
- `models/planner/`: checkpoint provenance and metadata, without checkpoint weights.
- `evidence/final_audit/`: final audit references and assertions. The audit manifest also references full model-input logs retained in the private archive; those logs are not part of this public snapshot.
- `SOURCE_MAP.json`: SHA-256 links to the verified complete package. Original copied research files are unchanged.
- `SHA256.json`: public-snapshot integrity catalogue.

Frozen code and integrity records can contain historical filesystem paths. The aggregate launcher resolves the supplied data relative to this checkout instead.

## Scope and runtime assets

This is a lightweight public research snapshot, **not the 24 GB complete offline execution archive**. Model weights, Docker images, offline Git mirrors/wheels, raw operational logs, manuscripts, old experiments, full model-input logs and the Planner training dataset are excluded. No credentials are required for the public aggregation commands.

The retained `tools/kcse.py` also documents the complete-package runtime interface. Its `verify`, generation, evaluation and training commands require the omitted complete-package assets; they are not supported by this public snapshot alone. Use `tools/verify_public.py` here. GPU reruns are not claimed to work from this clone alone. The third-party backbones are `bigcode/starcoder2-7b` and `intfloat/e5-small-v2`; obtaining them alone does not replace the pinned evaluator/offline assets or final Planner checkpoint. Third-party model/container assets are not redistributed by this public repository.

Packaging validation completed representative Full/Base GPU generation, new-candidate evaluation, reduced CPU Planner training and offline environment controls for **23/43 bugs**. The remaining 20 environments were not run. Full 258-run GPU replay and full Planner retraining were not performed. See `VALIDATION.md` for the complete-package validation scope; some detailed evidence paths there refer to excluded private-archive files.

## Attribution and rights

See `THIRD_PARTY.md` and `third_party/` for preserved upstream benchmark license/notice texts. Internal frozen manifests may reference Python cache files omitted from this source snapshot; the public integrity check uses `SHA256.json`.

Benchmark project names and code fragments originate from the corresponding BugsInPy target projects. Their upstream rights and licenses continue to apply. This snapshot does not grant a new blanket license to third-party code, corpora, patches or model assets. No project-wide license has been inferred from the private source repository.
