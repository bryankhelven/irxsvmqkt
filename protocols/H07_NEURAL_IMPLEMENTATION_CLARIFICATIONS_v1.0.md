# H07 Neural Implementation Clarifications v1.0

Status: **FROZEN BEFORE ANY H07 BERTWEET RESULT**
Date: 2026-09-02

These clauses resolve implementation details that METHOD FREEZE v1.0 left implicit.
They do not alter the research questions, model family, data conditions, learning rate,
batch size, seeds, selection metric, or test protocol.

## Base model artifact
- Hugging Face repository: `vinai/bertweet-base`.
- Resolve one repository snapshot before training.
- Record the resolved commit/snapshot identifier.
- All H07 runs load from that same local snapshot.
- `use_fast=False` for the tokenizer.

## Model input adapter
Exactly:
- `<USER>` -> `@USER`
- `<URL>` -> `HTTPURL`

No additional tweet normalizer is enabled.

## AdamW parameter groups
Weight decay `0.01` applies to parameters except:
- bias parameters;
- LayerNorm normalization parameters.

Those receive weight decay `0.0`.

## Scheduler
- linear warmup then linear decay;
- warmup ratio: 0.10;
- scheduler horizon is the number of optimizer updates planned for 5 epochs.

## Gradient handling
- gradient accumulation: 2 micro-batches;
- max gradient norm: 1.0;
- clip immediately before each optimizer step.

## Mixed precision
On CUDA, FP16 automatic mixed precision is enabled as a runtime optimization.
This does not alter the frozen effective batch size or learning-rate schedule.

If the frozen train micro-batch of 16 produces CUDA OOM during the pre-result smoke
test, the full experiment must STOP and return a failure handoff. The runner must not
silently change batch size.

## Early stopping
- validation once after every completed epoch;
- improvement means strictly greater validation Macro-F1;
- `min_delta = 0`;
- patience = 1 completed non-improving epoch;
- best validation checkpoint is restored before final-test inference.

## Determinism
For each run:
- Python RNG seed = model seed;
- NumPy seed = model seed;
- PyTorch CPU/CUDA seed = model seed;
- CuDNN benchmark disabled;
- deterministic algorithms requested in warn-only mode.

GPU kernels may still exhibit hardware/library-level nondeterminism; this is one reason
for the frozen three-seed neural summary.

## Run order
The full runner attempts, in this order:
1. A-C seed 42
2. A-L seed 42, evaluated on A-L and A-C test text
3. G-C seed 42
4. A-C seed 2026
5. A-C seed 7
6. G-C seed 2026
7. G-C seed 7

Completed runs are resumable/skippable by immutable DONE metadata.

## H07 acceptance target
A complete H07 handoff contains:
- A-C seeds 42 / 2026 / 7;
- G-C seeds 42 / 2026 / 7;
- A-L seed 42 evaluated on both A-L and A-C text;
- validation histories;
- model snapshot metadata;
- prediction files with the canonical 5,947 test IDs;
- environment/GPU snapshot.

No RoBERTa run is part of H07.
