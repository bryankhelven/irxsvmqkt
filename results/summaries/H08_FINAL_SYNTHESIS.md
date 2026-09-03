# H08 Final Qualitative Synthesis

## Status
**PASS — H08 qualitative stage closed.**

The formal 64-case pilot was selected deterministically in H08A before the qualitative
codebook was frozen. Two isolated Claude Code Haiku passes annotated only `case_id` and
`text_clean`.

These are replicate model passes, **not independent human annotators**.

## Replicate stability
- best-label exact agreement: 0.641
- best-label Cohen's kappa: 0.492
- interpretability exact agreement: 0.797
- interpretability kappa: 0.540
- primary-mechanism exact agreement: 0.625
- primary-mechanism kappa: 0.556
- confidence exact agreement: 0.750

The two passes disagreed on best label for 23/64 cases
(35.9%), which itself is evidence that many sampled hard cases are unstable under text-only interpretation.

## Consolidator adjudication
Best-label counts:
- sarcasm: 22
- irony: 8
- regular: 18
- uncertain: 16

Interpretability:
- clear from text: 42
- context-dependent: 16
- opaque fragment: 6

Most frequent challenge tags:
- EVENT_TOPIC_DEPENDENCY: 26
- IRONY_SARCASM_BOUNDARY: 26
- CONTEXT_WORLD_KNOWLEDGE: 23
- FIGURATIVE_REGULAR_BOUNDARY: 12
- FRAGMENT_OR_CORRUPTION: 7
- QUOTATION_ATTRIBUTION: 2
- HASHTAG_META_CUE: 1

## Important limitation
After unblinding, the adjudicated best label exactly matches the dataset gold in
25/64 cases.

**This is NOT an estimate that the corpus is only 39.1% correctly labeled.**
The pilot intentionally contains model errors and boundary cases, is not randomly sampled,
and the final adjudicator is an AI consolidator rather than a human gold-annotation panel.

The manuscript may use the replicate disagreement rate and selected examples to discuss
ambiguity/context dependence. It should not present the 25/64 figure as corpus label-noise prevalence.

## Manuscript-safe qualitative conclusion
The sampled errors repeatedly involve:
1. irony-sarcasm boundary ambiguity;
2. event/world-context dependence;
3. figurative-vs-literal boundary ambiguity;
4. missing linked or conversational context;
5. reversal/mock praise, incongruity, rhetorical questions, self-directed stance, echoic stance, and wordplay.

This supports the quantitative finding that contextual pretraining improves the clean
classification task while leaving a substantial residue of pragmatically and contextually
ambiguous cases.
