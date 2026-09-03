# H08A Statistical Analysis + Error Pilot

- status: **PASS**
- bootstrap: 10,000 stratified resamples, seed 20260902
- canonical test ID SHA-256: `35725f6a2d2d99ca65578fa838bae7c8be645b16e6c26fecde0449ef13e32f82`

## Main uncertainty anchors

- Surface-only A-C Macro-F1: 0.593520 [0.581384, 0.605646]
- S2 A-C Macro-F1: 0.825567 [0.816322, 0.834668]
- BERTweet A-C seed42 Macro-F1: 0.886816 [0.879261, 0.894526]
- BERTweet G-C seed42 Macro-F1: 0.887424 [0.879733, 0.895077]
- A-L-trained BERTweet evaluated on A-L: 1.000000 [1.000000, 1.000000]
- same A-L-trained BERTweet evaluated on A-C: 0.200593 [0.193419, 0.207916]

## Neural seed summary

- BERTweet A-C: **0.884344 ± 0.002266** Macro-F1
- BERTweet G-C: **0.886451 ± 0.001742** Macro-F1

## Required paired deltas

- sparse_matched_leaky_minus_clean: +0.173925 [+0.164864, +0.182811]
- bert_ac42_minus_s2_ac: +0.061249 [+0.051959, +0.070487]
- s2_gc_minus_ac: -0.001289 [-0.002595, -0.000001]
- s3_gc_minus_ac: -0.000455 [-0.001572, +0.000573]
- surface_gc_minus_ac: +0.000819 [+0.000157, +0.001643]
- bert_gc_minus_ac_seed42: +0.000609 [-0.004682, +0.006110]
- bert_gc_minus_ac_seed2026: +0.005124 [+0.000145, +0.010085]
- bert_gc_minus_ac_seed7: +0.000588 [-0.004909, +0.006125]
- same_leaky_trained_model_al_minus_ac_eval: +0.799407 [+0.791964, +0.806466]

## Error accounting

- S2 A-C direct irony<->sarcasm confusions: 962
- BERTweet A-C direct cross-confusions by seed: [666, 688, 672]
- BERTweet mean reduction vs S2: 29.80%
- S2/BERT seed42 both correct: 4610
- BERT-only correct: 630
- S2-only correct: 260
- both wrong: 447

## Error pilot

- blind unique cases: 64
- key selection rows: 64

The qualitative taxonomy is intentionally **not frozen** in H08A.
The consolidator must review `ERROR_PILOT_BLIND.tsv` and `ERROR_PILOT_KEY.tsv` first.
