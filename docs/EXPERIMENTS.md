# The experiments

| ID | Question | Function |
|---|---|---|
| E1 | Does two-channel fusion beat the obvious alternatives? | `baseline_ladder` |
| E2 | What does each channel catch that the other cannot? | `channel_matrix` |
| E3 | Which tamper classes are catchable at all? | `per_class_detection` |
| E4 | Does the learned layer generalise to an unseen attack? | `leave_one_class_out` |
| E5 | Did we learn tampering, or our own renderer? | `unseen_generator` |
| E6 | How does each edit-quality axis affect detection? | `strength_axes` |
| E7 | How does capture quality erode detection? | `degradation_curves` |
| E8 | How much decision error does the reading stage cause? | `ocr_sweep` |
| E9 | Is the score meaningful, and what is it worth? | `calibration_and_cost` |
| E10 | Which stage is responsible for the errors? | `attribution_study` |
| E11 | Can we say WHERE, not just whether? | `localisation_study` |

## Design decisions worth defending

**The baseline ladder is mandatory.** `always_benign`, `single_signal` and
`rules_only` are built and reported every run. Without a counterfactual,
"we built two channels and fused them" is unfalsifiable. If `fused` does not
beat `rules_only`, that is the headline and it gets reported as the headline.

**Strength axes are independent, not a ladder.** "Wrong font" and "recompressed
after the edit" are separate properties of an edit, not increasing points on one
scale. Each gets its own curve. Plotting them as a single ladder would imply an
ordering that does not exist.

**Reprint and full forgery are classes, not strength levels.** They have no
localisable edit region, so localisation metrics are undefined for them. They
are excluded from those metrics and reported separately rather than scored as
misses.

**Nothing is reported as a single headline accuracy.** Detection is broken down
by tamper class, and the classes that are effectively invisible are named.

## Operating points

Detection rates in E3, E6, E7 and the external harness are quoted at a **fixed
10 percent false-alarm budget on clean documents**: the threshold is the 90th
percentile of the clean score distribution, fixed on clean data before any
tampered document is scored.

## Quality gate bounds

The sub-score normalisation bounds in `quality/gate.py` come from the clean
render distribution rather than per-document tuning. The gate threshold is a
parameter; E7 shows how detection responds as capture quality degrades, which is
what the threshold should be chosen against.

## Reproducing

```bash
python run.py smoke   # ~3 min, used by CI
python run.py demo    # ~10 min
python run.py full    # the numbers in the README
```

Everything is seeded. `test_dataset_is_reproducible_from_a_seed` enforces it.
