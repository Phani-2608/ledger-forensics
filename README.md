# Ledger Forensics

**Multichannel document fraud forensics with error attribution.**

Receipt and invoice tampering is caught by two independent channels, pixel
forensics and semantic consistency. This project builds both, fuses them into a
calibrated risk score, and then measures the thing almost nobody measures: when
the final decision is wrong, **which stage caused it**.

Runs entirely locally. No cloud account, no API key, no dataset download.

```bash
pip install -r requirements.txt
python run.py demo        # ~8 minutes, writes artifacts/SUMMARY.md
pytest tests/ -q          # 34 tests
```

---

## The point

I was not trying to prove that a model can detect every forged receipt. I
wanted to measure **where document fraud detection actually breaks**.

So: generate controlled fraud with exact pixel-level ground truth, build
independent visual and semantic detectors, measure what each one sees, calibrate
the combination against dollars, and trace every wrong decision back to the
stage responsible for it.

The differentiator is not the OCR, the model, or the service wrapper. It is that
the system is built to understand its own failure boundary.

---

## Headline results

Numbers below are from `python run.py demo` (220 submissions, seed 0). Rerun it
and you will get these back; everything is seeded and CI enforces it.

### Fusion earns its complexity, but not for the reason you would guess

| model | ROC AUC | PR AUC | Brier |
|---|---|---|---|
| always_benign | 0.500 | 0.568 | 0.255 |
| rules_only | 0.572 | 0.610 | 0.343 |
| forensics_only | 0.749 | 0.770 | 0.211 |
| single_signal | 0.811 | 0.878 | 0.180 |
| semantics_only | 0.829 | 0.902 | 0.169 |
| **fused_logreg** | **0.905** | **0.947** | **0.138** |

Fused beats the best single channel by 0.076 AUC. That is the expected result.
The unexpected one is below.

### The pixel channel catches nothing on its own

| outcome | share of tampered documents |
|---|---|
| both channels see it | 48.1% |
| semantics only | 48.1% |
| **forensics only** | **0.0%** |
| neither | 3.8% |

This is the finding I did not want and am reporting anyway. **Visual forensics
never caught a single document that the semantic checks missed.** Its
contribution is entirely in the ranking, sharpening scores on documents the
semantic channel had already flagged, not in independent discovery.

If you are building this for real, that reorders your priorities: the ledger
reconciliation is the load-bearing component, and the pixel work is a
refinement, not a second line of defence.

### What is catchable, and what is not

At a fixed 10 percent false-alarm budget on clean documents:

| class | n | detection rate |
|---|---|---|
| digit_substitution | 8 | 100% |
| date_modification | 3 | 100% |
| merchant_substitution | 5 | 100% |
| line_item_insertion | 2 | 100% |
| splice | 1 | 100% |
| full_forgery | 3 | 100% |
| **copy_move** | 2 | **0%** |
| **reprint** | 1 | **0%** |
| none (clean) | 19 | 10.5% false alarm |

Two classes are effectively invisible, and for different reasons worth
separating.

**copy_move** changes no value on the document, so the semantic channel is blind
to it by construction, and it is precisely the case where forensics would have
had to carry the load alone. It did not. This is the honest weak point of the
system.

**reprint** is a document that was printed and photographed again with every
value intact. Nothing was edited. Arguably it should not be caught, and a
detector that fired on it would be penalising second-generation copies rather
than fraud. It is labelled positive here so the floor is visible.

### We did not just learn our own renderer

Trained on generators A, B and C. Tested on generator D, held out entirely,
with a different font family, metrics, layout, paper and ink.

| test set | ROC AUC |
|---|---|
| seen (A/B/C) | 0.851 |
| **unseen (D)** | **0.916** |

No collapse. Performance held, slightly improved. This is the experiment that
decides whether any other number here is worth reading.

### The reading stage sets the semantic error floor

Sweeping OCR character error rate with everything else fixed:

| OCR error rate | false alarms on clean | detection on tampered |
|---|---|---|
| 0.000 | 0.0% | 87.5% |
| 0.002 | 33.3% | 87.5% |
| 0.005 | 45.8% | 87.5% |
| 0.010 | 50.0% | 93.8% |
| 0.020 | 70.8% | 93.8% |

Detection barely moves. False alarms explode. **Every point of OCR error buys
you almost nothing in recall and costs you enormously in analyst time.** In a
real deployment this says: spend on the reader before you spend on the detector.

### Which stage actually caused the errors

5 wrong decisions out of 44 (11.4%), attributed by counterfactual repair:

| primary cause | share of errors |
|---|---|
| FORENSICS_FALSE_ALARM | 40% |
| NO_EVIDENCE | 40% |
| SEMANTIC_BLIND | 20% |

Forty percent of errors are the pixel channel firing on authentic documents.
Another 40 percent are the honest floor: not even a perfect reader and a perfect
detector would have flipped them. Only the middle band is addressable by better
modelling.

### Calibration and cost

Isotonic calibration on a held-out split reduced expected calibration error from
0.346 to 0.265. At a resampled 5 percent fraud prevalence, the loss-minimising
operating point reviews 15 percent of cases at 100 percent recall.

The prevalence matters and is stated: at the balanced 50 percent rate the
datasets are built with, reviewing every case is trivially optimal and the whole
threshold analysis would be meaningless.

### Localisation

Saying "this document was edited" is a binary classifier. Saying "**this region**
was edited" is falsifiable, and the tampering engine hands us exact masks to
score against.

| class | n | mean IoU | pointing accuracy |
|---|---|---|---|
| digit_substitution | 18 | 0.196 | 94% |
| merchant_substitution | 16 | 0.184 | 56% |
| date_modification | 19 | 0.160 | 47% |
| line_item_insertion | 14 | 0.138 | 64% |
| copy_move | 8 | 0.124 | 50% |
| splice | 11 | 0.111 | 9% |

IoU is low across the board. Pointing accuracy, whether the single hottest pixel
lands inside the true edit, is strong for digit substitution and near useless
for splice. Localisation is undefined for `reprint` and `full_forgery`, which
have no edited region; they are excluded rather than scored as misses.

### Two results that surprised me

**Edit quality barely matters.** Font matching, colour matching, compression
matching and region re-rendering were varied independently, and detection stayed
between 60 and 70 percent across every setting. I expected a clear gradient. The
absence of one says the forensic channel is not sensitive enough to distinguish
a careful edit from a sloppy one, which is consistent with it catching nothing
independently.

**Blur helps.** Detection rose from 20 percent at zero blur to 80 percent at
sigma 1.6, while JPEG quality behaved as expected, collapsing from 40 percent at
q95 to zero by q75. The blur result is counterintuitive and I have not fully
explained it; the likely mechanism is that blurring the whole image makes a
pasted region's differing sharpness profile stand out rather than hiding it. It
is reported as observed, not as understood.

---

## The pipeline

```
Document Generation  (4 configs: A, B, C, held-out D)
        |
Photograph           perspective, illumination, blur, scale, noise, JPEG
        |
Controlled Tampering  8 classes, exact pixel masks, independent strength axes
        |
Resave               identical for clean and tampered, so compression
        |            history cannot leak the label
Quality Gate         blur, exposure, resolution, skew -> reject or proceed
        |
   +----+--------------------------+
   |                               |
Visual Forensics            OCR + Semantic Checks
6 deterministic detectors   9 closed-world consistency checks
   |                               |
   +----+--------------------------+
        |
Evidence Fusion       model ladder, baselines always reported
        |
Calibration + Abstention   isotonic, ECE, expected-loss threshold
        |
Error Attribution     counterfactual repair, earliest sufficient cause
        |
Verified Evidence Report   LLM narrates, never decides, numbers checked
```

**The photograph stage runs before tampering.** This is not cosmetic. A
fraudster edits a file that has already been captured and compressed. If you
tamper first and compress afterwards, the edited region and the background share
an identical compression history and Error Level Analysis becomes structurally
incapable of detecting anything. I built it the wrong way round first and the
detectors looked broken until I found it.

### Visual forensics

Six deterministic algorithms. None has fitted parameters.

ELA, noise inconsistency, resampling energy, copy-move via word-template
correlation with displacement clustering, glyph consistency, baseline geometry.

**PRNU is deliberately absent.** Camera sensor fingerprinting requires many
images from one known physical device to build a reference pattern. On
synthetically rendered documents there is no such device, so it is not merely
hard to validate, it is inapplicable. Including it would have been the least
defensible thing in the repository.

Copy-move took three attempts. ORB keypoint self-matching fires everywhere on a
receipt because printed text repeats glyph shapes constantly. Exact block hashing
finds nothing because a pasted region rarely lands on the JPEG 8x8 grid, so
recompression gives it different pixels from its own source. What works is
correlating each detected word against the whole image and requiring several
words to share the same displacement vector.

### Semantic checks

Nine checks, all grounded in a registry this system defines: line arithmetic,
subtotal, tax against the jurisdiction rate, total, three payment-ledger
comparisons, near-duplicate submissions, and merchant template consistency.

They read **OCR output, not true values**, deliberately, so the attribution stage
can catch a misread digit masking a real inconsistency or inventing a false one.

### Error attribution

For each wrong decision, repair one stage at a time using that stage's oracle
output and re-run. If repairing stage S alone flips the decision, S is a
**sufficient cause**. Errors usually have several, so primary blame goes to the
**earliest sufficient cause in pipeline order** — fixing an upstream stage
removes the error without any downstream change, so it is the cheapest true fix.

Full method, including the oracle definitions, in [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md).

### The language model

It is a reporter. It receives a finished evidence packet and turns it into prose.
It never sees the image, never influences the risk score, never decides.

Every number in generated text is checked against the evidence packet; if any
number is absent, the text is discarded and a deterministic template is used.
It is off by default (`LEDGER_LLM=1` to enable). **No metric in this repository
depends on it**, which is the point.

---

## What this does not prove

Read [docs/SCOPE.md](docs/SCOPE.md) before believing any number above.

- **The documents are synthetic.** All results are internal validity only. The
  external validity harness (`ledger_forensics/experiments/external.py`) runs the
  same detectors against real receipts you supply, hand-edited. Until you run it,
  transfer to real documents is **unmeasured**.
- **The OCR is simulated.** It reads rendered values then injects character
  errors at a controlled rate. That makes OCR error a dial the attribution study
  can sweep, which is why it was done this way. No claim about real OCR accuracy
  follows. `TesseractOCR` is the swap point.
- **The semantic checks are closed world.** Tax rates, templates and the ledger
  are defined by our registry. Inside the simulation they are ground truth. They
  assert nothing about real merchants or real tax law.
- **Leave-one-attack-out is narrower than it looks.** Only the fusion layer is
  learned; the detectors and checks are deterministic and see no training data.
  E4 tests the combiner's generalisation, not the system's.
- **Prevalence and costs are assumptions**, stated in `results.json` and
  `decision/policy.py`, not measured.

---

## Repository

```
ledger_forensics/
  world/         closed-world registry: merchants, tax rates, ledger
  generation/    document model and 4 generator configs
  tampering/     8 tamper classes, exact masks, independent strength axes
  capture/       photograph and resave stages
  quality/       image quality gate
  extraction/    OCR interface, simulated engine + Tesseract adapter
  forensics/     6 detectors, text-box detection, localisation metrics
  semantics/     9 closed-world consistency checks
  fusion/        model ladder including mandatory baselines
  decision/      calibration, ECE, expected loss, abstention policy
  attribution/   counterfactual repair and blame assignment
  reporting/     evidence packet, template narrator, numeric verifier
  experiments/   11 experiments, staged runner, external validity harness
  api/           FastAPI service
dashboard/       Streamlit dashboard
tests/           34 tests
docs/            SCOPE, EXPERIMENTS, ATTRIBUTION
```

### Tests worth knowing about

Two invariants earn their keep. `test_local_tampers_produce_a_mask_over_changed_pixels`
checks every mask covers at least 75 percent of pixels that actually changed,
because otherwise localisation is graded against a wrong answer.
`test_compression_history_does_not_leak_the_label` checks that clean and tampered
submissions are indistinguishable by file artifacts, because if they were not, a
model could reach a high AUC by reading compression history and every result here
would be worthless.

### Running

```bash
python run.py smoke              # ~3 min, what CI runs
python run.py demo               # ~8 min
python run.py full               # the largest sweep
python run.py demo core          # stages are resumable; dataset is cached
python run.py api                # FastAPI on :8000, docs at /docs
python run.py dashboard          # Streamlit
docker compose up --build        # both services
```

## License

MIT
