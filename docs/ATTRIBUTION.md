# Error attribution: the method

Most pipelines report one end-to-end error rate. When a decision is wrong, that
number cannot tell you whether the reading stage misread a digit, the pixel
forensics missed an edit, the semantic checks were structurally blind, or the
fusion layer weighed correct evidence badly. Teams then improve whichever stage
is most fashionable rather than whichever stage is at fault.

This repository has ground truth at every stage, which almost no real pipeline
has. That makes a stronger question answerable.

## Counterfactual repair

For each wrong decision, repair one stage at a time by substituting that stage's
oracle output while holding everything else fixed, then re-run the decision.

> If repairing stage S **alone** flips the decision to correct, S is a
> **sufficient cause** of that error.

## The multi-cause rule

Errors often have several sufficient causes, so "which stage caused it" is
ill-posed without a tie-break. We record every sufficient cause and assign
**primary blame to the earliest one in pipeline order**.

The rationale is economic rather than philosophical: fixing an upstream stage
removes the error without requiring any downstream change, so the earliest
sufficient cause is the cheapest true fix.

Pipeline order used for the tie-break:

```
QUALITY_FALSE_REJECT -> OCR_ERROR -> FORENSICS_MISS / FORENSICS_FALSE_ALARM
  -> SEMANTIC_BLIND -> FUSION_ERROR -> NO_EVIDENCE
```

## The oracle definitions

These are the arguable part, so they are stated plainly rather than buried.

| Stage | Oracle |
|---|---|
| OCR | A perfect reader. Semantic checks re-run on error-free values. Computed at build time and stored on every submission. |
| Forensics | An ideal local-edit detector. Feature values set to the training distribution's 95th percentile when the document really has an edited region, 5th percentile when it does not. |
| Quality | The gate's decision inverted. |

## The two residual categories

**FUSION_ERROR** means no single repair flips the decision but repairing two
together does. Both channels supplied usable evidence and the combiner still got
it wrong. That is a modelling problem, not a sensing problem.

**NO_EVIDENCE** means not even a full repair flips it. This is the honest floor
of the system: the attack left nothing for either channel to find. Reporting a
high share here is not a failure of the study, it is the study working.

## SEMANTIC_BLIND is not a bug

Some attacks change no value that arithmetic or the ledger can see. A copied
block duplicates pixels and alters no number. Classifying those as
`SEMANTIC_BLIND` rather than as a semantic miss keeps the distinction between
"the check failed" and "there was nothing for the check to detect".
