# Pilot Evaluation Plan (proposed; nothing here has been run)

The dataset contains purchases and one satisfaction score per buyer. It contains **no campaign contact, response,
conversion, retention or time-series satisfaction data**, so no claim about business impact can be made from it.
This plan describes how to obtain valid evidence.

## Question
Do segment- and rule-based customer-experience actions improve outcomes versus the current approach, without
disadvantaging any group?

## Baseline
The solution being replaced (existing segmentation or generic communication), run in parallel on the same buyers.

## Design
- Randomise **individual buyers** (stratified by current segment and by purchase count) to: (A) baseline treatment, (B) this system's recommendations.
- Fixed duration set in advance (e.g. 90 days), the same contact-frequency cap in both arms.
- Pre-register outcomes and the analysis before launch.
- Human review required for all `OBS-LOW-SAT` and `AMBIG-ASSIGN` actions; opt-outs honoured; each recommendation must be explainable to the client.

## Outcomes (pre-registered)
- **Primary (choose one):** follow-on enquiry or purchase within the window, or a post-interaction satisfaction survey score.
- **Secondary:** response rate, complaints, opt-outs/unsubscribes, number of contacts per buyer, time to resolve low-satisfaction cases.
- **Guardrails:** no worsening of opt-outs or complaints; outcomes not materially worse for any gender, country group or age band (subgroup comparison, reported with intervals).

## Sample size (illustrative arithmetic, assumed baseline)
With 1,000 buyers per arm and an assumed 20% baseline event rate, 80% power and alpha 0.05, the minimum
detectable difference is about 5 percentage points (SE = sqrt(2 x 0.2 x 0.8 / 1000) = 0.018; MDE = 2.8 x SE).
The real baseline rate must be measured first. The 114-buyer segment is too small for a reliable segment-level
effect estimate; analyse it only descriptively or extend the pilot.

## Analysis
Intention-to-treat difference with confidence intervals; adjust for stratification factors; report segment-level
effects as exploratory. Declare in advance what result would stop the rollout.

## Data to log
Contact timestamps and channel, recommendation ID shown (`OBS-*`, `SEG-*`, `AMBIG-ASSIGN`), response, conversion, survey scores, complaints, opt-outs.

## What would count as success
A pre-specified improvement on the primary outcome with guardrails intact. A null result is informative: it would mean
the segments are descriptively useful but do not improve outcomes beyond the baseline.
