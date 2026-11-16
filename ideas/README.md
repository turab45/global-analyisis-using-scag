# SCAG research contribution proposals

**Status:** ideas for discussion, not agreed methods or established findings.  
**Prepared:** 10 November 2026.  
**Purpose:** identify a defensible contribution that addresses a limitation of existing analyses, rather than introduce a metric solely to give it a new name.

**Follow-up:** [Why a joint representation? Added-value experiments and novelty boundaries](README_joint_representation_value.md).

**Manuscript-specific review:** [Contribution gaps, reporting issues and proposed experiments](README_manuscript_review.md).

**Explanation and metric proposals:** [NeuroCartography-inspired analyses and evaluation](README_neurocartography_metrics.md).

## Main recommendation

**Make SCAG explain the different predictive roles of the same concept across communities, and measure whether preserving those communities improves the explanation.**

The central research question would be:

> Does collapsing neurons into a single anatomical concept conceal distinct predictive roles that their functional communities can reveal?

This builds directly on the community analysis already identified as a promising contribution by the supervisor.

## Conference-specific constraint

MIDL 2027's published author instructions explicitly prohibit LLM-generated research ideas, study designs and analyses. These AI-generated proposals cannot be presented as author-originated contributions compliant with that policy. Discuss this with the supervisor and, if necessary, clarify the scope with the chairs before adopting them for MIDL.

Source: [MIDL 2027 author instructions](https://2027.midl.io/author-instructions), checked 10 November 2026.

The published deadlines are **30 November 2026 for abstract registration** and **4 December 2026 for full-paper submission**, Anywhere on Earth. A focused extension is more realistic than several new experimental branches.

Source: [MIDL 2027 important dates](https://2027.midl.io/important-dates), checked 10 November 2026.

## 1. What we can—and cannot—claim as new

Community analysis and semantic graph construction are not broadly new. Particularly relevant is the paper coauthored by Alessandra Mileo that extracts semantic taxonomies from neuron coactivation graphs. Our distinction must be narrower than “we connect semantics to coactivation.”

- [Horta et al., Semantic Interpretability of Convolutional Neural Networks by Taxonomy Extraction, 2023](https://ceur-ws.org/Vol-3432/paper9.pdf)

Likewise, concept completeness already measures how sufficiently concepts explain model predictions. A new completeness-like formula alone would not establish novelty.

- [Yeh et al., On Completeness-aware Concept-Based Explanations in Deep Neural Networks, NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/ecb287ff763c169694f682af52c1f309-Abstract.html)

The opportunity is to establish what **community membership adds to an anatomical concept explanation**, particularly when neurons associated with the same concept play different predictive roles.

| Current measurement | What it establishes | What it leaves unanswered |
|---|---|---|
| IoU/selectivity | Association with an anatomical concept | How that association contributes to predictions |
| Purity/enrichment/SVE | Semantic organisation of communities | Whether preserving that organisation improves explanations |
| Coactivation | Statistical relationships between activations | Whether those neurons support or oppose a particular prediction |
| KL ablation | Magnitude of prediction disruption | Which class evidence was removed, and whether opposing effects cancelled |

These are limitations of the questions the metrics answer, not reasons to discard them.

### Comments

- User:
- Supervisor:
- Related work to check:

## 2. Preferred direction: community-resolved concept explanations

### Problem

Consider a **hypothetical** example:

- MYO-associated neurons in community A support the model's HCM prediction.
- MYO-associated neurons in community B oppose that prediction.
- Reporting one “MYO importance” merges these roles and may conceal both.

SCAG could instead explain:

> The model uses two MYO-associated groups with different contributions to this prediction.

This gives community analysis a purpose beyond describing enrichment.

### Opportunity in the current architecture

For the current **ResNet18 final-stage architecture**, global average pooling and the linear classifier make the logits additive in pooled channel activations.

For a fixed class margin—such as the original predicted class \(y\) against its original runner-up \(r\)—define:

\[
D_{m,c}(x)
=
\sum_{i\in C_m\cap T_c}
\bigl(w_{y,i}-w_{r,i}\bigr)z_i(x).
\]

Here:

- \(C_m\) is a training-derived functional community.
- \(T_c\) is a concept-associated channel set fixed using training data.
- \(z_i(x)\) is the pooled activation of channel \(i\).
- \(w_{y,i}-w_{r,i}\) is its classifier weight for the fixed class margin.

At this layer, \(D_{m,c}(x)\) equals the original margin minus the margin after zeroing those channels. Positive values support the margin; negative values oppose it. The class pair must remain fixed when measuring the intervention.

This exact interpretation depends on the downstream computation being linear. It must not be assumed for arbitrary earlier layers or other architectures.

### Why it is useful

The present KL analysis discards direction. Two groups can have substantial but opposing contributions, producing limited net disruption when ablated together.

A possible supporting diagnostic is:

\[
\operatorname{Cancellation}_c(x)
=
1-
\frac{\left|\sum_mD_{m,c}(x)\right|}
{\sum_m|D_{m,c}(x)|}.
\]

The score is undefined when its denominator is zero.

- Near zero: community contributions largely agree in direction.
- Near one: substantial opposing contributions cancel.

**The defensible novelty would be the community-resolved explanation and its validation, not a claim that this elementary cancellation formula is mathematically unprecedented.**

### Suggested figure

Show signed community contributions for each anatomical concept, linked back to neuron IDs and spatial evidence. This would explain why a concept-level intervention can have a small net effect, complementing the existing influence–IoU scatter.

### Required comparison and failure condition

Compare with size-matched random partitions and other appropriate partition controls. If arbitrary partitions expose the same structure just as well, the coactivation communities have not demonstrated additional value.

### Comments

- Does this answer the intended research question?
- What would count as convincing evidence?
- Potential confounders:
- User:
- Supervisor:

## 3. Evaluation proposal: community resolution gain

### Problem

A partition can look semantically organised without preserving information useful for explaining model behaviour. Enrichment and SVE do not directly measure that usefulness.

### Proposed representation

For patient \(x\), let \(z_i(x)\) be a pooled channel activation. Construct a concept response separately within each community:

\[
h_{m,c}(x)
=
\frac{\sum_{i\in C_m}S_{ic}z_i(x)}
{\sum_{i\in C_m}S_{ic}},
\]

excluding combinations with zero association mass. The semantic associations and communities are fixed from training data.

Compare small, regularised explanatory models that reproduce the original CNN's predictions using:

| Representation | Information retained |
|---|---|
| Semantic-only | One response per concept |
| Community-only | One response per functional community |
| Separate semantic + community summaries | Both summaries, without identifying which concept belongs to which community |
| Community-resolved semantics | One response per community–concept combination |

The third baseline is essential. Otherwise, “joint beats either alone” might merely show that supplying more features helps.

### Candidate evaluation score

Let \(L\) be held-out prediction-reconstruction error, such as mean KL divergence from the original model to the explanatory model. Define:

\[
G_{\mathrm{community}}
=
L_{\mathrm{separate}}
-
L_{\mathrm{community\text{-}resolved}}.
\]

Positive values mean that preserving **where concepts occur in the community structure** improves prediction reconstruction.

This is related to existing concept completeness. The contribution would be **a community-resolved completeness method and controlled evaluation**, not inventing completeness itself.

### Required controls

- Size-matched random partitions and shuffled semantic profiles.
- Comparable explanatory-model capacity and regularisation.
- Performance versus representation dimension: more communities create more features.
- A consistent treatment of poorly annotated channels.
- Evaluation on patients not used to fit the explanatory models.

The explanatory model should also predict responses to prespecified neuron interventions, not only intact predictions. This would test whether the compressed explanation captures behaviour beyond observational fit.

### Questions to resolve before implementation

- What is the smallest appropriate explanatory model?
- How should model capacity be controlled across representations?
- How should unannotated or weakly associated channels be represented?
- Which interventions should be fixed before evaluation?
- How should the limited patient count constrain fitting and validation?

### Comments

- User:
- Supervisor:
- Closest related methods:
- Reasons to accept or reject this direction:

## 4. Complementary validation: anatomy beyond simple anatomy size

### Problem

Neurons may coactivate because larger anatomical regions activate more channels, rather than because the graph captures richer organisation.

Segmentation masks and activation-coverage diagnostics could support the question:

> After accounting for concept area, activation magnitude and relevant acquisition variables, which concept-associated community relationships remain reproducible?

This would distinguish:

- Relationships explained by shared anatomical burden.
- Relationships persisting beyond those measured factors.

### Potential value and limitation

This could strengthen the medical relevance of SCAG. However, residualisation and partial correlation are established techniques. Use this as a validation component rather than market it as a newly invented metric. It also cannot remove unmeasured confounding.

### Comments

- Available adjustment variables:
- Risks of removing biologically meaningful variation:
- User:
- Supervisor:

## 5. Recommended combination

Combine **community-resolved concept explanations as the method** with **community resolution gain as its evaluation**:

> SCAG identifies community-specific roles of anatomical concepts and tests whether those roles preserve predictive information lost by concept-level summaries.

This is a more focused contribution than another general semantic-coherence score.

### First feasibility experiment

1. Use the existing ResNet18 final-stage communities and validated concept associations.
2. Measure signed concept contributions within each community.
3. Determine whether substantial, reproducible opposing roles exist.
4. Compare their organisation with matched random partitions.
5. Only then build the community-resolution evaluation.

### Controlled demonstration of the gap

Construct two models with the same activations, semantic associations and graph, but different classifier weights. Their purity, enrichment and SVE remain identical, while their concept-associated predictive roles differ.

This demonstrates why the existing structural measurements cannot answer the proposed behavioural question. It does not by itself prove that SCAG communities outperform alternative partitions; that requires the empirical controls above.

### Architectural caution about synergy

Avoid proposing “synergy” solely from non-additive KL effects at the current ResNet final stage. Its logits are additive there; non-additivity in probabilities or KL can arise from the output transformation itself. Earlier layers require a separate intervention analysis.

## 6. Relationship to the current results

The current results motivate these questions but do not establish them:

- Joint selection has stronger concept associations than functional-only selection in the observed comparison, while functional-only selection produces greater prediction disruption.
- Joint selection produces greater KL than semantic-only selection at the 10% budget in the current run.
- The saved bootstrap intervals do **not** clearly establish LV joint selection's IoU advantage over semantic-only selection, despite its higher point estimate.
- Semantic coverage differences and failed matching controls remain limitations.

The proposals should not be selected or tuned simply to make joint selection win.

Reference run: `scag_outputs/research/acdc/anatomy/20261109T154512_822872Z/`.

## 7. Decision and feedback

| Direction | Decision: pursue / revise / reject | Reason |
|---|---|---|
| Community-resolved concept explanations | | |
| Cancellation diagnostic | | |
| Community resolution gain | | |
| Anatomy-size conditional validation | | |

### Agreed research question

_To be completed after discussion._

### Agreed novelty claim

_To be completed after checking the closest related work._

### Minimum experiments needed

_To be completed after agreeing on the question and contribution._

### Open comments

- User:
- Supervisor:
- Follow-up discussion:

---

These proposals came from a targeted literature check, not an exhaustive novelty assessment. No proposed experiment has been implemented or validated by this document. None of these directions guarantees acceptance or scientific novelty.
