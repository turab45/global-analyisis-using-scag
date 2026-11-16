# SCAG manuscript review and contribution options

**Reviewed:** `SCAG__Connecting_Visual_Concepts_to_Functional_Organisation__Copy_ (3).pdf`, 13 pages, internal title “SCAG: Towards Understanding Deep Representations using Semantic Coactivation Graphs.”

**Status:** discussion proposals, not implemented experiments, validated findings or established novelty. Read alongside [the broader joint-representation discussion](README_joint_representation_value.md). The PDF uses the NeSy 2026 template; venue plans and submission status are not inferred from that header.

## 1. Overall assessment

The manuscript has a coherent empirical foundation: ten architectures, two datasets, semantic–functional alignment, community composition and interventions. It also appropriately distinguishes prediction changes from accuracy loss and acknowledges that concept dominance is not enrichment.

Its main contribution gap is specific: **the strongest intervention result does not test the contribution of semantics.** Communities are selected from functional coactivation; ablating those same communities would produce the same effects if their semantic annotations were erased or shuffled. Table 2 therefore supports functional community importance relative to size-matched random groups. It does not establish added intervention value from the joint representation.

Table 1 establishes correspondence between two representations, not the utility of combining them. Figure 2 uses the combination descriptively, but does not yet show that its semantic organisation is reliable or enables an independently useful operation. Figure 3 provides more detail about Table 2, rather than a new test of integration.

The missing result is an operation whose answer depends on **which concepts occupy which functional relationships**, followed by validation using independent evidence. This need not be a community operation or a new scalar metric.

## 2. Manuscript issues to resolve before expanding experiments

| Location | Finding | Action and relevance |
|---|---|---|
| Abstract; Table 1 | 0.46 and 0.29 are the highest reported joint-vocabulary test alignments, not overall correlations across all models | Identify them as maxima or report a clearly defined distribution; avoid implying a pooled result |
| Section 3.1 | Layer names, pooling choice, correlation estimator for R, activation calibration, IoU aggregation and zero-profile handling are underspecified | Provide exact definitions, training/calibration split, channel eligibility and per-model hook locations |
| Section 3.2 | Threshold includes a literal TODO; paper uses strict >0.30 while current adjacency uses >= | Document the actual executed convention, graph density, isolates, Louvain seed/resolution and sensitivity |
| Section 4.1 | Both datasets are described as central short-axis | Current `SCAGConfig.view` and M&Ms-2 loader use `lax_4c`; determine which pipeline generated the manuscript and correct the description. Do not attribute all between-dataset differences to scanners |
| Section 4.1 | M&Ms-2 counts are 156/28/108; configuration expects 160/30/110 | Verify retained patient manifests and exclusions. Configured counts need not equal usable counts; explain the difference rather than replacing values blindly |
| Section 3.1; Table 1 | Old and new semantic pipelines implement different vocabulary aggregation | The legacy `make_semantic_views` sums anatomy–phase IoUs; current `concept_profiles` computes direct IoU from masks. A sum of IoUs is generally not an IoU. Trace each table to its script, config and artifacts before mixing results |
| Table 1 | “Test” does not specify which semantic profiles were used | Legacy alignment uses training-derived semantics with evaluation coactivation. Distinguish this from independently recomputing semantics on held-out patients |
| Figure 2 | Only labelled neurons enter the percentage denominator; one MobileNetV3-L entry says composition not exported | Report labelled/total counts, unlabelled share, labelling threshold, dominant-label rule and enrichment. Resolve missing export or label it clearly |
| Section 4.3 | Refers to quantitative semantic-organisation measures without presenting numerical enrichment/SVE results here | Include the actual estimates, controls and uncertainty if making that claim |
| Table 2; Figure 3 | Validation/test pooled, size-weighted community averages, no uncertainty | Separate cohorts, identify graph selection split, show paired patient intervals and control variability; explain weighting and selection of tested communities |
| Figure 1 | Shows accuracy 86.5% to 57.3%, while reported interventions emphasise KL/flip | Cite its experimental provenance and conditions or explicitly mark the values illustrative. A flip rate is not an accuracy decrease |
| Experimental setup | Original classifier performance and checkpoint selection are missing | Report held-out performance and training/selection provenance so interpretation is grounded in the quality of each classifier |

These are findings about the uploaded draft and selected current code paths, not proof that the historical runs used today's code. The newer research run separates validation/test and adds controls; it cannot retroactively validate the older tables. The newer RESULTS.md also explicitly acknowledges prior test inspection.

## 3. Closest prior work: what the distinction must be

The targeted literature check makes a broad “first graph connecting concepts and neurons” claim difficult to defend:

- [Horta et al., 2021](https://www.sciencedirect.com/science/article/pii/S0167739X21000613) already studies coactivation graphs, communities and central neurons.
- [Horta et al., 2023, taxonomy extraction](https://ceur-ws.org/Vol-3432/paper9.pdf) combines coactivation structure with semantic class hierarchies and evaluates the resulting taxonomy. Its semantic objects are output classes and their hierarchy; SCAG associates hidden channels with spatially annotated visual concepts. This is a concrete distinction, not proof of novelty by itself.
- [Ferreira dos Santos and Mileo, concepts-to-relations extraction](https://ceur-ws.org/Vol-3910/aics2024_p69.pdf) uses extracted concepts, an external commonsense knowledge graph and relation validation. SCAG could distinguish directly measured neural coactivation relationships from external semantic relationships retrieved for detected concepts. Coactivation must not be relabelled as an anatomical relation.
- [SemanticLens](https://arxiv.org/abs/2501.05398) demonstrates operations enabled by semantic component representations, including search, comparison and auditing. SCAG needs its own demonstrated operation beyond annotated visualisation.
- [NeuroCartography](https://arxiv.org/abs/2108.12931) already summarises neuron concepts and their associations in visual graph views. A new annotated graph figure alone is insufficient differentiation.

The 2023 taxonomy paper, concepts-to-relations paper and SemanticLens are absent from the uploaded bibliography and merit explicit comparison. This is a targeted review, not an exhaustive novelty search.

## 4. Recommended main addition: validate the relations the joint graph exposes

### Research question

> Does the joint representation recover reproducible relationships between concept-associated neural groups that semantic association or functional structure alone cannot identify with the same reliability?

The distinction is between identifying a relation that requires both information sources and demonstrating that a retrieval procedure finds reliable examples. Neither implies new information beyond complete aligned A and S.

### Proposed operation

For a prespecified concept pair, retrieve supporting neuron pairs or small subgraphs. Retain both endpoint concept profiles, training coactivation, model/layer, thresholds and cohort. Do not replace the supporting graph with only a concept-level edge.

For example, a result could say “these MYO-associated channels coactivate with these LV-associated channels.” It should include which parts of the claim hold on held-out patients. It cannot say “MYO causes LV” or “the network uses ventricular interaction” on that evidence alone.

### Validation and comparison

1. Define endpoint association requirements, retrieval budgets and graph thresholds on development patients.
2. Select the same number of candidate neuron pairs for each strategy on the same candidate universe: semantic-only, functional-only, joint, and appropriate random/matched controls. The functional-only strategy receives no concept labels during selection. Its disadvantage on a concept-specific task is expected; report it as task utility, not universal superiority.
3. Independently recompute endpoint concept associations and pair coactivation on held-out patients using frozen activation calibration. Separate failure of semantics from failure of functional replication.
4. Use whole-profile shuffling to test correct correspondence, and graph controls to test topology. Clearly separate degree-preserving from strength-preserving randomisations; the existing uniform edge-position null preserves neither degree nor strength.
5. Report validation rates and their semantic/functional components, patient-bootstrap uncertainty, coverage, relation weights and sensitivity. Pairs sharing neurons/patients are dependent; do not count all pairs as independent samples.
6. Account for anatomy area, general activation magnitude and cohort composition in sensitivity analyses. Conditional residual correlations answer a different question from marginal correlations and need their own interpretation.

A prespecified validation rate is a useful performance measure, but is not itself a new metric contribution. With only three anatomical concepts, there are only three distinct between-concept categories; a correlation over those three values cannot support a broad generalisation claim.

### What would improve the manuscript?

A figure that shows one fully auditable relation, alongside quantitative results over all prespecified queries. The new contribution is the extraction-and-validation operation with evidence, not W = S-transpose A S in isolation.

Existing code provides concept projections, subgraph retrieval and some nulls. What remains is a fair retrieval comparison, independent validation of both parts of each retained claim, and a coherent reporting protocol. Verify the existing outputs before implementing duplicates.

## 5. Stronger, more ambitious addition: do graph relations identify intervention interactions?

### Why this addresses a real limitation

Current KL/flip ablation evaluates removal of groups but cannot distinguish independent contributions from non-additive joint effects. Coactivation alone cannot establish that two representations interact computationally. The joint graph makes concept-associated candidate interactions identifiable; interventions can then test them.

### Define a signed intervention outcome

For each patient x, fix a class y and comparator r before intervention and define the original logit margin m(x) = logit_y(x) - logit_r(x). Let

\[
D_x(U) = m(x) - m(x; U\text{ ablated}).
\]

For **disjoint** channel groups U and V at the same layer, measure

\[
I_x(U,V) = D_x(U\cup V) - D_x(U) - D_x(V).
\]

Positive I means the union causes a larger signed margin decrease than the sum of separate decreases; negative I means smaller. Interpret the sign alongside the individual effects. Do not automatically name these synergy and redundancy, or infer physiological mechanisms.

This finite-difference interaction is established mathematics, not a new formula. The research opportunity is whether concept-associated coactivation can locate reproducible non-additive intervention effects beyond its constituent evidence sources.

### Critical architecture constraint

For the current ResNet final-stage output followed only by average pooling and a linear classifier, disjoint channel ablations are additive in logits: I = 0 up to numerical precision. At that location this is a sanity check, not a meaningful synergy search. The stacked ResNet model in the current code has this structure; always verify the actual hook and checkpoint.

Use earlier stages with nonlinear downstream computation for the substantive experiment. Do not use KL non-additivity as evidence of neural interaction: the softmax/divergence calculation itself can generate it even when logits are additive. Do not assume all architectures have the same downstream head.

### Controls

- Compare connected and weakly connected groups matched on endpoint semantic profiles, group size and activation statistics. Check degree/strength as additional confounders; preserve failed matching as a limitation.
- Compare concept-guided versus unguided functional selection and correct versus shuffled concept assignment.
- Use development data for selection, evaluation patients for effects, and prespecify a small set of concept-pair queries and budgets. Do not select the largest evaluation interactions for inference.
- Report signed effects, their distribution and uncertainty; a near-zero mean can hide heterogeneous effects. If testing absolute magnitude, predefine the estimand and compare controls because absolute values have a positive noise baseline.
- Treat artificial hidden-state ablation as an intervention on the model, not an intervention on real anatomy. Consider zero versus development-mean replacement as sensitivity if the initial result warrants it.

### What constitutes a contribution?

Evidence that graph-linked concept-associated groups identify reproducible interaction effects more effectively than semantic-only and functional-only selection would connect descriptive relations to model behaviour. A null result is also informative: the graph describes statistical organisation without establishing computational interaction.

This is a higher-risk, higher-compute extension. It should follow semantic validation and a small earlier-layer pilot, rather than a full ten-model sweep.

## 6. An alternative to interaction experiments: test the value of semantic neighbourhood context

### Missing information in the current descriptors

The neuron's own semantic profile omits its neighbours. Functional centrality omits their semantic content. A simple descriptor combining them is

\[
H_{i,c}=\frac{\sum_j A_{ij}S_{jc}}{\sum_j A_{ij}},\qquad \sum_j A_{ij}>0.
\]

H describes the concept-association strengths of a neuron's coactivating neighbours. It is not a probability distribution over concepts and must not be treated as a replacement for that neuron's own concept profile. Isolates need an explicit missing-context indicator. Retain underlying edges because averaging can hide distinct neighbourhoods.

This is standard weighted neighbourhood aggregation, not novel mathematics. It makes a concrete broader graph question possible: do neurons with similar own-concept profiles have different functional roles depending on their semantic neighbourhood?

### Evaluate with a target not defined by H

Use held-out signed intervention responses as targets. Compare small explanatory predictors using semantic descriptors, functional descriptors, their concatenated summaries, and summaries augmented with H. Include activation magnitude, identical tuning opportunities and matched-capacity controls. A same-dimensional shuffled-context control helps detect gains caused merely by extra predictors.

An incremental measure is

\[
G = L_{\mathrm{separate\ summaries}}-L_{\mathrm{joint\ context}},
\]

where L is a prespecified held-out error, such as mean squared error for signed margin effects. Also compare each individual representation. G is an ordinary difference in predictive loss; its value is operational clarity, not mathematical novelty.

Fit explanatory models only on development interventions. Use held-out patients, and separately hold out intervention sets if claiming generalisation to new selections. Report those two generalisation questions separately. Since channels share the same trained model, uncertainty over patients alone does not establish across-model generality.

For a final linear head, full activations and classifier weights can determine the effect exactly. Include that as a reference when applicable: the objective is a useful constrained semantic explanation, not beating the full model's known computation. Complete aligned A and S also contain H, so “separate summaries” must never be described as all information available from both inputs.

This is a more manageable alternative to Section 5 if model-interaction work is too expensive, but it introduces an explanatory predictor and requires careful controls. Do not add both branches by default.

## 7. Supporting improvements that strengthen the claim

- **Semantic reliability first:** low IoU and split-dependent labelled coverage can undermine every graph interpretation. Use patient-level mask/activation coverage, overlap controls, representative overlays and vocabulary-dependent selectivity; keep unlabeled neurons visible.
- **Enrichment and SVE:** retain these as supporting evidence of community organisation. Separate a fixed training partition's held-out semantic organisation from communities reconstructed on evaluation data.
- **Cross-layer evaluation:** ask where validated relations and explanatory utility appear, with layer-specific spatial resolution, receptive field, channel count and graph density reported. A larger IoU at a higher-resolution layer is not by itself more interpretable computation.
- **Robustness:** compare correlation thresholds and density-matched graphs, patient resampling and a limited number of independently trained seeds. More architectures do not substitute for training-seed uncertainty.
- **Typed evidence:** preserve neuron IDs, masks, model/checkpoint/layer, cohort and statistical uncertainty. An association edge, intervention effect and external ontology assertion must remain different relation types.
- **Scope of generalisation:** independently trained ACDC and M&Ms-2 models provide replication across settings. They do not automatically provide external validation of a frozen classifier, especially when views and labels differ.

## 8. A focused revision package

1. Reconcile the manuscript with exact run provenance and repair semantic/data descriptions.
2. Add the held-out relation extraction experiment from Section 4 as the main integration result.
3. Choose **one** behavioural test: earlier-layer interaction retrieval (Section 5) or context-based intervention-response explanation (Section 6). Continue the already available fixed-budget joint/semantic/functional comparison as a simpler fallback.
4. Present enrichment/SVE, traceability and semantic reliability as supporting analyses.
5. Replicate the chosen primary result on a second model/setting after the pilot is interpretable. Treat previously inspected test patients as exploratory; use genuinely independent evidence for a confirmatory claim where available.

Suggested main-paper space allocation:

| Item | Purpose |
|---|---|
| Revised Figure 1 | Actual representation, the new operation and its validation; remove unsupported accuracy illustration |
| New Figure 2 | One auditable concept-associated subgraph plus held-out semantic and functional evidence |
| Main quantitative table | Semantic-only, functional-only, joint, correspondence-shuffled and relevant matched-control results at equal budgets |
| New behavioural figure | Either interaction validation or incremental response-prediction utility, with uncertainty |
| Compact supporting panel/appendix | Community enrichment, SVE, original alignment, coverage and sensitivity |

The present ranked-community-KL plot is a reasonable appendix candidate. The broad architecture alignment table can also be compressed to make space for the direct added-value experiment.

## 9. Claim boundaries and feedback

The current evidence supports “semantic similarity and coactivation are associated, and many functionally selected communities affect predictions.” The proposed experiments could support “the joint representation extracts reliable, traceable concept-associated functional relationships and improves a specified explanatory task.” They cannot establish complete understanding, physiological relations or automatic causal reasoning.

The novelty claim should identify a concrete operation and its independently validated benefit. It should not rest on graph annotation, a renamed classical statistic, or the number of architectures.

**User comments:**

**Supervisor comments:**

**Preferred primary experiment:**

**What would count as a negative result:**

**Related methods still requiring a direct comparison:**

### Venue note

The [MIDL 2027 author policy](https://2027.midl.io/author-instructions) currently prohibits LLM-generated research ideas, study designs, analyses and substantive text. These are AI-generated discussion proposals and cannot be represented as compliant author-originated contributions. This review does not resolve that venue constraint.

### Review limits

Read the full extracted manuscript text and visually inspected Figures 1–2. Inspected selected current code paths and the newer run's reporting notes. Did not reproduce historical tables, audit every implementation, inspect all original checkpoints, run interventions, or establish exhaustive novelty. No experimental code or results were changed.
