# Why a joint representation? Demonstrating the added value of SCAG

**Status:** proposals for discussion; not established findings or agreed experiments.  
**Prepared:** 10 November 2026.  
**Companion:** [Earlier contribution proposals](README.md).

## Revised scope: the value of the entire joint graph

The central question is **what integrating semantic associations with functional relationships enables**, regardless of whether communities are used. The community experiments below are optional instances of this broader question, not the definition of the contribution.

### Reading Mileo's paper

[Towards a neuro-symbolic cycle for human-centered explainability](https://journals.sagepub.com/doi/pdf/10.3233/NAI-240740) is a position paper. Section 3, objective O1, motivates neural-cognitive mapping; Section 4.1 connects graph analysis, semantic concepts and rule extraction. Section 4, H2, stresses accessible links between concepts and individual units. These motivate a broader representation than community summaries. They do not experimentally establish SCAG's effectiveness. The paper also cites earlier coactivation graphs and semantic local explanations, so combining graphs and semantics alone is not a defensible novelty claim.

The proposals below are our interpretation and possible operationalisation, not methods or findings demonstrated by that paper.

### What each representation makes available

| Representation | Direct evidence | Missing evidence |
|---|---|---|
| Semantic S | Which neurons are associated with which annotated concepts, and similarity between concept profiles | Whether different neurons actually coactivate, and the topology of those relationships |
| Functional A | Which neurons coactivate, with weights and graph structure | What annotated concepts those neurons represent |
| Aligned joint graph | Which concept-associated neurons have which functional relationships, with the contributing neurons and measurements retained | Prediction relevance, causality and clinical validity still require additional evidence |

**The added object of analysis is semantically identified functional structure.** It can include individual edges, neighbourhoods, paths, motifs, communities and whole-graph organisation. A concept list omits functional organisation; a coactivation graph omits concept identities. Joint queries use both.

For example: “Which MYO-associated neurons are directly connected to LV-associated neurons, how reproducible are those links, and which exact channels and images support the claim?”

The final question about images requires sample-level evidence beyond aggregate A and S. Likewise, prediction effects require actual model evaluation. The graph organises access to evidence; it does not manufacture missing evidence.

### A small argument for why neither input alone suffices

Take four neurons with fixed one-hot semantic labels: neurons 1 and 2 are LV, neurons 3 and 4 are MYO.

- Graph A1 contains undirected unit-weight edges (1,2) and (3,4): only within-concept connections.
- Graph A2 contains edges (1,3) and (2,4): only between-concept connections.

The semantic matrix is identical, and both graphs have the same degree sequence and unlabelled topology. Yet the answer to “are LV-associated neurons connected to MYO-associated neurons?” changes. S alone cannot determine it.

Conversely, keep A1 fixed and change labels to LV on neurons 1 and 3 and MYO on neurons 2 and 4. The functional graph is unchanged but the semantic interpretation of its edges changes. A alone cannot determine the answer either.

This demonstrates an information requirement for this relational question. It is not a new theorem, a biological discovery, or evidence that real CNN relationships are reliable. Complete aligned A and S remain equivalent in information to the joint graph; a graph file is not essential to perform the join.

### What operations should demonstrate this capability?

1. **Relational retrieval:** retrieve concept-associated neuron connections and neighbourhoods with auditable evidence paths. This extends concept retrieval from “find MYO neurons” to “find MYO-associated representations in a specified functional context.”
2. **Functional context of a concept:** distinguish neurons with similar concept profiles but different neighbours, or neurons with different concept profiles that coactivate. Communities are one possible summary, not a prerequisite.
3. **Comparison of relational organisation:** identify differences across layers, architectures or cohorts even when concept prevalence and individual association strengths are similar. Compare aligned concept categories rather than assuming neuron correspondence across models.
4. **Targeted intervention:** select neurons using both their concept association and relational position, then test the resulting prediction effects. A high coactivation edge is not itself evidence of computational interaction.
5. **Auditable statements and later reasoning:** preserve typed facts such as associated_with(neuron, concept) and coactivates_with(neuron, neuron), their weights, calibration, population and provenance. A query can derive a supported connection between concept-associated groups. It cannot derive anatomical causation or an ontology relation from correlation alone.

Multi-hop reachability is a graph property. In particular, A--B and B--C correlations do not imply A--C correlation or a causal pathway. Within-layer graphs also do not establish propagation across layers.

### How to show that this is useful, rather than merely possible

There are two separate claims to evaluate:

- **Representational capability:** a correctly implemented joint query returns information that requires both inputs. The example above and auditable outputs establish this limited claim.
- **Empirical utility:** the returned relationships are reliable and enable a useful task on independent evidence. This requires experiments, not simply counting returned edges or computing S-transpose A S.

Recommended first experiment: **extract and validate concept-associated functional relationships**, with individual neuron-pair provenance.

1. Freeze semantic calibration, graph construction, retrieval thresholds and a small family of queries on development data. Use all prespecified queries, not only attractive LV–MYO examples.
2. Extract candidate relationships and retain the exact neuron sets and edges supporting each claim. Distinguish a group-level aggregate claim from a claim about particular neuron pairs.
3. On held-out patients, independently evaluate the selected endpoints' concept associations and the selected pairs' coactivation. Use patient-level uncertainty, appropriate multiple-testing correction and controls for activation magnitude, anatomy size and relevant cohort composition.
4. Compare actual alignment with complete-profile semantic shuffling, degree-stratified shuffling where feasible, and suitable graph randomisation. Report exactly which graph properties each null preserves. Degree, strength and edge-weight distribution are different constraints.
5. To evaluate the added value of topology, let semantic-only selection nominate the same number of concept-associated pairs using semantic evidence; test whether graph-informed selection yields more independently supported relationships at comparable semantic strength. To evaluate the added value of semantics, use functional-only selection on the same candidate universe, keeping concept labels hidden until evaluation. Report the dependence of this result on the chosen concept-conditioned task.
6. Match selection budgets and relevant confounders. If matching is infeasible, report that limitation. Shuffling alignment is useful because it tests whether merely attaching labels suffices.

A transparent summary is the proportion of selected relational claims whose semantic and functional components both satisfy prespecified held-out validation criteria. Report the separate components and confidence intervals too; power and thresholds affect this proportion. This is a conventional validation rate, **not automatically a novel metric**. Use held-out confirmation language, not biological ground truth. With only three anatomy concepts, use patient and neuron-level evidence carefully; edges sharing patients and neurons are not independent replicates.

This experiment could demonstrate reliable relational knowledge extraction. It does not yet demonstrate that the relationships explain predictions.

For that stronger claim, add one downstream task: at fixed channel budget and adequate concept association, compare prediction effects of graph-informed selection against semantic-only, functional-only, random and matched controls. Alternatively, predict held-out intervention responses with equally constrained explanatory models. Joint selection need not maximise KL to be useful; it must improve a prespecified task or a meaningful semantic-association/influence trade-off.

### Where a contribution could lie

The candidate contribution is **an evaluated method for extracting, validating and using traceable semantic–functional relationships from CNNs**. The problem addressed is that neuron-wise concept scores omit measured relationships, while unannotated functional structure cannot identify their semantic content.

Novelty would need to reside in a specific extraction/validation algorithm, a useful reasoning operation, or a rigorous reusable evaluation protocol that advances existing graph-based explanation work. A graph schema, a product of existing scores, a new acronym, or the general act of merging A and S does not establish novelty.

The immediate recommendation is therefore to select a relational question, specify what evidence makes its answer trustworthy, and test whether answering it helps an independent explanatory task. Community analysis supports this programme alongside other graph operations.

**Possible reviewer response, conditional on validation:** “The joint representation links concept identity to measured neural relationships. It enables traceable queries about the functional organisation of concept-associated representations, which neither neuron–concept associations nor an unannotated coactivation graph determines alone. We evaluate the reliability of those extracted relationships and their utility for [the validated task], rather than assuming that integration guarantees better explanations.”

## 1. Earlier community-focused proposal: one possible experiment

> Why combine semantic associations and functional coactivation? What becomes possible, and what measurably improves, compared with using either representation separately?

One possible sub-question is **the added explanatory value of preserving concept–community correspondence**. This addresses one operation enabled by the joint graph; it does not exhaust the representation's purpose.

A possible hypothesis is:

> At a fixed explanation budget, retaining which concepts occur in which functional communities preserves predictive roles and intervention responses better than semantic-only, functional-only, or separate summary representations.

This is a hypothesis to test. The present results do not establish it.

## 2. A crucial distinction: integration does not manufacture information

Let A be the functional adjacency and S the neuron–concept association matrix. SCAG is a deterministic construction from A, S and their shared neuron identities.

- Compared with A alone, S supplies anatomical associations.
- Compared with S alone, A supplies statistical organisation not explicitly represented in S.
- Compared with **both complete, correctly aligned A and S**, SCAG contains no additional information simply because it is packaged as a joint graph.

A reviewer is correct to ask whether the same calculation could be performed from two separate files. If they retain the same identities, the answer is yes. Our claim must concern an explicit representation, analysis method, compression or evaluation benefit—not an information-theoretic advantage over its complete inputs.

Distinguish three levels:

| Level of claim | What would establish it? | What would not establish it? |
|---|---|---|
| Capability | A traceable operation requiring semantic and functional evidence | Claiming that an anatomical query is impossible for a baseline deliberately denied all concept information |
| Structural added value | Actual topology gives reproducible concept-associated structure beyond semantic co-association and appropriate controls | Merely computing W = SᵀAS |
| Practical explanatory value | Better held-out explanation quality at comparable complexity and information access | More features, larger interventions or a larger mean KL alone |

The strongest paper should show all three, with the third doing most of the argumentative work.

## 3. Define the comparison fairly

Use the same patients, model, layer, training-derived calibration, evaluation targets and intervention budgets.

| Comparator | Information/representation | Purpose |
|---|---|---|
| Semantic-only | Concept profiles and concept-weighted activation summaries | Tests the contribution of functional organisation |
| Functional-only | Functional communities/centrality and activation summaries | Tests the contribution of anatomical interpretation |
| Separate summaries | Concatenated concept summaries and community summaries | Tests whether retaining their correspondence helps beyond having both marginal summaries |
| Joint community–concept summaries | Concept responses resolved within functional communities | Proposed compressed explanation |
| Random-partition joint summaries | Same semantic profiles, with size-matched random communities | Tests whether an arbitrary subdivision would work equally well |
| Shuffled-semantics joint summaries | Actual graph, with complete semantic profiles reassigned to neurons | Tests dependence on correct semantic–functional correspondence |
| Full aligned inputs / full-channel readout | Original activation features and available metadata | Reference for information loss; not a semantically compressed explanation |

**Critical:** the separate-summary comparator loses correspondence through compression by design. Be explicit about that. It is not equivalent to giving a method complete A and S with neuron identities.

If a functional-only method needs concept labels to answer a concept-specific task, report that added information. Do not silently grant it semantic access and continue calling it functional-only. Conversely, do not use its inability to name anatomy as quantitative evidence of predictive superiority.

## 4. Priority experiment A: do communities preserve different roles of the same concept?

### The problem being addressed

A semantic summary can pool neurons associated with the same anatomical concept even when they contribute differently to predictions. A functional summary can retain groups without explaining what anatomy they represent.

The joint view can ask:

> Does MYO-associated activity in one community play a different predictive role from MYO-associated activity in another community?

This is more informative than asking only whether communities contain many MYO-labelled neurons.

### Analysis

1. Fix functional communities and reliable concept-associated channel sets using training data.
2. Within each community, identify the channels associated with each concept.
3. On held-out patients, measure signed changes in a fixed class margin when those groups are ablated.
4. Compare the resulting role structure with semantic-only grouping and matched random partitions.
5. Retain node IDs, training and evaluation IoUs, selected-channel counts, community membership and intervention effects.

For the current final ResNet18 stage, the downstream logits are linear in pooled channel activations. For an original class y and a fixed comparison class r:

\[
D_{m,c}(x)=\sum_{i\in C_m\cap T_c}(w_{y,i}-w_{r,i})z_i(x).
\]

This equals the original margin minus the margin after zeroing those channels. It does not establish a causal relationship between anatomy and diagnosis. Earlier layers require actual downstream intervention evaluation.

### What would support the joint representation?

The actual communities reveal reproducible concept-associated roles that are obscured by concept pooling and are better organised than roles obtained from matched arbitrary partitions.

Finding opposite signs somewhere is insufficient: arbitrary channel subsets can also have opposite signs. The graph must earn its role through controls and held-out evaluation.

### Suggested figure

A concept → community → signed prediction-effect diagram, with spatial evidence and neuron-level links. For example, show two MYO-associated groups with different effects on a fixed HCM-versus-comparator margin. That example must come from measured results, not the desired narrative.

## 5. Priority experiment B: measure the value of community resolution

This is one quantitative test of community-based compression, rather than a general definition of the joint graph's added value.

### Representations

Let z_i(x) be a channel's pooled activation. For community C_m and concept c, define:

\[
h_{m,c}(x)=\frac{\sum_{i\in C_m}S_{ic}z_i(x)}{\sum_{i\in C_m}S_{ic}},
\]

when the denominator is nonzero. Define global concept summaries analogously over all channels, and functional summaries as community activation summaries.

Compare small regularised explanatory readouts of the representations listed in Section 3. These readouts approximate the frozen original CNN; they do not retrain or replace it.

### Two evaluation targets

**Intact-prediction fidelity:** How closely does the explanation reproduce the original model's outputs on held-out patients?

**Intervention-response fidelity:** Can the explanation predict the change caused by prespecified channel interventions on held-out patients?

For the second target, recompute each representation from the intervened activations at the chosen layer, run the fixed explanatory readout, and compare its predicted change with the original model's actual change. Use the same intervention identities for all methods. Do not supply actual held-out intervention effects as features.

Where feasible, also reserve intervention sets not used in readout fitting. Otherwise label the result as generalisation to new patients under familiar interventions, not generalisation to new interventions.

### Candidate score: community resolution gain

Let L be held-out fidelity loss, with smaller values better:

\[
G_{\mathrm{community}}=L_{\mathrm{separate}}-L_{\mathrm{joint}}.
\]

Report an analogous gain against each relevant baseline. Use probability KL for intact prediction fidelity and a prespecified error on signed class-margin changes for intervention fidelity.

- Positive: the joint summary improves the selected explanatory task.
- Zero: no demonstrated benefit for that task.
- Negative: the joint summary performs worse.

Report absolute losses and paired patient-bootstrap intervals alongside gains. The difference-of-losses formula is conventional. The possible contribution is the community-resolved representation and the protocol that isolates its value.

### Avoid the “more parameters” explanation

- Match feature dimension where possible, and show fidelity versus representation dimension.
- Include random-partition joint representations with the same number and sizes of groups.
- Tune regularisation using development data only.
- Count effective parameters and report complexity explicitly.
- Define a common policy for weakly annotated/unannotated channels. An unconstrained residual channel can recover fidelity while bypassing the semantic explanation; report its contribution separately.
- Do not use the original classifier weights as features for one explanatory readout but not others.

Do not claim that higher output fidelity alone proves anatomical validity. The semantic reliability audit remains necessary.

## 6. Priority experiment C: quantify useful intervention trade-offs

The current data already supports examining concept association and prediction sensitivity together. It does not support the claim that joint selection always produces the largest disruption.

### Compare at matched operating points

At a fixed channel budget, compare the strategies on:

1. Held-out concept association and vocabulary-relative selectivity.
2. KL and signed class-margin effects.
3. Patient uncertainty and matching quality.

A possible task formulation is:

> Which method identifies influential channel groups while meeting a prespecified semantic-association requirement?

Alternatively, compare semantic association at a prespecified level of prediction disruption. Define operating points using training/validation data, then freeze them before test evaluation. If a method cannot reach the operating point, report it as infeasible rather than extrapolating or silently changing budgets.

Plot a trade-off frontier instead of multiplying IoU by KL. A product embeds arbitrary scaling and preferences; it also can conceal a weak component behind a strong one. A frontier is descriptive unless its uncertainty and selection procedure are evaluated.

This experiment tests intervention-selection utility. It does not prove concept-specific causality, and it must not be the only evidence for the community contribution.

## 7. Supporting experiment D: is there relational information beyond semantic co-association?

Compare:

\[
W=S^\top AS,\qquad B=S^\top S.
\]

W aggregates concept-associated coactivation between neurons. B captures semantic co-association within neuron profiles. Their different magnitudes do not constitute evidence that one is better; they measure different quantities.

### Useful comparisons

- Whole-profile semantic permutations with A fixed.
- Degree-stratified semantic permutations.
- Graph controls preserving specified structural properties, with their limitations documented.
- Stability of prespecified relation patterns across patients/splits and appropriate datasets.
- Sensitivity to anatomical mask area and activation magnitude.

The existing uniform edge-position rewiring preserves edge count and the weight multiset, **not degree or strength**. It cannot alone rule out hub/degree effects. A stronger rewiring comparison would need an explicit degree-preserving design; weight/strength preservation is a further, separate requirement.

Do not claim that a low correlation between W and B establishes useful added information. The difference must support a downstream task or a reproducible structural statement.

With only RV, MYO and LV there are three distinct between-concept pairs. A rank correlation of one over three values is weak evidence of generality. Do not enlarge the vocabulary merely to increase the number of observations; any added concepts need valid annotations and clear meanings.

## 8. Where could a defensible novel contribution lie?

### Strongest candidate: community-aware semantic abstraction

A compact explanatory representation that retains concept-specific responses within functional communities, together with evidence that this correspondence improves held-out prediction and intervention fidelity at comparable complexity.

Its novelty claim would concern **what is preserved during abstraction, why that matters, and how the benefit is isolated experimentally**.

### Supporting candidate: a controlled benchmark for the value of integration

A protocol that evaluates semantic grounding, functional fidelity and correct correspondence separately, using matched budgets, independent patients and disrupted-correspondence controls.

A benchmark contribution would require useful reusable tasks and sufficiently broad validation; collecting more metrics alone is not enough.

### Potential theoretical component

Construct a controlled example in which two activation states produce identical global concept and community summaries but different community–concept summaries and different model outputs. Then any deterministic predictor receiving only those marginal summaries must give the same prediction for both states, while a predictor receiving the joint summaries can potentially distinguish them.

This would demonstrate a specific information loss caused by marginal aggregation. The construction must satisfy the chosen representation definitions and be proved explicitly. It does not show that every SCAG graph has this property, that natural MRI data exhibits it, or that joint inputs exceed the information in full aligned A and S.

### What should not be claimed as novel on its own

- Matrix projection SᵀAS.
- Concatenating semantic and functional features.
- Purity, enrichment, SVE, bootstrap intervals or a difference of losses.
- A renamed product of semantic relevance and centrality.
- Discovering anatomical or physiological dependencies from coactivation.

## 9. A minimal experimental package

| Order | Experiment | Decision it supports |
|---|---|---|
| 1 | Complete semantic reliability, selectivity and representative-overlay audits | Are the anatomical labels defensible? |
| 2 | Signed concept roles within ResNet18 final-stage communities, with random-partition controls | Is community identity behaviourally informative? |
| 3 | Separate versus joint summaries at matched complexity, intact and intervention fidelity | Does preserving correspondence improve explanation quality? |
| 4 | Prespecified sensitivity checks and replication | Is the finding stable beyond one model/layer/cohort? |

Start with one architecture and one layer. If joint summaries fail to beat matched random partitions, investigate that outcome before expanding to every architecture.

The current ACDC test patients have already informed this exploratory discussion. Freezing a new analysis now does not restore an untouched test set. A future confirmatory claim needs genuinely independent evaluation, or the work must clearly report its exploratory status.

ACDC SAX and M&Ms-2 LAX differ in view, tasks and patient populations. Evaluating separately trained models on both supports broader replication; it is not automatically external validation of one frozen classifier.

## 10. Outcomes and the corresponding honest conclusion

| Observed outcome | Supported interpretation |
|---|---|
| Joint beats semantic-only but not separate summaries | Access to functional information helps; explicit correspondence has not shown added value |
| Joint beats separate summaries but not random-partition joint summaries | Finer subdivision helps; functional communities have not earned their role |
| Actual joint summaries beat both controls with reliable semantics | Evidence that functional community organisation improves semantic abstraction |
| Better intact fidelity but no intervention fidelity improvement | Observational approximation improves; behavioural generalisation remains unproven |
| Better fidelity but poor semantic validity | Compression may work, but an anatomically grounded explanation is not established |
| Higher IoU but lower KL | A trade-off, not universal superiority |
| No reproducible gain | The joint graph may remain useful for inspection; a quantitative superiority claim is unsupported |

A suitable reviewer response **if supported by results** would be:

> SCAG does not create information beyond its semantic and functional inputs. It preserves their neuron-level correspondence in an explicit community-based representation. Under matched explanation budgets, this representation retains concept-associated predictive roles and intervention responses that are lost by separate summaries, while supporting traceable anatomical evidence.

## 11. Related work and conference context

- [Horta et al., Semantic Interpretability of Convolutional Neural Networks by Taxonomy Extraction (2023)](https://ceur-ws.org/Vol-3432/paper9.pdf): prior use of coactivation graphs and semantics. Broad “semantic plus functional graph” novelty is insufficient.
- [Yeh et al., On Completeness-aware Concept-Based Explanations (NeurIPS 2020)](https://papers.neurips.cc/paper_files/paper/2020/hash/ecb287ff763c169694f682af52c1f309-Abstract.html): concept sufficiency and prediction behaviour. Community resolution gain must be positioned as an extension/application with distinct controls, not invention of completeness.
- [Koh et al., Concept Bottleneck Models (ICML 2020)](https://proceedings.mlr.press/v119/koh20a.html): predictions mediated by learned concepts and concept interventions. SCAG is a post-hoc explanation of a frozen model; channel ablation is not equivalent to correcting a concept bottleneck.

This is a targeted starting point, not an exhaustive novelty review. The earlier note contains further contribution ideas and the relevant MIDL policy caveat.

[MIDL 2027 author instructions](https://2027.midl.io/author-instructions) state that LLM-generated research ideas, designs and analyses are prohibited. This AI-generated discussion document must not be presented as author-originated research compliant with that policy. Clarify the implications with the supervisor/chairs before adoption for that venue.

## 12. Discussion and decisions

- Which added-value claim matters most: traceability, fidelity, intervention prediction, or constrained selection?
- Is the separate-summary baseline a fair representation of the practical alternative?
- What explanation budget should be matched: dimensions, parameters, number of groups, or displayed evidence?
- How will unreliable semantic channels be handled without hiding missing coverage?
- What result would make us reject the community-specific added-value hypothesis?
- What independent evidence could support a confirmatory claim?

**User comments:**

**Supervisor comments:**

**Agreed hypothesis:**

**Agreed primary experiment:**

**Agreed novelty boundary:**

No experimental code or results were changed in preparing this note. All proposed advantages remain hypotheses unless separately demonstrated.
