# SCAG explanations and candidate evaluation criteria inspired by NeuroCartography

**Status:** research discussion. Proposed names are working descriptions, not claims of established novelty. No new results, metric implementation or dashboard are provided here. This follows the [manuscript review](README_manuscript_review.md).

## 1. What to take from NeuroCartography

[NeuroCartography](https://arxiv.org/html/2108.12931) combines neuron clustering, coactivation-based embeddings, linked example/graph views, class exploration and concept cascades. Its clustering uses preferred-image similarity and spatial activation overlap; the graph's cross-layer influence estimates are different from SCAG's within-layer correlations. It also evaluates human recognition of coherent clusters and agreement in descriptions. See Sections 5–8.

The useful adaptation is an evidence-linked explanation workflow and evaluation. Copying a layout, adding UMAP, or calling correlation edges pathways would not establish novelty.

## 2. A concrete gap in the current SCAG evaluation

A graph explanation can be attractive, retain the same neurons and cause the same ablation effect, yet attach the wrong concept to a node or the wrong semantic interpretation to an edge.

- KL/flip cannot detect a label swap if the ablated channels are unchanged.
- IoU tests individual associations but does not test the relationships presented between them.
- Community purity can reward an easy-to-label diagram while hiding valid connections between different concepts.
- Global semantic–functional correlation does not evaluate an individual displayed explanation.

These are gaps in the present evaluation, not proof that the entire XAI literature lacks applicable metrics. The proposed target is **the accuracy and completeness of the relational statements conveyed by a compact graph explanation**.

## 3. Candidate A: grounded relational fidelity

### Question

> At the same explanation size, how accurately can the displayed representation answer independently evaluated questions about concept-associated neural relationships?

This is a candidate evaluation framework. Standard losses can implement it; a standard loss given a new name is not a novel mathematical metric.

### Make the explanation testable

Let E be the displayed explanation, including its node/group memberships, semantic profiles, displayed edges, expansion rules and explicitly linked evidence. Define a fixed query family before evaluation. Examples:

- Does a particular displayed neuron or group have supported association with MYO?
- Is there a supported direct coactivation connection between specified MYO- and LV-associated neuron sets?
- Which concept-associated neighbours does this selected neuron have?
- If an intervention effect is claimed, does the specified removal change the fixed class margin in the stated direction?

Specify exact semantics. An edge between two mixed groups means some measured neuron connections link the groups; it does not imply that every concept shown in one group connects to every concept in the other. A path means graph reachability, not signal propagation or endpoint correlation.

The explanation must answer from E and the evidence explicitly included in the allowed expansion budget. It must not secretly query the full graph while claiming to be a tiny explanation. A full graph is a reference ceiling, not a fair size-matched competitor.

### Held-out reference

Construct the reference answers from independent patients using frozen activation calibration and prespecified association/coactivation criteria. For a directed indexing of an undirected neuron pair, a basic conjunctive relation query has reference answer

\[
y_q=\mathbb{1}\{S^{eval}_{ic}\geq\gamma_c,\ R^{eval}_{ij}>\tau,\ S^{eval}_{jd}\geq\gamma_d\}.
\]

This is an operational reference, not physiological ground truth. The statistical evidence for the conjunction must account for uncertainty and dependence. Insufficient evidence is not automatically evidence of absence; retain an unresolved category and report how often it occurs. If binary scoring excludes unresolved queries, report the excluded share and test sensitivity to that policy.

Sample queries independently of the method's returned subgraph and include both present and absent relationships. Otherwise a method can score well by showing one easy relationship or choosing its own questions. Keep patient-disjoint development/evaluation and avoid treating all shared-neuron edges as independent observations.

### Score and reporting

For a declared query-answering rule, use standard per-family balanced accuracy for hard answers, or a proper scoring rule such as Brier loss for *calibrated* probabilities. IoU and correlation weights are not probabilities. Both classes must be present to define balanced accuracy; otherwise report class-specific results. Report abstentions, evidence coverage and the number of represented neurons/edges alongside accuracy.

A possible scalar summary is the macro-average of prespecified query-family scores at budget B. The macro-averaging and prediction metric are existing statistics. Do not silently combine intervention and structural questions into a single number: report the structural result and behavioural extension separately.

For scientific interpretation, the primary output should be a **fidelity-versus-size curve**, with separate semantic, relational and behavioural panels. This avoids compensating a completely wrong semantic label with a large ablation effect.

### Why this tests the joint representation

Semantic-only explanations can answer node-association questions but lack observed functional connectivity. Functional-only explanations contain connectivity without its concept identities. The joint graph can answer the conjunction, provided its claims are reliable.

Compare these as restricted-information baselines, but also compare a matched display of separate semantic and functional views, and a shuffled-correspondence control. Two complete aligned inputs can reconstruct the same joint information; any advantage over a separate display is an accessibility/compression benefit, not extra information created by a graph format.

### Necessary validity checks

1. Semantic-label corruption should harm semantic and relational answers when it changes their reference truth, while unchanged channel ablation scores can remain identical.
2. Topology corruption should harm changed relational answers while individual IoUs can remain identical.
3. Jointly renumbering neurons with all identities and evidence preserved must not change the score.
4. Repeating the same claim must not increase score or coverage; count unique queries.
5. A large uncompressed graph should not win a compactness comparison without paying the display cost.
6. A small explanation that abstains on almost everything must expose its low coverage.
7. Synthetic graphs with known labels/edges should give predictable scores; real-data tests then assess robustness under measurement uncertainty.

For a metric-focused paper contribution, establish these properties and show failure cases for the actual baseline metrics. If existing graph explanation accuracy already captures the desired property after a straightforward adaptation, present this as an evaluation protocol rather than claiming an unprecedented metric.

## 4. Candidate B: minimum grounded explanation size

### Question

> What is the smallest explanation that retains adequate semantic grounding, relational fidelity and—if claimed—behavioural fidelity?

For a fixed explanation family and predefined decoding rules, consider

\[
B^*(\epsilon_s,\epsilon_r,\epsilon_b)=\min_E |V_E|
\quad\text{subject to}\quad
L_s(E)\leq\epsilon_s,\quad L_r(E)\leq\epsilon_r,\quad L_b(E)\leq\epsilon_b.
\]

Here L_s measures semantic-query error, L_r relational-query error, and L_b error in claimed intervention responses. Tolerances are set on development data. The structural-only version omits L_b rather than implying behavioural validation was done.

The display protocol must cap edges, member lists and evidence per visible unit; otherwise a single supernode can encode the whole model. Report those sizes explicitly rather than hiding them in an arbitrary weighted cost. Comparable textual/semantic displays also require a fixed unit definition. Node count alone is only a proxy for cognitive load.

This formulation uses familiar constrained optimisation and rate–distortion ideas. The possible contribution is the grounded relational explanation task and an effective construction method, not the minimisation symbol. In practice report the smallest size *found by each method*, not a proven optimum, unless an optimisation certificate exists. If a method cannot meet all tolerances, report infeasibility rather than weakening them after seeing evaluation results.

A connected-subgraph or graph-summarisation method could target these requirements. It must compete with simple top-score and random summaries; do not build an elaborate optimiser before those baselines reveal a gap. More compact explanations are not automatically easier for people to understand; evaluate users separately.

## 5. What the explanation should actually show

Build an explanation around a concrete question, such as “Which concept-associated groups are functionally related to this MYO-associated group, and what happens when selected groups are removed?”

| Panel | Display | What it establishes |
|---|---|---|
| Overview | Small within-layer graph, node/group sizes, complete concept mixtures, explicit unlabelled share | Organisation represented by the summary |
| Visual evidence | MRI, anatomical masks, activation overlays and per-example overlaps, with a fixed sample-selection rule | Spatial support for concept associations |
| Relation detail | Exact supporting neuron pairs, development/evaluation correlations, uncertainty and relevant controls | Evidence for each displayed relation |
| Intervention detail | Original/ablated predictions, fixed-margin changes, KL/flip and matched controls | Measured effect of a specified channel intervention |

Keep the same identifiers and concept colours across panels. Clicking an edge in an interactive version should reveal its evidence; the paper should provide a readable static version without depending on a demo.

Use two kinds of examples: explicitly selected high-activation examples for exploration, and a reproducible representative audit sample including failures. Attractive examples alone do not establish population reliability. Never equate a high pooled activation with concept-specific spatial evidence.

**Explanation template, not a result:**

> “Group G1 contains channels associated with MYO under the stated spatial-overlap criterion. Its connection to G2 is supported by [listed neuron pairs] on [evaluation cohort]. G2's members have [concept profile]. Removing [specified channels] changes the fixed [class/comparator] margin by [measured effect and uncertainty]. These observations establish coactivation and intervention sensitivity; they do not establish an anatomical causal relationship.”

Separate population-level relationships from patient-specific observations. A cohort correlation cannot be estimated from one patient's pooled activations. A patient panel can highlight active members of a population graph and show that patient's intervention response, with those roles labelled explicitly.

For multilayer views, show within-layer graphs as separate panels until actual cross-layer evidence is implemented. Do not draw causal arrows simply by arranging layers from left to right. Cross-layer intervention effects are a possible later addition, not already present in the coactivation graph.

## 6. Analyses to adapt

The following are proposed SCAG analyses inspired by the source workflow, not reported NeuroCartography findings about cardiac models.

1. **Same anatomy, different functional context.** Retrieve neurons with similar own-concept profiles but different semantic neighbourhoods. Display matched examples and test whether held-out responses or interventions differ. This directly makes graph context inspectable without requiring communities.
2. **Different anatomy, strong coactivation.** Retrieve cross-concept connections, inspect spatial evidence and test replication. Control anatomy size, broad activation and cohort composition. Do not call this compositional computation without further evidence.
3. **Mixed-community explanations.** Show where different concepts occur inside a functional group and which edges connect them. Preserve mixtures rather than forcing a homogeneous label to make the plot look cleaner.
4. **Shared versus distinctive relational structure.** Compare layers, models or class-conditioned cohorts with matched sampling and uncertainty. Class-conditioned correlation estimates need enough independent patients; do not infer differences merely because one group is significant and another is not.
5. **Unexplained or isolated components.** Inspect weakly annotated yet influential neurons as gaps in vocabulary coverage. Isolation in a 2D layout alone does not imply graph isolation, irrelevance, or shortcut learning. Artifact claims require actual artifact evidence and annotations.
6. **Intervention response exploration.** On real inputs, ablate selected groups and measure downstream changes. This can support an interactive explanation, but perturbation magnitude, baseline replacement and off-distribution effects need explicit treatment. Setting activations to a constant is not the same as editing an anatomical concept.

## 7. Human evaluation: test understanding, not visual attractiveness

Use a small, properly scoped study with participants competent for the task: imaging experts for anatomy validity, and ML researchers or trained users for graph interpretation. Expert time should focus on questions requiring expertise. No participant recruitment or study is authorised by this note.

Two different tasks are needed:

- **Concept evidence task:** judge whether image/activation examples support a stated anatomical association. Include ambiguous and negative examples, conceal predicted labels when testing independent recognition, and use a rubric with unknown/mixed options. Do not treat the clustering algorithm's assignment as clinical ground truth.
- **Relational explanation task:** use the display to answer which concept-associated groups are connected, find evidence supporting an edge, identify an unsupported claim, or predict the direction of an intervention before revealing its result. The latter needs independent measured intervention outcomes as reference.

Compare semantic-only, functional-only, separate aligned views and the integrated view. Equalise display size, available examples, interaction time and training. The separate aligned view is essential: improvement over it supports the usability of integration rather than merely access to more information.

Measure accuracy, completion time, abstention/uncertainty and inter-rater agreement. Counterbalance display order and prevent repeated exposure to the same answer from inflating later conditions. Distinguish content-faithfulness results from usability results; neither alone establishes clinical usefulness. A pilot may reveal interface problems but is not evidence of broad expert agreement.

## 8. Prior work and the novelty boundary

Relevant primary sources to compare before claiming a new metric:

- [NeuroCartography](https://arxiv.org/html/2108.12931): source for the visual workflow and human-coherence evaluation.
- [Completeness-aware concept explanations](https://arxiv.org/abs/1910.07969): already evaluates whether concepts suffice to explain prediction behaviour.
- [Zorro](https://arxiv.org/abs/2105.08621): already uses rate–distortion-based fidelity for sparse graph explanations.
- [GraphFramEx fidelity definitions](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.explain.metric.fidelity.html): existing necessity/sufficiency-style evaluation of explanatory subgraphs.
- [LM–KG fidelity](https://aclanthology.org/2024.emnlp-main.1052/): existing work on reliability of graph-based explanations and fidelity–sparsity trade-offs in a different model setting.

SCAG's coactivation graph is an extracted representation of a CNN, not the graph consumed by a GNN. Removing an edge from the explanation does not remove a CNN computational connection. This affects what a faithful intervention or relation claim can mean, but is not sufficient by itself to establish novelty.

The candidate contribution is a **method for constructing compact, evidence-linked semantic–functional explanations, with evaluation of their relational fidelity and human usefulness**. Metric novelty is conditional on a demonstrated gap relative to the above work and other relevant literature. No exhaustive novelty search has been completed.

## 9. Recommended scope

Start with a static four-panel explanation and a prespecified relational query benchmark using existing saved outputs. Add representative overlays where evidence is missing. Assess label/edge corruption controls and matched-budget baselines. Only then decide whether a new summarisation algorithm or interactive tool is needed.

Prefer this coherent package to implementing a dashboard, UMAP, cascade experiments, an optimiser and several newly named metrics at once. The earlier interaction proposal remains optional; it is not a prerequisite for evaluating structural explanations.

**User comments:**

**Supervisor comments:**

**Chosen explanation task:**

**Proposed novelty beyond closest prior work:**

**Primary evaluation and failure criterion:**

The MIDL policy constraint described in the [manuscript review](README_manuscript_review.md#venue-note) still applies to these AI-generated research proposals. This note does not resolve it.
