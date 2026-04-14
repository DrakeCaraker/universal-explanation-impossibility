# Revolutionary Directions: Design Spec

**Goal:** Attempt three research directions that would elevate the Universal Explanation Impossibility from "precise characterization" to "revolutionary" — by making the framework *generative* (producing knowledge that wasn't accessible before it existed).

## Direction 1: G-Discovery Algorithm

**What:** An algorithm that discovers the explanation symmetry group G from data, without hand-specification, and uses it to predict explanation instability via η = dim(V^G)/dim(V).

**Algorithm (SAGE: Symmetry-Aware Group Estimation):**
1. Input: dataset X, model class (e.g., XGBoost), n_retrains M
2. Train M models on bootstrap resamples of X
3. For each model, compute feature importance vector → importance matrix I ∈ ℝ^(M×P)
4. For each feature pair (j,k): compute the flip rate across all M(M-1)/2 model pairs
5. Cluster features: hierarchical clustering on the flip-rate matrix. Features with high mutual flip rate (>0.3) are in the same orbit. Features with low mutual flip rate (<0.05) are in different orbits.
6. Output: g groups with sizes k₁,...,k_g
7. Predict: η = g/P, within-group flip rate ≈ 50%, between-group flip rate ≈ 0%

**Success criterion:** On 10+ real datasets (OpenML), the SAGE-predicted η achieves R² > 0.8 against the actual observed instability, without any hand-specification of groups.

**Compute:** ~30 min (100 models × 10 datasets × XGBoost = fast)

## Direction 2: Circuit Non-Uniqueness in Pythia

**What:** Demonstrate that neuron-level interpretations are non-unique across independently trained language models, using Pythia (which provides multiple seeds), confirming the interpretability ceiling in a real system.

**Experiment:**
1. Download Pythia-70M (two different checkpoints or seed variants)
2. For each model: train linear probes on layer-6 hidden states for 3 tasks:
   - Part-of-speech tagging (structural)
   - Sentiment (semantic)
   - Next-token entropy (functional)
3. For each task: identify top-10 "important" neurons (highest probe weight magnitude)
4. Measure agreement: Jaccard similarity of top-10 neuron sets between models
5. Measure subspace agreement: cosine similarity of the probe weight vectors (these live in the same dimensional space even if neuron identity differs)
6. Prediction: Jaccard (neuron-level) < 0.2, cosine (subspace-level) > 0.8

**Key insight:** Pythia-70M has d_model=512. If S_512 permutation symmetry holds, the ceiling predicts ≤ 1/512 ≈ 0.2% of neurons have stable roles. But training breaks symmetry — the question is HOW MUCH.

**Fallback:** If Pythia variants are too similar (both trained on The Pile with similar hyperparameters), use Pythia-70M vs GPT-2-small (different data, different architecture, same scale) as a more aggressive comparison.

**Compute:** ~1 hour (model download + probing)

## Direction 3: Quantum Measurement Impossibility

**What:** Formalize quantum state tomography as an ExplanationSystem and derive that dim(V^G)/dim(V) for the measurement symmetry predicts the accessible information fraction.

**Theoretical setup:**
- Θ = {density matrices ρ on ℂ^d}: configuration space
- Y = {probability distributions over measurement outcomes}: observable space
- H = {descriptions of the quantum state}: explanation space
- observe(ρ) = {Tr(ρ M_i)}_i for measurement operators {M_i}: the Born rule
- explain(ρ) = ρ itself
- incompatible(ρ₁, ρ₂) = (ρ₁ ≠ ρ₂)
- Rashomon: ∃ ρ₁ ≠ ρ₂ with Tr(ρ₁ M_i) = Tr(ρ₂ M_i) ∀i (informationally incomplete measurement)

**Key derivation:**
For a measurement {M_i} with symmetry group G (i.e., U M_i U† = M_{π(i)} for U ∈ G), the G-invariant subspace of state space has dimension dim(V^G). The fraction of state properties accessible from the measurement is dim(V^G)/dim(V) where V = ℝ^{d²-1} (the real vector space of traceless Hermitian operators, i.e., the Bloch vector space).

**Concrete example:** For a single Pauli-Z measurement on one qubit:
- d=2, V = ℝ³ (Bloch sphere)
- Measurement symmetry: U(1) rotations around Z-axis
- V^G = span{σ_z} has dimension 1
- η = 1/3: only 1/3 of state information is accessible from Z-measurement alone
- Compare to: you need 3 measurement bases (X, Y, Z) for full tomography → 1/3 from each, consistent

**Lean formalization:** Create QuantumMeasurement.lean with the ExplanationSystem instance for a finite-dimensional quantum system (can use Fin d for the Hilbert space dimension).

**Success criterion:** η = dim(V^G)/dim(V) matches the known sample complexity scaling for quantum tomography under specific measurement symmetries (Pauli measurements, random Clifford measurements, etc.)

**Compute:** Purely theoretical. ~2-4 hours for derivation + Lean formalization.

## Go/No-Go Criteria

| Direction | Go signal | No-go signal |
|-----------|-----------|-------------|
| G-Discovery | R² > 0.8 on 5+ datasets | R² < 0.5 (groups aren't discoverable from importance) |
| Pythia MI | Jaccard < 0.3 AND cosine > 0.7 | Models too different to compare (architecture mismatch) |
| Quantum | η matches known bounds for ≥2 measurement schemes | No non-trivial connection exists |
