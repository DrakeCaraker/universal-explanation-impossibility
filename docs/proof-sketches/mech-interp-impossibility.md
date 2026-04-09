# Mechanistic Interpretability Impossibility — Proof Sketch

## Setup
- Theta = neural network weight configurations (architecture + trained weights)
- H = circuit decompositions (subgraphs of the computational graph + interpretations of which components implement which sub-computations)
- Y = input-output functions (the function the network computes on the domain of interest)
- observe(theta) = the function f_theta : X -> Y
- explain(theta) = a circuit decomposition C_theta describing which components implement which sub-computations

## Rashomon Property
By Meloux et al. (ICLR 2025, "Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable?"):
- For a simple XOR task, they found **85 distinct valid circuits** with zero circuit error
- Each circuit admitted an average of **535.8 valid interpretations**
- This means: there exist theta_1, theta_2 such that:
  - f_{theta_1} = f_{theta_2} (identical input-output behavior)
  - C_{theta_1} and C_{theta_2} are incompatible (attribute computation to different components)

Additional evidence from Bricken et al. (2025): sparse autoencoders trained on the same data learn different features, demonstrating that even the feature-level decomposition is not unique.

Incompatibility: C_{theta_1} and C_{theta_2} attribute the same computation to disjoint or contradictory subgraphs of the network.

## Trilemma
- **Faithful**: report the actual circuit decomposition C_theta for network theta
- **Stable**: same circuit explanation for equivalent networks (same I/O function)
- **Decisive**: commit to a single circuit decomposition rather than returning the space of valid circuits

## 4-Step Proof
1. **Rashomon**: Meloux et al. witness two networks theta_1, theta_2 with identical I/O behavior but incompatible circuit decompositions (different components claimed responsible for the same computation).
2. **Decisiveness at theta_1**: Since explain(theta_1) and explain(theta_2) are incompatible, decisiveness forces E(theta_1) to also be incompatible with explain(theta_2).
3. **Stability**: observe(theta_1) = observe(theta_2) implies E(theta_1) = E(theta_2), so E(theta_2) is incompatible with explain(theta_2).
4. **Faithfulness at theta_2**: E(theta_2) must not contradict explain(theta_2). Contradiction with step 3.

## Resolution: Circuit Equivalence Classes
The constructive path forward mirrors the CPDAG resolution for causal discovery:
- Instead of searching for THE circuit, characterize the **equivalence class** of valid circuits
- Define circuit equivalence: C ~ C' iff they produce the same I/O behavior and the same computational invariants
- The G-invariant explanation is the equivalence class itself (analogous to CPDAGs for DAGs)
- This trades decisiveness for faithfulness + stability
- Practically: report which circuit features are shared across ALL valid decompositions (the "CPDAG of circuits") vs. which are orientation-dependent

This is exactly what the field is beginning to move toward: Anthropic's circuit tracing (2025) reports confidence levels rather than single circuits, and Conmy et al. (2023) frame automated circuit discovery as search over a space rather than identification of a unique answer.

## Axioms needed: Rashomon property only (via circuitSystem axiom).

## Lean reference: `mech_interp_impossibility` in `MechInterpInstance.lean`
