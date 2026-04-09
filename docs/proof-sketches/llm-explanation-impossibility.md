# LLM Self-Explanation Impossibility — Proof Sketch

## Setup
- Θ = LLM configurations (training initialization, fine-tuning seed, RLHF reward model, or generation sampling path)
- Y = output answers (classification labels, QA answers, final predictions)
- H = natural language reasoning chains (the "explanation" text)
- observe(θ) = the model's answer to a query
- explain(θ) = the model's chain-of-thought / stated reasoning

## Rashomon Property
Multiple LLM configurations produce the same answer but different reasoning:
- Different training initializations converge to models with different internal representations but equivalent output behavior
- Temperature sampling produces different reasoning paths that arrive at the same conclusion
- Different prompt framings ("explain why" vs "what factors led to") elicit different explanations for the same answer

Cite: Turpin et al. (2024) "Language Models Don't Always Say What They Think" — CoT explanations are unfaithful to actual computation. Lanham et al. (2023) — measuring faithfulness in chain-of-thought. Ye & Durrett (2022) — unreliability of few-shot explanations.

The key insight: LLM explanations are POST-HOC rationalizations of a computation that may use entirely different internal features. Two models that give the same answer may generate explanations citing different evidence because their internal representations differ.

## Incompatibility
Two explanations are incompatible if they cite different evidence for the same conclusion: e.g., one says "the word 'excellent' indicates positive sentiment" while another says "the phrase 'highly recommend' indicates positive sentiment."

## Trilemma
- Faithful: the explanation reflects the model's actual internal reasoning (which tokens/features actually drove the prediction)
- Stable: the same explanation for equivalent models (models that give the same answer)
- Decisive: the explanation commits to specific causal claims ("this word caused the prediction")

## Proof
Same 4-step structure as all other instances.

## Implications
This result is particularly significant for LLM deployment under regulation. The EU AI Act requires "meaningful explanations" for high-risk AI decisions. If the model's explanation changes upon retraining (or even upon re-prompting), the explanation cannot simultaneously be faithful to the model's actual reasoning, stable across equivalent models, and decisive in its causal claims.

The resolution: report explanation uncertainty. Instead of "the model predicted X because of Y," report "the model predicted X; across equivalent models, the most commonly cited evidence is {Y₁, Y₂, Y₃} with confidence levels."

## Verdict: GO
## Axioms needed: Rashomon property only.
