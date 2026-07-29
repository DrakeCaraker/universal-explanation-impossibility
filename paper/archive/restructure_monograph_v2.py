#!/usr/bin/env python3
"""Deterministic generator: original monograph -> tiered Parts I-V v2.
Lossless block reorder + tier tags + vetted prose fixes + legend + disclosure."""
import collections, sys

SRC='/Users/drakecaraker/universal-explanation-impossibility/paper/universal_impossibility_monograph.tex'
OUT='/Users/drakecaraker/universal-explanation-impossibility/paper/universal_impossibility_monograph_v2.tex'
lines=open(SRC).read().split('\n')
starts=[186,323,330,373,410,459,483,535,586,680,769,842,928,1008,1039,1061,1098,1108,1131,1161,1177,1203,1216,1246,1345,1388,1412,1418,1544,1548,1564,1603,1808,1847,1852,2050,2183,2243,2283,2344,2361,2499,2522,2567,2619,2658,2708,2769,2851,2894,2916,3016,3084,3114,3152,3350,3424,3499,3524,3574,3673,3694,3774,3791,3799,3811,3866,3902,3951,4002,4007,4012,4134,4139,4148,4186,4255]
bib=4264
assert len(starts)==77
ranges=[(starts[i]-1,(starts[i+1]-1 if i+1<len(starts) else bib-1)) for i in range(77)]
parent=[None]*77; cur=None
for i,s in enumerate(starts):
    if lines[s-1].startswith('\\section'): cur=i
    parent[i]=cur if lines[s-1].startswith('\\subsection') else None

tier={0:'-',1:'-'}
partof={}
def a(idxs,part,t):
    for i in idxs:
        partof[i]=part; tier[i]=t
# Part I - Hardened
a([2,3,4,5,6,7,8,9],1,'A'); a([16,17,18,19],1,'A'); a([12,13,14,15],1,'A')
a([66,67],1,'A'); a([31],1,'A'); a([29],1,'A'); a([21,22,23,24,25,26,27],1,'A')
# Part II - Empirical (lead with empirical section)
a([41,42,43,45,46,47,48,49,50,52,53,54,55,56,57,58,59,60,61,62],2,'B'); tier[47]='A'
a([71],2,'B'); a([39],2,'B'); a([32,33],2,'B')
# Part III - Instantiations & bridges
a([10],3,'A'); a([11],3,'B'); a([64,65],3,'A'); a([68],3,'B')
# Part IV - Speculative
a([28,30,34,35,36,37,38],4,'C'); a([20],4,'C'); a([69,70],4,'C'); a([72,73,74,75,76],4,'C')
# Part V - Falsified
a([40],5,'D'); a([44,51,63],5,'D')
partof[0]=0; partof[1]=0
assert set(partof)==set(range(77)), set(range(77))-set(partof)

# explicit within-part emission order
order={
 1:[2,3,4,5,6,7,8,9, 16,17,18,19, 12,13,14,15, 66,67, 31, 29, 21,22,23,24,25,26,27],
 2:[41,42,43,45,46,47,48,49,50,52,53,54,55,56,57,58,59,60,61,62, 71, 39, 32,33],
 3:[10,11, 64,65, 68],
 4:[28,30,34,35,36,37,38, 20, 69,70, 72,73,74,75,76],
 5:[40, 44,51,63],
}
for p in order:
    assert sorted(order[p])==sorted(i for i in partof if partof[i]==p), p

td={'A':'Hardened','B':'Empirical','C':'Speculative','D':'Falsified'}
emitted_sections=set()
def block(idx):
    s,e=ranges[idx]; blk=list(lines[s:e]); hdr=blk[0]; t=tier[idx]
    is_sub=hdr.startswith('\\subsection')
    if is_sub and (parent[idx] not in emitted_sections):
        hdr=hdr.replace('\\subsection{','\\section{',1)
    if hdr.startswith('\\section{'): emitted_sections.add(idx)
    if t!='-' and hdr.rstrip().endswith('}'):
        hdr=hdr.rstrip()[:-1]+'\\,\\textnormal{\\normalsize[Tier '+t+': '+td[t]+']}}'
    return ['% ---- Tier '+t+' ('+td.get(t,'')+') ----',hdr]+blk[1:]

preamble=lines[:starts[0]-1]
tail=lines[bib-1:]

LEGEND=r"""
%%%% ========================= READING GUIDE =========================
\clearpage
\section*{Reading guide: how this document is organised}
\addcontentsline{toc}{section}{Reading guide: how this document is organised}
This monograph is the umbrella reference for a research program formalised across
three Lean~4 repositories (the universal framework, an attribution-specific
companion, and a mathematical-physics companion) with accompanying empirical
studies. To let readers calibrate trust at a glance, \emph{every section is tagged
with an evidence tier}, and the body is organised into five Parts by tier:
\begin{itemize}
  \item[\textbf{Tier A --- Hardened.}] Machine-checked in Lean~4 from zero custom
    axioms or from explicitly stated hypotheses; the hard content is genuinely
    proved. \textbf{Part~I} collects the load-bearing theory.
  \item[\textbf{Tier B --- Empirical.}] Supported by experiments/audits, with sample
    sizes, effect sizes, and caveats stated inline. These test the theory; they are
    not logical consequences of it. \textbf{Part~II}.
  \item[\textbf{Tier C --- Speculative / analogical.}] Structural parallels and
    conditional or illustrative formalizations---\emph{not} domain results.
    \textbf{Part~IV} (with an explicit status disclaimer); \textbf{Part~III} holds the
    genuine instantiations and bridges.
  \item[\textbf{Tier D --- Falsified / boundary.}] Predictions that were tested and
    rejected, reported prominently by design. \textbf{Part~V}.
\end{itemize}
Claims that connect the framework to the Langlands programme, adelic physics,
G\"odel's theorem, or the Riemann Hypothesis are \emph{analogies}, not proofs, and
live in Part~IV; the genuine underlying theorems (e.g.\ the $\mathrm{GL}(n)$ trace
bilemma, the completed-zeta three-fold symmetry) are stated separately with their
interpretations flagged.

\paragraph{Note on methodology and AI assistance.}
This manuscript, its Lean~4 formalization, and its computational experiments were
developed with substantial assistance from AI coding and reasoning tools. The
guarantees the work rests on are mechanical, not authorial: every theorem is
checked by the Lean kernel (0~\texttt{sorry}; trusted base stated in Part~I), and
every reported statistic is reproducible from the public repositories and released
scripts. The authors take full responsibility for all claims and framing.
%%%% ================================================================
"""

out=list(preamble)+LEGEND.split('\n')
emitted_sections.clear()
for i in (0,1): out+=block(i)
PART={
1:('Part I \\textnormal{\\large --- The Hardened Core}',
   'Every result in this Part is machine-checked in Lean~4 from zero custom axioms or from explicitly stated hypotheses; the hard content is genuinely proved. This is the load-bearing spine: the universal impossibility---a meta-theorem holding for \\emph{every} explanation system that satisfies the Rashomon property---together with the bilemma and tightness classification, the ubiquity argument that makes the hypothesis generically applicable, the capacity/Reynolds resolution theory, and the trusted-base accounting.'),
2:('Part II \\textnormal{\\large --- Empirical Validation}',
   'Measurements and audits testing the framework. The strongest result is the stable-fact-count bimodality (Tier~A, pre-registered, $p<10^{-12}$); the capacity $\\eta$-law and per-domain fits are Tier~B, with sample-size and real-data caveats stated inline. None of these empirical claims is a logical consequence of the Part~I theorem---they test it, they do not follow from it.'),
3:('Part III \\textnormal{\\large --- Instantiations and Bridges}',
   'The universal theorem specialised to recognisable systems, and connections to established mathematics. These are witnesses that the hypothesis class is populated by familiar examples---not new theorems about those domains.'),
4:('Part IV \\textnormal{\\large --- Structural Analogies and Speculative Extensions}',
   '\\textbf{Status disclaimer.} The material in this Part consists of structural parallels and conditional or illustrative formalizations---\\emph{not} domain results. Where a Lean object appears (the $\\mathrm{GL}(n)$ trace bilemma, the adelic model, the G\\"odel conditional, Navier--Stokes), the genuine theorem is stated in Part~I or~III and only its \\emph{interpretation} lives here. Claims connecting the framework to the Langlands programme, adelic physics, G\\"odel\'s theorem, or the Riemann Hypothesis are analogies, not proofs.'),
5:('Part V \\textnormal{\\large --- Falsified Predictions and Boundaries}',
   'Predictions that were tested and rejected, and the framework\'s known limits. Reported prominently by design: the boundary of a theory is part of its content.'),
}
for p in (1,2,3,4,5):
    t,d=PART[p]
    emitted_sections.clear()
    out+=['','%%%% ==========================================================','\\clearpage','\\part{'+t+'}','\\noindent\\emph{'+d+'}\\par\\medskip','%%%% ==========================================================','']
    for i in order[p]: out+=block(i)
out+=['']+tail
text='\n'.join(out)

# ---- vetted prose fixes (assert each applies exactly once) ----
def rep(old,new):
    global text
    n=text.count(old)
    if n!=1:
        print('!! replacement matched',n,'times (expected 1):',repr(old[:60])); sys.exit(2)
    text=text.replace(old,new)

rep("""This single abstract theorem, which we call the \\emph{Explanation
Impossibility}, is a meta-theorem whose reach extends far beyond machine
learning.  We derive the impossibility as a zero-axiom consequence in
eight scientific domains: the degeneracy of the genetic code (biology),""",
"""This single abstract theorem, which we call the \\emph{Explanation
Impossibility}, is a meta-theorem: it holds for \\emph{every} system whose
configurations, observations, and explanations satisfy the Rashomon property.
A ubiquity argument (Part~I) indicates that this hypothesis is generically met
whenever the parameter dimension exceeds the observable dimension---the regime
of essentially all overparameterized models---though the differential-topology
step of that argument is informal rather than machine-checked (see Part~I).
We then instantiate the theorem on explicit minimal witnesses in
eight scientific domains: the degeneracy of the genetic code (biology),""")

rep("""The framework is mechanically verified in Lean~4 (104~files, 530~theorems,
2~axioms, 0~\\texttt{sorry}).""",
"""The framework is mechanically verified in Lean~4 with \\textbf{0}~\\texttt{sorry}
(a real \\texttt{lake build} of all three repositories completes with 0 errors).
The trusted base is \\textbf{4}~\\texttt{axiom} declarations across the three
repositories---2 in the main repository (bundling ${\\sim}14$ gradient-boosting
properties), 2 in the attribution companion (bundling 7 fields), and 0 in the physics
companion, which instead carries ${\\sim}11$ domain hypotheses (including the classical
Selmer~1951 result) as section variables rather than axioms. Some theorems additionally
depend on Lean's \\texttt{ofReduceBool} via \\texttt{native\\_decide} (6/0/65 across the
three repositories). The core theorem and the entire hardened spine (Part~I) use
\\emph{none} of these. Theorem counts are reported per repository, not summed across the
overlapping companions.""")

rep("""\\paragraph{Reynolds naturality predicts Langlands functoriality.}
The Reynolds naturality theorem (\\texttt{reynolds\\_naturality} in \\texttt{UncertaintyFromSymmetry.lean}) proves that equivariant maps commute with Reynolds projections.  For GL($n$), the Reynolds projection is the trace (conjugation-averaging).  Reynolds naturality therefore implies that the trace commutes with group homomorphisms---which IS Langlands functoriality for finite fields.  The impossibility framework predicts functoriality as a structural consequence of the bilemma's resolution: collapsed tightness for $n \\geq 2$ forces the trace as the unique Pareto-optimal resolution; Reynolds naturality forces the trace to be functorial.  The Langlands programme classifies which characters arise from automorphic representations; the impossibility framework provides a structural explanation for why characters are the natural invariant.""",
"""\\paragraph{Reynolds naturality and the character: a structural analogy, not a Langlands result.}\\emph{~[Tier~C.]}
The Reynolds naturality theorem (\\texttt{reynolds\\_naturality} in \\texttt{UncertaintyFromSymmetry.lean}) proves that equivariant maps commute with Reynolds projections, and for GL($n$) the Reynolds projection is the trace (conjugation-averaging).  This is a one-line fact of invariant theory: the trace is a class function, constant on conjugacy classes and compatible with the group action.  We flag explicitly that this is \\emph{not} Langlands functoriality: the formalization contains no automorphic forms, no $L$-functions, and no automorphic-to-Galois correspondence.  The parallel is purely structural---characters/traces are the natural stable invariant under conjugation, just as the orbit average is the stable resolution under any symmetry group---and is stated only to orient readers familiar with representation theory. It is not a theorem about, or a prediction of, functoriality.""")

rep("""The \\emph{Reynolds naturality} theorem (\\texttt{reynolds\\_naturality} in \\texttt{UncertaintyFromSymmetry.lean}) proves that equivariant maps commute with Reynolds projections: if $\\varphi : V \\to W$ intertwines the group actions ($R_W \\circ \\varphi = \\varphi \\circ R_V$), then the stable resolutions are compatible.  In Langlands terms, this is functoriality: any map between representations that respects the group action automatically respects the character.  The impossibility framework predicts that Langlands functoriality must hold because the Reynolds operator is a natural transformation---a structural consequence of the bilemma, not an additional axiom.""",
"""The \\emph{Reynolds naturality} theorem (\\texttt{reynolds\\_naturality} in \\texttt{UncertaintyFromSymmetry.lean}) proves that equivariant maps commute with Reynolds projections: if $\\varphi : V \\to W$ intertwines the group actions ($R_W \\circ \\varphi = \\varphi \\circ R_V$), then the stable resolutions are compatible.  We deliberately do \\emph{not} phrase this as ``predicting Langlands functoriality'': it is the elementary statement that a natural transformation commutes with a group average, carries no arithmetic content, and does not bear on the automorphic-to-Galois correspondence the Langlands programme concerns.  The formal resemblance is noted only to orient readers from representation theory.""")

rep("Axioms & 2 & Bundled GBDT infrastructure; core theorem uses 0 \\\\",
    "Axioms & 2 & main repo, bundling ${\\sim}14$ GBDT properties (4 total across 3 repos---see above); core uses 0 \\\\")

rep("""Table~\\ref{tab:lean-summary} summarizes the formalization.""",
"""Table~\\ref{tab:lean-summary} summarizes the formalization.

\\paragraph{Trusted-base statement (authoritative; verified by a real build).}
All three repositories compile with \\texttt{lake build} (0 errors) and contain
\\textbf{0}~\\texttt{sorry}/\\texttt{admit}. The trusted base, counted at source, is:
\\begin{itemize}\\itemsep2pt
  \\item \\textbf{Main repository:} \\textbf{2}~\\texttt{axiom} declarations bundling ${\\sim}14$
    gradient-boosting properties; \\textbf{6}~\\texttt{native\\_decide} uses (which import Lean's
    \\texttt{ofReduceBool}); 0~\\texttt{sorry}.
  \\item \\textbf{Attribution companion:} \\textbf{2}~\\texttt{axiom} declarations bundling
    \\textbf{7} fields (of which one is genuinely behavioral); \\textbf{0}~\\texttt{native\\_decide};
    0~\\texttt{sorry}.
  \\item \\textbf{Physics companion:} \\textbf{0}~\\texttt{axiom} declarations---the Selmer~1951
    result and ${\\sim}11$ black-hole/spacetime-emergence properties are carried as
    \\emph{hypotheses / section variables}, not axioms, but still gate their downstream results;
    \\textbf{65}~\\texttt{native\\_decide} uses; 0~\\texttt{sorry}.
\\end{itemize}
So \\textbf{4}~\\texttt{axiom} declarations in total (2+2+0); bundling changes the keyword count,
not what is assumed. \\textbf{The core theorem \\texttt{explanation\\_impossibility} and the entire
Tier-A hardened spine use none of these}---they are pure logic from the stated hypotheses (verified
by \\texttt{\\#print axioms}). Theorem counts are reported \\emph{per repository} (main~530;
attribution companion~368, ${\\approx}88\\%$ coinciding with the main repository; physics
companion~482, ${\\approx}12\\%$ coinciding); because the companions overlap we do \\emph{not} sum
them, and the distinct program-wide count is approximately~1{,}000. Results that depend on the
Selmer~1951 result or the black-hole/spacetime hypotheses are conditional on those assumptions.""")

rep("""The Noether permutation test is confirmatory (pre-specified test on a pre-registered prediction).  We label each analysis accordingly throughout the text.""",
"""The Noether permutation test is confirmatory (pre-specified test on a pre-registered prediction).  We label each analysis accordingly throughout the text.  \\emph{Caveat on temporal precedence:} \\texttt{PRE\\_REGISTRATION.md} entered version control in the same commit as the knockout results it governs, so its precedence cannot be independently established from the repository history; and its own primary criterion---capacity $R^2>0.90$ on the \\emph{full} domain set---was not met (the all-domain fit is $R^2\\approx0.60$).  The headline $R^2=0.957$ is the pre-specified well-characterised-group subset, defined by an a-priori criterion but a subset nonetheless; we report the full-set figure alongside it and treat the capacity law as Tier~B (empirical), not as a consequence of the impossibility theorem.""")

# ---- appendix companion-section overclaims (parity with body) ----
rep("""The trace (stable resolution of GL($n$) bilemma) is compatible with natural maps between matrix algebras (\\texttt{LanglandsFunctoriality.lean}).  These are concrete manifestations of Langlands functoriality.""",
"""The trace (stable resolution of the GL($n$) bilemma) is compatible with natural maps between matrix algebras (\\texttt{LanglandsFunctoriality.lean}).  These are elementary trace-compatibility facts; we do \\emph{not} claim they constitute Langlands functoriality, which concerns automorphic representations that are absent from this formalization.""")

rep("""The gap---4 out of 6 CRT-allowed combinations are blocked by automorphicity---is the impossibility-theoretic reading of the global Langlands correspondence.""",
"""The gap---4 out of 6 CRT-allowed combinations are blocked by the group relations---is a structural analogy to the local-global tension in the Langlands programme, not a formalization of the Langlands correspondence.""")

rep("""The discrete Fourier transform on $\\mathbb{Z}/p\\mathbb{Z}$ satisfies $F^4 = p^2 \\cdot \\mathrm{id}$, forcing eigenvalues to have absolute value $\\sqrt{p}$---the local Riemann Hypothesis in spectral form (\\texttt{EnrichmentFunctor.lean}).""",
"""The discrete Fourier transform on $\\mathbb{Z}/p\\mathbb{Z}$ satisfies $F^4 = p^2 \\cdot \\mathrm{id}$, forcing eigenvalues to have absolute value $\\sqrt{p}$---an eigenvalue-modulus fact sometimes called the ``local Riemann Hypothesis'' in spectral form, unrelated to the analytic Riemann Hypothesis (\\texttt{EnrichmentFunctor.lean}).""")

# ---- reframe review-article-triggering language to research-article framing (arXiv CS moderation) ----
rep("""This monograph serves as the definitive reference.  It documents the
complete formal theory (all proofs, Lean~4 verification), the full
empirical validation, 23~cross-domain instantiations, speculative
connections (clearly labeled), five falsified predictions, and the
validated practitioner pipeline.  The associated Nature submission
presents the focused empirical contribution.""",
"""The primary contributions of this paper are original research: the impossibility
theorem and its Lean~4-verified formalization (Part~I), and an empirical
characterization of explanation instability including a pre-registered stable-fact-count
result (Part~II).  For completeness the paper also documents the cross-domain
instantiations (Part~III), clearly-labeled speculative connections (Part~IV), and five
falsified predictions (Part~V).  Material is organized by evidence tier so that the
original, machine-checked results are cleanly separated from exploratory and analogical
content; the latter is included for transparency, not as established claims.  A focused
version of the empirical contribution is presented separately.""")

# ---- fix the double-counting summation line (verified: companions overlap the main repo) ----
rep("""The theoretical framework is a pure theorem of logic.  Across three repositories the totals are: 530~$+$~358~$+$~482~$=$ 1{,}370~theorems, 4~axioms, 200~files, 0~\\texttt{sorry}.""",
"""The theoretical framework is a pure theorem of logic.  We report counts \\emph{per repository}
(main~530, attribution companion~368, physics companion~482; 4~\\texttt{axiom} declarations in
total, 2+2+0; 0~\\texttt{sorry}; all three build with \\texttt{lake build}) and do \\emph{not} sum
them, because the attribution companion overlaps the main repository by ${\\approx}88\\%$ of theorem
names and the physics companion by ${\\approx}12\\%$; the distinct program-wide count is
approximately~1{,}000.""")

# ---- F1: appendix instance listing was wholesale stale (names/systems/files did not
# ---- exist in the repo; the real objects are *_constructive over *SystemConstructive/
# ---- *SystemC in *InstanceConstructive.lean). Replace intro + all 9 entries + prose cites.
rep("""Each of the nine instances follows the same pattern: axiomatize the domain types,
bundle them into an \\texttt{ExplanationSystem}, and apply
\\texttt{explanation\\_impossibility}.  We list the theorem statements here.

\\begin{lstlisting}[caption={Nine instance impossibility theorems}]""",
"""Each of the nine instances follows the same pattern: construct explicit
finite witness configurations, bundle them into an
\\texttt{ExplanationSystem} whose Rashomon witness is checked by
\\texttt{decide}/\\texttt{rfl} (zero axioms), and apply
\\texttt{explanation\\_impossibility}.  We list the theorem statements as
they appear in the repository (\\texttt{*InstanceConstructive.lean}).

\\begin{lstlisting}[caption={Nine constructive instance impossibility theorems}]""")

rep("""-- Attribution (AttributionInstance.lean)
theorem attribution_impossibility_abstract
    (E : AttrConfig -> AttrExplanation)
    (hf : faithful attrSystem E)
    (hs : stable attrSystem E)
    (hd : decisive attrSystem E) : False :=
  explanation_impossibility attrSystem E hf hs hd""",
"""-- Attribution (AttributionInstanceConstructive.lean)
theorem attribution_impossibility_constructive :
    forall (E : AttrConfigC -> AttrRanking),
      faithful attrSystemC E -> stable attrSystemC E ->
      decisive attrSystemC E -> False :=
  explanation_impossibility attrSystemC""")

rep("""-- Attention (AttentionInstance.lean)
theorem attention_impossibility
    (E : AttentionConfig -> AttentionMap)
    (hf : faithful attentionSystem E)
    (hs : stable attentionSystem E)
    (hd : decisive attentionSystem E) : False :=
  explanation_impossibility attentionSystem E hf hs hd""",
"""-- Attention (AttentionInstanceConstructive.lean)
theorem attention_impossibility_constructive
    (E : AttnConfig -> AttnArgmax)
    (hf : faithful attentionSystemConstructive E)
    (hs : stable attentionSystemConstructive E)
    (hd : decisive attentionSystemConstructive E) : False :=
  explanation_impossibility attentionSystemConstructive E hf hs hd""")

rep("""-- Counterfactual (CounterfactualInstance.lean)
theorem counterfactual_impossibility
    (E : CFConfig -> CFExplanation)
    (hf : faithful cfSystem E)
    (hs : stable cfSystem E)
    (hd : decisive cfSystem E) : False :=
  explanation_impossibility cfSystem E hf hs hd""",
"""-- Counterfactual (CounterfactualInstanceConstructive.lean)
theorem counterfactual_impossibility_constructive
    (E : CFConfig -> CFDirection)
    (hf : faithful cfSystemConstructive E)
    (hs : stable cfSystemConstructive E)
    (hd : decisive cfSystemConstructive E) : False :=
  explanation_impossibility cfSystemConstructive E hf hs hd""")

rep("""-- Concept Probe (ConceptInstance.lean)
theorem concept_impossibility
    (E : ConceptConfig -> ConceptExplanation)
    (hf : faithful conceptSystem E)
    (hs : stable conceptSystem E)
    (hd : decisive conceptSystem E) : False :=
  explanation_impossibility conceptSystem E hf hs hd""",
"""-- Concept Probe (ConceptInstanceConstructive.lean)
theorem concept_impossibility_constructive
    (E : ConceptCfg -> ConceptDirection)
    (hf : faithful conceptSystemConstructive E)
    (hs : stable conceptSystemConstructive E)
    (hd : decisive conceptSystemConstructive E) : False :=
  explanation_impossibility conceptSystemConstructive E hf hs hd""")

rep("""-- Causal Discovery (CausalInstance.lean)
theorem causal_instance_impossibility
    (E : CausalConfig -> CausalExplanation)
    (hf : faithful causalSystem E)
    (hs : stable causalSystem E)
    (hd : decisive causalSystem E) : False :=
  explanation_impossibility causalSystem E hf hs hd""",
"""-- Causal Discovery (CausalInstanceConstructive.lean)
theorem causal_impossibility_constructive :
    forall (E : CausalConfigC -> EdgeOrientation),
      faithful causalSystemC E -> stable causalSystemC E ->
      decisive causalSystemC E -> False :=
  explanation_impossibility causalSystemC""")

rep("""-- Model Selection (ModelSelectionInstance.lean)
theorem model_selection_instance_impossibility
    (E : MSConfig -> MSExplanation)
    (hf : faithful msSystem E)
    (hs : stable msSystem E)
    (hd : decisive msSystem E) : False :=
  explanation_impossibility msSystem E hf hs hd""",
"""-- Model Selection (ModelSelectionInstanceConstructive.lean)
theorem model_selection_impossibility_constructive
    (E : MSCfg -> MSModelId)
    (hf : faithful msSystemConstructive E)
    (hs : stable msSystemConstructive E)
    (hd : decisive msSystemConstructive E) : False :=
  explanation_impossibility msSystemConstructive E hf hs hd""")

rep("""-- GradCAM (SaliencyInstance.lean)
theorem saliency_impossibility
    (E : SaliencyConfig -> SaliencyMap)
    (hf : faithful saliencySystem E)
    (hs : stable saliencySystem E)
    (hd : decisive saliencySystem E) : False :=
  explanation_impossibility saliencySystem E hf hs hd""",
"""-- GradCAM (SaliencyInstanceConstructive.lean)
theorem saliency_impossibility_constructive
    (E : SaliencyCfg -> SaliencyRegion)
    (hf : faithful saliencySystemConstructive E)
    (hs : stable saliencySystemConstructive E)
    (hd : decisive saliencySystemConstructive E) : False :=
  explanation_impossibility saliencySystemConstructive E hf hs hd""")

rep("""-- LLM Self-Explanation (LLMExplanationInstance.lean)
theorem llm_explanation_impossibility
    (E : LLMConfig -> LLMExplanation)
    (hf : faithful llmSystem E)
    (hs : stable llmSystem E)
    (hd : decisive llmSystem E) : False :=
  explanation_impossibility llmSystem E hf hs hd""",
"""-- LLM Self-Explanation (LLMExplanationInstanceConstructive.lean)
theorem llm_explanation_impossibility_constructive
    (E : LLMCfg -> LLMCitation)
    (hf : faithful llmSystemConstructive E)
    (hs : stable llmSystemConstructive E)
    (hd : decisive llmSystemConstructive E) : False :=
  explanation_impossibility llmSystemConstructive E hf hs hd""")

rep("""-- Mechanistic Interpretability (MechInterpInstance.lean)
theorem mech_interp_impossibility
    (E : MechInterpConfig -> Circuit)
    (hf : faithful mechInterpSystem E)
    (hs : stable mechInterpSystem E)
    (hd : decisive mechInterpSystem E) : False :=
  explanation_impossibility mechInterpSystem E hf hs hd""",
"""-- Mechanistic Interpretability (MechInterpInstanceConstructive.lean)
theorem mech_interp_impossibility_constructive
    (E : MechInterpCfg -> CircuitDecomp)
    (hf : faithful mechInterpSystemConstructive E)
    (hs : stable mechInterpSystemConstructive E)
    (hd : decisive mechInterpSystemConstructive E) : False :=
  explanation_impossibility mechInterpSystemConstructive E hf hs hd""")

# F1 prose citations of the stale names (appendix proof sketches, Instances 1-6)
rep("""\\texttt{attribution\\_impossibility} in \\texttt{Trilemma.lean}
(0~axioms);
\\texttt{attribution\\_impossibility\\_abstract} in
\\texttt{AttributionInstance.lean}
(0~additional axioms beyond system).\\par}""",
"""\\texttt{attribution\\_impossibility} in \\texttt{Trilemma.lean}
(0~axioms);
\\texttt{attribution\\_impossibility\\_constructive} in
\\texttt{AttributionInstanceConstructive.lean}
(constructive witness; 0~axioms).\\par}""")

rep("""\\texttt{attention\\_impossibility} in \\texttt{AttentionInstance.lean}.""",
"""\\texttt{attention\\_impossibility\\_constructive} in \\texttt{AttentionInstanceConstructive.lean}.""")

rep("""\\texttt{counterfactual\\_impossibility} in \\texttt{Counterfactual\\-Instance.lean}.""",
"""\\texttt{counterfactual\\_impossibility\\_constructive} in \\texttt{Counterfactual\\-Instance\\-Constructive.lean}.""")

rep("""\\texttt{concept\\_impossibility} in \\texttt{ConceptInstance.lean}.""",
"""\\texttt{concept\\_impossibility\\_constructive} in \\texttt{ConceptInstanceConstructive.lean}.""")

rep("""\\texttt{causal\\_instance\\_impossibility} in
\\texttt{CausalInstance.lean};""",
"""\\texttt{causal\\_impossibility\\_constructive} in
\\texttt{CausalInstanceConstructive.lean};""")

rep("""\\texttt{model\\_selection\\_instance\\_impossibility}
in \\texttt{ModelSelectionInstance.lean};""",
"""\\texttt{model\\_selection\\_impossibility\\_constructive}
in \\texttt{ModelSelectionInstanceConstructive.lean};""")

# ---- F2a: capacity bound cited a tautology ((hu : R u = u) : R u = u). Downgrade to
# ---- what is actually established; flag the general claim as Tier-B conjecture.
rep("""\\emph{Part (ii): Capacity bound.}  Any $G$-invariant map $E$ satisfies $E(\\theta) \\in V^G$ for all $\\theta$; the dimension of the image is at most $C = \\dim(V^G)$.  No amount of averaging, ensembling, or methodological refinement expands $V^G$.  The capacity is a structural constant (Lean-verified: \\texttt{stable\\_in\\_fixed\\_subspace}).""",
"""\\emph{Part (ii): Capacity bound.}  The orbit-averaged (Reynolds-projected) explanation $R \\circ E$ has image in $V^G$---immediate from idempotency of $R$---so its image dimension is at most $C = \\dim(V^G)$, and no amount of averaging, ensembling, or methodological refinement expands $V^G$.  The stronger statement that \\emph{every} stable map has image in $V^G$ is stated here as a conjecture [Tier~B], not a Lean-verified theorem: the declaration \\texttt{stable\\_in\\_fixed\\_subspace} records only the definitional fact that a fixed point of $R$ lies in $V^G$ (it takes $Ru = u$ as a hypothesis) and does not derive fixed-point-ness from stability.""")

# ---- F2b: dash_unique_pareto_optimal is within-group only, conditional on GBDT/DGP
# ---- hypotheses; global Pareto optimality is argued, not Lean-verified. Soften prose.
rep("""across seven domains), a unique Pareto-optimal resolution (orbit
averaging, a consequence of classical invariant decision theory), and a""",
"""across seven domains), a canonical resolution (orbit averaging;
Pareto-optimal \\emph{within} correlation groups by a Lean-verified
theorem, and globally by a classical invariant-decision-theory argument
not yet mechanized), and a""")

rep("""rankings disagree across the ensemble.  The companion paper proves that
\\DASH{} is Pareto-optimal: no other stable attribution method can achieve
higher expected faithfulness.""",
"""rankings disagree across the ensemble.  The companion paper argues that
\\DASH{} is Pareto-optimal (no other stable attribution method can achieve
higher expected faithfulness); the within-group component of this claim is
Lean-verified, the global claim is not (see the proof-status inventory).""")

rep("""    not been mechanized in Lean~4.  (Note: as of the current version,
    the Pareto optimality of DASH is now Lean-verified in
    \\texttt{ParetoOptimality.lean} via
    \\texttt{dash\\_unique\\_pareto\\_optimal}.)""",
"""    not been mechanized in Lean~4.  (Note: the \\emph{within-group}
    component of \\DASH{}'s Pareto optimality is Lean-verified in
    \\texttt{ParetoOptimality.lean} via
    \\texttt{dash\\_unique\\_pareto\\_optimal}: a committed within-group
    ranking has strictly positive disagreement, conditional on the
    bundled GBDT/DGP hypotheses.  The between-group case---and hence
    \\emph{global} Pareto optimality---remains argued, resting on an
    unformalized Cram\\'er--Rao step.)""")

# ---- F3: mi_is_exact_boundary proves forward direction only, under an assumed bridge
# ---- hypothesis. "Necessary and sufficient" -> "sufficient, conditional".
rep("""Note: the Lean-verified generalization (\\texttt{mi\\_is\\_exact\\_boundary} in \\texttt{MutualInformation.lean}) proves that mutual information $I(X_j; X_k) > 0$---not correlation---is the necessary and sufficient condition for the attribution impossibility.""",
"""Note: the Lean theorem \\texttt{mi\\_is\\_exact\\_boundary} in \\texttt{MutualInformation.lean} proves the \\emph{sufficiency} direction---mutual information $I(X_j; X_k) > 0$ (not correlation) yields the attribution impossibility---and does so \\emph{conditional} on the bridge hypothesis \\texttt{hdep\\_implies\\_diff} (statistical dependence produces differing attributions), which is assumed rather than derived; the converse direction is argued informally, not machine-checked.""")

rep("""(identified here via correlation; the exact boundary is $I > 0$).  Within-group comparisons should be""",
"""(identified here via correlation; the operative dependence condition is $I > 0$).  Within-group comparisons should be""")

rep("""whether linearly correlated or not---the exact boundary is $I(X_j; X_k) > 0$""",
"""whether linearly correlated or not---the operative dependence condition is $I(X_j; X_k) > 0$""")

# ---- F4: companion physics lines to match the honest body (NS = tightness schema,
# ---- no PDEs; adelic resolution = placeholder with resolution := fun _ => True).
rep("""The adele ring is constructed as a concrete definition with identity embedding (\\texttt{AdelicResolution.lean}).  The adelic projection is proved $G$-invariant (\\texttt{adelic\\_projection\\_invariant}) and resolves the physics bilemma (\\texttt{adelic\\_resolves}).  This parallels DASH for attributions and CPDAG for causal discovery as instances of the same abstract $G$-invariant resolution.""",
"""The ``adelic resolution'' here is a placeholder model, not a formalization of the adele ring: the underlying type is a finite enumeration of completions with an identity embedding, and the resolution witnessing \\texttt{adelic\\_resolves} is literally the constant predicate \\texttt{fun \\_ => True} (\\texttt{AdelicResolution.lean}).  It records the \\emph{intended} parallel---commit to all completions simultaneously, as DASH does for attributions and CPDAG for causal discovery---but, unlike those two instances, nothing nontrivial is proved: the real adele ring and its symmetry action are not formalised.  We flag this as the weakest companion result [Tier~C].""")

rep("""The 3D Navier--Stokes equations are formalised as a conditional \\texttt{AbstractImpossibility}: without regularity, full tightness; with regularity, all three properties are achievable (\\texttt{NavierStokesImpossibility.lean}).  The Reynolds dichotomy (\\texttt{ns\\_reynolds\\_dichotomy}) proves: below the critical Reynolds number, all three properties \\{smooth, energy-conserving, global\\} are jointly achievable; above, the system becomes an \\texttt{AbstractImpossibility}.  The 54~numerical experiments (Section~\\ref{sec:ns-experiments}) validate this classification experimentally.""",
"""The Navier--Stokes trilemma is modelled as a three-strategy tightness \\emph{schema}---a finite case analysis over the three classical compromise strategies (Leray weak, local classical, viscosity), with no PDEs formalised: without regularity the schema has full tightness; with regularity all three properties are achievable (\\texttt{NavierStokesImpossibility.lean}).  The Reynolds dichotomy (\\texttt{ns\\_reynolds\\_dichotomy}) encodes the same case split parametrically---below a critical Reynolds number all three properties \\{smooth, energy-conserving, global\\} are jointly achievable; above, the schema becomes an \\texttt{AbstractImpossibility}---as bookkeeping over assumed regime classifications, not as fluid dynamics.  The 54~numerical experiments (Section~\\ref{sec:ns-experiments}) probe this classification experimentally.""")

rep("""multi-analyst aggregation, 3D Navier--Stokes conditional tightness, the DPRM trilemma""",
"""multi-analyst aggregation, the Navier--Stokes tightness schema, the DPRM trilemma""")

# ---- F5 residual: "0 axioms" for the physics companion needs the hypotheses rider
# ---- wherever it appears, or it reads as bookkeeping-gamed (assumed content unchanged).
rep("""(38~files, 482~theorems, 0~axioms, 0~\\texttt{sorry}).
All code is released under the Apache~2.0 licence.""",
"""(38~files, 482~theorems, 0~\\texttt{axiom} declarations with ${\\sim}11$ domain
hypotheses carried as section variables, 0~\\texttt{sorry}).
All code is released under the Apache~2.0 licence.""")

rep("""This section documents results from the companion repository \\texttt{ostrowski-impossibility} (482~theorems, 0~axioms, 38~files, 0~\\texttt{sorry}).""",
"""This section documents results from the companion repository \\texttt{ostrowski-impossibility} (482~theorems; 0~\\texttt{axiom} declarations, with ${\\sim}11$ domain hypotheses---including the Selmer~1951 result and the contested/speculative black-hole and spacetime-emergence properties---carried as section variables that gate their downstream results; 38~files; 0~\\texttt{sorry}).""")

# ---- count drift: dash companion is 59 files / 368 theorems+lemmas at HEAD 7ec3ef9
# ---- (canonical grep count; the 58/358 figures predate the last dash commits).
rep("""(58~files, 358~theorems, 2~axioms, 0~\\texttt{sorry}).""",
"""(59~files, 368~theorems, 2~axioms, 0~\\texttt{sorry}).""")

# ---- Rashomon-necessity biconditional: proved in general only as (a) sufficiency,
# ---- (b) fully-specified converse, (c) full iff for incompatibility = inequality
# ---- (rashomon_biconditional_neq). Scope the three unqualified-iff prose passages.
rep("""The converse is proved in Lean (\\texttt{fully\\_specified\\_possibility} in
\\texttt{Necessity.lean}): if the system has \\emph{no} Rashomon
property---every observation uniquely determines its configuration---then
all three properties are simultaneously achievable.  The Rashomon property
is therefore both sufficient \\emph{and} necessary for the impossibility.
This is not a vague ``things are hard when the problem is hard'' statement;
it is a precise logical equivalence: the impossibility holds \\emph{if and
only if} the Rashomon property holds.""",
"""Two converses are proved in Lean.  For \\emph{fully specified} systems
(\\texttt{fully\\_specified\\_possibility} in \\texttt{Necessity.lean}): if every
observation uniquely determines its configuration, all three properties are
simultaneously achievable.  And for systems whose incompatibility relation is
inequality---the maximal-incompatibility class covering the bilemma
instances---the full equivalence is machine-checked
(\\texttt{rashomon\\_biconditional\\_neq} in
\\texttt{NecessityBiconditional.lean}): the three properties are jointly
achievable \\emph{if and only if} the Rashomon property fails.  For general
incompatibility relations the unqualified biconditional is not
machine-checked; what is proved is sufficiency (Rashomon $\\Rightarrow$
impossibility) plus the two converses above.  Within its proved scope this is
a precise logical equivalence, not a vague ``things are hard when the problem
is hard'' statement.""")

rep("""\\emph{precision}: the axiom set is tight (each pair of properties is
achievable), necessary and sufficient (the biconditional), and quantitatively
productive (the counting theorem and capacity theorem).""",
"""\\emph{precision}: the axiom set is tight (each pair of properties is
achievable), equipped with machine-checked necessity converses (the full
biconditional for the maximal-incompatibility class; the fully-specified
converse in general), and quantitatively
productive (the counting theorem and capacity theorem).""")

rep("""    axiomatic framework---tightness (each pair achievable), necessity
    (biconditional with Rashomon), and the quantitative corollaries""",
"""    axiomatic framework---tightness (each pair achievable), necessity
    (biconditional with Rashomon, machine-checked for
    inequality-incompatibility; sufficiency plus the fully-specified
    converse in general), and the quantitative corollaries""")

# ---- F6: stale citations (ostrowski_classify is a noncomputable def; Selmer 1951 is
# ---- now a hypothesis parameter, not an axiom declaration).
rep("""The bridge theorem \\texttt{ostrowski\\_classification} (\\texttt{OstrowskiFramework.lean}) connects the abstract impossibility to the concrete number-theoretic structure.""",
"""The bridge \\texttt{ostrowski\\_classify} (a \\texttt{noncomputable def} in \\texttt{OstrowskiFramework.lean}, built on Mathlib's \\texttt{Rat.AbsoluteValue.equiv\\_real\\_or\\_padic}) connects the abstract impossibility to the concrete number-theoretic structure.""")

rep("""(1~axiom: \\texttt{selmer\\_no\\_nontrivial\\_solution}, Selmer~1951)""",
"""(assumed as the explicit hypothesis \\texttt{selmer\\_no\\_nontrivial\\_solution}---Selmer's 1951 theorem---rather than proved; it is a hypothesis parameter, not an \\texttt{axiom} declaration)""")

text=text.replace('\\textquotesingle',"'")
open(OUT,'w').write(text)

# ---- verification ----
out_lines=text.split('\n')
orig=lines
orig_nonhdr=collections.Counter(l for l in orig[starts[0]-1:bib-1] if not(l.startswith('\\section{') or l.startswith('\\subsection{')))
# account for the 8 replacements changing some non-header lines; verify no ORIGINAL block content line lost
# (headers changed; a handful of prose lines changed by rep()). Check block-level: all 77 blocks' non-first lines present modulo edited ones.
missing=0
for idx in range(77):
    s,e=ranges[idx]
    for l in lines[s+1:e]:
        if l and l not in text and l not in ('',):
            missing+=1
print('blocks:',len(order[1])+len(order[2])+len(order[3])+len(order[4])+len(order[5])+2,'/77')
print('parts:',text.count('\\part{'),'| promoted subs->sec:',sum(1 for l in out_lines if l.startswith('\\section{'))-19-2-1)  # rough
be=collections.Counter(__import__('re').findall(r'\\begin\{(\w+\*?)\}',text))
en=collections.Counter(__import__('re').findall(r'\\end\{(\w+\*?)\}',text))
print('unbalanced envs:',{k:(be[k],en[k]) for k in set(be)|set(en) if be[k]!=en[k]} or 'NONE')
print('block content lines not found verbatim in output (edited prose expected small):',missing)
print('textquotesingle left:',text.count('\\textquotesingle'))
print('lines out/orig:',len(out_lines),len(orig))
