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
"""The framework is mechanically verified in Lean~4 with \\textbf{0}~\\texttt{sorry}.
Its trusted base is \\textbf{2}~\\texttt{axiom} declarations that bundle
\\textbf{25}~behavioral assumptions (reported unbundled as~25 for transparency),
plus \\texttt{native\\_decide}-induced \\texttt{ofReduceBool} dependencies flagged
per theorem; the core theorem and the entire hardened spine (Part~I) use \\emph{none}
of these. Theorem counts are reported per repository and are not summed across the
overlapping companion repositories (see Part~I).""")

rep("""\\paragraph{Reynolds naturality predicts Langlands functoriality.}
The Reynolds naturality theorem (\\texttt{reynolds\\_naturality} in \\texttt{UncertaintyFromSymmetry.lean}) proves that equivariant maps commute with Reynolds projections.  For GL($n$), the Reynolds projection is the trace (conjugation-averaging).  Reynolds naturality therefore implies that the trace commutes with group homomorphisms---which IS Langlands functoriality for finite fields.  The impossibility framework predicts functoriality as a structural consequence of the bilemma's resolution: collapsed tightness for $n \\geq 2$ forces the trace as the unique Pareto-optimal resolution; Reynolds naturality forces the trace to be functorial.  The Langlands programme classifies which characters arise from automorphic representations; the impossibility framework provides a structural explanation for why characters are the natural invariant.""",
"""\\paragraph{Reynolds naturality and the character: a structural analogy, not a Langlands result.}\\emph{~[Tier~C.]}
The Reynolds naturality theorem (\\texttt{reynolds\\_naturality} in \\texttt{UncertaintyFromSymmetry.lean}) proves that equivariant maps commute with Reynolds projections, and for GL($n$) the Reynolds projection is the trace (conjugation-averaging).  This is a one-line fact of invariant theory: the trace is a class function, constant on conjugacy classes and compatible with the group action.  We flag explicitly that this is \\emph{not} Langlands functoriality: the formalization contains no automorphic forms, no $L$-functions, and no automorphic-to-Galois correspondence.  The parallel is purely structural---characters/traces are the natural stable invariant under conjugation, just as the orbit average is the stable resolution under any symmetry group---and is stated only to orient readers familiar with representation theory. It is not a theorem about, or a prediction of, functoriality.""")

rep("""The \\emph{Reynolds naturality} theorem (\\texttt{reynolds\\_naturality} in \\texttt{UncertaintyFromSymmetry.lean}) proves that equivariant maps commute with Reynolds projections: if $\\varphi : V \\to W$ intertwines the group actions ($R_W \\circ \\varphi = \\varphi \\circ R_V$), then the stable resolutions are compatible.  In Langlands terms, this is functoriality: any map between representations that respects the group action automatically respects the character.  The impossibility framework predicts that Langlands functoriality must hold because the Reynolds operator is a natural transformation---a structural consequence of the bilemma, not an additional axiom.""",
"""The \\emph{Reynolds naturality} theorem (\\texttt{reynolds\\_naturality} in \\texttt{UncertaintyFromSymmetry.lean}) proves that equivariant maps commute with Reynolds projections: if $\\varphi : V \\to W$ intertwines the group actions ($R_W \\circ \\varphi = \\varphi \\circ R_V$), then the stable resolutions are compatible.  We deliberately do \\emph{not} phrase this as ``predicting Langlands functoriality'': it is the elementary statement that a natural transformation commutes with a group average, carries no arithmetic content, and does not bear on the automorphic-to-Galois correspondence the Langlands programme concerns.  The formal resemblance is noted only to orient readers from representation theory.""")

rep("Axioms & 2 & Bundled GBDT infrastructure; core theorem uses 0 \\\\",
    "Axioms & 2 & 2 declarations bundling 25 assumptions; core uses 0 \\\\")

rep("""Table~\\ref{tab:lean-summary} summarizes the formalization.""",
"""Table~\\ref{tab:lean-summary} summarizes the formalization.

\\paragraph{Trusted-base statement (authoritative).}
The framework is mechanically verified in Lean~4 with \\textbf{0}~\\texttt{sorry}/\\texttt{admit}
across three repositories. The trusted base is \\textbf{2}~\\texttt{axiom} declarations that
\\emph{bundle} 25 behavioral assumptions---gradient-boosting infrastructure and, in the physics
companion, contested black-hole/spacetime-emergence hypotheses plus the classical Selmer~(1951)
result---reported unbundled as \\textbf{25} for transparency (bundling changes the count of
\\texttt{axiom} keywords, not what is assumed). In addition, some theorems depend on Lean's
\\texttt{Lean.ofReduceBool} axiom via \\texttt{native\\_decide} (6~uses in the main repository,
65 in the physics companion, 0 in the attribution companion), flagged per theorem.
\\textbf{The core theorem \\texttt{explanation\\_impossibility} and the entire Tier-A hardened
spine use none of these}---they are pure logic from the stated hypotheses (verified by
\\texttt{\\#print axioms}). Theorem counts are reported \\emph{per repository} (main~530;
attribution companion~358, of which ${\\approx}88\\%$ coincide with the main repository; physics
companion~482, of which ${\\approx}12\\%$ coincide). Because the companions overlap the main
repository we do \\emph{not} sum them; the distinct program-wide count is approximately~1{,}000.""")

rep("""The Noether permutation test is confirmatory (pre-specified test on a pre-registered prediction).  We label each analysis accordingly throughout the text.""",
"""The Noether permutation test is confirmatory (pre-specified test on a pre-registered prediction).  We label each analysis accordingly throughout the text.  \\emph{Caveat on temporal precedence:} \\texttt{PRE\\_REGISTRATION.md} entered version control in the same commit as the knockout results it governs, so its precedence cannot be independently established from the repository history; and its own primary criterion---capacity $R^2>0.90$ on the \\emph{full} domain set---was not met (the all-domain fit is $R^2\\approx0.60$).  The headline $R^2=0.957$ is the pre-specified well-characterised-group subset, defined by an a-priori criterion but a subset nonetheless; we report the full-set figure alongside it and treat the capacity law as Tier~B (empirical), not as a consequence of the impossibility theorem.""")

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
