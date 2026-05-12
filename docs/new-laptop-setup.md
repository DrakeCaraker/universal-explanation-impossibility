# New Laptop Setup — Drake Caraker's Claude Code Environment

## 1. System Dependencies

```bash
# Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Core tools
brew install git node claude-code

# Python (system python3 is fine on macOS, or install via brew)
# Key: use pip3 --user for packages

# Node (if you prefer nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
nvm install 20

# Lean 4 (elan)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# LaTeX (TeX Live)
brew install --cask mactex
```

## 2. Python Packages

```bash
pip3 install --user \
  numpy==1.26.4 scipy==1.13.1 pandas==2.3.3 \
  scikit-learn matplotlib==3.9.4 \
  xgboost==2.1.4 shap==0.49.1 \
  pmlb openml seaborn
```

## 3. Clone Repos

```bash
mkdir -p ~/ds_projects && cd ~/ds_projects
git clone https://github.com/DrakeCaraker/universal-explanation-impossibility.git
git clone https://github.com/DrakeCaraker/ostrowski-impossibility.git
git clone https://github.com/DrakeCaraker/dash-impossibility-lean.git
git clone https://github.com/DrakeCaraker/dash-shap.git
git clone https://github.com/DrakeCaraker/alfred.git
```

## 4. Lean Setup (per repo)

```bash
cd ~/ds_projects/universal-explanation-impossibility && lake build
cd ~/ds_projects/dash-impossibility-lean && lake build
cd ~/ds_projects/ostrowski-impossibility && lake build
```

## 5. Claude Code Settings

```bash
mkdir -p ~/.claude

cat > ~/.claude/settings.json << 'SETTINGS'
{
  "permissions": {
    "allow": [
      "Read",
      "Write",
      "Edit",
      "Glob",
      "Grep",
      "Agent",
      "Bash(*)",
      "Skill(*)",
      "WebFetch(*)",
      "WebSearch",
      "NotebookEdit"
    ]
  },
  "model": "opus[1m]",
  "enabledPlugins": {
    "superpowers@claude-plugins-official": true,
    "alfred@alfred-marketplace": false
  },
  "extraKnownMarketplaces": {
    "alfred-marketplace": {
      "source": {
        "source": "git",
        "url": "https://github.com/DrakeCaraker/alfred.git"
      }
    }
  }
}
SETTINGS

cat > ~/.claude/settings.local.json << 'LOCAL'
{
  "permissions": {
    "allow": [
      "Bash(git:*)",
      "Bash(gh:*)",
      "Bash(cd:*)",
      "Bash(ls:*)",
      "Bash(find:*)",
      "Bash(mkdir:*)",
      "Bash(chmod:*)",
      "Bash(cat:*)",
      "Bash(head:*)",
      "Bash(tail:*)",
      "Bash(wc:*)",
      "Bash(diff:*)",
      "Bash(sort:*)",
      "Bash(python3:*)",
      "Bash(pip:*)",
      "Bash(pip3:*)",
      "Bash(pytest:*)",
      "Bash(make:*)",
      "Bash(bash:*)",
      "Bash(ruff:*)",
      "Bash(aws:*)",
      "Bash(jq:*)",
      "Bash(grep:*)",
      "Bash(mv:*)",
      "Bash(cp:*)",
      "Bash(echo:*)",
      "Bash(touch:*)",
      "Bash(which:*)",
      "Bash(command:*)",
      "Bash(date:*)",
      "Bash(stat:*)",
      "Bash(du:*)",
      "Bash(env:*)",
      "Bash(sed:*)",
      "Bash(awk:*)",
      "Bash(tr:*)",
      "Bash(cut:*)",
      "Bash(tee:*)",
      "Bash(xargs:*)",
      "Bash(pgrep:*)",
      "Bash(lake:*)",
      "Bash(pdflatex:*)",
      "Bash(bibtex:*)",
      "Bash(latexmk:*)",
      "Bash(texcount:*)",
      "Bash(for:*)",
      "Bash(do:*)",
      "Bash(conda:*)",
      "Bash(npm:*)",
      "Skill(update-config)",
      "WebSearch",
      "WebFetch(domain:arxiv.org)"
    ],
    "deny": [
      "Bash(rm -rf /)",
      "Bash(rm -rf ~)"
    ]
  }
}
LOCAL
```

## 6. Install Plugins

```bash
claude plugin install superpowers@claude-plugins-official
# Alfred (optional — currently disabled):
# claude plugin install alfred@alfred-marketplace
```

## 7. Claude Code Memory

```bash
mkdir -p ~/.claude/projects/-Users-$(whoami)/memory

# MEMORY.md index
cat > ~/.claude/projects/-Users-$(whoami)/memory/MEMORY.md << 'EOF'
# Memory Index

## User
- [drake-profile](user_drake.md) — Drake Caraker, data scientist at Oura, systems thinker, builds Alfred and dash-shap

## Projects
- [alfred-project](project_alfred.md) — Claude Code plugin for development habit teaching, encrypted collective learning, 7-layer automation stack
- [knockout-experiments](project_knockout_experiments.md) — Nature knockout prediction results: 3 confirmed (Noether, η, interp ceiling), 2 falsified (phase transition, uncertainty), 1 negative (mol evo)
- [nature-paper-state](project_nature_paper_state.md) — Nature paper "The Limits of Explanation" current state: 5 instances, tightness classification, capacity theorem

## Reference
- [higher-impact-experiments](reference_higher_impact_experiments.md) — GPT-2 IOI, scaling law, LLM self-explanation — higher-impact experiment ideas for Nature paper
- [naming-conventions](reference_naming_conventions.md) — Canonical naming for the explanation impossibility framework (established 2026-04-30)

## Feedback
- [security-requirements-first](feedback_security_requirements_first.md) — Ask about security/access/encryption BEFORE building data transport
- [branch-scope](feedback_branch_scope.md) — Watch for multiple features on one branch, suggest splitting
- [project-directory](feedback_project_directory.md) — Always launch Claude Code from project root for slash commands to work
EOF

# User profile
cat > ~/.claude/projects/-Users-$(whoami)/memory/user_drake.md << 'EOF'
Drake is a data scientist at Oura Ring. Advanced coder — Python/ML/data science, git, CI/CD.

Active projects: Alfred (Claude Code plugin), dash-shap (DASH-SHAP framework), universal-explanation-impossibility (Nature paper), ostrowski-impossibility (FoP paper), dash-impossibility-lean (NeurIPS paper).

Working style: Systems thinker. Uses "ultrathink" for deep analysis with vet passes. Makes fast decisions with A/B/C options. Delegates implementation, owns architecture. Demands audits at checkpoints. Prefers terse interactions. Says "do it" when design is approved.
EOF

# Naming conventions
cat > ~/.claude/projects/-Users-$(whoami)/memory/reference_naming_conventions.md << 'EOF'
# Naming Conventions — The Limits of Explanation

Established 2026-04-30. Canonical reference: docs/naming-conventions.md in each repo.

Key terms (use these, not deprecated versions):
- Explanation capacity (C = dim V^G)
- Explanation loss rate (η = 1 − C/dim V)
- Explanation Capacity Theorem — NOT "Explanation Capacity Law"
- Explanation uncertainty bound — NOT "tradeoff bound" or "uncertainty relation"
- The stable projection — NOT "the explanation code"
- Stable fact count — NOT "Noether counting law"
- Explanatory information loss — NOT "Pythagorean decomposition" as prose name
- Over-explanation penalty — NOT "beyond-capacity penalty"
- Explanation Stability Theorem (4-part theorem) — NOT "Explanation Coding Theorem"
- Explanation stability convergence rate
- Stability threshold (M*(ε))
EOF
```

## 8. Verify Setup

```bash
# Lean
cd ~/ds_projects/universal-explanation-impossibility && lake env lean UniversalImpossibility/ExplanationSystem.lean

# LaTeX
cd paper && pdflatex -interaction=nonstopmode nature_article.tex

# Python + packages
python3 -c "import xgboost, shap, sklearn, scipy, numpy, pmlb; print('All packages OK')"

# Claude Code
claude --version
```

## 9. Current State Summary

- Nature paper: ~3400 words, 5 instances (genomics, mech interp, quantum contextuality, social choice, 3D Navier-Stokes), compiles clean
- Monograph: 128 pages, compiles clean
- Lean: 519 theorems (universal) + 357 (attribution) + 482 (ostrowski) = 1,358 total, 0 sorry
- Audit: 149 datasets, 53 domains, 27:1 confirmation ratio
- All repos on main, pushed to GitHub
