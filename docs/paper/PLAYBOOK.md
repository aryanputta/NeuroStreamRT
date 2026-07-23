# Paper Playbook — build the project, then write the paper to learn it

The loop you are running on every project: **build → measure honestly → explain
→ write → critique → repeat.** The paper is not paperwork; it is the forcing
function that makes you actually understand what you built. Writing the method
section is how you find out whether you understand the method.

This playbook is model-agnostic: use Claude Opus 4.8 or Codex 5.5 as the
co-author. Two roles, used in this order for every section:
- **Tutor** (learn): make the model explain the concept from first principles
  until you can re-derive it.
- **Drafter** (write): make the model draft the section *from your actual code
  and numbers*, never from generic knowledge.
- **Critic** (review): make the model attack the draft for overclaiming and gaps.

## The one rule that makes the papers good

**Report where the method loses as carefully as where it wins.** A negative
result that isolates the mechanism ("compression beats eviction *only* when the
cold region is redundant") is worth more than a vague win, teaches you more, and
survives review. If you cannot yet run the decisive experiment, say so and name
it; do not imply you ran it.

## Workflow per project

0. **Finish a runnable slice of the project** with a benchmark harness that
   prints real numbers (compression, latency, accuracy, fidelity — whatever the
   project optimizes). The paper draws only from this.
1. **Scaffold:** `python3 ~/paper-kit/scaffold.py --project . --title "..." --slug <name>`
2. Fill sections **in this order**: Method → Results → Setup → Related →
   Intro → Abstract → Limitations. (Write what you know cold first; the abstract
   is last because it summarizes what you actually found.)
3. For each section run Tutor → Drafter → Critic (prompts below).
4. **Build:** `cd docs/paper && ./build.sh <name>` then open the PDF.
5. Update the project's Brain page + log with what you learned, not just what you did.

## Section-by-section prompts

Paste these to Opus/Codex. Replace [PROJECT] and attach/point at the relevant
code file or benchmark output.

### Method
- Tutor: "Explain the core algorithm in [file] from first principles. Derive any
  formula step by step, and give me one tiny worked numeric example. Then ask me
  two questions to check I understand it."
- Drafter: "Write the Method section. Define notation first, then number the
  steps so each maps 1:1 to the code in [file]. Put the single key design choice
  in its own paragraph and justify it physically (memory, compute, or
  statistics), not hand-wavily."
- Critic: "Find every place the Method claims something the code in [file] does
  not actually do. List mismatches with line refs."

### Results
- Tutor: "Given this benchmark output [paste], what is the controlled variable,
  and what comparison is actually fair? What would a skeptical reviewer say I am
  hiding?"
- Drafter: "Write the Results section around this table [paste]. State the
  controlled variable. Report the regime where the method loses and explain the
  mechanism. No adjectives that the numbers do not support."
- Critic: "Codex: review this Results section for overclaiming, cherry-picked
  budgets, and missing baselines. Be adversarial."

### Setup
- Drafter: "Write Experimental Setup in two parts: (a) what I can run now and
  what it validates; (b) the decisive real experiment — models, datasets,
  baselines, metrics — with the explicit hypothesis it tests."

### Related Work (ground citations in YOUR Brain library)
First pull real candidate sources from papers you have already ingested:
```sh
python3 ~/paper-kit/cite_from_brain.py "<topic>" "<keyword>" "<keyword>"
brain-wiki query "<topic> related work and prior approaches"   # deeper context
```
- Tutor: "Cluster these prior works [paste the cite_from_brain list] into 2-4
  named families. For each, one sentence of what it does and one of why it does
  not close my gap: [gap]."
- Drafter: "Write Related Work from that clustering, citing ONLY sources that
  exist in my Brain library (the list above) plus canonical works I name
  explicitly. End with a one-sentence gap statement. Do not invent citations."
- Cite discipline: every reference must be a paper you can point to in Brain or
  a canonical work you have read. If a needed source is missing, ingest it first
  (`brain-wiki ingest <file>`), then cite it. Never fabricate a citation.

### Intro
- Drafter: "Write the Introduction as a funnel: domain cost → families of prior
  approaches → the structural limit they share → my stance in one sentence →
  numbered contributions. Keep contributions to exactly what the paper delivers."

### Abstract (write last)
- Drafter: "Write a 6-8 sentence abstract covering: problem+cost, prior limit,
  my idea in one sentence, what I built, the honest result (including where it
  loses), and the decisive next experiment."

### Whole-paper critique (run before every build)
- Codex: "Read the whole paper. List, by severity: (1) claims unsupported by the
  results, (2) method/code mismatches, (3) missing baselines or ablations, (4)
  places a reviewer would reject. Then give the 3 highest-leverage fixes."

## Reusable structure (already in template.tex)
Abstract · Introduction (+contributions) · Background/Related · Method ·
Implementation · Experimental Setup · Results · Limitations & Next Steps · References.

## Example to copy
`EigenKache/docs/paper/eigenkache.tex` is a worked instance of this template:
two-regime results (random → eviction wins; structured → compression wins) that
isolate the mechanism honestly. Read it before starting a new one.
