# Student Contributor Guide
## LLM-Augmented Bibliometric Pipeline for Trauma Acute Care Surgery

Welcome! This guide is written for a medical student or resident joining this project.
Your goal is to produce the human validation data and write the manuscript for a
JMIR Methods paper. You do **not** need to understand every line of code — this guide
tells you exactly what to run and what to write.

---

## 1. What the Pipeline Does (Plain English)

The pipeline answers: *"What are the most important trauma surgery papers from 2020–2026,
and what topics are trending?"*

It works in five automated steps:

```
Scopus database
      ↓  fetch_abstracts.py        — downloads up to ~10,000 paper titles/abstracts
      ↓  filter_results.py         — keyword filter removes obvious off-topic papers
      ↓  llm_pipeline.py screen    — LLM reads each abstract, decides: relevant or not?
      ↓  llm_pipeline.py classify  — LLM assigns clinical concepts (e.g., "REBOA", "TBI")
      ↓  llm_pipeline.py extract   — LLM extracts study design, sample size, key finding
      ↓  statistical_tests.py      — generates 30 publication-quality figures
```

**Your job:** Run the pipeline on real data, annotate 200 papers by hand
(the gold standard), and write the paper.

---

## 2. Setup (One-Time)

### 2a. Prerequisites
- Python 3.11+ installed (`python3 --version`)
- A terminal / command line (Terminal on Mac, Command Prompt on Windows)
- Git installed (`git --version`)

### 2b. Clone the Repository
```bash
git clone https://github.com/YOUR_REPO_URL.git
cd "Trauma Presentation"
```

### 2c. Install Dependencies
```bash
pip install -r requirements.txt
```
This installs all required libraries (pandas, pydantic, openai, etc.) at the exact
versions used for publication reproducibility.

### 2d. API Keys
You need a Scopus API key AND an LLM provider key.

**Scopus API key** (free for academic use):
1. Go to https://dev.elsevier.com → Register → Create API key
2. Your institution likely already has Scopus access — check with the library

**OpenAI API key** (recommended provider):
1. Go to https://platform.openai.com/api-keys → Create key
2. Add funds (~$20 is enough for the full dataset)

Set your keys in the terminal before running any script:
```bash
export ELSEVIER_API_KEY="your-scopus-key-here"
export OPENAI_API_KEY="your-openai-key-here"
```
> **Security:** Never paste API keys into code files or commit them to GitHub.
> Always set them as environment variables.

### 2e. Verify Setup
```bash
python validate.py
```
Expected output: `✓ All X concepts synchronized. No errors found.`

---

## 3. Running the Pipeline

### Step 1: Fetch Papers from Scopus
```bash
python search_trauma_v3_unbiased.py
```
- Downloads trauma papers (2020–2026, US + Canada) from Scopus
- Runtime: 10–30 minutes depending on API rate limits
- Output: `results_refined/all_results.csv`

### Step 2: Keyword Filter
```bash
python filter_results.py
```
- Removes off-topic papers (materials science, psychiatry, veterinary, etc.)
- Output: `results_curated/all_filtered.csv`
- Console will show: `After exclusion filter: XXXX` and `After inclusion filter: XXXX`
- Expected survival rate: 60–80% of raw papers

### Step 3: LLM Screening (largest step — ~$5–15 in API costs)
```bash
python llm_pipeline.py --provider openai --steps screen
```
- LLM reads each abstract and decides: relevant to trauma surgery or not?
- Progress updates every 50 papers
- Runtime: 1–3 hours for ~5,000 papers
- Cache saves automatically — if interrupted, re-run the same command to resume

### Step 4: LLM Classification
```bash
python llm_pipeline.py --provider openai --steps classify
```
- Assigns clinical concepts (REBOA, TBI, Hemorrhage Control, etc.) to each paper
- Runtime: 1–2 hours

### Step 5: LLM Data Extraction
```bash
python llm_pipeline.py --provider openai --steps extract
```
- Extracts study design, sample size, key finding from each abstract
- Runtime: 1–2 hours

### Step 6: Run All Steps Together (after testing individually)
```bash
python llm_pipeline.py --provider openai --steps screen,classify,extract
```

### Step 7: Generate Figures
```bash
python statistical_tests.py
```
- Generates all 30 publication figures at 300 DPI
- Output: `figures_v3/` folder
- Runtime: 2–5 minutes

### Step 8: Generate PRISMA Checklist
```bash
python prisma_compliance.py
```
- Output: `stats_output/prisma_traice_checklist.md` — you will complete this manually
- Output: `stats_output/prisma_traice_checklist.json` — machine-readable version

---

## 4. Human Validation (Your Primary Academic Contribution)

This is the most important part of your work. The LLM made decisions on thousands
of papers — you will verify a sample of 200 to measure its accuracy.

### 4a. Generate the Validation Sample
```bash
python llm_pipeline.py --generate-validation-sample
```
Output: `llm_cache/validation_sample.csv`

This CSV contains 200 papers stratified across confidence levels and concepts.
It has blank columns for you to fill in.

### 4b. What to Fill In

Open `validation_sample.csv` in Excel or Google Sheets.
For each paper, read the **title** and **abstract** and fill in:

| Column | What to enter | Example |
|--------|--------------|---------|
| `human_relevant` | Is this paper relevant to trauma surgery? | `yes` or `no` |
| `human_concepts` | Which clinical concepts apply? (comma-separated) | `REBOA, Hemorrhage Control` |
| `human_study_design` | Study design | `retrospective_cohort` |
| `human_notes` | Any comments or edge cases | `Borderline — EGS, not pure trauma` |

**Valid study design values:**
`RCT`, `prospective_observational`, `retrospective_cohort`, `case_control`,
`cross_sectional`, `systematic_review`, `meta_analysis`, `case_series`,
`case_report`, `narrative_review`, `guideline`, `registry_study`,
`experimental`, `qualitative`, `other`

**Valid concept names** (copy-paste from this list):
`Geriatric / Frailty`, `Trauma Systems / QI`, `Hemorrhage Control`,
`Pediatric Trauma`, `REBOA`, `Prehospital / EMS`, `Whole Blood / MTP`,
`Penetrating Trauma`, `AI / Machine Learning`, `Blunt Trauma`,
`Damage Control`, `Simulation / Training`, `POCUS / eFAST`,
`Splenic Injury`, `Coagulopathy (TIC)`, `Liver Injury`,
`Rib Fixation (SSRF)`, `Non-Operative Mgmt`, `Angioembolization`,
`TBI / Neurotrauma`, `TEG / ROTEM`, `Fibrinogen / Cryo`,
`Resuscitative Thoracotomy`, `Fracture Management`, `Orthopaedic Trauma`,
`Pelvic / Acetabular`, `Hip / Femur Fracture`, `Military / Combat`,
`Polytrauma`, `Spinal Cord Injury`, `Thoracic Trauma`, `Abdominal Trauma`,
`Vascular Injury`, `Teletrauma / Remote`, `COVID-19 Impact`,
`Triage Systems`, `VTE Prevention`, `Airway / Tracheostomy`,
`ECMO in Trauma`, `Mass Casualty / Disaster`, `Firearm / Gun Violence`,
`Emergency General Surgery`, `Tranexamic Acid (TXA)`, `Burn Injury`

### 4c. Second Annotator (Required for Publication)

You need a **second reviewer** to annotate the same 200 papers independently.
This is called inter-rater reliability (IRR) — without it, JMIR will request it.

- Suitable second annotators: another medical student, a senior resident, a fellow
- They should annotate the same CSV **without seeing your answers first**
- Add their answers in columns: `human2_relevant`, `human2_concepts`

### 4d. Compute Agreement Statistics
```bash
python llm_pipeline.py --compute-human-agreement
```
- Computes Cohen's kappa between LLM and human, and between human 1 and human 2
- Target: κ > 0.70 for screening, κ > 0.60 for concept classification
- Output goes into the console and `stats_output/validation_report.json`

---

## 5. Writing the Paper

### Target Journal
**JMIR Medical Informatics** (https://medinform.jmir.org)
- Open access, PubMed-indexed, impact factor ~3.9
- Article type: **Methods / Technical Paper**
- Word limit: ~4,000 words (excluding references)
- Figures: up to 10 in main paper; rest as supplements

### Paper Structure

**Title (suggestion):**
*"LLM-Augmented Bibliometric Pipeline for Trauma Surgery Literature Surveillance:
Development and Validation of an Open-Source Tool"*

**Abstract** (~250 words, structured):
- Background: Manual literature monitoring is unsustainable at scale
- Objective: Develop and validate an automated LLM-augmented pipeline
- Methods: Scopus search + 5-layer LLM processing + human validation
- Results: [paste numbers from your run]
- Conclusions: Pipeline achieves κ=X for screening with $Y total cost

**Introduction** (~500 words):
- Why literature surveillance matters in trauma surgery
- Current approaches and limitations (manual review, keyword-only tools)
- What LLMs add (semantic understanding, structured extraction)
- Cite the literature review provided to you by the senior author

**Methods** (~1,200 words):
Use `stats_output/prisma_traice_checklist.md` as your outline.
Key subsections:
1. Search Strategy (Scopus, date range, geographic filter)
2. Keyword Filtering (cite `filter_results.py` inclusions/exclusions)
3. LLM Screening (model version, temperature, prompt strategy, confidence thresholds)
4. LLM Classification (hierarchical: domain → concept, 44 concepts)
5. LLM Data Extraction (study design, sample size, key finding)
6. Validation Framework (200-paper stratified sample, two annotators, kappa)
7. Cost Analysis (per-step breakdown from pipeline output)

**Results** (~800 words):
- Table 1: Pipeline statistics (papers at each stage)
- Table 2: LLM vs human agreement (κ, sensitivity, specificity, F1 — from validation output)
- Table 3: Per-step cost breakdown
- Figure 1: Pipeline flow diagram (Fig 30)
- Figure 2: Consensus heatmap (Fig 27)
- Figure 3: Validation dashboard (Fig 29)
- Figure 4: Top concepts over time (Fig from statistical_tests.py)

**Discussion** (~600 words):
- What the results mean clinically
- Comparison to manual review (cost, time, accuracy)
- Limitations (complete `prisma_traice_checklist.md` items 27–28):
  - Single database (Scopus only)
  - English-language bias
  - Model non-determinism
  - Geographic filter (US + Canada only)
  - Abstract-only (no full-text)
  - COVID-19 temporal confounding (2020–2021 spike)

**Conclusion** (~100 words):
One paragraph. Restate the contribution and call for adaptation to other specialties.

**Data Availability:**
Link to GitHub repo + Zenodo DOI (see Step 6 below)

---

## 6. Before Submission

### Register PROSPERO Protocol
1. Go to https://www.crd.york.ac.uk/prospero/
2. Register a systematic review protocol (takes ~30 min)
3. Use the senior author's ORCID
4. Record the PROSPERO ID — goes in Methods section

### Create Zenodo Code Deposit
1. Go to https://zenodo.org → Log in with GitHub
2. Enable the Trauma Presentation repository
3. Create a new release on GitHub → Zenodo auto-mints a DOI
4. The DOI goes in the Data Availability statement

### Ethics Statement
Most institutions classify this as a bibliometric study — **no ethics approval required**.
Confirm with your institution's REB/IRB office. One sentence in Methods:
*"This study used publicly available bibliometric data and did not involve human subjects."*

### Submission Checklist
```
□ All 29 PRISMA-trAIce items completed in prisma_traice_checklist.md
□ Kappa values computed (κ > 0.70 screening, κ > 0.60 classification)
□ Two annotators completed validation independently
□ GitHub repo public with MIT license
□ Zenodo DOI obtained
□ PROSPERO ID obtained
□ Figures at 300 DPI (already set in pipeline)
□ Word count < 4,000 (excluding references)
□ Senior author reviewed and approved final draft
```

---

## 7. Questions and Troubleshooting

**Pipeline crashes mid-run:**
Re-run the exact same command. The cache system saves every 50 papers — it will
resume from where it stopped.

**"API key not found" error:**
```bash
export OPENAI_API_KEY="sk-..."   # Mac/Linux
set OPENAI_API_KEY=sk-...        # Windows
```

**Validation kappa is below 0.60:**
This is important data — don't hide it. Describe it in the Limitations section and
investigate which concepts caused the most disagreement. This is actually interesting
for the Discussion.

**"python: command not found":**
Try `python3` instead of `python`.

**A figure doesn't generate:**
The figure may require data from a prior step. Run all pipeline steps first, then
`python statistical_tests.py`. Some figures (26–30) require provider comparison or
validation data — they will skip gracefully with a warning if that data doesn't exist.

**I don't understand what a concept means:**
See `concept_definitions.py` — every concept has a comment explaining what it matches
and why. The regex patterns show exactly what the pipeline looks for.

---

## 8. Contact

For questions about the pipeline code or data: contact the senior author
For questions about clinical content: use your clinical judgment — you're the MD

---

*Pipeline version: see `requirements.txt` for pinned dependency versions*
*License: MIT — see `LICENSE` for terms*
