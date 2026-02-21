# Adapting This Pipeline to Any Medical Field
## A Complete Step-by-Step Guide

This pipeline was built for **Trauma Acute Care Surgery** but is intentionally designed
to be adapted to any medical specialty with minimal technical effort. Every
field-specific value lives in exactly three files. Everything else is generic.

---

## The Three-File Rule

| File | What you change | Time |
|------|----------------|------|
| `config.py` | Field name, dates, geography, scoring | 10 min |
| `concept_definitions.py` | Clinical concepts for your specialty | 1–3 hours |
| `search_trauma_v3_unbiased.py` | Journals and search terms for your field | 30–60 min |

**Everything else** (pipeline logic, validation, statistics, figures, LLM prompts)
runs unchanged.

---

## Step 1: Fork the Repository

```bash
git clone https://github.com/Handicam007/trauma-bibliometric-pipeline.git
cd trauma-bibliometric-pipeline
# Rename your fork on GitHub to match your field, e.g.:
# cardiology-bibliometric-pipeline
# oncology-bibliometric-pipeline
```

---

## Step 2: Edit `config.py` (10 minutes)

This is the **only** file most settings live in. Open it and change:

### 2a. Field Identity
```python
# CHANGE THESE:
FIELD_NAME = "Interventional Cardiology"   # Full name (used in figure titles)
FIELD_SHORT = "Cardiology"                  # Short name (used in axis labels)
```

### 2b. Time Window
```python
YEAR_MIN = 2018          # First year of interest
YEAR_MAX = 2026          # Last year (can be current year, even if partial)
YEAR_STATS_MAX = 2025    # Last COMPLETE year (exclude partial current year)
```

### 2c. Geography
```python
# North America only:
GEO_FILTER_SCOPUS = ' AND (AFFILCOUNTRY(United States) OR AFFILCOUNTRY(Canada))'

# Global (no filter):
GEO_FILTER_SCOPUS = None

# Single country:
GEO_FILTER_SCOPUS = ' AND AFFILCOUNTRY(France)'

# Europe:
GEO_FILTER_SCOPUS = (' AND (AFFILCOUNTRY(United Kingdom) OR AFFILCOUNTRY(Germany)'
                     ' OR AFFILCOUNTRY(France) OR AFFILCOUNTRY(Italy)'
                     ' OR AFFILCOUNTRY(Netherlands))')

GEO_HIGHLIGHT_COUNTRIES = ["United States", "Canada"]  # Which countries to highlight in bar chart
GEO_LABEL = "US + Canada"                               # Label for figure titles
```

### 2d. Word Cloud Stop Words
Remove words that are so obvious in your field they drown out interesting words:
```python
# Trauma:
WORDCLOUD_STOPWORDS = ["trauma", "injury", "surgical", "surgery", ...]

# Cardiology example:
WORDCLOUD_STOPWORDS = ["cardiac", "heart", "cardiology", "patient", "patients",
                       "study", "clinical", "outcome", "outcomes", "results",
                       "analysis", "retrospective", "coronary"]

# Oncology example:
WORDCLOUD_STOPWORDS = ["cancer", "tumor", "tumour", "oncology", "patient",
                       "patients", "treatment", "therapy", "clinical", "study"]
```

### 2e. Journal Highlighting
```python
# Flag "core field" journals in bar charts:
JOURNAL_HIGHLIGHT_KEYWORDS = ["cardiology", "cardiac", "heart", "circulation"]
JOURNAL_HIGHLIGHT_LABEL = "Cardiology-focused journals"
JOURNAL_OTHER_LABEL = "Other journals"
```

---

## Step 3: Define Your Clinical Concepts (1–3 hours)

Open `concept_definitions.py`. This is where you define what topics matter in your field.

### 3a. Start Fresh vs. Adapt

**Option A — Start fresh** (recommended for very different fields):
Delete all entries from `CLINICAL_CONCEPTS = {` down to the closing `}` and
replace with your field's concepts (see examples below).

**Option B — Adapt** (for overlapping fields, e.g., vascular surgery):
Keep relevant concepts (AI / Machine Learning, VTE Prevention, etc.) and
replace field-specific ones.

### 3b. How to Write a Concept

Each concept is one line:
```python
"Display Name":  r"regex_pattern",
```

**Simple concept (single keyword):**
```python
"Heart Failure":  r"heart failure",
```

**Multiple synonyms:**
```python
"Atrial Fibrillation":  r"atrial fibrillation|\bafib\b|\baf\b.*cardiac|atrial flutter",
```

**Exact abbreviation (prevent partial matches):**
```python
"TAVR":  r"\btavr\b|\btavi\b|transcatheter aortic valve",
```

**Concept that needs context (prevent false positives):**
```python
"Pediatric Cardiology": (
    r"(?:pediatric|paediatric|congenital).*(?:cardiac|heart|cardiology)"
    r"|(?:cardiac|heart).*(?:pediatric|paediatric|congenital)"
    r"|congenital heart disease"
),
```

**Always run after editing:**
```bash
python validate.py
```

### 3c. Example Concept Sets by Specialty

---

#### CARDIOLOGY (example — ~20 starter concepts)
```python
CLINICAL_CONCEPTS = {
    "Atrial Fibrillation":        r"atrial fibrillation|\bafib\b|atrial flutter",
    "Heart Failure":              r"heart failure|cardiac failure|cardiomyopathy",
    "Coronary Artery Disease":    r"coronary artery disease|\bcad\b|coronary atherosclerosis",
    "Acute MI":                   r"myocardial infarction|\bstemi\b|\bnstemi\b|acute coronary",
    "PCI / Stenting":             r"\bpci\b|percutaneous coronary|coronary stent|drug.eluting stent",
    "TAVR / Valve":               r"\btavr\b|\btavi\b|transcatheter aortic valve|valve replacement|valve repair",
    "Heart Transplant":           r"heart transplant|cardiac transplant",
    "LVAD / MCS":                 r"\blvad\b|left ventricular assist|mechanical circulatory support|\brvad\b",
    "Cardiogenic Shock":          r"cardiogenic shock",
    "Cardiac Arrest":             r"cardiac arrest|cardiopulmonary resuscitation|\bcpr\b|\brosc\b",
    "Electrophysiology":          r"electrophysiology|\bep\b.*catheter|ablation.*arrhythmia|arrhythmia.*ablation",
    "Hypertension":               r"hypertension|blood pressure control|antihypertensive",
    "Aortic Disease":             r"aortic aneurysm|aortic dissection|aortic stenosis",
    "Anticoagulation":            r"anticoagulat|direct oral anticoagulant|\bdoac\b|\bnoac\b|\bwarfarin\b",
    "AI / Machine Learning":      r"artificial intelligence|machine learning|deep learning|neural network",
    "Cardiac Imaging":            r"cardiac \bct\b|cardiac mri|echocardiograph|cardiac imaging",
    "Congenital Heart":           r"congenital heart|congenital cardiac|fontan|tetralogy",
    "Preventive Cardiology":      r"statin|lipid.lower|cardiovascular prevention|primary prevention.*cardiac",
    "Cardiac Rehabilitation":     r"cardiac rehab|cardiovascular rehab|exercise.*cardiac",
    "COVID-19 Impact":            r"\bcovid.?19\b|\bcovid\b.*cardiac|cardiac.*pandemic|myocarditis.*covid",
}
```

---

#### ONCOLOGY (example — ~20 starter concepts)
```python
CLINICAL_CONCEPTS = {
    "Immunotherapy":              r"immunotherapy|checkpoint inhibitor|\bpd.1\b|\bpd.l1\b|\bctla.4\b",
    "Targeted Therapy":           r"targeted therapy|tyrosine kinase|\begfr\b|\balk\b|\bkras\b|targeted agent",
    "Chemotherapy":               r"chemotherapy|cytotoxic|platinum.based|taxane|anthracycline",
    "CAR-T":                      r"\bcar.t\b|chimeric antigen receptor",
    "Radiation Therapy":          r"radiation therapy|radiotherapy|radiosurgery|\bsbrt\b|\bsrs\b",
    "Surgical Oncology":          r"oncologic resection|cancer surgery|surgical oncology|tumor resection",
    "Breast Cancer":              r"breast cancer|breast carcinoma|breast tumor|\bher2\b",
    "Lung Cancer":                r"lung cancer|lung carcinoma|non.small cell|small cell lung|\bnsclc\b",
    "Colorectal Cancer":          r"colorectal cancer|colon cancer|rectal cancer|colorectal carcinoma",
    "Hematologic Malignancy":     r"leukemia|lymphoma|myeloma|hematologic malignancy|\ball\b.*oncol|acute myeloid",
    "Precision Medicine":         r"precision medicine|biomarker.driven|genomic.*oncol|liquid biopsy|\bngS\b",
    "Palliative Care":            r"palliative|end.of.life|hospice.*cancer|symptom management.*cancer",
    "Neoadjuvant Therapy":        r"neoadjuvant|preoperative.*chemo|preoperative.*radiation",
    "Cancer Screening":           r"cancer screening|early detection|cancer surveillance|colonoscopy.*screen",
    "AI / Machine Learning":      r"artificial intelligence|machine learning|deep learning|neural network",
    "Tumor Microenvironment":     r"tumor microenvironment|immunosuppressive|tumor immune",
    "Clinical Trials":            r"randomized.*oncol|phase \b[123]\b.*trial.*cancer|clinical trial.*cancer",
    "Survivorship":               r"cancer survivor|survivorship|long.term.*cancer|late effect.*cancer",
    "COVID-19 Impact":            r"\bcovid.?19\b.*cancer|cancer.*pandemic|oncology.*covid",
    "Health Disparities":         r"health disparit|racial disparit|cancer.*disparit|equity.*cancer",
}
```

---

#### NEUROSURGERY (example — ~18 starter concepts)
```python
CLINICAL_CONCEPTS = {
    "Glioma / Brain Tumor":       r"glioma|glioblastoma|meningioma|brain tumor|brain tumour|intracranial tumor",
    "TBI / Neurotrauma":          r"traumatic brain|\btbi\b|neurotrauma|diffuse axonal",
    "Cerebrovascular":            r"cerebrovascular|ischemic stroke|\bcva\b|cerebral ischemia|carotid stenosis",
    "Aneurysm / SAH":             r"aneurysm|subarachnoid hemorrhage|\bsah\b|cerebral aneurysm",
    "Spine / Disc":               r"spinal fusion|disc herniation|lumbar disc|cervical disc|spondylosis|laminectomy",
    "Spinal Cord Injury":         r"spinal cord injur|cervical spine injur|spinal trauma",
    "Epilepsy Surgery":           r"epilepsy surgery|temporal lobectomy|corpus callosotomy|seizure surgery",
    "Hydrocephalus":              r"hydrocephalus|ventriculoperitoneal|\bvp\b shunt|cerebrospinal fluid",
    "DBS / Neuromodulation":      r"deep brain stimulation|\bdbs\b|neuromodulation|spinal cord stimulation",
    "Minimally Invasive":         r"minimally invasive.*neuro|endoscopic.*neuro|keyhole.*neuro|tubular.*neuro",
    "Intraoperative Imaging":     r"intraoperative mri|intraoperative ultrasound|\biomri\b|5.ALA|fluorescence.guided",
    "AI / Machine Learning":      r"artificial intelligence|machine learning|deep learning|neural network",
    "Radiosurgery":               r"radiosurgery|\bgamma knife\b|\bcyberknife\b|\blinac\b|stereotactic radiosurgery",
    "Skull Base":                 r"skull base|pituitary|sellar|transsphenoidal|cranial base",
    "Pediatric Neurosurgery":     r"pediatric.*neurosurg|paediatric.*neurosurg|pediatric.*brain|congenital.*neuro",
    "Simulation / Training":      r"neurosurgery.*simulat|simulat.*neurosurg|surgical training|virtual reality.*surg",
    "Outcomes / QI":              r"neurosurgery.*outcome|neurosurgical.*quality|morbidity.*neurosurg|complications.*neurosurg",
    "COVID-19 Impact":            r"\bcovid.?19\b.*neurosurg|neurosurg.*pandemic",
}
```

---

#### EMERGENCY MEDICINE (example — ~18 starter concepts)
```python
CLINICAL_CONCEPTS = {
    "Sepsis / Septic Shock":      r"sepsis|septic shock|septicemia",
    "Resuscitation":              r"resuscitation|\bcpr\b|cardiac arrest.*ED|emergency resuscitat",
    "Point-of-Care Ultrasound":   r"\bpocus\b|\befast\b|point.of.care ultrasound|bedside ultrasound",
    "Airway Management":          r"rapid sequence|\brsi\b|emergency intubation|airway management.*ED|difficult airway",
    "Toxicology / Overdose":      r"overdose|toxicology|poisoning|opioid.*overdose|naloxone",
    "Stroke":                     r"acute stroke|\btpa\b.*stroke|stroke.*ED|ischemic stroke.*emergency",
    "Chest Pain / ACS":           r"chest pain.*ED|acute coronary.*ED|STEMI.*ED|emergency.*\bacs\b",
    "Pediatric Emergency":        r"pediatric.*emergency|paediatric.*emergency|pediatric.*ED",
    "Mental Health Crisis":       r"psychiatric emergency|mental health.*ED|suicidal.*ED|agitation.*ED",
    "Ultrasound-Guided":          r"ultrasound.guided|echo.guided|sonography.guided",
    "Pain Management":            r"pain management.*ED|analgesia.*ED|opioid.*ED|ketamine.*ED",
    "Syncope":                    r"\bsyncope\b|vasovagal|near.syncope",
    "Pulmonary Embolism":         r"pulmonary embolism|\bpe\b.*emergency|\bdvt\b|thromboembolism",
    "AI / Machine Learning":      r"artificial intelligence|machine learning|deep learning|neural network",
    "Crowding / Operations":      r"ED crowding|emergency department overcrowd|length of stay.*ED|throughput.*ED",
    "Simulation / Training":      r"simulat.*emergency|emergency.*simulat|\batls\b|\bacls\b|high.fidelity simulat",
    "COVID-19 Impact":            r"\bcovid.?19\b.*emergency|emergency.*pandemic|ED.*pandemic",
    "Geriatric Emergency":        r"geriatric.*emergency|older adult.*ED|elderly.*emergency|frailty.*ED",
}
```

---

#### ORTHOPEDIC SURGERY (full working example — 31 concepts, ready to run)

A complete Orthopedic Surgery configuration is available in the `configs/` directory:
- `configs/ortho_config.py` — field identity, geography, scoring
- `configs/ortho_concepts.py` — 31 clinical concepts with domain groupings
- `configs/ortho_search.py` — 3-layer Scopus search (13 queries, 5 journal groups)
- `configs/ortho_filters.py` — exclusion/inclusion keyword lists, journal list

To activate it:
```bash
cp configs/ortho_config.py config.py
cp configs/ortho_concepts.py concept_definitions.py
# Then update llm_schemas.py CONCEPT_NAMES + ConceptLiteral
# Then copy search and filter keyword lists from configs/ortho_*.py
python validate.py
```

```python
# Excerpt (6 of 31 concepts):
CLINICAL_CONCEPTS = {
    "Total Knee Arthroplasty (TKA)": r"\btka\b|total knee arthroplasty|total knee replacement|unicompartmental knee",
    "ACL / Knee Ligament":           r"\bacl\b|anterior cruciate|posterior cruciate|\bpcl\b|knee ligament|multiligament",
    "Spine Surgery / Fusion":        r"spinal fusion|lumbar fusion|cervical fusion|\bacdf\b|\btlif\b|laminectomy",
    "Rotator Cuff / Shoulder":       r"rotator cuff|supraspinatus|shoulder instability|labral tear|\bbankart\b",
    "Robotics / Navigation":         r"robotic.*(?:surg|arthroplasty)|robot.assisted|computer.assisted.*navigation",
    "Fragility Fracture / Osteoporosis": r"fragility fracture|osteoporotic fracture|osteoporosis.*fracture|fracture liaison",
    # ... 25 more concepts in configs/ortho_concepts.py
}
```

---

### 3d. Domain Groupings

After defining concepts, update `DOMAIN_GROUPS` to match your field's natural clusters.
This determines the subplot layout in multi-panel figures:

```python
# Example for Cardiology:
DOMAIN_GROUPS = {
    "Interventional": ["PCI / Stenting", "TAVR / Valve", "LVAD / MCS"],
    "Electrophysiology": ["Atrial Fibrillation", "Electrophysiology", "Cardiac Arrest"],
    "Heart Failure": ["Heart Failure", "Cardiogenic Shock", "Heart Transplant"],
    "Vascular": ["Coronary Artery Disease", "Acute MI", "Aortic Disease"],
    "Prevention & Population": ["Hypertension", "Anticoagulation", "Preventive Cardiology"],
    "Technology": ["AI / Machine Learning", "Cardiac Imaging", "Cardiac Rehabilitation"],
}

# Example for Orthopedic Surgery:
DOMAIN_GROUPS = {
    "Arthroplasty & Joint Replacement": ["Total Knee Arthroplasty (TKA)", "Total Hip Arthroplasty (THA)", "Shoulder Arthroplasty", "Revision Arthroplasty"],
    "Trauma & Fracture": ["Hip / Femur Fracture", "Distal Radius Fracture", "Ankle Fracture", "Pelvic / Acetabular Fracture", "Polytrauma / Ortho Trauma", "Fragility Fracture / Osteoporosis"],
    "Spine": ["Spine Surgery / Fusion", "Degenerative Disc Disease", "Spinal Deformity (Scoliosis)"],
    "Sports Medicine": ["ACL / Knee Ligament", "Rotator Cuff / Shoulder", "Meniscus / Cartilage", "Sports Medicine (General)"],
    "Subspecialties": ["Pediatric Orthopedics", "Hand / Upper Extremity", "Foot / Ankle", "Musculoskeletal Oncology"],
    "Technology, Quality & Systems": ["AI / Machine Learning", "Robotics / Navigation", "3D Printing / Patient-Specific", "Biologics / PRP / Stem Cells", "Outcomes / PROMs", "Infection / SSI"],
}
```

Also update `TOP_CONCEPTS_FOR_CHARTS` to be the same list of concept names in your
preferred display order.

---

## Step 4: Update the Search Strategy (30–60 min)

Open `search_trauma_v3_unbiased.py`. Replace the journal lists and topic terms.

### 4a. Find Your Field's Core Journals

Search Scopus directly for your field's top journals. Tips:
- Go to https://www.scopus.com/sources → search by subject area
- Check the latest SCImago Journal Rankings for your specialty
- Ask a subject librarian for the canonical journal list

### 4b. Replace Layer 1 (Journal Sweep)

```python
# CARDIOLOGY EXAMPLE:
QUERIES = {
    "L1_core_journals": SearchQuery(
        '(SRCTITLE("Journal of the American College of Cardiology") '
        'OR SRCTITLE("Circulation") '
        'OR SRCTITLE("European Heart Journal") '
        'OR SRCTITLE("JACC: Cardiovascular Interventions") '
        'OR SRCTITLE("Heart Rhythm") '
        'OR SRCTITLE("Journal of Heart and Lung Transplantation"))'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),
    ...
}
```

### 4c. Replace Layer 2 (Topic Sweep)

Replace the broad topic keywords with your field's core terminology:

```python
# Cardiology:
'TITLE-ABS-KEY(cardiac OR cardiovascular OR "heart failure" OR "coronary artery")'

# Oncology:
'TITLE-ABS-KEY(cancer OR tumor OR carcinoma OR oncology OR malignancy)'

# Neurosurgery:
'TITLE-ABS-KEY(neurosurgery OR "brain tumor" OR "spinal cord" OR glioma OR aneurysm)'
```

### 4d. Import Config Variables

Make sure the search file reads from config.py instead of hardcoding values:
```python
from config import YEAR_MIN, YEAR_MAX, GEO_FILTER_SCOPUS
NA_FILTER = GEO_FILTER_SCOPUS or ''
```

---

## Step 5: Update Filter Keywords

Open `filter_results.py`. Update three lists:

```python
# EXCLUDE_TITLE_KEYWORDS — papers definitely NOT in your field
# For cardiology, remove trauma-specific exclusions and add:
EXCLUDE_TITLE_KEYWORDS = [
    "veterinary", "animal model", "in vitro", "in vivo",
    "materials science", "dental", "dermatology",
]

# REQUIRE_ANY_KEYWORD — at least one of these must appear for inclusion
# For cardiology:
REQUIRE_ANY_KEYWORD = [
    "cardiac", "cardiovascular", "coronary", "heart",
    "atrial", "ventricular", "arrhythmia", "myocardial",
]

# TOP_TRAUMA_JOURNALS → rename and update for your field
TOP_FIELD_JOURNALS = [
    "journal of the american college of cardiology",
    "circulation",
    "european heart journal",
    ...
]
```

---

## Step 6: Update PRISMA Checklist Caveats

Open `prisma_compliance.py` and update item 27 (Limitations) to reflect your field:

```python
(27, "Discussion / Limitations",
 "Limitations specific to this review: single-database (Scopus), "
 "English-language bias, [your geography] filter, model non-determinism, "
 "[field-specific confounders, e.g., 'COVID-19 surge in cardiology 2020-2021']"),
```

---

## Step 7: Run validate.py

```bash
python validate.py
```
This checks:
- All concept names are consistent across files
- All regex patterns compile without errors
- CONCEPT_NAMES and ConceptLiteral are in sync

Fix any errors before running the pipeline.

---

## Step 8: Run the Pipeline

```bash
python search_trauma_v3_unbiased.py    # Step 1: fetch papers
python filter_results.py               # Step 2: keyword filter
python llm_pipeline.py --steps screen classify extract   # Steps 3-5: LLM
python statistical_tests.py            # Step 6: figures
python prisma_compliance.py            # Step 7: PRISMA checklist
```

---

## Estimated Effort by Specialty Similarity

| Your field vs. Trauma | Estimated adaptation time |
|-----------------------|--------------------------|
| Acute Care Surgery, EGS, Burns | 1–2 hours (very similar) |
| Emergency Medicine, Critical Care | 2–4 hours |
| Vascular Surgery, Orthopaedics | 3–5 hours |
| General Surgery, Neurosurgery | 4–6 hours |
| Cardiology, Oncology, Pulmonology | 5–8 hours |
| Psychiatry, Dermatology, Endocrinology | 8–12 hours |

The bottleneck is always **defining good concepts** — not technical setup.

---

## What You Can Cite

When you publish your adapted pipeline, cite:

1. **The original trauma paper** (this project's JMIR publication, forthcoming)
2. **The GitHub repository** (MIT license — cite as software)
3. **PRISMA-trAIce** checklist (AI-augmented systematic reviews)
4. **LLM provider** (OpenAI, Anthropic, or Google)
5. **Pybliometrics** (Scopus API wrapper)

---

## Common Mistakes

**"My kappa is 0.45 — the LLM is terrible"**
Check your concept definitions first. Overlapping concepts (e.g., both
"Heart Failure" and "Cardiogenic Shock" could match the same paper) inflate
apparent disagreement. Tighten the regexes or merge concepts.

**"I'm getting 80% exclusion rate at the filter step"**
Your `REQUIRE_ANY_KEYWORD` list is too narrow, or your search already filtered
too aggressively. Lower the inclusion threshold or broaden Layer 2 keywords.

**"The LLM keeps assigning the wrong concepts"**
The LLM uses your concept names as context. Rename vague concepts to be
more specific. "Cardiac Procedures" → "Percutaneous Coronary Intervention"
gives the LLM much better signal.

**"My figures look wrong"**
Run `python validate.py` — mismatched concept names between files is the
most common cause of empty figures.

---

*This pipeline is MIT licensed. Fork it, adapt it, cite it.*
*See CONTRIBUTING.md for the student onboarding guide.*
