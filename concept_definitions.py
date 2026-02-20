"""
SINGLE SOURCE OF TRUTH for clinical concept regex patterns.
All analysis scripts must import from here — no duplicating patterns.

Each pattern is designed for case-insensitive matching against paper titles.

v3 — tightened all regexes after audit found overcounting.
     Every pattern reviewed against sample matches for specificity.

HOW TO EDIT THIS FILE
=====================
- Run `python validate.py` AFTER every edit to catch mistakes.
- Each concept is: "Display Name": r"regex_pattern",
- Don't forget the comma at the end of each line!
  (A missing comma silently merges two lines — Python quirk.)

REGEX QUICK REFERENCE (for non-programmers)
============================================
  |           means "OR"            e.g.  cat|dog  matches "cat" or "dog"
  \\b          word boundary          e.g.  \\bteg\\b  matches "teg" but NOT "integral"
  (?:...)     groups terms           e.g.  (?:cat|dog)  groups "cat" or "dog"
  .*          any characters between e.g.  cat.*dog  matches "cat and dog", "cat chases dog"
  r"..."      raw string (required)  tells Python not to interpret backslashes

COMMON PATTERNS
  Simple keyword:           r"damage control"
  Multiple synonyms:        r"hemorrha|haemorrha|hemostatic"
  Exact abbreviation:       r"\\breboa\\b"    (word boundary prevents partial matches)
  Word A near Word B:       r"(?:ecmo|ecls).*trauma|trauma.*(?:ecmo|ecls)"
                            (matches either order: "ECMO in trauma" or "trauma ECMO")

VALIDATION
  After editing, run:  python validate.py
  This checks all regexes compile and all concept names are consistent.
"""

# ── CLINICAL CONCEPTS ─────────────────────────────────────────────
# IMPORTANT: every line MUST end with a comma. A missing comma
# silently merges two strings. Run `python validate.py` after editing.

CLINICAL_CONCEPTS = {
    # --- Resuscitation & Hemostasis ---

    # Matches: geriatric, frailty, elderly, older adult
    "Geriatric / Frailty":        r"geriatric|frailty|\belderly\b|older adult",
    # Matches: trauma system, trauma centre, TQIP, trauma quality
    "Trauma Systems / QI":        r"trauma system|trauma cent|trauma network|\btqip\b|trauma quality",
    # Matches: hemorrhage, haemorrhage, hemostatic, tourniquet
    "Hemorrhage Control":         r"hemorrha|haemorrha|hemostatic|hemorrhage control|tourniquet",
    # Matches: "pediatric" near trauma-related words (either order), or "child/children" near trauma words
    # Prevents matching non-trauma pediatric papers (e.g., MIS-C, PICU nutrition)
    "Pediatric Trauma": (
        r"(?:pediatric|paediatric).*(?:trauma|injur|fracture|emergency|resuscitat|surgery|critical)"
        r"|(?:trauma|injur|fracture|emergency|resuscitat).*(?:pediatric|paediatric)"
        r"|\bchildren\b.*(?:trauma|injur|fracture)"
        r"|\bchild\b.*(?:trauma|injur|fracture)"
        r"|(?:trauma|injur|fracture).*\bchildren\b"
        r"|(?:trauma|injur|fracture).*\bchild\b"
        r"|pediatric trauma|paediatric trauma"
    ),
    "REBOA":                      r"\breboa\b|resuscitative endovascular balloon|aortic occlusion",
    "Prehospital / EMS":          r"prehospital|pre-hospital|\bems\b|paramedic|air ambulance|air medical",
    "Whole Blood / MTP":          r"whole blood|massive transfusion",
    "Penetrating Trauma":         r"penetrating|\bstab\b|gunshot|stab wound",
    # Matches: artificial intelligence, machine learning, deep learning, neural network
    "AI / Machine Learning":      r"artificial intelligence|machine learning|deep learning|\bneural network\b",
    # Matches: "blunt" near trauma/injury context (prevents matching "blunt dissection technique")
    "Blunt Trauma":               r"blunt.*(?:trauma|injur|abdominal|thoracic|aortic|splenic|hepatic|cardiac|cerebrovascular)|(?:trauma|injur).*\bblunt\b|blunt force",
    "Damage Control":             r"damage control",
    # Matches: simulation near trauma/surgical context, VR, cadaver-based, ATLS
    "Simulation / Training":      r"(?:trauma|surgical|resuscitation).*simulat|simulat.*(?:trauma|surgical|resuscitation)|virtual reality|cadaver.based|\batls\b",
    "POCUS / eFAST":              r"\bpocus\b|\befast\b|point.of.care ultrasound",
    "Splenic Injury":             r"\bspleen\b|\bsplenic\b",
    "Coagulopathy (TIC)":         r"coagulopathy|trauma.induced coag",
    "Liver Injury":               r"liver trauma|liver injur|hepatic injur|hepatic trauma|liver laceration",
    "Rib Fixation (SSRF)":        r"rib fixation|rib fracture|\bssrf\b|rib stabiliz|flail chest",
    "Non-Operative Mgmt":         r"non.operative|nonoperative",
    "Angioembolization":          r"angioemboliz|(?:splenic|hepatic|pelvic).*embolization",
    "TBI / Neurotrauma":          r"traumatic brain|\btbi\b|neurotrauma",
    "TEG / ROTEM":                r"\bteg\b|\brotem\b|viscoelastic|thromboelast",
    "Fibrinogen / Cryo":          r"fibrinogen|cryoprecipitate",
    "Resuscitative Thoracotomy":  r"resuscitative thoracotomy|emergency thoracotomy",

    # --- Expanded concepts ---
    "Fracture Management":        r"\bfracture\b|osteosynthesis|\borif\b|intramedullary|\bnailing\b",
    "Orthopaedic Trauma":         r"orthopaedic.*trauma|orthopedic.*trauma|orthopaedic.*injur|orthopedic.*injur|orthopaedic.*fracture|orthopedic.*fracture",
    "Pelvic / Acetabular":        r"\bpelvi[cs]\b|\bacetabul",
    "Hip / Femur Fracture":       r"hip fracture|femoral neck|femur fracture|proximal femur|intertrochanteric",
    "Military / Combat":          r"\bmilitary\b|\bcombat\b|battlefield|tactical combat|combat casualty",
    "Polytrauma":                 r"polytrauma|multiple trauma|multiply injured",
    "Spinal Cord Injury":         r"spinal cord injur|cervical spine injur|spine fracture|spinal trauma|spinal injur",
    "Thoracic Trauma":            r"chest trauma|pneumothorax|hemothorax|thoracic trauma|thoracic injur",
    "Abdominal Trauma":           r"abdominal trauma|abdominal injur|bowel injur|\blaparotomy\b.*trauma|trauma.*\blaparotomy\b",
    "Vascular Injury":            r"vascular injur|arterial injur|venous injur|vascular trauma",
    "Teletrauma / Remote":        r"teletrauma|tele.?trauma|telemedicine.*trauma|trauma.*telemedicine|rural trauma|telementoring",
    # Matches: COVID-19, or COVID near trauma/surgery/pandemic context
    "COVID-19 Impact": (
        r"\bcovid.?19\b"
        r"|\bcovid\b.*(?:trauma|injur|surgery|pandemic|impact|era|period)"
        r"|\bsars.cov.2\b"
        r"|(?:trauma|injur|surgery).*\bcovid\b"
        r"|coronavirus.*(?:trauma|surgery|impact)"
        r"|pandemic.*(?:trauma|surgery|injur)"
        r"|(?:trauma|surgery|injur).*pandemic"
    ),
    "Triage Systems":             r"\btriage\b|undertriage|overtriage|field triage",
    "VTE Prevention":             r"thromboembolism|thromboembolic|\bvte\b|\bdvt\b|pulmonary embolism",
    "Airway / Tracheostomy":      r"\bairway\b.*trauma|trauma.*\bairway\b|\bintubat.*trauma|trauma.*\bintubat|\btracheostom",
    # Matches: ECMO/ECLS near "trauma" (either order). Prevents matching non-trauma ECMO papers.
    "ECMO in Trauma":             r"(?:ecmo|ecls|extracorporeal membrane|extracorporeal life support).*trauma|trauma.*(?:ecmo|ecls|extracorporeal membrane|extracorporeal life support)",
    "Mass Casualty / Disaster":   r"mass casualty|mass shooting|blast injur|\bmci\b|active shooter",
    "Firearm / Gun Violence":     r"firearm|gun violence|gunshot wound|shooting victim|bullet",
}

# ── CONCEPTS FOR CHARTS ─────────────────────────────────────────────
TOP_CONCEPTS_FOR_CHARTS = [
    "Geriatric / Frailty",
    "Trauma Systems / QI",
    "Hemorrhage Control",
    "Pediatric Trauma",
    "REBOA",
    "Prehospital / EMS",
    "Whole Blood / MTP",
    "Penetrating Trauma",
    "AI / Machine Learning",
    "Blunt Trauma",
    "Damage Control",
    "Simulation / Training",
    "POCUS / eFAST",
    "Splenic Injury",
    "Coagulopathy (TIC)",
    "Liver Injury",
    "Rib Fixation (SSRF)",
    "Non-Operative Mgmt",
    "Angioembolization",
    "TBI / Neurotrauma",
    "TEG / ROTEM",
    "Fracture Management",
    "Orthopaedic Trauma",
    "Pelvic / Acetabular",
    "Hip / Femur Fracture",
    "Military / Combat",
    "Polytrauma",
    "Abdominal Trauma",
    "Triage Systems",
    "COVID-19 Impact",
    "Thoracic Trauma",
    "Spinal Cord Injury",
    "Teletrauma / Remote",
    "Vascular Injury",
    "VTE Prevention",
    "Airway / Tracheostomy",
    "Firearm / Gun Violence",
]

# ── DOMAIN GROUPINGS (for subplot grid) ─────────────────────────────
DOMAIN_GROUPS = {
    "Resuscitation & Blood Products": [
        "Damage Control", "Whole Blood / MTP",
        "TEG / ROTEM", "Coagulopathy (TIC)", "Fibrinogen / Cryo",
        "Hemorrhage Control",
    ],
    "Surgical Techniques & Approaches": [
        "REBOA", "Non-Operative Mgmt", "Angioembolization",
        "Rib Fixation (SSRF)", "Resuscitative Thoracotomy",
        "Thoracic Trauma", "Abdominal Trauma", "Vascular Injury",
    ],
    "Technology & Innovation": [
        "AI / Machine Learning", "POCUS / eFAST", "Simulation / Training",
        "Teletrauma / Remote", "ECMO in Trauma",
    ],
    "Populations & Systems": [
        "Geriatric / Frailty", "Pediatric Trauma", "Penetrating Trauma",
        "Trauma Systems / QI", "Prehospital / EMS", "Triage Systems",
        "Military / Combat", "Mass Casualty / Disaster",
    ],
    "Injury Patterns & Ortho": [
        "Fracture Management", "Orthopaedic Trauma", "Pelvic / Acetabular",
        "Hip / Femur Fracture", "Polytrauma", "Blunt Trauma",
        "Splenic Injury", "Liver Injury", "TBI / Neurotrauma",
        "Spinal Cord Injury",
    ],
    "Other Emerging Topics": [
        "COVID-19 Impact", "VTE Prevention", "Airway / Tracheostomy",
        "Firearm / Gun Violence",
    ],
}
