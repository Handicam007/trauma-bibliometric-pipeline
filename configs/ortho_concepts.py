"""
CLINICAL CONCEPT DEFINITIONS — ORTHOPEDIC SURGERY
====================================================
Copy this file to the project root as concept_definitions.py to switch
the pipeline from Trauma Acute Care Surgery to Orthopedic Surgery.

    cp configs/ortho_concepts.py concept_definitions.py

IMPORTANT: After copying, you must also update llm_schemas.py to sync
CONCEPT_NAMES and ConceptLiteral with the concepts defined here.
Run `python validate.py` after any changes.

REGEX QUICK REFERENCE (for non-programmers)
============================================
  |           means "OR"            e.g.  cat|dog  matches "cat" or "dog"
  \\b          word boundary          e.g.  \\btka\\b  matches "tka" but NOT "stka"
  (?:...)     groups terms           e.g.  (?:cat|dog)  groups "cat" or "dog"
  .*          any characters between e.g.  cat.*dog  matches "cat and dog"
  r"..."      raw string (required)  tells Python not to interpret backslashes
"""

# ── CLINICAL CONCEPTS ─────────────────────────────────────────────
# 30 concepts across 6 domains of orthopedic surgery.
# IMPORTANT: every line MUST end with a comma.

CLINICAL_CONCEPTS = {
    # --- Arthroplasty & Joint Replacement ---

    # Matches: total knee arthroplasty, TKA, knee replacement, unicompartmental
    "Total Knee Arthroplasty (TKA)": (
        r"\btka\b|total knee arthroplasty|total knee replacement"
        r"|unicompartmental knee|unicondylar knee|\buka\b"
    ),
    # Matches: total hip arthroplasty, THA, hip replacement, hemiarthroplasty
    "Total Hip Arthroplasty (THA)": (
        r"\btha\b|total hip arthroplasty|total hip replacement"
        r"|hemiarthroplasty|hip arthroplasty"
    ),
    # Matches: shoulder arthroplasty, reverse shoulder, TSA, RSA
    "Shoulder Arthroplasty": (
        r"shoulder arthroplasty|shoulder replacement"
        r"|reverse total shoulder|\brsa\b.*shoulder|\btsa\b.*shoulder"
        r"|reverse shoulder arthroplasty"
    ),
    # Matches: revision arthroplasty, revision TKA/THA, periprosthetic, aseptic loosening
    "Revision Arthroplasty": (
        r"revision arthroplasty|revision.*\btka\b|revision.*\btha\b"
        r"|\btka\b.*revision|\btha\b.*revision"
        r"|periprosthetic fracture|periprosthetic joint"
        r"|aseptic loosening|prosthetic joint infection"
    ),

    # --- Trauma & Fracture ---

    # Matches: hip fracture, femoral neck, intertrochanteric, proximal femur, femur fracture
    "Hip / Femur Fracture": (
        r"hip fracture|femoral neck|femur fracture|proximal femur"
        r"|intertrochanteric|subtrochanteric|pertrochanteric"
    ),
    # Matches: distal radius, wrist fracture, Colles, Smith fracture
    "Distal Radius Fracture": (
        r"distal radius|wrist fracture|colles fracture|smith fracture"
        r"|radius fracture"
    ),
    # Matches: ankle fracture, malleolar, pilon, tibial plafond
    "Ankle Fracture": (
        r"ankle fracture|malleolar fracture|pilon fracture"
        r"|tibial plafond|bimalleolar|trimalleolar"
    ),
    # Matches: pelvic fracture, acetabular fracture, acetabulum
    "Pelvic / Acetabular Fracture": (
        r"\bpelvi[cs]\b.*fracture|fracture.*\bpelvi[cs]\b"
        r"|\bacetabul.*fracture|fracture.*\bacetabul"
        r"|pelvic ring|sacral fracture"
    ),
    # Matches: polytrauma, orthopaedic trauma, orthopedic trauma, multiple fractures
    "Polytrauma / Ortho Trauma": (
        r"polytrauma|multiple trauma|orthopaedic trauma|orthopedic trauma"
        r"|multiply injured|multiple fracture"
    ),
    # Matches: fragility fracture, osteoporotic fracture, osteoporosis, atypical femoral
    "Fragility Fracture / Osteoporosis": (
        r"fragility fracture|osteoporotic fracture|osteoporosis.*fracture"
        r"|fracture.*osteoporosis|atypical femoral fracture|\bfls\b.*fracture"
        r"|fracture liaison"
    ),

    # --- Spine ---

    # Matches: spinal fusion, lumbar fusion, cervical fusion, ACDF, PLIF, TLIF, laminectomy
    "Spine Surgery / Fusion": (
        r"spinal fusion|lumbar fusion|cervical fusion"
        r"|\bacdf\b|\bplif\b|\btlif\b|\balif\b"
        r"|laminectomy|laminoplasty|spine surgery"
        r"|posterior spinal|anterior cervical"
    ),
    # Matches: degenerative disc, disc herniation, lumbar stenosis, spondylosis, sciatica
    "Degenerative Disc Disease": (
        r"degenerative disc|disc herniation|lumbar stenosis"
        r"|spinal stenosis|spondylosis|disc degeneration"
        r"|radiculopathy|sciatica|lumbar disc"
    ),
    # Matches: scoliosis, spinal deformity, kyphosis, adolescent idiopathic
    "Spinal Deformity (Scoliosis)": (
        r"scoliosis|spinal deformity|kyphosis|kyphoscoliosis"
        r"|adolescent idiopathic|adult spinal deformity"
    ),

    # --- Sports Medicine ---

    # Matches: ACL, anterior cruciate ligament, PCL, MCL, knee ligament, multiligament
    "ACL / Knee Ligament": (
        r"\bacl\b|anterior cruciate|posterior cruciate|\bpcl\b"
        r"|\bmcl\b|knee ligament|multiligament|cruciate ligament"
        r"|acl reconstruction|acl repair"
    ),
    # Matches: rotator cuff, supraspinatus, shoulder instability, labral tear, Bankart, SLAP
    "Rotator Cuff / Shoulder": (
        r"rotator cuff|supraspinatus|infraspinatus|subscapularis"
        r"|shoulder instability|labral tear|labrum"
        r"|\bbankart\b|\bslap\b.*shoulder|shoulder dislocation"
    ),
    # Matches: meniscus, meniscal, cartilage repair, osteochondral, microfracture, MACI
    "Meniscus / Cartilage": (
        r"\bmeniscus\b|\bmeniscal\b|cartilage repair|cartilage defect"
        r"|osteochondral|chondral|microfracture|\bmaci\b"
        r"|autologous chondrocyte"
    ),
    # Matches: sports medicine, sports injury, athletic injury, return to sport/play
    "Sports Medicine (General)": (
        r"sports medicine|sports injur|athletic injur"
        r"|return to sport|return to play|sports surgery"
    ),

    # --- Subspecialties ---

    # Matches: pediatric orthop/orthopaed, congenital, DDH, clubfoot, Perthes, SCFE
    "Pediatric Orthopedics": (
        r"(?:pediatric|paediatric).*(?:orthop|orthopaed|fracture|musculoskeletal)"
        r"|(?:orthop|orthopaed).*(?:pediatric|paediatric)"
        r"|developmental dysplasia.*hip|\bddh\b"
        r"|clubfoot|perthes|\bscfe\b|slipped capital femoral"
        r"|congenital.*(?:orthop|limb|skeletal)"
    ),
    # Matches: hand surgery, wrist, carpal tunnel, finger, upper extremity, elbow
    "Hand / Upper Extremity": (
        r"hand surg|wrist surg|carpal tunnel|finger fracture"
        r"|upper extremity|distal humerus|\belbow\b.*fracture"
        r"|fracture.*\belbow\b|trigger finger|dupuytren"
        r"|cubital tunnel|hand injur|hand trauma"
    ),
    # Matches: foot surgery, ankle surgery, hallux valgus, bunion, Achilles, plantar fasci
    "Foot / Ankle": (
        r"foot surg|foot injur|hallux valgus|\bbunion\b"
        r"|achilles.*(?:tendon|repair|rupture)|plantar fasci"
        r"|ankle arthroscopy|ankle arthroplasty|ankle instability"
        r"|lisfranc|calcaneal fracture|midfoot"
    ),
    # Matches: musculoskeletal oncology, bone tumor, sarcoma, giant cell, metastatic bone
    "Musculoskeletal Oncology": (
        r"musculoskeletal oncology|bone tumor|bone tumour"
        r"|osteosarcoma|chondrosarcoma|ewing.*sarcoma"
        r"|giant cell tumor|metastatic bone|bone metasta"
        r"|musculoskeletal tumor|musculoskeletal tumour"
    ),

    # --- Technology & Innovation ---

    # Matches: artificial intelligence, machine learning, deep learning, neural network
    "AI / Machine Learning": r"artificial intelligence|machine learning|deep learning|\bneural network\b",
    # Matches: robotic surgery, robot-assisted, navigation, computer-assisted
    "Robotics / Navigation": (
        r"robotic.*(?:surg|arthroplasty|assist)|robot.assisted"
        r"|computer.assisted.*(?:surg|navigation|ortho)"
        r"|surgical navigation|image.guided.*surg"
        r"|robotic.*(?:knee|hip|spine|orthop)"
    ),
    # Matches: 3D printing, patient-specific, custom implant, additive manufacturing
    "3D Printing / Patient-Specific": (
        r"3d print|three.dimensional print|patient.specific.*(?:implant|instrument|guide)"
        r"|custom implant|additive manufactur|bioprinting"
        r"|patient.matched|personalized implant"
    ),
    # Matches: PRP, platelet-rich plasma, stem cell, growth factor, biologics, bone graft substitute
    "Biologics / PRP / Stem Cells": (
        r"\bprp\b|platelet.rich plasma|stem cell.*(?:orthop|bone|cartilage|musculoskeletal)"
        r"|(?:orthop|bone|cartilage|musculoskeletal).*stem cell"
        r"|growth factor.*bone|bone morphogenetic|\bbmp\b"
        r"|bone graft substitute|biologic.*(?:augment|treat)"
    ),

    # --- Systems & Quality ---

    # Matches: patient-reported outcome, PROM, WOMAC, KOOS, Harris hip, VAS, EQ-5D
    "Outcomes / PROMs": (
        r"patient.reported outcome|\bprom\b|\bproms\b"
        r"|\bwomac\b|\bkoos\b|harris hip score|\bvas\b.*pain"
        r"|\beq.5d\b|\bsf.36\b|\bdash\b.*score|functional outcome"
        r"|patient satisfaction.*(?:orthop|surg|arthroplasty)"
    ),
    # Matches: surgical site infection, SSI, periprosthetic infection, PJI, osteomyelitis
    "Infection / SSI": (
        r"surgical site infection|\bssi\b.*(?:orthop|surg|fracture)"
        r"|periprosthetic.*infection|prosthetic joint infection|\bpji\b"
        r"|osteomyelitis|septic arthritis|wound infection.*(?:orthop|surg|fracture)"
    ),
    # Matches: VTE, DVT, PE, thromboembolism, thromboprophylaxis
    "VTE Prevention": r"thromboembolism|thromboembolic|\bvte\b|\bdvt\b|pulmonary embolism|thromboprophylaxis",
    # Matches: geriatric, frailty, elderly, older adult (near ortho/fracture/surgery context)
    "Geriatric / Frailty": r"geriatric|frailty|\belderly\b|older adult",
    # Matches: simulation, virtual reality, surgical training, cadaver-based
    "Simulation / Training": (
        r"(?:orthop|orthopaed|surgical).*simulat|simulat.*(?:orthop|orthopaed|surgical)"
        r"|virtual reality.*(?:orthop|surg)|(?:orthop|surg).*virtual reality"
        r"|cadaver.based.*train|surgical training|surgical education"
    ),
    # Matches: COVID-19 or COVID near ortho/surgery context
    "COVID-19 Impact": (
        r"\bcovid.?19\b"
        r"|\bcovid\b.*(?:orthop|orthopaed|surgery|fracture|arthroplasty)"
        r"|\bsars.cov.2\b"
        r"|(?:orthop|orthopaed|surgery|arthroplasty).*\bcovid\b"
        r"|pandemic.*(?:orthop|surgery|fracture)"
        r"|(?:orthop|surgery|fracture).*pandemic"
    ),
}

# ── CONCEPTS FOR CHARTS ─────────────────────────────────────────────
# Same concept names in preferred display order (top → bottom in bar charts)
TOP_CONCEPTS_FOR_CHARTS = [
    "Total Knee Arthroplasty (TKA)",
    "Total Hip Arthroplasty (THA)",
    "Shoulder Arthroplasty",
    "Revision Arthroplasty",
    "Hip / Femur Fracture",
    "Distal Radius Fracture",
    "Ankle Fracture",
    "Pelvic / Acetabular Fracture",
    "Polytrauma / Ortho Trauma",
    "Fragility Fracture / Osteoporosis",
    "Spine Surgery / Fusion",
    "Degenerative Disc Disease",
    "Spinal Deformity (Scoliosis)",
    "ACL / Knee Ligament",
    "Rotator Cuff / Shoulder",
    "Meniscus / Cartilage",
    "Sports Medicine (General)",
    "Pediatric Orthopedics",
    "Hand / Upper Extremity",
    "Foot / Ankle",
    "Musculoskeletal Oncology",
    "AI / Machine Learning",
    "Robotics / Navigation",
    "3D Printing / Patient-Specific",
    "Biologics / PRP / Stem Cells",
    "Outcomes / PROMs",
    "Infection / SSI",
    "VTE Prevention",
    "Geriatric / Frailty",
    "Simulation / Training",
    "COVID-19 Impact",
]

# ── DOMAIN GROUPINGS (for subplot grid) ─────────────────────────────
DOMAIN_GROUPS = {
    "Arthroplasty & Joint Replacement": [
        "Total Knee Arthroplasty (TKA)", "Total Hip Arthroplasty (THA)",
        "Shoulder Arthroplasty", "Revision Arthroplasty",
    ],
    "Trauma & Fracture": [
        "Hip / Femur Fracture", "Distal Radius Fracture",
        "Ankle Fracture", "Pelvic / Acetabular Fracture",
        "Polytrauma / Ortho Trauma", "Fragility Fracture / Osteoporosis",
    ],
    "Spine": [
        "Spine Surgery / Fusion", "Degenerative Disc Disease",
        "Spinal Deformity (Scoliosis)",
    ],
    "Sports Medicine": [
        "ACL / Knee Ligament", "Rotator Cuff / Shoulder",
        "Meniscus / Cartilage", "Sports Medicine (General)",
    ],
    "Subspecialties": [
        "Pediatric Orthopedics", "Hand / Upper Extremity",
        "Foot / Ankle", "Musculoskeletal Oncology",
    ],
    "Technology, Quality & Systems": [
        "AI / Machine Learning", "Robotics / Navigation",
        "3D Printing / Patient-Specific", "Biologics / PRP / Stem Cells",
        "Outcomes / PROMs", "Infection / SSI", "VTE Prevention",
        "Geriatric / Frailty", "Simulation / Training", "COVID-19 Impact",
    ],
}
