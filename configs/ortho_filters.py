"""
FILTER KEYWORDS — ORTHOPEDIC SURGERY
====================================================
Copy this file to the project root as filter_results.py (replacing the
Trauma version) when switching to the Orthopedic Surgery field.

NOTE: This file provides the keyword lists only. The load_and_filter()
function and scoring logic in the original filter_results.py are generic
and work for any field. When adapting:
  1. Copy these keyword lists into filter_results.py (replace existing lists)
  2. Rename TOP_TRAUMA_JOURNALS → TOP_ORTHO_JOURNALS and update references
  3. Run: python validate.py --dry-run  (check survival rate is 60-80%)
"""

# ============================================================
# ORTHO-RELEVANT JOURNALS (high weight in scoring)
# ============================================================
TOP_ORTHO_JOURNALS = [
    # Core orthopaedic
    "journal of bone and joint surgery",
    "bone and joint journal",
    "clinical orthopaedics and related research",
    "journal of orthopaedic trauma",
    "journal of orthopaedic research",
    "injury",
    "acta orthopaedica",
    "international orthopaedics",
    "musculoskeletal surgery",
    # Arthroplasty
    "journal of arthroplasty",
    "arthroplasty today",
    # Sports medicine / arthroscopy
    "american journal of sports medicine",
    "arthroscopy",
    "knee surgery sports traumatology arthroscopy",
    "journal of shoulder and elbow surgery",
    "sports health",
    "british journal of sports medicine",
    # Spine
    "spine",
    "spine journal",
    "european spine journal",
    "global spine journal",
    "journal of neurosurgery spine",
    # Subspecialty
    "journal of hand surgery",
    "hand",
    "foot and ankle international",
    "foot and ankle surgery",
    "journal of pediatric orthopaedics",
    # General surgery / high-impact
    "annals of surgery",
    "jama surgery",
    "jama",
    "new england journal of medicine",
    "lancet",
    "bmj",
    "cochrane database of systematic reviews",
]

# ============================================================
# EXCLUSION KEYWORDS (remove off-topic noise)
# ============================================================
EXCLUDE_TITLE_KEYWORDS = [
    # Materials science / chemistry noise
    "photoelectron spectroscopy", "ti3c2", "mxene",
    "tribological", "coatings", "ceramics", "nanocomposite",
    "electrochemical", "corrosion", "photocatalyst", "semiconductor",
    # Unrelated medical specialties
    "ivf", "in vitro fertilization", "assisted reproduction",
    "breast augmentation", "rhinoplasty", "cosmetic surgery",
    "alzheimer", "parkinson",
    # Psychology / social (not surgical ortho)
    "childhood adversity", "adverse childhood experience",
    "post-traumatic stress disorder", "ptsd",
    "psychological trauma", "emotional trauma",
    "moral injury", "telemental health",
    # Veterinary
    "veterinary", "canine", "feline", "equine",
    # Oncology (non-MSK)
    "hepatocellular carcinoma", "breast cancer", "prostate cancer",
    "colorectal cancer", "melanoma", "lymphoma",
    "chemotherapy", "radiotherapy",
    # Transplant
    "organ transplant", "kidney transplant", "liver transplant",
    "heart transplant",
    # Unrelated specialties
    "dental implant", "orthodontic", "periodontal", "endodontic",
    "dermatol", "psoriasis", "eczema",
    "ophthalmol", "cataract", "glaucoma", "retinal",
    "infertility", "prenatal", "obstetric", "neonatal",
    "chronic obstructive", "asthma", "copd",
    "diabetes mellitus", "insulin resistance",
    "multiple sclerosis", "amyotrophic lateral",
    "dialysis", "urethral stricture",
    # Cardiac (non-ortho)
    "cardiac catheter", "coronary artery", "atrial fibrillation",
    "myocardial infarction",
]

EXCLUDE_ABSTRACT_KEYWORDS = [
    "photoelectron", "mxene", "nanoparticle", "catalyst",
    "semiconductor", "electrode",
]

# ============================================================
# INCLUSION: must match at least one of these in title or abstract
# ============================================================
REQUIRE_ANY_KEYWORD = [
    # Core orthopaedic terms
    "orthop", "orthopaed", "musculoskeletal",
    "fracture", "arthroplasty", "joint replacement",
    # Arthroplasty specific
    "total knee", "total hip", "shoulder replacement",
    "hemiarthroplasty", "unicompartmental", "revision",
    "periprosthetic", "prosthetic joint",
    # Fracture types
    "femoral neck", "intertrochanteric", "distal radius",
    "ankle fracture", "tibial", "humeral", "calcaneal",
    "pelvic fracture", "acetabular", "pilon",
    # Fixation / hardware
    "intramedullary", "plate fixation", "screw",
    "open reduction", "internal fixation", "ORIF",
    "external fixation", "nailing",
    # Spine
    "spinal fusion", "lumbar", "cervical", "laminectomy",
    "disc herniation", "scoliosis", "kyphosis",
    "stenosis", "spondylosis", "ACDF", "TLIF",
    # Sports medicine
    "ACL", "anterior cruciate", "posterior cruciate",
    "rotator cuff", "meniscus", "meniscal",
    "labral", "labrum", "ligament reconstruction",
    "cartilage repair", "osteochondral",
    "return to sport", "sports medicine",
    # Subspecialty
    "hand surg", "wrist", "carpal tunnel",
    "foot surg", "hallux valgus", "Achilles",
    "pediatric orthop", "pediatric fracture",
    "bone tumor", "osteosarcoma", "sarcoma",
    # Technology
    "robotic", "robot-assisted", "navigation",
    "3D print", "patient-specific", "custom implant",
    "artificial intelligence", "machine learning", "deep learning",
    # Biologics
    "PRP", "platelet-rich", "stem cell", "bone graft",
    "bone morphogenetic", "BMP",
    # Outcomes / quality
    "patient-reported outcome", "PROM", "WOMAC", "KOOS",
    "surgical site infection", "SSI", "periprosthetic infection",
    "VTE", "DVT", "thromboembolism",
    # General surgical
    "osteotomy", "arthrodesis", "tendon repair",
    "bone healing", "nonunion", "malunion",
    "implant", "prosthesis", "osteoporosis",
    "fragility fracture", "geriatric", "frailty",
    "rehabilitation", "physical therapy",
    "complication", "readmission",
]
