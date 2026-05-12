"""
radlex.py — NarrateRad
=======================
RadLex vocabulary integration — two functions:

1. RADLEX_PROMPT
   An expanded Whisper context prompt containing hundreds of RSNA RadLex
   preferred terms organised by body system. Drop this into transcribe.py
   to replace RADIOLOGY_PROMPT for immediate accuracy improvement.

2. standardise(text) -> tuple[str, list[Correction]]
   Post-transcription term standardiser. Maps informal or colloquial
   radiology language to RadLex preferred terms.
   e.g. "big heart" → "cardiomegaly"
        "fluid around the lung" → "pleural effusion"
        "air in the chest" → "pneumothorax"

Usage:
    from radlex import RADLEX_PROMPT, standardise

    # In transcribe.py — replace RADIOLOGY_PROMPT with RADLEX_PROMPT
    RADIOLOGY_PROMPT = RADLEX_PROMPT

    # After transcription — standardise the text
    standardised_text, corrections = standardise(result.text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ── Expanded RadLex vocabulary prompt ────────────────────────────────────────
# Contains RSNA RadLex preferred terms organised by body system.
# Use this as the initial_prompt for Whisper to improve recognition of
# medical terminology across all radiology subspecialties.

RADLEX_PROMPT: str = (
    "Radiology report dictation using RadLex standardised terminology. "

    # ── Chest / Pulmonary ──
    "Pulmonary terms: pneumothorax, tension pneumothorax, hydropneumothorax, "
    "pleural effusion, empyema, haemothorax, chylothorax, pneumomediastinum, "
    "consolidation, airspace opacity, ground glass opacity, reticular opacity, "
    "nodular opacity, interstitial opacity, alveolar opacity, "
    "atelectasis, subsegmental atelectasis, lobar atelectasis, "
    "bronchiectasis, cylindrical bronchiectasis, cystic bronchiectasis, "
    "pulmonary embolism, pulmonary infarction, pulmonary oedema, "
    "pulmonary fibrosis, honeycombing, traction bronchiectasis, "
    "pulmonary nodule, solitary pulmonary nodule, pulmonary mass, "
    "cavitation, pulmonary cavity, lung abscess, "
    "hilar enlargement, bilateral hilar lymphadenopathy, "
    "mediastinal widening, mediastinal mass, mediastinal lymphadenopathy, "
    "costophrenic angle, blunting, hemidiaphragm, tracheal deviation, "
    "hyperinflation, air trapping, mosaic attenuation, "

    # ── Cardiac ──
    "cardiomegaly, cardiac enlargement, pericardial effusion, "
    "pericardial thickening, cardiac tamponade, "
    "left ventricular enlargement, right ventricular enlargement, "
    "left atrial enlargement, right atrial enlargement, "
    "aortic dilatation, aortic aneurysm, aortic dissection, "
    "aortic stenosis, mitral valve calcification, "
    "coronary artery calcification, cardiac calcification, "

    # ── Abdominal / GI ──
    "hepatomegaly, splenomegaly, hepatosplenomegaly, "
    "hepatic steatosis, cirrhosis, portal hypertension, "
    "hepatic lesion, hepatic mass, hepatocellular carcinoma, "
    "focal nodular hyperplasia, hepatic haemangioma, "
    "biliary dilatation, intrahepatic biliary dilatation, "
    "common bile duct dilatation, cholelithiasis, choledocholithiasis, "
    "cholecystitis, gallbladder wall thickening, "
    "pancreatitis, pancreatic duct dilatation, pancreatic mass, "
    "bowel obstruction, small bowel obstruction, large bowel obstruction, "
    "pneumoperitoneum, free air, free fluid, ascites, "
    "bowel wall thickening, mesenteric fat stranding, "
    "appendicitis, periappendiceal fat stranding, "
    "diverticulitis, diverticulosis, "
    "adrenal adenoma, adrenal mass, adrenal enlargement, "

    # ── Genitourinary / Renal ──
    "nephrolithiasis, ureterolithiasis, renal calculus, ureteric calculus, "
    "hydronephrosis, hydroureter, hydro-ureteronephrosis, "
    "renal mass, renal cell carcinoma, angiomyolipoma, "
    "renal cyst, simple renal cyst, complex renal cyst, "
    "pyelonephritis, renal abscess, perinephric stranding, "
    "bladder wall thickening, bladder mass, "
    "prostatic enlargement, benign prostatic hyperplasia, "
    "uterine fibroid, leiomyoma, ovarian cyst, ovarian mass, "

    # ── Musculoskeletal ──
    "fracture, acute fracture, stress fracture, pathological fracture, "
    "comminuted fracture, displaced fracture, non-displaced fracture, "
    "compression fracture, vertebral body fracture, "
    "dislocation, subluxation, "
    "osteoporosis, osteopenia, bone mineral density, "
    "osteophyte, spondylosis, spondylolisthesis, anterolisthesis, retrolisthesis, "
    "disc herniation, disc protrusion, disc extrusion, disc sequestration, "
    "intervertebral disc, disc space narrowing, "
    "spinal stenosis, central canal stenosis, foraminal stenosis, "
    "cord compression, cauda equina compression, "
    "lytic lesion, sclerotic lesion, periosteal reaction, cortical breach, "
    "joint effusion, synovial thickening, cartilage loss, "
    "rotator cuff tear, meniscal tear, ligament tear, "

    # ── Neurological ──
    "intracranial haemorrhage, subarachnoid haemorrhage, "
    "subdural haematoma, extradural haematoma, intraparenchymal haemorrhage, "
    "intraventricular haemorrhage, "
    "ischaemic stroke, cerebral infarction, lacunar infarction, "
    "cerebral oedema, midline shift, herniation, transtentorial herniation, "
    "hydrocephalus, ventriculomegaly, cerebral atrophy, "
    "white matter changes, leukoaraiosis, periventricular white matter, "
    "cerebral mass, brain metastasis, meningioma, glioma, "
    "cerebral aneurysm, arteriovenous malformation, "
    "sinusitis, mastoiditis, "

    # ── Vascular ──
    "deep vein thrombosis, venous thrombosis, "
    "arterial occlusion, arterial stenosis, "
    "aortic aneurysm, abdominal aortic aneurysm, thoracic aortic aneurysm, "
    "dissection, intramural haematoma, penetrating aortic ulcer, "
    "atheromatous disease, atherosclerosis, calcification, "

    # ── Interventional Radiology ──
    "percutaneous drainage, image guided biopsy, "
    "arterial access, venous access, central venous catheter, "
    "PICC line, port-a-cath, tunnelled catheter, "
    "angioplasty, stenting, embolisation, "
    "thrombolysis, thrombectomy, "
    "nephrostomy, biliary drainage, cholecystostomy, "
    "vertebroplasty, kyphoplasty, "

    # ── General descriptors ──
    "heterogeneous, homogeneous, hyperdense, hypodense, hyperechoic, hypoechoic, "
    "enhancing, non-enhancing, avid enhancement, rim enhancement, "
    "well-defined, ill-defined, spiculated, lobulated, "
    "calcified, calcification, ossification, "
    "bilateral, unilateral, ipsilateral, contralateral, "
    "proximal, distal, medial, lateral, superior, inferior, "
    "adjacent, abutting, invading, displacing, compressing."
)


# ── Synonym mapper ────────────────────────────────────────────────────────────

@dataclass
class Correction:
    """A single term standardisation applied to the transcript."""
    original: str
    standardised: str
    radlex_concept: str

    def __str__(self) -> str:
        return f'"{self.original}" → "{self.standardised}" ({self.radlex_concept})'


# Informal / colloquial → RadLex preferred term
# Format: "informal phrase" : ("RadLex preferred term", "RadLex concept label")
_SYNONYM_MAP: dict[str, tuple[str, str]] = {
    # Cardiac
    "big heart":                    ("cardiomegaly", "RID1385 Cardiomegaly"),
    "enlarged heart":               ("cardiomegaly", "RID1385 Cardiomegaly"),
    "heart is enlarged":            ("cardiomegaly is present", "RID1385 Cardiomegaly"),
    "small heart":                  ("reduced cardiac silhouette", "RID1386 Cardiac size"),
    "fluid around the heart":       ("pericardial effusion", "RID1384 Pericardial effusion"),
    "water around the heart":       ("pericardial effusion", "RID1384 Pericardial effusion"),

    # Pulmonary
    "air in the chest":             ("pneumothorax", "RID5352 Pneumothorax"),
    "collapsed lung":               ("atelectasis", "RID28493 Atelectasis"),
    "lung collapse":                ("atelectasis", "RID28493 Atelectasis"),
    "partial collapse":             ("subsegmental atelectasis", "RID28493 Atelectasis"),
    "fluid in the chest":           ("pleural effusion", "RID1339 Pleural effusion"),
    "fluid around the lung":        ("pleural effusion", "RID1339 Pleural effusion"),
    "water on the lung":            ("pleural effusion", "RID1339 Pleural effusion"),
    "white patch":                  ("airspace opacity", "RID5350 Opacity"),
    "white out":                    ("consolidation", "RID28748 Consolidation"),
    "fluffy":                       ("airspace opacity", "RID5350 Opacity"),
    "haziness":                     ("opacity", "RID5350 Opacity"),
    "hazy":                         ("opacification", "RID5350 Opacity"),
    "patchy":                       ("patchy airspace opacity", "RID5350 Opacity"),
    "spot on the lung":             ("pulmonary nodule", "RID3530 Pulmonary nodule"),
    "shadow":                       ("opacity", "RID5350 Opacity"),

    # Abdominal
    "big liver":                    ("hepatomegaly", "RID5074 Hepatomegaly"),
    "enlarged liver":               ("hepatomegaly", "RID5074 Hepatomegaly"),
    "big spleen":                   ("splenomegaly", "RID5099 Splenomegaly"),
    "enlarged spleen":              ("splenomegaly", "RID5099 Splenomegaly"),
    "gallstones":                   ("cholelithiasis", "RID5060 Cholelithiasis"),
    "kidney stones":                ("nephrolithiasis", "RID5098 Nephrolithiasis"),
    "ureteric stone":               ("ureterolithiasis", "RID34588 Ureterolithiasis"),
    "fluid in the belly":           ("ascites", "RID34539 Ascites"),
    "free fluid abdomen":           ("ascites", "RID34539 Ascites"),
    "dilated bowel":                ("bowel obstruction", "RID5053 Bowel obstruction"),
    "air under the diaphragm":      ("pneumoperitoneum", "RID5363 Pneumoperitoneum"),
    "free air":                     ("pneumoperitoneum", "RID5363 Pneumoperitoneum"),

    # Musculoskeletal
    "slipped disc":                 ("intervertebral disc herniation", "RID50127 Disc herniation"),
    "disc slip":                    ("intervertebral disc herniation", "RID50127 Disc herniation"),
    "bone spur":                    ("osteophyte", "RID5196 Osteophyte"),
    "bone spurs":                   ("osteophytes", "RID5196 Osteophyte"),
    "wear and tear":                ("spondylosis", "RID5191 Spondylosis"),
    "cracked":                      ("fracture", "RID3565 Fracture"),
    "crack":                        ("fracture", "RID3565 Fracture"),
    "broken":                       ("fracture", "RID3565 Fracture"),
    "hairline fracture":            ("stress fracture", "RID3565 Fracture"),
    "thinning of bones":            ("osteopenia", "RID5195 Osteopenia"),
    "brittle bones":                ("osteoporosis", "RID5197 Osteoporosis"),
    "fluid in the joint":           ("joint effusion", "RID5201 Joint effusion"),

    # Neurological
    "bleed in the brain":           ("intracranial haemorrhage", "RID4700 Intracranial haemorrhage"),
    "brain bleed":                  ("intracranial haemorrhage", "RID4700 Intracranial haemorrhage"),
    "blood clot brain":             ("intracranial haemorrhage", "RID4700 Intracranial haemorrhage"),
    "stroke":                       ("ischaemic stroke", "RID4700 Cerebral infarction"),
    "water on the brain":           ("hydrocephalus", "RID4710 Hydrocephalus"),
    "brain shrinkage":              ("cerebral atrophy", "RID4799 Cerebral atrophy"),

    # Vascular
    "blood clot leg":               ("deep vein thrombosis", "RID34609 Deep vein thrombosis"),
    "clot in vein":                 ("venous thrombosis", "RID34609 Venous thrombosis"),
    "blocked artery":               ("arterial occlusion", "RID5361 Arterial occlusion"),
    "bulge in aorta":               ("aortic aneurysm", "RID5358 Aortic aneurysm"),
    "aorta tear":                   ("aortic dissection", "RID5357 Aortic dissection"),
}


def standardise(text: str) -> tuple[str, list[Correction]]:
    """
    Map informal radiology language to RadLex preferred terms.

    Parameters
    ----------
    text : str
        Raw transcript text from the radiologist's dictation.

    Returns
    -------
    tuple[str, list[Correction]]
        (standardised_text, list of corrections applied)

    Example
    -------
    >>> text, corrections = standardise("There is a big heart and fluid around the lung.")
    >>> print(text)
    "There is cardiomegaly and pleural effusion."
    >>> for c in corrections: print(c)
    "big heart" → "cardiomegaly" (RID1385 Cardiomegaly)
    "fluid around the lung" → "pleural effusion" (RID1339 Pleural effusion)
    """
    corrections: list[Correction] = []
    result = text

    # Sort by length descending so longer phrases match before sub-phrases
    sorted_map = sorted(_SYNONYM_MAP.items(), key=lambda x: len(x[0]), reverse=True)

    for informal, (preferred, concept) in sorted_map:
        pattern = re.compile(r'\b' + re.escape(informal) + r'\b', re.IGNORECASE)
        if pattern.search(result):
            # Preserve capitalisation of first letter if at start of sentence
            def replace(match: re.Match) -> str:
                if match.start() == 0 or result[match.start() - 2] in '.!?':
                    return preferred.capitalize()
                return preferred

            new_result = pattern.sub(replace, result)
            if new_result != result:
                corrections.append(Correction(
                    original=informal,
                    standardised=preferred,
                    radlex_concept=concept,
                ))
                result = new_result

    return result, corrections


# ── Smoke test ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    TEST_CASES = [
        "There is a big heart and fluid around the lung. No air in the chest.",
        "The liver is enlarged. There are gallstones. Free fluid in the belly.",
        "There is a slipped disc at L4 L5 with bone spurs.",
        "Brain bleed noted on the right side. No water on the brain.",
        "Spot on the lung in the right upper lobe. No collapse.",
    ]

    print("=" * 60)
    print("NarrateRad — radlex.py smoke test")
    print("=" * 60)

    for text in TEST_CASES:
        standardised, corrections = standardise(text)
        print(f"\nOriginal:     {text}")
        if corrections:
            print(f"Standardised: {standardised}")
            for c in corrections:
                print(f"  ✓ {c}")
        else:
            print("  No informal terms found.")
