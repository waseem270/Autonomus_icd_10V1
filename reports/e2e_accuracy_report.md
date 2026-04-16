# End-to-End Accuracy Report (Evidence-Based MEAT)

**Generated**: 2026-04-10 11:03:26 UTC  
**Pipeline Mode**: Evidence-Only (No Fallback Enrichment)  
**Pipeline Status**: ⚠️ SOME CRITERIA NOT MET

---

## 1. Database Connection

| Metric | Value |
|--------|-------|
| DB Type | sqlite |
| Host | local |
| Connected | ✅ |
| Tables Created | audit_log, detected_diseases, documents, icd10_codes, icd_mappings, meat_validation, rejected_diagnoses |
| Total Tables | 7 |

## 2. PDF Processing Summary

| Metric | Value |
|--------|-------|
| Total PDFs | 10 |
| Successful | 10 |
| Failed | 0 |
| Total Diseases Found | 76 |
| Total ICD Mapped | 76 |
| Total Processing Time | 257.6s |
| Avg Time per PDF | 25.8s |

## 3. ICD Accuracy Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ICD Precision | 98.7% | > 90% | ✅ |
| Hallucination Rate | 1.3% | < 10% | ✅ |
| ICD Specificity | 98.7% | > 80% | ✅ |
| ICD Mismatch Rate | 0.0% | < 20% | ✅ |
| Duplicates | 0 | 0 | ✅ |
| Symptom Overcoding | 6 | 0 | ⚠️ |
| Confidence Average | 0.672 | > 0.5 | ✅ |

## 4. Evidence-Based MEAT Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MEAT Coverage (Strong+Moderate) | 42.7% | > 50% | ⚠️ |
| Evidence-Based Rate | 100.0% | > 60% | ✅ |
| MEAT Validation Pass Rate | 42.7% | > 50% | ⚠️ |

### MEAT Tier Distribution (All PDFs)

| Tier | Count | % |
|------|-------|---|
| Strong Evidence (3-4 components) | 19 | 25.0% |
| Moderate Evidence (2 components) | 16 | 21.1% |
| Weak Evidence (1 component) | 41 | 53.9% |
| No MEAT (0 components) | 0 | 0.0% |
| **Evidence-Based Total** | **76** | **100.0%** |

## 5. Per-PDF Summary

| PDF | Diseases | ICD | MEAT% | Strong | Mod | Weak | None | Halluc | Conf | Time |
|-----|----------|-----|-------|--------|-----|------|------|--------|------|------|
| ✅ 1477127_2.19.2025.pdf | 6 | 6 | 66.7% | 2 | 2 | 2 | 0 | 0 | 0.74 | 32.01s |
| ✅ 2137395_2.18.2025.pdf | 12 | 12 | 25.0% | 2 | 1 | 9 | 0 | 0 | 0.614 | 35.49s |
| ✅ 4195560_2.19.2025.pdf | 7 | 7 | 57.1% | 3 | 1 | 3 | 0 | 0 | 0.743 | 23.06s |
| ✅ 5509215_2.14.2025.pdf | 12 | 12 | 33.3% | 2 | 2 | 8 | 0 | 0 | 0.639 | 39.05s |
| ✅ 5590960_2.18.2025.pdf | 2 | 2 | 0.0% | 0 | 0 | 2 | 0 | 0 | 0.525 | 9.14s |
| ✅ 5841830_2.17.2025.pdf | 12 | 12 | 83.3% | 3 | 7 | 2 | 0 | 0 | 0.761 | 28.72s |
| ✅ 7750233_2.18.2025.pdf | 7 | 7 | 28.6% | 1 | 1 | 5 | 0 | 0 | 0.606 | 26.52s |
| ✅ 8331786_2.18.2025.pdf | 5 | 5 | 60.0% | 2 | 1 | 2 | 0 | 1 | 0.738 | 16.69s |
| ✅ 8884485_2.14.2025.pdf | 3 | 3 | 33.3% | 1 | 0 | 2 | 0 | 0 | 0.678 | 12.76s |
| ✅ 945509_2.17.2025.pdf | 10 | 10 | 40.0% | 3 | 1 | 6 | 0 | 0 | 0.674 | 34.16s |

## 6. Per-PDF Disease Details

### 1477127_2.19.2025.pdf

| # | Disease | ICD Code | MEAT Tier | Score | M | E | A | T | Conf | Section |
|---|---------|----------|-----------|-------|---|---|---|---|------|---------|
| 1 | Back pain | M54.9 | strong_evidence | 3 | ✓ | ✗ | ✓ | ✓ | 0.945 | assessment_and_plan |
| 2 | Urinary frequency | R35.0 | strong_evidence | 3 | ✓ | ✗ | ✓ | ✓ | 0.945 | assessment_and_plan |
| 3 | Chronic coronary artery disease | I25.10 | moderate_evidence | 2 | ✗ | ✗ | ✓ | ✓ | 0.75 | assessment_and_plan |
| 4 | Hypothyroidism, postradioiodine therapy | E89.0 | moderate_evidence | 2 | ✗ | ✗ | ✓ | ✓ | 0.75 | assessment_and_plan |
| 5 | Dysuria | R30.0 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | Assessment |
| 6 | Chronic pancreatitis | K86.1 | weak_evidence | 1 | ✗ | ✗ | ✗ | ✓ | 0.525 | past_medical_history |

### 2137395_2.18.2025.pdf

| # | Disease | ICD Code | MEAT Tier | Score | M | E | A | T | Conf | Section |
|---|---------|----------|-----------|-------|---|---|---|---|------|---------|
| 1 | Colovesical fistula | N32.1 | strong_evidence | 3 | ✓ | ✓ | ✓ | ✗ | 0.945 | assessment_and_plan |
| 2 | Ulcerative pancolitis with unspecified complications | K51.019 | strong_evidence | 3 | ✓ | ✗ | ✓ | ✓ | 0.945 | assessment_and_plan |
| 3 | Chronic obstructive pulmonary disease | J44.9 | moderate_evidence | 2 | ✓ | ✗ | ✗ | ✓ | 0.75 | problem_list |
| 4 | Coronary artery disease | I25.10 | weak_evidence | 1 | ✗ | ✗ | ✗ | ✓ | 0.525 | problem_list |
| 5 | Gastroesophageal reflux disease | K21.9 | weak_evidence | 1 | ✗ | ✗ | ✗ | ✓ | 0.525 | problem_list |
| 6 | Mixed anxiety and depressive disorder | F41.8 | weak_evidence | 1 | ✗ | ✗ | ✗ | ✓ | 0.525 | problem_list |
| 7 | Mixed hyperlipidemia | E78.2 | weak_evidence | 1 | ✗ | ✗ | ✗ | ✓ | 0.525 | problem_list |
| 8 | Class 1 obesity | E66.01 | weak_evidence | 1 | ✓ | ✗ | ✗ | ✗ | 0.525 | problem_list |
| 9 | Colostomy in place (HCC) | Z93.3 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | Assessment |
| 10 | Presence of coronary artery stent | Z95.5 | weak_evidence | 1 | ✗ | ✗ | ✗ | ✓ | 0.525 | problem_list |
| 11 | Ulcerative pancolitis with complication (HCC) Overview: Ulcerative Colitis: microscopic | K51.90 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | Assessment |
| 12 | Urge incontinence | N39.41 | weak_evidence | 1 | ✓ | ✗ | ✗ | ✗ | 0.525 | problem_list |

### 4195560_2.19.2025.pdf

| # | Disease | ICD Code | MEAT Tier | Score | M | E | A | T | Conf | Section |
|---|---------|----------|-----------|-------|---|---|---|---|------|---------|
| 1 | Overweight | E66.3 | strong_evidence | 4 | ✓ | ✓ | ✓ | ✓ | 0.985 | assessment |
| 2 | Irritable bowel syndrome, unspecified type | K58.9 | strong_evidence | 3 | ✓ | ✗ | ✓ | ✓ | 0.945 | assessment |
| 3 | Hyperlipidemia | E78.5 | strong_evidence | 3 | ✓ | ✓ | ✗ | ✓ | 0.945 | medications |
| 4 | Ulcerative colitis | K51.90 | moderate_evidence | 2 | ✓ | ✗ | ✓ | ✗ | 0.75 | HPI |
| 5 | Primary osteoarthritis of right hip | M16.11 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment |
| 6 | Mitral valve disorder | I34.9 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment |
| 7 | Rosacea | L71.9 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment |

### 5509215_2.14.2025.pdf

| # | Disease | ICD Code | MEAT Tier | Score | M | E | A | T | Conf | Section |
|---|---------|----------|-----------|-------|---|---|---|---|------|---------|
| 1 | Anemia, unspecified type | D64.9 | strong_evidence | 4 | ✓ | ✓ | ✓ | ✓ | 0.985 | assessment_and_plan |
| 2 | Wound of right lower extremity | S81.801A | strong_evidence | 4 | ✓ | ✓ | ✓ | ✓ | 0.985 | assessment_and_plan |
| 3 | Acute respiratory failure with hypoxia | J96.01 | moderate_evidence | 2 | ✓ | ✗ | ✓ | ✗ | 0.75 | assessment_and_plan |
| 4 | Pneumonia of both lungs | J18.9 | moderate_evidence | 2 | ✗ | ✓ | ✓ | ✗ | 0.75 | assessment_and_plan |
| 5 | Chronic heart failure with preserved ejection fraction | I50.32 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment |
| 6 | Chronic Kidney Disease, stage 3b | N18.32 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment |
| 7 | Aspiration into airway, sequela | T17.908S | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment_and_plan |
| 8 | Atrial fibrillation | I48.91 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment |
| 9 | Body Mass Index 40.0-44.9 | Z68.41 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment |
| 10 | Class 3 severe obesity | E66.01 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment |
| 11 | Peripheral vascular disease | I73.9 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment |
| 12 | Vitamin B-12 deficiency | E53.8 | weak_evidence | 1 | ✗ | ✗ | ✗ | ✓ | 0.525 | assessment_and_plan |

### 5590960_2.18.2025.pdf

| # | Disease | ICD Code | MEAT Tier | Score | M | E | A | T | Conf | Section |
|---|---------|----------|-----------|-------|---|---|---|---|------|---------|
| 1 | Influenza A | J10.1 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment_and_plan |
| 2 | Selective mutism | F94.0 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment_and_plan |

### 5841830_2.17.2025.pdf

| # | Disease | ICD Code | MEAT Tier | Score | M | E | A | T | Conf | Section |
|---|---------|----------|-----------|-------|---|---|---|---|------|---------|
| 1 | Chronic obstructive pulmonary disease | J44.9 | strong_evidence | 3 | ✓ | ✗ | ✓ | ✓ | 0.945 | assessment_and_plan |
| 2 | Iron deficiency anemia | D50.9 | strong_evidence | 3 | ✗ | ✓ | ✓ | ✓ | 0.945 | assessment_and_plan |
| 3 | Pressure injury of skin of right buttock, unspecified injury stage | L89.319 | strong_evidence | 3 | ✓ | ✗ | ✓ | ✓ | 0.945 | assessment_and_plan |
| 4 | Atherosclerotic cardiovascular disease | I25.10 | moderate_evidence | 2 | ✗ | ✗ | ✓ | ✓ | 0.75 | assessment_and_plan |
| 5 | Hypertensive heart and chronic kidney disease with heart failure | I13.0 | moderate_evidence | 2 | ✗ | ✗ | ✓ | ✓ | 0.75 | assessment_and_plan |
| 6 | Atrial fibrillation | I48.91 | moderate_evidence | 2 | ✗ | ✗ | ✓ | ✓ | 0.75 | assessment_and_plan |
| 7 | Chronic respiratory failure with hypoxia | J96.11 | moderate_evidence | 2 | ✗ | ✗ | ✓ | ✓ | 0.75 | assessment_and_plan |
| 8 | Pressure ulcer of left buttock, stage 4 | L89.324 | moderate_evidence | 2 | ✓ | ✗ | ✓ | ✗ | 0.75 | assessment_and_plan |
| 9 | Thrombocytosis | D75.839 | moderate_evidence | 2 | ✓ | ✗ | ✓ | ✗ | 0.75 | assessment_and_plan |
| 10 | Urinary incontinence | R32 | moderate_evidence | 2 | ✓ | ✗ | ✓ | ✗ | 0.75 | assessment_and_plan |
| 11 | Atrial flutter | I48.92 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment_and_plan |
| 12 | Foley catheter in place | Z97.8 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment_and_plan |

### 7750233_2.18.2025.pdf

| # | Disease | ICD Code | MEAT Tier | Score | M | E | A | T | Conf | Section |
|---|---------|----------|-----------|-------|---|---|---|---|------|---------|
| 1 | Hyperglycemia | R73.9 | strong_evidence | 3 | ✓ | ✓ | ✓ | ✗ | 0.945 | Assessment |
| 2 | Personal history of adenomatous polyp of colon | Z86.010 | moderate_evidence | 2 | ✓ | ✗ | ✓ | ✗ | 0.675 | Assessment |
| 3 | Blood pressure is better after resting.? Whitecoat hypertension. Blood pressure at home | I10 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | Assessment |
| 4 | Calcific tendinitis of left shoulder | M65.222 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | Assessment |
| 5 | Internal hemorrhoids | K64.8 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | Assessment |
| 6 | Overweight | E66.3 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | Assessment |
| 7 | Degenerative disease involving the distal interphalangeal joint of the left little finger | M19.042 | weak_evidence | 1 | ✗ | ✓ | ✗ | ✗ | 0.525 | imaging |

### 8331786_2.18.2025.pdf

| # | Disease | ICD Code | MEAT Tier | Score | M | E | A | T | Conf | Section |
|---|---------|----------|-----------|-------|---|---|---|---|------|---------|
| 1 | Allergic dermatitis, forearms | L23.9 | strong_evidence | 3 | ✗ | ✓ | ✓ | ✓ | 0.945 | Assessment & Plan |
| 2 | Eyelid dermatitis, right upper eyelid | H01.111 | strong_evidence | 3 | ✗ | ✓ | ✓ | ✓ | 0.945 | Assessment & Plan |
| 3 | Allergy, Northeast Profile IgE; Future | T78.40XA | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | Assessment |
| 4 | Polycystic ovarian syndrome | E28.2 | moderate_evidence | 2 | ✗ | ✗ | ✓ | ✓ | 0.75 | Assessment & Plan |
| 5 | Eyelid dermatitis, left upper eyelid | H01.114 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | Assessment & Plan |

### 8884485_2.14.2025.pdf

| # | Disease | ICD Code | MEAT Tier | Score | M | E | A | T | Conf | Section |
|---|---------|----------|-----------|-------|---|---|---|---|------|---------|
| 1 | Cellulitis of right leg | L03.115 | strong_evidence | 4 | ✓ | ✓ | ✓ | ✓ | 0.985 | assessment_plan |
| 2 | History of throat cancer | Z85.21 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | history_present_illness |
| 3 | Stage 3a chronic kidney disease | N18.31 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment_plan |

### 945509_2.17.2025.pdf

| # | Disease | ICD Code | MEAT Tier | Score | M | E | A | T | Conf | Section |
|---|---------|----------|-----------|-------|---|---|---|---|------|---------|
| 1 | Dermatitis | L30.9 | strong_evidence | 3 | ✓ | ✗ | ✓ | ✓ | 0.945 | assessment |
| 2 | Iron deficiency anemia | D50.9 | strong_evidence | 3 | ✗ | ✓ | ✓ | ✓ | 0.945 | assessment |
| 3 | Nausea | R11.0 | strong_evidence | 3 | ✗ | ✓ | ✓ | ✓ | 0.945 | assessment |
| 4 | Elevated blood pressure reading | R03.0 | moderate_evidence | 2 | ✓ | ✗ | ✓ | ✗ | 0.75 | assessment |
| 5 | Atherosclerosis | I70.90 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment |
| 6 | Hypokalemia | E87.6 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment |
| 7 | Pleural thickening | J92.9 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment |
| 8 | Subdural hygroma | G96.11 | weak_evidence | 1 | ✗ | ✗ | ✓ | ✗ | 0.525 | assessment |
| 9 | Herpes zoster | B02.9 | weak_evidence | 1 | ✗ | ✗ | ✗ | ✓ | 0.525 | Problem List |
| 10 | Seasonal allergic rhinitis | J30.2 | weak_evidence | 1 | ✗ | ✗ | ✗ | ✓ | 0.525 | Problem List |

## 7. Final Criteria

| Criterion | Result |
|-----------|--------|
| DB Connected | ✅ |
| Tables Created (≥5) | ✅ |
| All PDFs Processed | ✅ |
| No Crashes | ✅ |
| Hallucination < 10% | ✅ |
| MEAT Coverage > 50% | ⚠️ |
| Duplicates Removed | ✅ |
| Evidence-Based > 60% | ✅ |

**Overall**: ⚠️ PARTIAL
