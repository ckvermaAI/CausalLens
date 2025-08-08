-- 1. Select 50 random adult patients 
WITH random_patients AS (
  SELECT subject_id
  FROM `physionet-data.mimiciv_2_2_hosp.patients`
  WHERE anchor_age BETWEEN 18 AND 89
  ORDER BY RAND()
  LIMIT 50
),

-- 2. Discharge notes
discharge_notes AS (
  SELECT 
    d.subject_id,
    d.hadm_id,
    d.charttime,
    d.text AS discharge_summary
  FROM `physionet-data.mimiciv_note.discharge` d
  JOIN `physionet-data.mimiciv_note.discharge_detail` dd
    ON d.note_id = dd.note_id
  JOIN random_patients rp 
    ON d.subject_id = rp.subject_id
),

-- 3. Pre-aggregated diagnoses
diagnoses_agg AS (
  SELECT 
    subject_id,
    hadm_id,
    ARRAY_AGG(DISTINCT ddi.long_title IGNORE NULLS) AS diagnoses
  FROM `physionet-data.mimiciv_2_2_hosp.diagnoses_icd` d
  LEFT JOIN `physionet-data.mimiciv_2_2_hosp.d_icd_diagnoses` ddi
    ON d.icd_code = ddi.icd_code AND d.icd_version = ddi.icd_version
  GROUP BY subject_id, hadm_id
),

-- 4. Pre-aggregated prescriptions
prescriptions_agg AS (
  SELECT 
    subject_id,
    hadm_id,
    ARRAY_AGG(DISTINCT drug IGNORE NULLS) AS medications
  FROM `physionet-data.mimiciv_2_2_hosp.prescriptions`
  GROUP BY subject_id, hadm_id
),

-- 5. Pre-aggregated lab results
labs_agg AS (
  SELECT 
    l.subject_id,
    l.hadm_id,
    ARRAY_AGG(DISTINCT CONCAT(li.label, ': ', CAST(l.valuenum AS STRING), ' ', l.valueuom) IGNORE NULLS) AS lab_results
  FROM `physionet-data.mimiciv_2_2_hosp.labevents` l
  LEFT JOIN `physionet-data.mimiciv_2_2_hosp.d_labitems` li
    ON l.itemid = li.itemid
  GROUP BY l.subject_id, l.hadm_id
)

-- 6. Final join (no explosion)
SELECT 
  dn.subject_id,
  dn.hadm_id,
  dn.charttime AS discharge_time,
  dn.discharge_summary,
  IF(dx.diagnoses IS NOT NULL, ARRAY_TO_STRING(dx.diagnoses, '; '), NULL) AS diagnoses,
  IF(pr.medications IS NOT NULL, ARRAY_TO_STRING(pr.medications, '; '), NULL) AS medications,
  IF(lb.lab_results IS NOT NULL, ARRAY_TO_STRING(lb.lab_results, '; '), NULL) AS lab_results
FROM discharge_notes dn
LEFT JOIN diagnoses_agg dx ON dn.subject_id = dx.subject_id AND dn.hadm_id = dx.hadm_id
LEFT JOIN prescriptions_agg pr ON dn.subject_id = pr.subject_id AND dn.hadm_id = pr.hadm_id
LEFT JOIN labs_agg lb ON dn.subject_id = lb.subject_id AND dn.hadm_id = lb.hadm_id
