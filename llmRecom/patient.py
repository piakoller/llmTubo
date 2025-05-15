from collections import namedtuple
from dataclasses import dataclass

import pandas as pd


@dataclass
class Patient:
    id: str
    main_diagnosis_text: str
    secondary_diagnoses: str
    clinical_info: str
    pet_ct_report: str
    presentation_type: str
    main_diagnosis: str
    ann_arbor_stage: str
    accompanying_symptoms: str
    prognosis_score: str

    def __str__(self) -> str:
        return f"""
# Patient
- Vorstellungsart: {self.presentation_type}
- Ann-Arbor Stadium: {self.ann_arbor_stage}
- Begleitsymptome: {self.accompanying_symptoms}
- Prognose-Scores: {self.prognosis_score}

# Hauptdiagnose
{self.main_diagnosis_text}

# Relevante Nebendiagnosen
{self.secondary_diagnoses}

# Relevante anamnestische und klinische Angaben (z.B. Beschwerden) / Fragestellung an Tumorboard / Behandlungsvorschlag
{self.clinical_info}

# PET-CT-Bericht
{self.pet_ct_report}
"""

    @classmethod
    def from_namedtuple(cls, row: namedtuple) -> "Patient":
        return cls(
            id=row.ID,
            main_diagnosis_text=row.main_diagnosis_text,
            secondary_diagnoses=row.secondary_diagnoses,
            clinical_info=row.clinical_info,
            pet_ct_report=row.pet_ct_report,
            presentation_type=row.presentation_type,
            main_diagnosis=row.main_diagnosis,
            ann_arbor_stage=row.ann_arbor_stage,
            accompanying_symptoms=row.accompanying_symptoms,
            prognosis_score=row.prognosis_score,
        )
