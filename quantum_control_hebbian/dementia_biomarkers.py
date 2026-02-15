"""
Dementia-Specific Biomarker Tracking
Comprehensive biomarker management for dementia care
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime, timedelta


@dataclass
class CognitiveAssessment:
    """Cognitive assessment scores"""
    date: datetime
    mmse: float  # 0-30
    moca: float  # 0-30
    clock_drawing: float  # 0-10
    verbal_fluency: float  # Number of words in 60s
    notes: str = ""


@dataclass
class BiochemicalMarkers:
    """Biochemical biomarkers"""
    date: datetime
    acetylcholine_percent: float  # Percentage of normal
    bdnf_ng_ml: float  # Brain-Derived Neurotrophic Factor
    amyloid_beta_42_40_ratio: float  # CSF or plasma
    p_tau_pg_ml: float  # Phosphorylated tau


@dataclass
class NeuroimagingMarkers:
    """Neuroimaging biomarkers"""
    date: datetime
    hippocampal_volume_mm3: float
    entorhinal_thickness_mm: float
    white_matter_fa: float  # Fractional anisotropy (DTI)
    glucose_metabolism_suvr: float  # FDG-PET standardized uptake value ratio


@dataclass
class FunctionalMarkers:
    """Functional assessment scores"""
    date: datetime
    adl_score: float  # Activities of Daily Living (0-100)
    iadl_score: float  # Instrumental ADL (0-100)
    npi_score: float  # Neuropsychiatric Inventory (0-144)
    caregiver_burden: float  # 0-100


class DementiaBiomarkerTracker:
    """
    Comprehensive biomarker tracking for dementia patients
    """
    
    def __init__(self, patient_id: str = "DEMO-001"):
        self.patient_id = patient_id
        self.cognitive_assessments: List[CognitiveAssessment] = []
        self.biochemical_markers: List[BiochemicalMarkers] = []
        self.neuroimaging_markers: List[NeuroimagingMarkers] = []
        self.functional_markers: List[FunctionalMarkers] = []
        
        # Initialize with baseline
        self._initialize_baseline()
    
    def _initialize_baseline(self):
        """Initialize with baseline measurements"""
        baseline_date = datetime.now() - timedelta(days=180)  # 6 months ago
        
        # Baseline cognitive assessment (moderate Alzheimer's)
        self.cognitive_assessments.append(CognitiveAssessment(
            date=baseline_date,
            mmse=18.0,
            moca=16.0,
            clock_drawing=4.0,
            verbal_fluency=8.0,
            notes="Baseline assessment - moderate Alzheimer's disease"
        ))
        
        # Baseline biochemical markers
        self.biochemical_markers.append(BiochemicalMarkers(
            date=baseline_date,
            acetylcholine_percent=40.0,  # Reduced
            bdnf_ng_ml=15.0,  # Low (normal ~25-30)
            amyloid_beta_42_40_ratio=0.05,  # Low ratio indicates AD
            p_tau_pg_ml=85.0  # Elevated (normal <60)
        ))
        
        # Baseline neuroimaging
        self.neuroimaging_markers.append(NeuroimagingMarkers(
            date=baseline_date,
            hippocampal_volume_mm3=5500,  # Reduced (normal ~7000-8000)
            entorhinal_thickness_mm=2.8,  # Thinned (normal ~3.5-4.0)
            white_matter_fa=0.35,  # Reduced integrity (normal ~0.45-0.55)
            glucose_metabolism_suvr=0.75  # Hypometabolism (normal ~1.0)
        ))
        
        # Baseline functional assessment
        self.functional_markers.append(FunctionalMarkers(
            date=baseline_date,
            adl_score=70.0,  # Some impairment
            iadl_score=50.0,  # Moderate impairment
            npi_score=35.0,  # Moderate neuropsychiatric symptoms
            caregiver_burden=60.0  # Moderate burden
        ))
    
    def add_cognitive_assessment(self, mmse: float, moca: float,
                                 clock_drawing: float = None,
                                 verbal_fluency: float = None,
                                 notes: str = ""):
        """Add new cognitive assessment"""
        assessment = CognitiveAssessment(
            date=datetime.now(),
            mmse=mmse,
            moca=moca,
            clock_drawing=clock_drawing or mmse / 3.0,
            verbal_fluency=verbal_fluency or mmse * 0.5,
            notes=notes
        )
        self.cognitive_assessments.append(assessment)
        return assessment
    
    def calculate_mmse(self, responses: Dict[str, int]) -> float:
        """
        Calculate MMSE score from responses
        
        Domains:
        - Orientation (10 points)
        - Registration (3 points)
        - Attention/Calculation (5 points)
        - Recall (3 points)
        - Language (9 points)
        """
        total = 0
        total += responses.get('orientation', 0)  # Max 10
        total += responses.get('registration', 0)  # Max 3
        total += responses.get('attention', 0)  # Max 5
        total += responses.get('recall', 0)  # Max 3
        total += responses.get('language', 0)  # Max 9
        
        return min(30, total)
    
    def calculate_moca(self, responses: Dict[str, int]) -> float:
        """
        Calculate MoCA score from responses
        
        Domains:
        - Visuospatial/Executive (5 points)
        - Naming (3 points)
        - Attention (6 points)
        - Language (3 points)
        - Abstraction (2 points)
        - Delayed Recall (5 points)
        - Orientation (6 points)
        """
        total = 0
        total += responses.get('visuospatial', 0)  # Max 5
        total += responses.get('naming', 0)  # Max 3
        total += responses.get('attention', 0)  # Max 6
        total += responses.get('language', 0)  # Max 3
        total += responses.get('abstraction', 0)  # Max 2
        total += responses.get('recall', 0)  # Max 5
        total += responses.get('orientation', 0)  # Max 6
        
        # Add 1 point if education ≤ 12 years
        if responses.get('education_years', 13) <= 12:
            total += 1
        
        return min(30, total)
    
    def get_cognitive_trajectory(self) -> Dict:
        """Get cognitive decline trajectory"""
        if len(self.cognitive_assessments) < 2:
            return {'insufficient_data': True}
        
        assessments = sorted(self.cognitive_assessments, key=lambda x: x.date)
        
        # Calculate rates of change
        first = assessments[0]
        last = assessments[-1]
        days_elapsed = (last.date - first.date).days
        years_elapsed = days_elapsed / 365.25
        
        if years_elapsed == 0:
            return {'insufficient_data': True}
        
        mmse_change_per_year = (last.mmse - first.mmse) / years_elapsed
        moca_change_per_year = (last.moca - first.moca) / years_elapsed
        
        return {
            'baseline_mmse': first.mmse,
            'current_mmse': last.mmse,
            'mmse_change_per_year': mmse_change_per_year,
            'baseline_moca': first.moca,
            'current_moca': last.moca,
            'moca_change_per_year': moca_change_per_year,
            'years_tracked': years_elapsed,
            'num_assessments': len(assessments),
            'trajectory': 'improving' if mmse_change_per_year > 0 else 'declining'
        }
    
    def get_biomarker_summary(self) -> Dict:
        """Get comprehensive biomarker summary"""
        latest_cognitive = self.cognitive_assessments[-1] if self.cognitive_assessments else None
        latest_biochemical = self.biochemical_markers[-1] if self.biochemical_markers else None
        latest_neuroimaging = self.neuroimaging_markers[-1] if self.neuroimaging_markers else None
        latest_functional = self.functional_markers[-1] if self.functional_markers else None
        
        summary = {
            'patient_id': self.patient_id,
            'assessment_date': datetime.now().isoformat()
        }
        
        if latest_cognitive:
            summary['cognitive'] = {
                'mmse': latest_cognitive.mmse,
                'moca': latest_cognitive.moca,
                'clock_drawing': latest_cognitive.clock_drawing,
                'verbal_fluency': latest_cognitive.verbal_fluency,
                'disease_stage': self._get_disease_stage(latest_cognitive.mmse)
            }
        
        if latest_biochemical:
            summary['biochemical'] = {
                'acetylcholine_percent': latest_biochemical.acetylcholine_percent,
                'bdnf_ng_ml': latest_biochemical.bdnf_ng_ml,
                'amyloid_beta_ratio': latest_biochemical.amyloid_beta_42_40_ratio,
                'p_tau_pg_ml': latest_biochemical.p_tau_pg_ml,
                'alzheimers_biomarker_positive': latest_biochemical.amyloid_beta_42_40_ratio < 0.08
            }
        
        if latest_neuroimaging:
            summary['neuroimaging'] = {
                'hippocampal_volume_mm3': latest_neuroimaging.hippocampal_volume_mm3,
                'hippocampal_atrophy_percent': (1 - latest_neuroimaging.hippocampal_volume_mm3 / 7500) * 100,
                'entorhinal_thickness_mm': latest_neuroimaging.entorhinal_thickness_mm,
                'white_matter_fa': latest_neuroimaging.white_matter_fa,
                'glucose_metabolism_suvr': latest_neuroimaging.glucose_metabolism_suvr
            }
        
        if latest_functional:
            summary['functional'] = {
                'adl_score': latest_functional.adl_score,
                'iadl_score': latest_functional.iadl_score,
                'npi_score': latest_functional.npi_score,
                'caregiver_burden': latest_functional.caregiver_burden
            }
        
        summary['trajectory'] = self.get_cognitive_trajectory()
        
        return summary
    
    def _get_disease_stage(self, mmse: float) -> str:
        """Determine disease stage from MMSE"""
        if mmse >= 24:
            return "Mild Cognitive Impairment (MCI)"
        elif mmse >= 20:
            return "Mild Dementia"
        elif mmse >= 10:
            return "Moderate Dementia"
        else:
            return "Severe Dementia"
    
    def simulate_treatment_effect(self, treatment_efficacy: float):
        """Simulate biomarker changes from DBS treatment"""
        if not self.cognitive_assessments:
            return
        
        latest = self.cognitive_assessments[-1]
        
        # Improved cognitive scores
        new_mmse = min(30, latest.mmse * (1 + treatment_efficacy * 0.2))
        new_moca = min(30, latest.moca * (1 + treatment_efficacy * 0.2))
        
        self.add_cognitive_assessment(
            mmse=new_mmse,
            moca=new_moca,
            notes=f"Post-DBS treatment (efficacy: {treatment_efficacy:.2%})"
        )
        
        # Improved biochemical markers
        if self.biochemical_markers:
            latest_bio = self.biochemical_markers[-1]
            self.biochemical_markers.append(BiochemicalMarkers(
                date=datetime.now(),
                acetylcholine_percent=min(100, latest_bio.acetylcholine_percent * (1 + treatment_efficacy * 0.3)),
                bdnf_ng_ml=min(30, latest_bio.bdnf_ng_ml * (1 + treatment_efficacy * 0.15)),
                amyloid_beta_42_40_ratio=latest_bio.amyloid_beta_42_40_ratio,  # Doesn't change quickly
                p_tau_pg_ml=latest_bio.p_tau_pg_ml  # Doesn't change quickly
            ))


if __name__ == "__main__":
    print("="*60)
    print("Dementia Biomarker Tracking System")
    print("="*60)
    
    tracker = DementiaBiomarkerTracker(patient_id="DEMO-001")
    
    print("\nBaseline Biomarker Summary:")
    summary = tracker.get_biomarker_summary()
    import json
    print(json.dumps(summary, indent=2, default=str))
    
    print("\n" + "="*60)
    print("Simulating DBS Treatment Effect...")
    print("="*60)
    
    tracker.simulate_treatment_effect(treatment_efficacy=0.25)  # 25% efficacy
    
    print("\nPost-Treatment Biomarker Summary:")
    summary = tracker.get_biomarker_summary()
    print(json.dumps(summary, indent=2, default=str))
    
    print("\nCognitive Trajectory:")
    trajectory = tracker.get_cognitive_trajectory()
    print(json.dumps(trajectory, indent=2))
