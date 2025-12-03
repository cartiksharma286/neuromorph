"""
Structured Reporting Templates Engine
Defines medical reporting templates for multiple specialties
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class StructuredTemplates:
    """
    Template engine for structured medical reporting across specialties
    Supports radiology, cardiology, pathology, and general medicine
    """
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.terminology = self._load_standard_terminology()
    
    def _initialize_templates(self) -> Dict[str, Dict]:
        """Initialize all medical reporting templates"""
        return {
            'radiology_ct_brain': self._create_ct_brain_template(),
            'radiology_mri_spine': self._create_mri_spine_template(),
            'radiology_chest_xray': self._create_chest_xray_template(),
            'cardiology_echo': self._create_echo_template(),
            'cardiology_ecg': self._create_ecg_template(),
            'pathology_histology': self._create_histology_template(),
            'general_hp': self._create_hp_template()
        }
    
    def _create_ct_brain_template(self) -> Dict:
        """CT Brain structured template (Radiology)"""
        return {
            'id': 'radiology_ct_brain',
            'name': 'CT Brain',
            'specialty': 'radiology',
            'modality': 'CT',
            'age_range': [0, 120],
            'sections': [
                {
                    'name': 'Clinical Information',
                    'fields': {
                        'indication': {
                            'type': 'text',
                            'required': True,
                            'label': 'Clinical Indication'
                        },
                        'clinical_history': {
                            'type': 'textarea',
                            'required': False,
                            'label': 'Clinical History'
                        }
                    }
                },
                {
                    'name': 'Technique',
                    'fields': {
                        'contrast': {
                            'type': 'select',
                            'required': True,
                            'label': 'Contrast',
                            'options': ['Non-contrast', 'With contrast', 'With and without contrast']
                        },
                        'slice_thickness': {
                            'type': 'numeric',
                            'required': False,
                            'label': 'Slice Thickness (mm)',
                            'min': 0.5,
                            'max': 10
                        }
                    }
                },
                {
                    'name': 'Findings',
                    'fields': {
                        'brain_parenchyma': {
                            'type': 'textarea',
                            'required': True,
                            'label': 'Brain Parenchyma',
                            'suggestions': [
                                'Normal attenuation',
                                'No acute intracranial abnormality',
                                'Age-appropriate volume loss'
                            ]
                        },
                        'hemorrhage': {
                            'type': 'select',
                            'required': True,
                            'label': 'Hemorrhage',
                            'options': ['None', 'Acute', 'Subacute', 'Chronic', 'Indeterminate']
                        },
                        'hemorrhage_location': {
                            'type': 'text',
                            'required': False,
                            'label': 'Hemorrhage Location',
                            'conditional': {'hemorrhage': ['Acute', 'Subacute', 'Chronic']}
                        },
                        'mass_effect': {
                            'type': 'boolean',
                            'required': True,
                            'label': 'Mass Effect'
                        },
                        'midline_shift': {
                            'type': 'numeric',
                            'required': False,
                            'label': 'Midline Shift (mm)',
                            'min': 0,
                            'max': 30
                        },
                        'ventricles': {
                            'type': 'select',
                            'required': True,
                            'label': 'Ventricles',
                            'options': ['Normal', 'Dilated', 'Compressed', 'Asymmetric']
                        },
                        'extra_axial': {
                            'type': 'select',
                            'required': True,
                            'label': 'Extra-axial Spaces',
                            'options': ['Normal', 'Subdural collection', 'Epidural collection', 'Subarachnoid hemorrhage']
                        }
                    }
                },
                {
                    'name': 'Impression',
                    'fields': {
                        'impression': {
                            'type': 'textarea',
                            'required': True,
                            'label': 'Impression',
                            'min_length': 10
                        },
                        'birads': {
                            'type': 'select',
                            'required': False,
                            'label': 'BIRADS Category',
                            'options': ['N/A', '0', '1', '2', '3', '4', '5', '6']
                        }
                    }
                }
            ],
            'standard_terminology': 'RadLex',
            'report_template': '{clinical_info}\n\nTECHNIQUE:\n{technique}\n\nFINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}'
        }
    
    def _create_mri_spine_template(self) -> Dict:
        """MRI Spine structured template"""
        return {
            'id': 'radiology_mri_spine',
            'name': 'MRI Spine',
            'specialty': 'radiology',
            'modality': 'MRI',
            'age_range': [0, 120],
            'sections': [
                {
                    'name': 'Clinical Information',
                    'fields': {
                        'indication': {'type': 'text', 'required': True, 'label': 'Indication'},
                        'spine_level': {
                            'type': 'select',
                            'required': True,
                            'label': 'Spine Level',
                            'options': ['Cervical', 'Thoracic', 'Lumbar', 'Entire Spine']
                        }
                    }
                },
                {
                    'name': 'Findings',
                    'fields': {
                        'alignment': {
                            'type': 'select',
                            'required': True,
                            'label': 'Alignment',
                            'options': ['Normal', 'Scoliosis', 'Kyphosis', 'Lordosis', 'Listhesis']
                        },
                        'disc_degeneration': {
                            'type': 'text',
                            'required': True,
                            'label': 'Disc Degeneration/Herniation'
                        },
                        'spinal_canal': {
                            'type': 'select',
                            'required': True,
                            'label': 'Spinal Canal',
                            'options': ['Patent', 'Stenosis - mild', 'Stenosis - moderate', 'Stenosis - severe']
                        },
                        'cord_signal': {
                            'type': 'select',
                            'required': True,
                            'label': 'Cord Signal',
                            'options': ['Normal', 'Abnormal - specify in findings']
                        }
                    }
                }
            ]
        }
    
    def _create_chest_xray_template(self) -> Dict:
        """Chest X-Ray structured template"""
        return {
            'id': 'radiology_chest_xray',
            'name': 'Chest X-Ray',
            'specialty': 'radiology',
            'modality': 'XR',
            'sections': [
                {
                    'name': 'Findings',
                    'fields': {
                        'lungs': {
                            'type': 'select',
                            'required': True,
                            'label': 'Lungs',
                            'options': ['Clear', 'Infiltrate', 'Effusion', 'Pneumothorax', 'Mass/Nodule']
                        },
                        'heart_size': {
                            'type': 'select',
                            'required': True,
                            'label': 'Heart Size',
                            'options': ['Normal', 'Mild cardiomegaly', 'Moderate cardiomegaly', 'Severe cardiomegaly']
                        },
                        'mediastinum': {
                            'type': 'select',
                            'required': True,
                            'label': 'Mediastinum',
                            'options': ['Normal', 'Widened', 'Mass']
                        }
                    }
                }
            ]
        }
    
    def _create_echo_template(self) -> Dict:
        """Echocardiogram structured template (Cardiology)"""
        return {
            'id': 'cardiology_echo',
            'name': 'Echocardiogram',
            'specialty': 'cardiology',
            'modality': 'Echo',
            'sections': [
                {
                    'name': 'Measurements',
                    'fields': {
                        'lvef': {
                            'type': 'numeric',
                            'required': True,
                            'label': 'LVEF (%)',
                            'min': 0,
                            'max': 100
                        },
                        'lv_function': {
                            'type': 'select',
                            'required': True,
                            'label': 'LV Systolic Function',
                            'options': ['Normal', 'Mildly reduced', 'Moderately reduced', 'Severely reduced']
                        },
                        'wall_motion': {
                            'type': 'text',
                            'required': True,
                            'label': 'Wall Motion Abnormalities'
                        },
                        'mitral_valve': {
                            'type': 'select',
                            'required': True,
                            'label': 'Mitral Valve',
                            'options': ['Normal', 'Regurgitation - mild', 'Regurgitation - moderate', 
                                       'Regurgitation - severe', 'Stenosis']
                        },
                        'aortic_valve': {
                            'type': 'select',
                            'required': True,
                            'label': 'Aortic Valve',
                            'options': ['Normal', 'Regurgitation - mild', 'Regurgitation - moderate', 
                                       'Regurgitation - severe', 'Stenosis']
                        },
                        'pericardial_effusion': {
                            'type': 'select',
                            'required': True,
                            'label': 'Pericardial Effusion',
                            'options': ['None', 'Small', 'Moderate', 'Large']
                        }
                    }
                }
            ]
        }
    
    def _create_ecg_template(self) -> Dict:
        """ECG structured template"""
        return {
            'id': 'cardiology_ecg',
            'name': 'ECG Interpretation',
            'specialty': 'cardiology',
            'modality': 'ECG',
            'sections': [
                {
                    'name': 'Measurements',
                    'fields': {
                        'heart_rate': {
                            'type': 'numeric',
                            'required': True,
                            'label': 'Heart Rate (bpm)',
                            'min': 20,
                            'max': 300
                        },
                        'rhythm': {
                            'type': 'select',
                            'required': True,
                            'label': 'Rhythm',
                            'options': ['Sinus rhythm', 'Atrial fibrillation', 'Atrial flutter', 
                                       'SVT', 'Ventricular tachycardia', 'Paced']
                        },
                        'pr_interval': {
                            'type': 'numeric',
                            'required': True,
                            'label': 'PR Interval (ms)',
                            'min': 80,
                            'max': 300
                        },
                        'qrs_duration': {
                            'type': 'numeric',
                            'required': True,
                            'label': 'QRS Duration (ms)',
                            'min': 60,
                            'max': 200
                        },
                        'qt_qtc': {
                            'type': 'text',
                            'required': True,
                            'label': 'QT/QTc (ms)'
                        },
                        'axis': {
                            'type': 'select',
                            'required': True,
                            'label': 'Axis',
                            'options': ['Normal', 'Left axis deviation', 'Right axis deviation', 'Indeterminate']
                        },
                        'st_changes': {
                            'type': 'text',
                            'required': True,
                            'label': 'ST Segment Changes'
                        }
                    }
                }
            ]
        }
    
    def _create_histology_template(self) -> Dict:
        """Histopathology structured template"""
        return {
            'id': 'pathology_histology',
            'name': 'Histopathology Report',
            'specialty': 'pathology',
            'modality': 'Histology',
            'sections': [
                {
                    'name': 'Specimen',
                    'fields': {
                        'specimen_type': {'type': 'text', 'required': True, 'label': 'Specimen Type'},
                        'specimen_site': {'type': 'text', 'required': True, 'label': 'Site'}
                    }
                },
                {
                    'name': 'Diagnosis',
                    'fields': {
                        'diagnosis': {'type': 'textarea', 'required': True, 'label': 'Microscopic Diagnosis'},
                        'grade': {
                            'type': 'select',
                            'required': False,
                            'label': 'Grade',
                            'options': ['N/A', 'Well differentiated', 'Moderately differentiated', 'Poorly differentiated']
                        },
                        'margins': {
                            'type': 'select',
                            'required': False,
                            'label': 'Margins',
                            'options': ['N/A', 'Negative', 'Positive', 'Close']
                        }
                    }
                }
            ]
        }
    
    def _create_hp_template(self) -> Dict:
        """History & Physical structured template"""
        return {
            'id': 'general_hp',
            'name': 'History & Physical',
            'specialty': 'general',
            'sections': [
                {
                    'name': 'Chief Complaint',
                    'fields': {
                        'chief_complaint': {'type': 'text', 'required': True, 'label': 'Chief Complaint'}
                    }
                },
                {
                    'name': 'History of Present Illness',
                    'fields': {
                        'hpi': {'type': 'textarea', 'required': True, 'label': 'HPI'}
                    }
                },
                {
                    'name': 'Physical Examination',
                    'fields': {
                        'vitals': {'type': 'text', 'required': True, 'label': 'Vital Signs'},
                        'general': {'type': 'text', 'required': True, 'label': 'General Appearance'},
                        'heent': {'type': 'text', 'required': False, 'label': 'HEENT'},
                        'cardiovascular': {'type': 'text', 'required': False, 'label': 'Cardiovascular'},
                        'respiratory': {'type': 'text', 'required': False, 'label': 'Respiratory'},
                        'abdomen': {'type': 'text', 'required': False, 'label': 'Abdomen'},
                        'neurological': {'type': 'text', 'required': False, 'label': 'Neurological'}
                    }
                },
                {
                    'name': 'Assessment & Plan',
                    'fields': {
                        'assessment': {'type': 'textarea', 'required': True, 'label': 'Assessment'},
                        'plan': {'type': 'textarea', 'required': True, 'label': 'Plan'}
                    }
                }
            ]
        }
    
    def _load_standard_terminology(self) -> Dict[str, List[str]]:
        """Load standard medical terminology (RadLex, SNOMED CT)"""
        return {
            'radlex': [
                'No acute intracranial abnormality',
                'Age-appropriate volume loss',
                'Chronic small vessel ischemic changes',
                'Acute infarct',
                'Hemorrhagic transformation'
            ],
            'snomed': [
                'Normal examination',
                'Abnormal findings present',
                'Clinical correlation recommended'
            ]
        }
    
    def get_template(self, template_id: str) -> Optional[Dict]:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    def get_templates_by_specialty(self, specialty: str) -> List[Dict]:
        """Get all templates for a specialty"""
        return [
            template for template in self.templates.values()
            if template.get('specialty') == specialty
        ]
    
    def list_all_templates(self) -> List[Dict]:
        """List all available templates"""
        return [
            {
                'id': tid,
                'name': t['name'],
                'specialty': t.get('specialty', 'general'),
                'modality': t.get('modality', 'N/A')
            }
            for tid, t in self.templates.items()
        ]
    
    def validate_report(self, template_id: str, report_data: Dict) -> Dict[str, Any]:
        """Validate report data against template"""
        template = self.get_template(template_id)
        if not template:
            return {'valid': False, 'errors': ['Template not found']}
        
        errors = []
        warnings = []
        
        for section in template.get('sections', []):
            for field_name, field_config in section.get('fields', {}).items():
                # Check required fields
                if field_config.get('required') and field_name not in report_data:
                    errors.append(f"Required field '{field_name}' is missing")
                
                # Check field types
                if field_name in report_data:
                    value = report_data[field_name]
                    field_type = field_config.get('type')
                    
                    if field_type == 'numeric':
                        if not isinstance(value, (int, float)):
                            errors.append(f"Field '{field_name}' must be numeric")
                        else:
                            min_val = field_config.get('min')
                            max_val = field_config.get('max')
                            if min_val is not None and value < min_val:
                                warnings.append(f"Field '{field_name}' is below minimum value")
                            if max_val is not None and value > max_val:
                                warnings.append(f"Field '{field_name}' exceeds maximum value")
                    
                    elif field_type == 'select':
                        options = field_config.get('options', [])
                        if value not in options:
                            errors.append(f"Field '{field_name}' has invalid option")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'completeness': self._calculate_completeness(template, report_data)
        }
    
    def _calculate_completeness(self, template: Dict, report_data: Dict) -> float:
        """Calculate report completeness percentage"""
        total_fields = 0
        filled_fields = 0
        
        for section in template.get('sections', []):
            for field_name in section.get('fields', {}).keys():
                total_fields += 1
                if field_name in report_data and report_data[field_name]:
                    filled_fields += 1
        
        return filled_fields / total_fields if total_fields > 0 else 0.0


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("EMR Platform - Structured Templates Engine")
    print("=" * 80)
    
    st = StructuredTemplates()
    
    print("\n📋 Available Templates:")
    for template in st.list_all_templates():
        print(f"  • {template['name']} ({template['specialty']}) - ID: {template['id']}")
    
    print("\n\n🏥 CT Brain Template Details:")
    ct_brain = st.get_template('radiology_ct_brain')
    print(f"  Specialty: {ct_brain['specialty']}")
    print(f"  Modality: {ct_brain['modality']}")
    print(f"  Sections: {len(ct_brain['sections'])}")
    
    for section in ct_brain['sections']:
        print(f"\n  [{section['name']}]")
        for field_name, field_config in section['fields'].items():
            required = "✓" if field_config.get('required') else " "
            print(f"    [{required}] {field_config['label']} ({field_config['type']})")
    
    print("\n\n✅ Validating Sample Report:")
    sample_report = {
        'indication': 'Headache',
        'contrast': 'Non-contrast',
        'brain_parenchyma': 'No acute intracranial abnormality',
        'hemorrhage': 'None',
        'mass_effect': False,
        'ventricles': 'Normal',
        'extra_axial': 'Normal',
        'impression': 'No acute intracranial abnormality'
    }
    
    validation = st.validate_report('radiology_ct_brain', sample_report)
    print(f"  Valid: {validation['valid']}")
    print(f"  Completeness: {validation['completeness'] * 100:.1f}%")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    
    print("\n" + "=" * 80)
