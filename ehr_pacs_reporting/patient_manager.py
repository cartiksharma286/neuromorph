"""
Patient Management System
Handles patient records, medical history, and report associations
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid


class PatientManager:
    """
    HIPAA-compliant patient data management system
    Manages patient records, medical history, and associated reports
    """
    
    def __init__(self):
        self.patients = {}  # In production, use secure database
        self.reports = {}  # Report storage
    
    def create_patient(self, patient_data: Dict[str, Any]) -> str:
        """Create a new patient record"""
        patient_id = str(uuid.uuid4())
        
        patient_record = {
            'id': patient_id,
            'mrn': patient_data.get('mrn', self._generate_mrn()),
            'demographics': {
                'first_name': patient_data.get('first_name', ''),
                'last_name': patient_data.get('last_name', ''),
                'date_of_birth': patient_data.get('date_of_birth', ''),
                'gender': patient_data.get('gender', ''),
                'email': patient_data.get('email', ''),
                'phone': patient_data.get('phone', '')
            },
            'medical_history': {
                'allergies': patient_data.get('allergies', []),
                'medications': patient_data.get('medications', []),
                'conditions': patient_data.get('conditions', []),
                'surgical_history': patient_data.get('surgical_history', [])
            },
            'reports': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        self.patients[patient_id] = patient_record
        return patient_id
    
    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """Retrieve patient record by ID"""
        return self.patients.get(patient_id)
    
    def search_patients(self, query: Dict[str, Any]) -> List[Dict]:
        """Search patients by various criteria"""
        results = []
        
        for patient in self.patients.values():
            match = True
            
            # Search by MRN
            if 'mrn' in query:
                if patient['mrn'] != query['mrn']:
                    match = False
            
            # Search by name
            if 'name' in query:
                search_name = query['name'].lower()
                full_name = f"{patient['demographics']['first_name']} {patient['demographics']['last_name']}".lower()
                if search_name not in full_name:
                    match = False
            
            # Search by DOB
            if 'date_of_birth' in query:
                if patient['demographics']['date_of_birth'] != query['date_of_birth']:
                    match = False
            
            if match:
                results.append(patient)
        
        return results
    
    def update_patient(self, patient_id: str, updates: Dict[str, Any]) -> bool:
        """Update patient record"""
        if patient_id not in self.patients:
            return False
        
        patient = self.patients[patient_id]
        
        # Update demographics
        if 'demographics' in updates:
            patient['demographics'].update(updates['demographics'])
        
        # Update medical history
        if 'medical_history' in updates:
            patient['medical_history'].update(updates['medical_history'])
        
        patient['updated_at'] = datetime.now().isoformat()
        return True
    
    def delete_patient(self, patient_id: str) -> bool:
        """Delete patient record (soft delete in production)"""
        if patient_id in self.patients:
            del self.patients[patient_id]
            return True
        return False
    
    def add_report_to_patient(self, patient_id: str, report_id: str) -> bool:
        """Associate a report with a patient"""
        if patient_id not in self.patients:
            return False
        
        self.patients[patient_id]['reports'].append(report_id)
        self.patients[patient_id]['updated_at'] = datetime.now().isoformat()
        return True
    
    def get_patient_reports(self, patient_id: str) -> List[Dict]:
        """Get all reports for a patient"""
        patient = self.get_patient(patient_id)
        if not patient:
            return []
        
        return [
            self.reports.get(report_id)
            for report_id in patient.get('reports', [])
            if report_id in self.reports
        ]
    
    def calculate_age(self, patient_id: str) -> Optional[int]:
        """Calculate patient age from date of birth"""
        patient = self.get_patient(patient_id)
        if not patient:
            return None
        
        dob_str = patient['demographics'].get('date_of_birth')
        if not dob_str:
            return None
        
        try:
            dob = datetime.fromisoformat(dob_str)
            today = datetime.now()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return age
        except:
            return None
    
    def get_patient_context(self, patient_id: str) -> Dict[str, Any]:
        """Get patient context for AI-enhanced reasoning and workflow optimization"""
        patient = self.get_patient(patient_id)
        if not patient:
            return {}
        
        age = self.calculate_age(patient_id)
        
        return {
            'patient_id': patient_id,
            'age': age,
            'gender': patient['demographics'].get('gender'),
            'allergies': patient['medical_history'].get('allergies', []),
            'conditions': patient['medical_history'].get('conditions', []),
            'previous_reports': len(patient.get('reports', []))
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get patient database statistics"""
        total_patients = len(self.patients)
        total_reports = sum(len(p.get('reports', [])) for p in self.patients.values())
        
        gender_dist = {'M': 0, 'F': 0, 'Other': 0, 'Unknown': 0}
        for patient in self.patients.values():
            gender = patient['demographics'].get('gender', 'Unknown')
            if gender in gender_dist:
                gender_dist[gender] += 1
            else:
                gender_dist['Other'] += 1
        
        return {
            'total_patients': total_patients,
            'total_reports': total_reports,
            'gender_distribution': gender_dist,
            'average_reports_per_patient': total_reports / total_patients if total_patients > 0 else 0
        }
    
    def _generate_mrn(self) -> str:
        """Generate unique Medical Record Number"""
        # Simple MRN generation (use more sophisticated method in production)
        return f"MRN{datetime.now().strftime('%Y%m%d')}{len(self.patients):04d}"
    
    def export_patient_data(self, patient_id: str, format: str = 'json') -> Optional[str]:
        """Export patient data in specified format"""
        patient = self.get_patient(patient_id)
        if not patient:
            return None
        
        if format == 'json':
            return json.dumps(patient, indent=2)
        elif format == 'hl7':
            return self._convert_to_hl7(patient)
        
        return None
    
    def _convert_to_hl7(self, patient: Dict) -> str:
        """Convert patient data to HL7 format (simplified)"""
        demographics = patient['demographics']
        
        # PID segment (Patient Identification)
        pid = f"PID|1||{patient['mrn']}||{demographics['last_name']}^{demographics['first_name']}||" \
              f"{demographics['date_of_birth']}|{demographics['gender']}|||||||||||"
        
        return pid


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("EMR Platform - Patient Management System")
    print("=" * 80)
    
    pm = PatientManager()
    
    # Create sample patients
    print("\n👤 Creating sample patients...")
    
    patient1_id = pm.create_patient({
        'first_name': 'John',
        'last_name': 'Doe',
        'date_of_birth': '1975-05-15',
        'gender': 'M',
        'email': 'john.doe@example.com',
        'phone': '555-1234',
        'allergies': ['Penicillin'],
        'conditions': ['Hypertension', 'Type 2 Diabetes'],
        'medications': ['Metformin', 'Lisinopril']
    })
    
    patient2_id = pm.create_patient({
        'first_name': 'Jane',
        'last_name': 'Smith',
        'date_of_birth': '1982-11-23',
        'gender': 'F',
        'email': 'jane.smith@example.com',
        'phone': '555-5678',
        'allergies': [],
        'conditions': ['Migraine'],
        'medications': ['Sumatriptan']
    })
    
    print(f"✓ Created patient: {pm.get_patient(patient1_id)['demographics']['first_name']} "
          f"{pm.get_patient(patient1_id)['demographics']['last_name']} "
          f"(MRN: {pm.get_patient(patient1_id)['mrn']})")
    print(f"✓ Created patient: {pm.get_patient(patient2_id)['demographics']['first_name']} "
          f"{pm.get_patient(patient2_id)['demographics']['last_name']} "
          f"(MRN: {pm.get_patient(patient2_id)['mrn']})")
    
    # Search patients
    print("\n\n🔍 Searching for patients named 'Doe'...")
    results = pm.search_patients({'name': 'Doe'})
    print(f"✓ Found {len(results)} patient(s)")
    
    # Get patient context
    print("\n\n📋 Patient Context for Gemini 3.0 Reasoning:")
    context = pm.get_patient_context(patient1_id)
    print(f"  Age: {context['age']} years")
    print(f"  Gender: {context['gender']}")
    print(f"  Conditions: {', '.join(context['conditions'])}")
    print(f"  Allergies: {', '.join(context['allergies'])}")
    
    # Statistics
    print("\n\n📊 Patient Database Statistics:")
    stats = pm.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # HL7 Export
    print("\n\n📤 HL7 Export Sample:")
    hl7_data = pm.export_patient_data(patient1_id, format='hl7')
    print(f"  {hl7_data}")
    
    print("\n" + "=" * 80)
