"""
Report Generation Engine
Quantum-optimized report generation with multi-format export
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid
from quantum_reporter import QuantumReporter
from structured_templates import StructuredTemplates
from patient_manager import PatientManager


class ReportGenerator:
    """
    Advanced report generation engine with quantum optimization
    Supports template-based creation and multi-format export
    """
    
    def __init__(self):
        self.quantum_reporter = QuantumReporter(num_qubits=6)
        self.templates = StructuredTemplates()
        self.reports = {}
    
    def create_report(self, template_id: str, patient_id: str, 
                     initial_data: Optional[Dict] = None,
                     use_quantum_optimization: bool = True) -> str:
        """
        Create a new report from template
        Optionally uses quantum optimization for field suggestions
        """
        report_id = str(uuid.uuid4())
        
        template = self.templates.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        report = {
            'id': report_id,
            'template_id': template_id,
            'template_name': template['name'],
            'patient_id': patient_id,
            'status': 'draft',
            'data': initial_data or {},
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'created_by': 'Current User',  # In production, use actual user
            'quantum_optimized': use_quantum_optimization,
            'optimization_metrics': {}
        }
        
        # Apply quantum optimization if requested
        if use_quantum_optimization and patient_id:
            try:
                patient_manager = PatientManager()
                patient_context = patient_manager.get_patient_context(patient_id)
                
                optimization_result = self.quantum_reporter.optimize_report_fields(
                    template, patient_context
                )
                
                report['optimization_metrics'] = {
                    'confidence': optimization_result['confidence'],
                    'quantum_advantage': optimization_result['quantum_advantage'],
                    'suggestions_count': len(optimization_result['suggestions'])
                }
                
                report['suggestions'] = optimization_result['suggestions']
                
            except Exception as e:
                print(f"Warning: Quantum optimization failed: {e}")
                report['quantum_optimized'] = False
        
        self.reports[report_id] = report
        return report_id
    
    def update_report(self, report_id: str, field_updates: Dict[str, Any]) -> bool:
        """Update report field values"""
        if report_id not in self.reports:
            return False
        
        report = self.reports[report_id]
        report['data'].update(field_updates)
        report['updated_at'] = datetime.now().isoformat()
        
        # Recalculate quality score
        quality_score = self.quantum_reporter.generate_report_quality_score(report)
        report['quality_score'] = quality_score
        
        return True
    
    def finalize_report(self, report_id: str) -> bool:
        """Finalize report (mark as complete)"""
        if report_id not in self.reports:
            return False
        
        report = self.reports[report_id]
        
        # Validate report
        validation = self.templates.validate_report(
            report['template_id'],
            report['data']
        )
        
        if not validation['valid']:
            report['validation_errors'] = validation['errors']
            return False
        
        report['status'] = 'finalized'
        report['finalized_at'] = datetime.now().isoformat()
        report['completeness'] = validation['completeness']
        
        # Generate final quality score
        quality_score = self.quantum_reporter.generate_report_quality_score(report)
        report['final_quality_score'] = quality_score
        
        return True
    
    def get_report(self, report_id: str) -> Optional[Dict]:
        """Retrieve report by ID"""
        return self.reports.get(report_id)
    
    def export_report(self, report_id: str, format: str = 'json') -> Optional[str]:
        """
        Export report in specified format
        Supports: json, pdf, dicom_sr, hl7_fhir
        """
        report = self.get_report(report_id)
        if not report:
            return None
        
        if format == 'json':
            return self._export_json(report)
        elif format == 'pdf':
            return self._export_pdf(report)
        elif format == 'dicom_sr':
            return self._export_dicom_sr(report)
        elif format == 'hl7_fhir':
            return self._export_hl7_fhir(report)
        
        return None
    
    def _export_json(self, report: Dict) -> str:
        """Export report as JSON"""
        export_data = {
            'report_id': report['id'],
            'template': report['template_name'],
            'patient_id': report['patient_id'],
            'status': report['status'],
            'created_at': report['created_at'],
            'data': report['data'],
            'quality_score': report.get('final_quality_score', report.get('quality_score', 0)),
            'quantum_optimized': report['quantum_optimized']
        }
        return json.dumps(export_data, indent=2)
    
    def _export_pdf(self, report: Dict) -> str:
        """Export report as PDF (returns formatted text representation)"""
        template = self.templates.get_template(report['template_id'])
        
        pdf_content = []
        pdf_content.append("=" * 80)
        pdf_content.append(f"{report['template_name']}")
        pdf_content.append("=" * 80)
        pdf_content.append(f"\nReport ID: {report['id']}")
        pdf_content.append(f"Patient ID: {report['patient_id']}")
        pdf_content.append(f"Date: {report['created_at']}")
        pdf_content.append(f"Status: {report['status'].upper()}")
        
        if report.get('quantum_optimized'):
            pdf_content.append(f"\n[QUANTUM OPTIMIZED - Confidence: {report['optimization_metrics'].get('confidence', 0):.3f}]")
        
        pdf_content.append("\n" + "-" * 80)
        
        # Add sections
        for section in template.get('sections', []):
            pdf_content.append(f"\n{section['name'].upper()}")
            pdf_content.append("-" * len(section['name']))
            
            for field_name, field_config in section.get('fields', {}).items():
                value = report['data'].get(field_name, '[Not provided]')
                pdf_content.append(f"\n{field_config['label']}:")
                pdf_content.append(f"  {value}")
        
        if 'final_quality_score' in report:
            pdf_content.append(f"\n\n" + "=" * 80)
            pdf_content.append(f"Quality Score: {report['final_quality_score']:.3f}")
            pdf_content.append(f"Completeness: {report.get('completeness', 0) * 100:.1f}%")
        
        pdf_content.append("\n" + "=" * 80)
        
        return "\n".join(pdf_content)
    
    def _export_dicom_sr(self, report: Dict) -> str:
        """Export as DICOM Structured Report (simplified representation)"""
        sr_content = []
        sr_content.append("DICOM Structured Report")
        sr_content.append(f"SOP Instance UID: {report['id']}")
        sr_content.append(f"Series Description: {report['template_name']}")
        sr_content.append(f"Content Date: {datetime.now().strftime('%Y%m%d')}")
        sr_content.append(f"Content Time: {datetime.now().strftime('%H%M%S')}")
        sr_content.append("")
        sr_content.append("CONTENT SEQUENCE:")
        
        for field_name, value in report['data'].items():
            sr_content.append(f"  {field_name}: {value}")
        
        return "\n".join(sr_content)
    
    def _export_hl7_fhir(self, report: Dict) -> str:
        """Export as HL7 FHIR DiagnosticReport"""
        fhir_report = {
            "resourceType": "DiagnosticReport",
            "id": report['id'],
            "status": "final" if report['status'] == 'finalized' else "preliminary",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "RAD",
                    "display": "Radiology"
                }]
            }],
            "code": {
                "text": report['template_name']
            },
            "subject": {
                "reference": f"Patient/{report['patient_id']}"
            },
            "effectiveDateTime": report['created_at'],
            "issued": report.get('finalized_at', report['created_at']),
            "conclusion": report['data'].get('impression', ''),
            "extension": [{
                "url": "http://example.org/fhir/StructureDefinition/quantum-optimized",
                "valueBoolean": report['quantum_optimized']
            }]
        }
        
        return json.dumps(fhir_report, indent=2)
    
    def generate_report_summary(self, report_id: str) -> Dict[str, Any]:
        """Generate comprehensive report summary"""
        report = self.get_report(report_id)
        if not report:
            return {}
        
        template = self.templates.get_template(report['template_id'])
        validation = self.templates.validate_report(report['template_id'], report['data'])
        
        return {
            'report_id': report_id,
            'template': report['template_name'],
            'status': report['status'],
            'created': report['created_at'],
            'last_updated': report['updated_at'],
            'completeness': validation['completeness'] * 100,
            'quality_score': report.get('final_quality_score', report.get('quality_score', 0)),
            'quantum_optimized': report['quantum_optimized'],
            'validation': {
                'valid': validation['valid'],
                'errors': len(validation['errors']),
                'warnings': len(validation['warnings'])
            }
        }
    
    def get_all_reports(self, filters: Optional[Dict] = None) -> List[Dict]:
        """Get all reports with optional filters"""
        reports = list(self.reports.values())
        
        if filters:
            if 'patient_id' in filters:
                reports = [r for r in reports if r['patient_id'] == filters['patient_id']]
            
            if 'status' in filters:
                reports = [r for r in reports if r['status'] == filters['status']]
            
            if 'template_id' in filters:
                reports = [r for r in reports if r['template_id'] == filters['template_id']]
        
        return reports


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("EMR Platform - Report Generation Engine")
    print("=" * 80)
    
    rg = ReportGenerator()
    
    # Create a report
    print("\n📝 Creating CT Brain report...")
    report_id = rg.create_report(
        template_id='radiology_ct_brain',
        patient_id='patient_123',
        initial_data={
            'indication': 'Headache, rule out intracranial hemorrhage',
            'contrast': 'Non-contrast'
        },
        use_quantum_optimization=True
    )
    
    print(f"✓ Report created: {report_id}")
    
    report = rg.get_report(report_id)
    if report.get('quantum_optimized'):
        metrics = report['optimization_metrics']
        print(f"\n🔬 Quantum Optimization Metrics:")
        print(f"  Confidence: {metrics['confidence']:.3f}")
        print(f"  Quantum Advantage: {metrics['quantum_advantage']:.3f}")
        print(f"  Suggestions: {metrics['suggestions_count']}")
    
    # Update report with findings
    print("\n\n✏️ Updating report with findings...")
    rg.update_report(report_id, {
        'brain_parenchyma': 'No acute intracranial abnormality. Age-appropriate volume loss.',
        'hemorrhage': 'None',
        'mass_effect': False,
        'ventricles': 'Normal',
        'extra_axial': 'Normal',
        'impression': 'No acute intracranial abnormality.'
    })
    
    print("✓ Report updated")
    
    # Finalize report
    print("\n\n✅ Finalizing report...")
    if rg.finalize_report(report_id):
        print("✓ Report finalized successfully")
        
        final_report = rg.get_report(report_id)
        print(f"  Quality Score: {final_report.get('final_quality_score', 0):.3f}")
        print(f"  Completeness: {final_report.get('completeness', 0) * 100:.1f}%")
    
    # Generate summary
    print("\n\n📊 Report Summary:")
    summary = rg.generate_report_summary(report_id)
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Export formats
    print("\n\n📤 Export Demonstrations:")
    
    print("\n[PDF Export]")
    pdf = rg.export_report(report_id, format='pdf')
    print(pdf[:500] + "..." if len(pdf) > 500 else pdf)
    
    print("\n\n[HL7 FHIR Export]")
    fhir = rg.export_report(report_id, format='hl7_fhir')
    fhir_data = json.loads(fhir)
    print(json.dumps(fhir_data, indent=2)[:500] + "...")
    
    print("\n" + "=" * 80)
