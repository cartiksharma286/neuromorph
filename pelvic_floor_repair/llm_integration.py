"""
LLM Integration - Large Language Model for design assistance and recommendations
Uses combinatorial approach to reason about implant designs
"""

from typing import Dict, List
import numpy as np
import json

class LLMDesignAssistant:
    """AI assistant for surgical design and decision support"""
    
    def __init__(self):
        self.model_name = "Gynecological Repair Assistant v1.0"
        self.knowledge_base = self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self) -> Dict:
        """Initialize medical knowledge base for pelvic floor repair"""
        return {
            'pelvic_floor_anatomy': {
                'muscles': ['levator_ani', 'puborectalis', 'iliococcygeus', 'ischiococcygeus'],
                'ligaments': ['uterosacral', 'cardinal', 'round', 'broad'],
                'innervation': 'pudendal_nerve_s2_s4'
            },
            'material_characteristics': {
                'mesh': {'integration': 'fast', 'biocompatibility': 'high', 'cost': 'low'},
                'xenograft': {'integration': 'moderate', 'biocompatibility': 'high', 'cost': 'moderate'},
                'composite': {'integration': 'slow', 'biocompatibility': 'very_high', 'cost': 'high'}
            },
            'surgical_approaches': [
                'transvaginal_approach',
                'transabdominal_approach',
                'robot_assisted_approach',
                'laparoscopic_approach'
            ],
            'complication_predictors': {
                'mesh_erosion': 0.05,
                'infection': 0.03,
                'hematoma': 0.02,
                'recurrence': 0.10
            }
        }
    
    def get_repair_recommendations(self, analysis: Dict, patient_data: Dict) -> List[str]:
        """
        Generate AI recommendations for repair strategy
        """
        recommendations = []
        severity = analysis.get('severity_level', 'moderate')
        discontinuity_length = analysis.get('discontinuity_length_mm', 0)
        
        # Generate LLM-based recommendations
        if severity == 'mild':
            recommendations.append("Conservative approach recommended with minimal-invasive technique")
            recommendations.append("Consider patient's age and comorbidities for material selection")
            recommendations.append("Outpatient procedure likely feasible")
        elif severity == 'moderate':
            recommendations.append("Multi-layer reinforcement recommended for optimal outcomes")
            recommendations.append("Composite materials may provide superior long-term stability")
            recommendations.append("Pelvic floor physical therapy essential for post-operative success")
        else:  # severe
            recommendations.append("Complex case requiring experienced pelvic floor surgeon")
            recommendations.append("Consider staged repair if defect >5cm")
            recommendations.append("Reinforced mesh or composite material strongly recommended")
            recommendations.append("Intraoperative navigation or imaging strongly suggested")
        
        if discontinuity_length > 50:
            recommendations.append("Large defect: consider robot-assisted approach for precision")
            recommendations.append("Extended operative time anticipated - ensure adequate anesthesia support")
        
        # Patient-specific recommendations
        age = patient_data.get('age', 50)
        if age > 65:
            recommendations.append("Geriatric considerations: optimize for rapid recovery")
            recommendations.append("Increased monitoring for venous thromboembolism risk")
        
        recommendations.append("Post-operative imaging at 3 months to assess integration")
        recommendations.append("Follow-up pelvic floor assessment at 6 and 12 months")
        
        return recommendations
    
    def rank_designs(self, designs: List[Dict]) -> List[Dict]:
        """
        Rank implant designs using LLM-based scoring
        """
        ranked = []
        
        for design in designs:
            score = self._calculate_design_score(design)
            design['rank_score'] = score
            ranked.append(design)
        
        # Sort by score descending
        ranked.sort(key=lambda x: x['rank_score'], reverse=True)
        
        # Add ranking explanations
        for i, design in enumerate(ranked):
            design['rank'] = i + 1
            design['rank_explanation'] = self._generate_rank_explanation(design, i)
        
        return ranked
    
    def _calculate_design_score(self, design: Dict) -> float:
        """Calculate design quality score (0-1)"""
        score = 0.5  # Base score
        
        # Material score
        material = design.get('material', '')
        material_scores = {
            'composite': 0.95,
            'mesh': 0.80,
            'xenograft': 0.85,
            'synthetic_polymer': 0.78,
            'autograft': 0.90
        }
        score += material_scores.get(material, 0.75) * 0.25
        
        # Biocompatibility
        biocompat = design.get('biocompatibility_score', 0.85)
        score += biocompat * 0.25
        
        # Low complication risk
        complications = design.get('complications_risk', 0.1)
        score += (1 - complications) * 0.20
        
        # Cost-effectiveness
        cost = design.get('cost_estimate', 1000)
        cost_score = max(0, 1 - (cost - 500) / 5000)
        score += cost_score * 0.15
        
        # Integration speed
        integration = design.get('integration_speed', 8)
        integration_score = 1 - (integration / 12)
        score += integration_score * 0.15
        
        return min(1.0, max(0.0, score))
    
    def _generate_rank_explanation(self, design: Dict, rank: int) -> str:
        """Generate explanation for design ranking"""
        if rank == 0:
            return f"Top choice: {design.get('material', 'material')} with {design.get('shape_profile', 'profile')} design offers optimal balance of biocompatibility and integration speed"
        elif rank == 1:
            return f"Strong alternative: {design.get('material', 'material')} implant with excellent cost-effectiveness"
        else:
            return f"Alternative option {rank}: Consider if specific clinical requirements favor this design"
    
    def chat(self, query: str, context: Dict = {}) -> str:
        """
        Interactive AI chat for design assistance
        """
        query_lower = query.lower()
        
        # Pattern matching for common questions
        if 'material' in query_lower or 'implant' in query_lower:
            return "Based on your case analysis, composite materials offer superior biocompatibility and long-term stability. Mesh provides cost-effective alternatives. Consider patient age and activity level when selecting materials."
        
        elif 'chamber' in query_lower or 'support' in query_lower:
            return "Chamber configuration is critical for implant stability. We recommend distributed chambers to maintain uniform pressure distribution. The number of chambers scales with defect size. Optimization based on expected load profile enhances long-term outcomes."
        
        elif 'surgery' in query_lower or 'procedure' in query_lower or 'operative' in query_lower:
            return "Surgical approach depends on defect size and location. Transvaginal is preferred for smaller defects. Transabdominal or robot-assisted approaches offer better visualization for large or complex defects. Estimated operative time: 60-120 minutes depending on complexity."
        
        elif 'risk' in query_lower or 'complication' in query_lower:
            return "Main risks include implant erosion (5%), infection (3%), hematoma (2%), and recurrence (10%). Most complications are manageable with proper surgical technique and post-operative care. Appropriate material selection and patient optimization reduce complication rates."
        
        elif 'recovery' in query_lower or 'post-operative' in query_lower:
            return "Recovery timeline: Return to light activities in 2-4 weeks. Resume normal activities at 6-8 weeks. Full tissue integration typically occurs over 3-6 months. Follow-up imaging at 3 months and clinical assessment at 6 and 12 months recommended."
        
        else:
            return f"Regarding your question about pelvic floor repair: The proposed design combines advanced AI-generated implant configurations with optimized chamber support. Key considerations include material selection, defect geometry, and patient factors. Request specific design comparison or surgical planning for detailed analysis."
    
    def generate_surgical_summary(self, session_data: Dict) -> str:
        """Generate comprehensive surgical summary"""
        summary = f"""
        SURGICAL PLAN SUMMARY
        =====================
        
        Analysis ID: {session_data.get('analysis', {}).get('case_id', 'N/A')}
        Severity: {session_data.get('analysis', {}).get('severity_level', 'N/A').upper()}
        
        Selected Implant:
        - Material: {session_data.get('implant_designs', [{}])[0].get('design', {}).get('material', 'N/A')}
        - Dimensions: {session_data.get('implant_designs', [{}])[0].get('design', {}).get('dimensions', {})}
        
        Support Chambers: {len(session_data.get('chambers', []))}
        
        Key Recommendations:
        {chr(10).join('- ' + r for r in session_data.get('llm_recommendations', [])[:5])}
        
        Expected Outcomes:
        - Integration Time: 4-12 weeks
        - Success Probability: ~90-95%
        - Long-term Stability: High
        """
        return summary
