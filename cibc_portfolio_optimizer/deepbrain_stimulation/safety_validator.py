"""
Safety Validation System for DBS Parameters
Ensures compliance with medical device safety standards
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class SafetyLevel(Enum):
    """Safety assessment levels"""
    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"


@dataclass
class SafetyLimits:
    """Safety parameter limits based on medical standards"""
    # Current and voltage limits
    max_current_ma: float = 25.5
    max_voltage_v: float = 10.5
    
    # Charge density limits (Shannon 1992)
    max_charge_density_uc_cm2: float = 30.0
    max_current_density_ma_cm2: float = 2.0
    
    # Frequency and pulse width
    min_frequency_hz: float = 2.0
    max_frequency_hz: float = 250.0
    min_pulse_width_us: float = 60.0
    max_pulse_width_us: float = 450.0
    
    # Thermal limits
    max_temperature_c: float = 38.0
    max_power_mw: float = 100.0
    
    # Impedance limits
    min_impedance_ohms: float = 200.0
    max_impedance_ohms: float = 5000.0
    normal_impedance_range: Tuple[float, float] = (500.0, 2000.0)
    
    # Duty cycle
    max_duty_cycle: float = 0.9


class SafetyValidator:
    """Comprehensive safety validation for DBS parameters"""
    
    def __init__(self, limits: Optional[SafetyLimits] = None):
        self.limits = limits or SafetyLimits()
        self.validation_history = []
    
    def validate_parameters(self, amplitude_ma: float, frequency_hz: float,
                          pulse_width_us: float, electrode_area_cm2: float = 0.06,
                          impedance_ohms: float = 1000.0) -> Dict:
        """
        Comprehensive parameter validation
        Returns safety assessment and recommendations
        """
        violations = []
        warnings = []
        
        # 1. Current amplitude check
        if amplitude_ma > self.limits.max_current_ma:
            violations.append(f"Current {amplitude_ma} mA exceeds maximum {self.limits.max_current_ma} mA")
        elif amplitude_ma > self.limits.max_current_ma * 0.9:
            warnings.append(f"Current {amplitude_ma} mA approaching maximum limit")
        
        # 2. Frequency check
        if frequency_hz < self.limits.min_frequency_hz or frequency_hz > self.limits.max_frequency_hz:
            violations.append(f"Frequency {frequency_hz} Hz outside range [{self.limits.min_frequency_hz}, {self.limits.max_frequency_hz}]")
        
        # 3. Pulse width check
        if pulse_width_us < self.limits.min_pulse_width_us or pulse_width_us > self.limits.max_pulse_width_us:
            violations.append(f"Pulse width {pulse_width_us} μs outside range [{self.limits.min_pulse_width_us}, {self.limits.max_pulse_width_us}]")
        
        # 4. Charge density check (CRITICAL)
        charge_per_phase_uc = (amplitude_ma * 1000) * (pulse_width_us / 1e6)  # Convert to μC
        charge_density = charge_per_phase_uc / electrode_area_cm2
        
        if charge_density > self.limits.max_charge_density_uc_cm2:
            violations.append(
                f"Charge density {charge_density:.2f} μC/cm² exceeds Shannon limit "
                f"{self.limits.max_charge_density_uc_cm2} μC/cm² - TISSUE DAMAGE RISK"
            )
        elif charge_density > self.limits.max_charge_density_uc_cm2 * 0.8:
            warnings.append(f"Charge density {charge_density:.2f} μC/cm² approaching safety limit")
        
        # 5. Current density check
        current_density = amplitude_ma / electrode_area_cm2
        if current_density > self.limits.max_current_density_ma_cm2:
            violations.append(
                f"Current density {current_density:.2f} mA/cm² exceeds limit "
                f"{self.limits.max_current_density_ma_cm2} mA/cm²"
            )
        
        # 6. Power dissipation check
        voltage_v = (amplitude_ma / 1000) * impedance_ohms  # Ohm's law
        power_mw = voltage_v * amplitude_ma
        
        if power_mw > self.limits.max_power_mw:
            violations.append(f"Power dissipation {power_mw:.2f} mW exceeds limit {self.limits.max_power_mw} mW")
        elif power_mw > self.limits.max_power_mw * 0.8:
            warnings.append(f"Power dissipation {power_mw:.2f} mW approaching limit")
        
        # 7. Impedance check
        if impedance_ohms < self.limits.min_impedance_ohms:
            violations.append(f"Impedance {impedance_ohms} Ω too low - possible short circuit")
        elif impedance_ohms > self.limits.max_impedance_ohms:
            violations.append(f"Impedance {impedance_ohms} Ω too high - possible open circuit")
        elif not (self.limits.normal_impedance_range[0] <= impedance_ohms <= self.limits.normal_impedance_range[1]):
            warnings.append(f"Impedance {impedance_ohms} Ω outside normal range {self.limits.normal_impedance_range}")
        
        # 8. Voltage compliance check
        if voltage_v > self.limits.max_voltage_v:
            violations.append(f"Required voltage {voltage_v:.2f} V exceeds compliance voltage {self.limits.max_voltage_v} V")
        
        # Determine overall safety level
        if violations:
            safety_level = SafetyLevel.UNSAFE
        elif warnings:
            safety_level = SafetyLevel.WARNING
        else:
            safety_level = SafetyLevel.SAFE
        
        result = {
            'safety_level': safety_level.value,
            'violations': violations,
            'warnings': warnings,
            'is_safe': safety_level == SafetyLevel.SAFE,
            'metrics': {
                'charge_density_uc_cm2': round(charge_density, 2),
                'current_density_ma_cm2': round(current_density, 2),
                'power_dissipation_mw': round(power_mw, 2),
                'voltage_v': round(voltage_v, 2),
                'charge_per_phase_uc': round(charge_per_phase_uc, 4)
            },
            'recommendations': self._generate_recommendations(
                amplitude_ma, frequency_hz, pulse_width_us, 
                charge_density, violations, warnings
            )
        }
        
        # Log validation
        self.validation_history.append({
            'parameters': {
                'amplitude_ma': amplitude_ma,
                'frequency_hz': frequency_hz,
                'pulse_width_us': pulse_width_us,
                'impedance_ohms': impedance_ohms
            },
            'result': result
        })
        
        return result
    
    def _generate_recommendations(self, amplitude_ma: float, frequency_hz: float,
                                 pulse_width_us: float, charge_density: float,
                                 violations: List[str], warnings: List[str]) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []
        
        if charge_density > self.limits.max_charge_density_uc_cm2:
            # Calculate safe amplitude
            safe_amplitude = (self.limits.max_charge_density_uc_cm2 * 0.06) / (pulse_width_us / 1e6) / 1000
            recommendations.append(
                f"Reduce amplitude to ≤{safe_amplitude:.2f} mA or reduce pulse width to maintain safe charge density"
            )
        
        if amplitude_ma > self.limits.max_current_ma * 0.9:
            recommendations.append("Consider reducing amplitude to increase safety margin")
        
        if pulse_width_us > 200:
            recommendations.append("Consider shorter pulse width for equivalent charge delivery")
        
        if not violations and not warnings:
            recommendations.append("Parameters within safe operating range")
            recommendations.append("Monitor impedance regularly for changes")
            recommendations.append("Start with minimum effective dose and titrate gradually")
        
        return recommendations
    
    def validate_charge_balance(self, cathodic_charge_uc: float, 
                               anodic_charge_uc: float,
                               tolerance_percent: float = 1.0) -> Dict:
        """
        Validate charge balance for biphasic stimulation
        Critical for preventing electrochemical damage
        """
        charge_imbalance = abs(cathodic_charge_uc - anodic_charge_uc)
        total_charge = cathodic_charge_uc + anodic_charge_uc
        imbalance_percent = (charge_imbalance / (total_charge / 2)) * 100
        
        is_balanced = imbalance_percent <= tolerance_percent
        
        return {
            'is_balanced': is_balanced,
            'imbalance_percent': round(imbalance_percent, 2),
            'cathodic_charge_uc': cathodic_charge_uc,
            'anodic_charge_uc': anodic_charge_uc,
            'charge_imbalance_uc': round(charge_imbalance, 4),
            'status': 'SAFE' if is_balanced else 'UNSAFE',
            'recommendation': (
                'Charge balance within tolerance' if is_balanced else
                f'Adjust pulse parameters to achieve <{tolerance_percent}% imbalance'
            )
        }
    
    def validate_thermal_safety(self, power_mw: float, duration_s: float,
                               tissue_thermal_conductivity: float = 0.5) -> Dict:
        """
        Validate thermal safety
        Simplified thermal model
        """
        # Energy delivered
        energy_mj = power_mw * duration_s
        
        # Estimate temperature rise (simplified)
        # ΔT ≈ P * t / (k * V)
        # Assuming small volume (~1 cm³)
        volume_cm3 = 1.0
        temp_rise_c = (power_mw / 1000) * duration_s / (tissue_thermal_conductivity * volume_cm3)
        
        final_temp_c = 37.0 + temp_rise_c  # Body temperature + rise
        
        is_safe = final_temp_c < self.limits.max_temperature_c
        
        return {
            'is_safe': is_safe,
            'estimated_temperature_c': round(final_temp_c, 2),
            'temperature_rise_c': round(temp_rise_c, 2),
            'energy_delivered_mj': round(energy_mj, 2),
            'status': 'SAFE' if is_safe else 'THERMAL RISK',
            'recommendation': (
                'Thermal safety within limits' if is_safe else
                'Reduce power or duration to prevent thermal damage'
            )
        }
    
    def check_biocompatibility(self, electrode_material: str) -> Dict:
        """Check electrode material biocompatibility"""
        approved_materials = {
            'Platinum': {'biocompatible': True, 'mri_safe': True, 'corrosion_resistant': True},
            'Platinum-Iridium': {'biocompatible': True, 'mri_safe': True, 'corrosion_resistant': True},
            'Titanium': {'biocompatible': True, 'mri_safe': True, 'corrosion_resistant': True},
            'Iridium Oxide': {'biocompatible': True, 'mri_safe': True, 'corrosion_resistant': True},
            'Gold': {'biocompatible': True, 'mri_safe': False, 'corrosion_resistant': True}
        }
        
        if electrode_material in approved_materials:
            properties = approved_materials[electrode_material]
            return {
                'material': electrode_material,
                'approved': True,
                'biocompatible': properties['biocompatible'],
                'mri_safe': properties['mri_safe'],
                'corrosion_resistant': properties['corrosion_resistant'],
                'recommendation': f'{electrode_material} is approved for chronic implantation'
            }
        else:
            return {
                'material': electrode_material,
                'approved': False,
                'recommendation': f'{electrode_material} not in approved materials list. Use Platinum-Iridium or Iridium Oxide.'
            }
    
    def validate_regulatory_compliance(self, device_specs: Dict) -> Dict:
        """
        Validate compliance with regulatory standards
        """
        compliance_checks = {
            'IEC 60601-1': self._check_iec_60601_1(device_specs),
            'IEC 60601-2-10': self._check_iec_60601_2_10(device_specs),
            'ISO 14708-3': self._check_iso_14708_3(device_specs),
            'FDA 21 CFR 820': self._check_fda_qsr(device_specs)
        }
        
        all_compliant = all(check['compliant'] for check in compliance_checks.values())
        
        return {
            'overall_compliance': all_compliant,
            'standards': compliance_checks,
            'status': 'COMPLIANT' if all_compliant else 'NON-COMPLIANT',
            'recommendations': self._get_compliance_recommendations(compliance_checks)
        }
    
    def _check_iec_60601_1(self, specs: Dict) -> Dict:
        """Check IEC 60601-1 (General medical electrical equipment safety)"""
        checks = []
        
        # Check for electrical isolation
        if specs.get('electrical_isolation', False):
            checks.append('Electrical isolation: PASS')
        else:
            checks.append('Electrical isolation: FAIL - Required for patient safety')
        
        # Check for leakage current limits
        leakage_current_ua = specs.get('leakage_current_ua', 0)
        if leakage_current_ua < 10:
            checks.append('Leakage current: PASS')
        else:
            checks.append(f'Leakage current: FAIL - {leakage_current_ua} μA exceeds 10 μA limit')
        
        compliant = all('PASS' in check for check in checks)
        
        return {'compliant': compliant, 'checks': checks}
    
    def _check_iec_60601_2_10(self, specs: Dict) -> Dict:
        """Check IEC 60601-2-10 (Nerve and muscle stimulators)"""
        checks = []
        
        # Output current limits
        if specs.get('max_current_ma', 0) <= 25.5:
            checks.append('Maximum current: PASS')
        else:
            checks.append('Maximum current: FAIL')
        
        # Emergency shutoff
        if specs.get('emergency_shutoff', False):
            checks.append('Emergency shutoff: PASS')
        else:
            checks.append('Emergency shutoff: FAIL - Required safety feature')
        
        compliant = all('PASS' in check for check in checks)
        
        return {'compliant': compliant, 'checks': checks}
    
    def _check_iso_14708_3(self, specs: Dict) -> Dict:
        """Check ISO 14708-3 (Active implantable medical devices - DBS)"""
        checks = []
        
        # Hermetic sealing
        if specs.get('hermetic_seal', False):
            checks.append('Hermetic sealing: PASS')
        else:
            checks.append('Hermetic sealing: FAIL - Required for implantable devices')
        
        # MRI compatibility
        if specs.get('mri_conditional', False):
            checks.append('MRI conditional: PASS')
        else:
            checks.append('MRI conditional: RECOMMENDED')
        
        # Battery safety
        if specs.get('battery_protection', False):
            checks.append('Battery protection: PASS')
        else:
            checks.append('Battery protection: FAIL')
        
        compliant = all('PASS' in check or 'RECOMMENDED' in check for check in checks)
        
        return {'compliant': compliant, 'checks': checks}
    
    def _check_fda_qsr(self, specs: Dict) -> Dict:
        """Check FDA 21 CFR Part 820 (Quality System Regulation)"""
        checks = []
        
        # Design controls
        if specs.get('design_validation', False):
            checks.append('Design validation: PASS')
        else:
            checks.append('Design validation: FAIL - Required for FDA approval')
        
        # Risk management
        if specs.get('risk_analysis', False):
            checks.append('Risk analysis: PASS')
        else:
            checks.append('Risk analysis: FAIL - ISO 14971 required')
        
        compliant = all('PASS' in check for check in checks)
        
        return {'compliant': compliant, 'checks': checks}
    
    def _get_compliance_recommendations(self, compliance_checks: Dict) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        for standard, result in compliance_checks.items():
            if not result['compliant']:
                recommendations.append(f"Address {standard} non-compliance issues")
                for check in result['checks']:
                    if 'FAIL' in check:
                        recommendations.append(f"  - {check}")
        
        if not recommendations:
            recommendations.append("Device meets all regulatory requirements")
        
        return recommendations
    
    def generate_safety_report(self, parameters: Dict, device_specs: Dict) -> str:
        """Generate comprehensive safety report"""
        validation = self.validate_parameters(**parameters)
        compliance = self.validate_regulatory_compliance(device_specs)
        
        report = f"""
================================================================
          DBS DEVICE SAFETY VALIDATION REPORT                 
================================================================

STIMULATION PARAMETERS:
  Amplitude: {parameters['amplitude_ma']} mA
  Frequency: {parameters['frequency_hz']} Hz
  Pulse Width: {parameters['pulse_width_us']} us
  Impedance: {parameters.get('impedance_ohms', 1000)} Ohm

SAFETY ASSESSMENT: {validation['safety_level'].upper()}

SAFETY METRICS:
  Charge Density: {validation['metrics']['charge_density_uc_cm2']} uC/cm2
    (Limit: {self.limits.max_charge_density_uc_cm2} uC/cm2)
  Current Density: {validation['metrics']['current_density_ma_cm2']} mA/cm2
    (Limit: {self.limits.max_current_density_ma_cm2} mA/cm2)
  Power Dissipation: {validation['metrics']['power_dissipation_mw']} mW
    (Limit: {self.limits.max_power_mw} mW)
  Voltage: {validation['metrics']['voltage_v']} V
    (Limit: {self.limits.max_voltage_v} V)

VIOLATIONS:
"""
        if validation['violations']:
            for v in validation['violations']:
                report += f"  ! {v}\n"
        else:
            report += "  * None\n"
        
        report += "\nWARNINGS:\n"
        if validation['warnings']:
            for w in validation['warnings']:
                report += f"  ! {w}\n"
        else:
            report += "  * None\n"
        
        report += f"\nREGULATORY COMPLIANCE: {compliance['status']}\n"
        for standard, result in compliance['standards'].items():
            status = "*" if result['compliant'] else "X"
            report += f"  {status} {standard}\n"
        
        report += "\nRECOMMENDATIONS:\n"
        for rec in validation['recommendations']:
            report += f"  - {rec}\n"
        
        report += "\n" + "="*64 + "\n"
        report += "WARNING: FOR RESEARCH AND EDUCATIONAL USE ONLY\n"
        report += "Clinical use requires regulatory approval and medical oversight\n"
        report += "="*64 + "\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    validator = SafetyValidator()
    
    # Test parameters
    params = {
        'amplitude_ma': 3.0,
        'frequency_hz': 130,
        'pulse_width_us': 90,
        'electrode_area_cm2': 0.06,
        'impedance_ohms': 1000
    }
    
    device_specs = {
        'electrical_isolation': True,
        'leakage_current_ua': 5,
        'max_current_ma': 25.5,
        'emergency_shutoff': True,
        'hermetic_seal': True,
        'mri_conditional': True,
        'battery_protection': True,
        'design_validation': True,
        'risk_analysis': True
    }
    
    # Generate report
    report = validator.generate_safety_report(params, device_specs)
    print(report)
    
    # Test charge balance
    print("\nCharge Balance Validation:")
    balance = validator.validate_charge_balance(
        cathodic_charge_uc=0.27,
        anodic_charge_uc=0.27
    )
    print(f"Status: {balance['status']}")
    print(f"Imbalance: {balance['imbalance_percent']}%")
