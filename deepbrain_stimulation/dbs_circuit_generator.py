"""
Deep Brain Stimulation Circuit Schematic Generator
Generates professional-grade DBS device circuit schematics with safety systems
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json


@dataclass
class ElectrodeConfig:
    """Configuration for DBS electrode array"""
    num_contacts: int = 4
    contact_spacing_mm: float = 1.5
    contact_diameter_mm: float = 1.3
    contact_length_mm: float = 1.5
    impedance_ohms: float = 1000
    material: str = "Platinum-Iridium"


@dataclass
class PulseGeneratorConfig:
    """Configuration for pulse generator circuit"""
    voltage_range_v: Tuple[float, float] = (0.0, 10.5)
    current_range_ma: Tuple[float, float] = (0.0, 25.5)
    frequency_range_hz: Tuple[float, float] = (2, 250)
    pulse_width_range_us: Tuple[float, float] = (60, 450)
    mode: str = "current_controlled"  # or "voltage_controlled"
    waveform: str = "biphasic"  # or "monophasic"


@dataclass
class SafetyConfig:
    """Safety system configuration"""
    max_charge_density_uc_cm2: float = 30.0  # Microcoulombs per cm²
    max_current_density_ma_cm2: float = 2.0
    max_temperature_c: float = 38.0
    impedance_check_enabled: bool = True
    emergency_shutoff_enabled: bool = True


class DBSCircuitGenerator:
    """Generates DBS circuit schematics with component specifications"""
    
    def __init__(self):
        self.electrode_config = ElectrodeConfig()
        self.pulse_config = PulseGeneratorConfig()
        self.safety_config = SafetyConfig()
        
    def generate_electrode_array_schematic(self) -> Dict:
        """Generate electrode array circuit schematic"""
        
        # Calculate electrode surface area
        contact_area_cm2 = (np.pi * self.electrode_config.contact_diameter_mm * 
                           self.electrode_config.contact_length_mm) / 100
        
        schematic = {
            "component": "Electrode Array",
            "type": "Multi-contact DBS Lead",
            "specifications": {
                "contacts": self.electrode_config.num_contacts,
                "spacing_mm": self.electrode_config.contact_spacing_mm,
                "diameter_mm": self.electrode_config.contact_diameter_mm,
                "length_mm": self.electrode_config.contact_length_mm,
                "surface_area_cm2": round(contact_area_cm2, 4),
                "impedance_ohms": self.electrode_config.impedance_ohms,
                "material": self.electrode_config.material
            },
            "connections": [
                {"contact": i, "wire": f"Lead_{i}", "color": self._get_wire_color(i)}
                for i in range(self.electrode_config.num_contacts)
            ],
            "design_notes": [
                "Directional steering capability with segmented contacts",
                "Low impedance for efficient charge delivery",
                "Biocompatible platinum-iridium alloy",
                "MRI-conditional design with RF filtering"
            ]
        }
        
        return schematic
    
    def generate_pulse_generator_schematic(self) -> Dict:
        """Generate pulse generator circuit schematic"""
        
        schematic = {
            "component": "Programmable Pulse Generator",
            "architecture": "Microcontroller-based Current Source",
            "main_components": {
                "microcontroller": {
                    "type": "ARM Cortex-M4",
                    "purpose": "Parameter control and timing",
                    "features": ["DSP", "Low power", "Wireless communication"]
                },
                "current_source": {
                    "type": "Precision DAC + Op-Amp",
                    "topology": "Howland current pump",
                    "components": {
                        "DAC": "16-bit DAC8568 (Texas Instruments)",
                        "op_amp": "OPA2140 (Low noise, rail-to-rail)",
                        "feedback_resistor": "10kΩ precision (0.1%)",
                        "output_capacitor": "100nF ceramic"
                    },
                    "specifications": {
                        "resolution": "16-bit (390 nA steps)",
                        "accuracy": "±1%",
                        "compliance_voltage": "±10V"
                    }
                },
                "h_bridge": {
                    "purpose": "Biphasic waveform generation",
                    "components": {
                        "mosfets": "4x BSS138 (N-channel)",
                        "gate_drivers": "2x TC4427 (Dual MOSFET driver)"
                    },
                    "switching_frequency": "Up to 1 MHz"
                },
                "output_stage": {
                    "coupling_capacitor": "10µF tantalum (DC blocking)",
                    "series_resistor": "100Ω (current limiting)",
                    "protection_diodes": "2x BAV99 (Schottky, bidirectional)"
                }
            },
            "timing_circuit": {
                "pulse_width_generator": "Hardware timer with 1µs resolution",
                "frequency_generator": "Programmable divider from 32kHz crystal",
                "duty_cycle_control": "PWM modulation"
            },
            "parameter_ranges": {
                "voltage_v": list(self.pulse_config.voltage_range_v),
                "current_ma": list(self.pulse_config.current_range_ma),
                "frequency_hz": list(self.pulse_config.frequency_range_hz),
                "pulse_width_us": list(self.pulse_config.pulse_width_range_us)
            },
            "design_notes": [
                "Constant current mode ensures consistent charge delivery",
                "Biphasic pulses prevent charge accumulation",
                "Active charge balancing with <1% mismatch",
                "Isolated output stage for patient safety"
            ]
        }
        
        return schematic
    
    def generate_power_management_schematic(self) -> Dict:
        """Generate power management system schematic"""
        
        schematic = {
            "component": "Power Management System",
            "primary_power": {
                "battery": {
                    "type": "Lithium-ion rechargeable",
                    "capacity_mah": 150,
                    "voltage_v": 3.7,
                    "chemistry": "LiCoO2 (medical grade)",
                    "protection": "Built-in PCM (Protection Circuit Module)"
                },
                "charging": {
                    "method": "Inductive (wireless)",
                    "frequency_khz": 125,
                    "coil": "Ferrite core, 10mm diameter",
                    "rectifier": "Full-bridge with Schottky diodes",
                    "charge_controller": "BQ25570 (Texas Instruments)"
                }
            },
            "voltage_regulators": {
                "digital_3v3": {
                    "ic": "TPS62740 (Ultra-low quiescent)",
                    "output_v": 3.3,
                    "current_ma": 300,
                    "efficiency": "95%"
                },
                "analog_5v": {
                    "ic": "TPS7A4700 (Low noise LDO)",
                    "output_v": 5.0,
                    "current_ma": 100,
                    "psrr_db": 80
                },
                "stimulation_boost": {
                    "ic": "TPS61099 (Boost converter)",
                    "output_v": 12.0,
                    "current_ma": 50,
                    "purpose": "High voltage for stimulation compliance"
                }
            },
            "power_monitoring": {
                "battery_gauge": "BQ27441 (Fuel gauge IC)",
                "current_sense": "INA219 (High-side current monitor)",
                "voltage_monitor": "ADC channels on MCU"
            },
            "estimated_battery_life": {
                "continuous_stimulation_hours": 48,
                "standby_days": 30,
                "recharge_time_hours": 2
            },
            "design_notes": [
                "Ultra-low power design for extended battery life",
                "Wireless charging eliminates infection risk from connectors",
                "Multiple voltage rails for clean analog/digital separation",
                "Battery protection prevents over-discharge and thermal runaway"
            ]
        }
        
        return schematic
    
    def generate_safety_system_schematic(self) -> Dict:
        """Generate safety monitoring and protection system"""
        
        schematic = {
            "component": "Safety and Monitoring System",
            "current_monitoring": {
                "sensor": "Hall effect current sensor (ACS712)",
                "range_ma": "±30",
                "resolution_ma": 0.1,
                "sampling_rate_hz": 10000,
                "safety_action": "Shutoff if >110% of programmed current"
            },
            "impedance_monitoring": {
                "method": "AC impedance measurement at 1kHz",
                "frequency_hz": 1000,
                "test_current_ua": 10,
                "normal_range_ohms": [500, 2000],
                "alert_conditions": [
                    "Open circuit (>5kΩ): Lead disconnection",
                    "Short circuit (<200Ω): Insulation breach",
                    "Rapid change (>20%/hour): Tissue reaction"
                ]
            },
            "temperature_monitoring": {
                "sensor": "NTC thermistor (10kΩ at 25°C)",
                "location": "Near pulse generator output",
                "sampling_rate_hz": 1,
                "alert_threshold_c": self.safety_config.max_temperature_c,
                "shutoff_threshold_c": 40.0
            },
            "charge_density_protection": {
                "calculation": "Q = I × PW / Area",
                "max_charge_uc_cm2": self.safety_config.max_charge_density_uc_cm2,
                "enforcement": "Firmware limits on I and PW combination",
                "reference": "Shannon 1992 safe stimulation limits"
            },
            "emergency_shutoff": {
                "triggers": [
                    "Temperature > 40°C",
                    "Current > 28 mA",
                    "Impedance out of range",
                    "Battery voltage < 3.0V",
                    "Watchdog timer timeout"
                ],
                "action": "Open all output switches, disable DAC",
                "recovery": "Manual reset required via programmer",
                "indicator": "LED blink pattern + wireless alert"
            },
            "redundancy": {
                "dual_microcontrollers": "Primary + safety supervisor",
                "independent_watchdog": "External watchdog IC (TPS3823)",
                "hardware_interlocks": "Analog comparators for instant shutoff"
            },
            "design_notes": [
                "Multi-layer safety approach: firmware + hardware",
                "Fail-safe design: defaults to OFF state",
                "Continuous monitoring during stimulation",
                "Compliant with IEC 60601-1 medical device safety"
            ]
        }
        
        return schematic
    
    def generate_signal_processing_schematic(self) -> Dict:
        """Generate signal processing and feedback system"""
        
        schematic = {
            "component": "Signal Processing and Feedback System",
            "sensing_channels": {
                "num_channels": 4,
                "purpose": "Local field potential (LFP) recording",
                "amplifier": {
                    "type": "Instrumentation amplifier",
                    "ic": "INA333 (Texas Instruments)",
                    "gain": 1000,
                    "bandwidth_hz": [0.5, 500],
                    "input_impedance_ohms": 10e9,
                    "cmrr_db": 100
                },
                "filters": {
                    "high_pass": {
                        "type": "2nd order Butterworth",
                        "cutoff_hz": 0.5,
                        "purpose": "Remove DC offset"
                    },
                    "low_pass": {
                        "type": "4th order Butterworth",
                        "cutoff_hz": 500,
                        "purpose": "Anti-aliasing"
                    },
                    "notch": {
                        "type": "Twin-T notch filter",
                        "frequency_hz": 60,
                        "q_factor": 30,
                        "purpose": "Power line interference rejection"
                    }
                },
                "adc": {
                    "ic": "ADS1299 (8-channel, 24-bit)",
                    "resolution_bits": 24,
                    "sampling_rate_hz": 1000,
                    "input_referred_noise_uv": 1.0
                }
            },
            "artifact_rejection": {
                "blanking_window_ms": 2.0,
                "purpose": "Blank ADC during stimulation pulse",
                "recovery_time_ms": 5.0,
                "adaptive_filtering": "Wiener filter for residual artifacts"
            },
            "closed_loop_control": {
                "biomarker": "Beta band power (13-30 Hz)",
                "feature_extraction": "FFT with Welch's method",
                "control_algorithm": "PID controller",
                "update_rate_hz": 10,
                "parameters": {
                    "kp": 0.5,
                    "ki": 0.1,
                    "kd": 0.05
                }
            },
            "wireless_communication": {
                "protocol": "Bluetooth Low Energy 5.0",
                "ic": "nRF52832 (Nordic Semiconductor)",
                "data_rate_kbps": 1000,
                "range_m": 10,
                "encryption": "AES-128",
                "transmitted_data": [
                    "LFP signals (compressed)",
                    "Stimulation parameters",
                    "Battery status",
                    "Safety alerts"
                ]
            },
            "design_notes": [
                "Closed-loop capability for adaptive stimulation",
                "High-quality LFP recording for research",
                "Artifact rejection enables simultaneous stim/record",
                "Secure wireless communication for patient privacy"
            ]
        }
        
        return schematic
    
    def generate_complete_system_schematic(self) -> Dict:
        """Generate complete DBS system schematic"""
        
        system = {
            "system_name": "Deep Brain Stimulation Device for PTSD Treatment",
            "architecture": "Implantable Pulse Generator (IPG) + Lead System",
            "components": {
                "electrode_array": self.generate_electrode_array_schematic(),
                "pulse_generator": self.generate_pulse_generator_schematic(),
                "power_management": self.generate_power_management_schematic(),
                "safety_system": self.generate_safety_system_schematic(),
                "signal_processing": self.generate_signal_processing_schematic()
            },
            "system_specifications": {
                "dimensions_mm": [50, 40, 10],
                "weight_g": 25,
                "implant_location": "Subclavicular pocket",
                "lead_length_cm": 40,
                "mri_conditional": "1.5T and 3T with restrictions",
                "biocompatibility": "ISO 10993 compliant",
                "hermetic_seal": "Titanium case with ceramic feedthrough"
            },
            "target_brain_regions": {
                "primary": "Basolateral Amygdala (BLA)",
                "secondary": "Ventromedial Prefrontal Cortex (vmPFC)",
                "tertiary": "Hippocampus (CA1 region)",
                "rationale": "Fear extinction and emotional regulation circuits"
            },
            "clinical_parameters": {
                "typical_settings": {
                    "amplitude_ma": 3.0,
                    "frequency_hz": 130,
                    "pulse_width_us": 90,
                    "mode": "Continuous bilateral stimulation"
                },
                "adjustment_range": {
                    "amplitude_ma": [0.5, 8.0],
                    "frequency_hz": [20, 185],
                    "pulse_width_us": [60, 210]
                }
            },
            "regulatory_compliance": [
                "FDA 21 CFR Part 820 (Quality System Regulation)",
                "IEC 60601-1 (Medical electrical equipment safety)",
                "IEC 60601-2-10 (Nerve and muscle stimulators)",
                "ISO 14708-3 (Active implantable medical devices - DBS)",
                "ISO 13485 (Medical devices quality management)"
            ]
        }
        
        return system
    
    def generate_ocd_specific_schematic(self) -> Dict:
        """Generate OCD-specific circuit schematic and specifications"""
        
        schematic = {
            "application": "Obsessive-Compulsive Disorder (OCD) Treatment",
            "target_structures": [
                "Anterior Limb of Internal Capsule (ALIC)",
                "Ventral Capsule / Ventral Striatum (VC/VS)",
                "Nucleus Accumbens (NAc)",
                "Subthalamic Nucleus (STN) - Antero-medial"
            ],
            "electrical_specifications": {
                "stimulation_mode": "Voltage-Controlled or Current-Controlled (Constant Current preferred)",
                "waveform": "Biphasic, Charge-Balanced, Square Wave",
                "frequency_range": {
                    "min": "100 Hz",
                    "max": "180 Hz",
                    "optimal": "130-140 Hz (High Frequency for functional lesioning)"
                },
                "amplitude_range": {
                    "voltage_mode": "2.0 V - 8.0 V",
                    "current_mode": "1.0 mA - 6.0 mA",
                    "typical_therapeutic": "3.0 - 5.0 V / 2.5 - 4.5 mA"
                },
                "pulse_width": {
                    "range": "60 µs - 210 µs",
                    "typical": "90 µs - 120 µs"
                },
                "impedance_load": "500 Ω - 1500 Ω"
            },
            "electrode_configuration": {
                "type": "Quadripolar Lead (4 contacts)",
                "spacing": "1.5 mm or 3.0 mm (Wider spacing often used for ALIC)",
                "contact_active_selection": "Monopolar (Case Return) or Bipolar (Adjacent Contact)",
                "field_shaping": "Current Steering required for precise VC/VS targeting"
            },
            "safety_limits": {
                "max_charge_density": "30 µC/cm²/phase",
                "max_voltage_compliance": "12.0 V",
                "protection_features": [
                    "Soft-start ramping (2-4 seconds)",
                    "Open-circuit detection",
                    "Thermal monitoring"
                ]
            },
            "mechanism_of_action": "Disruption of hyperactive orbitofrontal-thalamic connectivity loops via high-frequency depolarization block or axonal jamming.",
            "svg_schematic": self.generate_svg_schematic(component="ocd")
        }
        
        return schematic
    
    def generate_svg_schematic(self, component: str = "complete") -> str:
        """Generate SVG circuit diagram"""
        
        # Simplified SVG generation - in production, use a proper circuit drawing library
        svg_header = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800" width="1200" height="800">
  <defs>
    <style>
      .component { fill: #2a2a2a; stroke: #00d4ff; stroke-width: 2; }
      .wire { stroke: #00ff88; stroke-width: 2; fill: none; }
      .text { fill: #ffffff; font-family: Arial, sans-serif; font-size: 14px; }
      .label { fill: #00d4ff; font-family: Arial, sans-serif; font-size: 12px; }
      .title { fill: #00d4ff; font-family: Arial, sans-serif; font-size: 24px; font-weight: bold; }
    </style>
  </defs>
  
  <rect width="1200" height="800" fill="#0a0a0a"/>
  <text x="600" y="40" class="title" text-anchor="middle">DBS Circuit Schematic</text>
'''
        
        # Draw main components
        components_svg = '''
  <!-- Electrode Array -->
  <g id="electrode-array">
    <rect x="100" y="100" width="150" height="200" class="component" rx="10"/>
    <text x="175" y="130" class="text" text-anchor="middle">Electrode Array</text>
    <text x="175" y="150" class="label" text-anchor="middle">4-Contact Lead</text>
    <circle cx="175" cy="180" r="8" fill="#00ff88"/>
    <circle cx="175" cy="210" r="8" fill="#00ff88"/>
    <circle cx="175" cy="240" r="8" fill="#00ff88"/>
    <circle cx="175" cy="270" r="8" fill="#00ff88"/>
  </g>
  
  <!-- Pulse Generator -->
  <g id="pulse-generator">
    <rect x="350" y="100" width="200" height="200" class="component" rx="10"/>
    <text x="450" y="130" class="text" text-anchor="middle">Pulse Generator</text>
    <text x="450" y="155" class="label" text-anchor="middle">16-bit DAC</text>
    <text x="450" y="175" class="label" text-anchor="middle">Howland Current Pump</text>
    <text x="450" y="195" class="label" text-anchor="middle">H-Bridge</text>
    <rect x="380" y="210" width="140" height="60" fill="#1a1a1a" stroke="#00d4ff" stroke-width="1"/>
    <text x="450" y="245" class="label" text-anchor="middle">0-25.5 mA</text>
  </g>
  
  <!-- Power Management -->
  <g id="power-management">
    <rect x="650" y="100" width="180" height="200" class="component" rx="10"/>
    <text x="740" y="130" class="text" text-anchor="middle">Power System</text>
    <text x="740" y="155" class="label" text-anchor="middle">Li-ion Battery</text>
    <text x="740" y="175" class="label" text-anchor="middle">150 mAh</text>
    <text x="740" y="200" class="label" text-anchor="middle">Wireless Charging</text>
    <circle cx="740" cy="240" r="30" fill="none" stroke="#00ff88" stroke-width="3"/>
    <text x="740" y="248" class="label" text-anchor="middle">⚡</text>
  </g>
  
  <!-- Safety System -->
  <g id="safety-system">
    <rect x="900" y="100" width="200" height="200" class="component" rx="10"/>
    <text x="1000" y="130" class="text" text-anchor="middle">Safety System</text>
    <text x="1000" y="155" class="label" text-anchor="middle">Current Monitor</text>
    <text x="1000" y="175" class="label" text-anchor="middle">Impedance Check</text>
    <text x="1000" y="195" class="label" text-anchor="middle">Temperature Sensor</text>
    <text x="1000" y="215" class="label" text-anchor="middle">Emergency Shutoff</text>
    <rect x="930" y="230" width="140" height="50" fill="#ff3333" fill-opacity="0.3" stroke="#ff3333" stroke-width="2"/>
    <text x="1000" y="260" class="text" text-anchor="middle">PROTECTION</text>
  </g>
  
  <!-- Signal Processing -->
  <g id="signal-processing">
    <rect x="350" y="400" width="200" height="200" class="component" rx="10"/>
    <text x="450" y="430" class="text" text-anchor="middle">Signal Processing</text>
    <text x="450" y="455" class="label" text-anchor="middle">LFP Recording</text>
    <text x="450" y="475" class="label" text-anchor="middle">24-bit ADC</text>
    <text x="450" y="495" class="label" text-anchor="middle">Artifact Rejection</text>
    <text x="450" y="515" class="label" text-anchor="middle">Closed-Loop Control</text>
    <text x="450" y="540" class="label" text-anchor="middle">BLE 5.0</text>
  </g>
  
  <!-- Microcontroller -->
  <g id="microcontroller">
    <rect x="650" y="400" width="180" height="200" class="component" rx="10"/>
    <text x="740" y="430" class="text" text-anchor="middle">Microcontroller</text>
    <text x="740" y="455" class="label" text-anchor="middle">ARM Cortex-M4</text>
    <text x="740" y="480" class="label" text-anchor="middle">Parameter Control</text>
    <text x="740" y="500" class="label" text-anchor="middle">Safety Monitoring</text>
    <text x="740" y="520" class="label" text-anchor="middle">Wireless Comm</text>
  </g>
  
  <!-- Wiring -->
  <g id="wiring">
    <path d="M 250 200 L 350 200" class="wire"/>
    <path d="M 550 200 L 650 200" class="wire"/>
    <path d="M 830 200 L 900 200" class="wire"/>
    <path d="M 450 300 L 450 400" class="wire"/>
    <path d="M 550 500 L 650 500" class="wire"/>
    <path d="M 740 400 L 740 300" class="wire"/>
    <path d="M 1000 300 L 1000 350 L 740 350" class="wire"/>
  </g>
  
  <!-- Annotations -->
  <text x="100" y="650" class="label">⚠ Medical Device - Research/Educational Use Only</text>
  <text x="100" y="670" class="label">Compliant with IEC 60601-1, ISO 14708-3</text>
  <text x="100" y="690" class="label">Charge Density: &lt;30 µC/cm² (Shannon limit)</text>
'''
        
        svg_footer = '''
</svg>
'''
        
        return svg_header + components_svg + svg_footer
    
    def _get_wire_color(self, index: int) -> str:
        """Get wire color for electrode contact"""
        colors = ["Red", "Yellow", "Green", "Blue", "White", "Black", "Orange", "Purple"]
        return colors[index % len(colors)]
    
    def export_schematics(self, output_dir: str = "."):
        """Export all schematics to JSON and SVG files"""
        import os
        
        # Generate complete system schematic
        system = self.generate_complete_system_schematic()
        
        # Save JSON
        json_path = os.path.join(output_dir, "dbs_circuit_schematic.json")
        with open(json_path, 'w') as f:
            json.dump(system, f, indent=2)
        
        # Save SVG
        svg_path = os.path.join(output_dir, "dbs_circuit_diagram.svg")
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_svg_schematic())
        
        return {
            "json_path": json_path,
            "svg_path": svg_path,
            "system": system
        }


if __name__ == "__main__":
    # Example usage
    generator = DBSCircuitGenerator()
    
    # Generate and print complete system schematic
    system = generator.generate_complete_system_schematic()
    print(json.dumps(system, indent=2))
    
    # Export to files
    result = generator.export_schematics()
    print(f"\nSchematics exported to:")
    print(f"  JSON: {result['json_path']}")
    print(f"  SVG: {result['svg_path']}")
