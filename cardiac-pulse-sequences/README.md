# Cardiac Parallel Imaging Pulse Sequence Generator with LLM Integration

A professional-grade web-based application for designing, optimizing, and visualizing cardiac MRI parallel imaging pulse sequences with integrated Large Language Model capabilities for intelligent protocol generation.

## Features

### Parallel Imaging Techniques
- **SENSE** (Sensitivity Encoding) - Coil-based parallel imaging
- **GRAPPA** (Generalized Autocalibrating Partial Parallel Acquisition)
- **Compressed Sensing** - Incoherent undersampling with sparsity constraints
- **SMS** (Simultaneous Multi-Slice) - Multi-band excitation
- **Hybrid** - Combined SENSE + Compressed Sensing

### Cardiac Sequences
- **CINE** - Balanced SSFP for dynamic cardiac imaging
- **Perfusion** - First-pass myocardial perfusion
- **LGE** (Late Gadolinium Enhancement) - Scar/fibrosis detection
- **Parametric Mapping** - T1, T2, T2*, ECV quantification
- **Phase Contrast Flow** - 2D/4D velocity encoding

### LLM-Powered Features
- Natural language protocol generation from clinical scenarios
- Intelligent parameter optimization based on patient factors
- Real-time explanations of trade-offs and recommendations
- Conversational interface for protocol refinement

### Visualizations
- Interactive k-space undersampling patterns
- Real-time g-factor calculations and maps
- ECG-gated cardiac phase timing diagrams
- Pulse sequence diagrams
- SNR and quality metrics

### Export Capabilities
- **PyPulseq** - Complete Python code generation
- **Vendor Formats** - Siemens, GE, Philips parameter files
- **Clinical Reports** - PDF-ready HTML protocol summaries
- **JSON** - Machine-readable configuration files

## Quick Start

1. **Open the Application**
   ```bash
   # Simply open index.html in a modern web browser
   # Or use a local server:
   python -m http.server 8000
   # Then navigate to http://localhost:8000
   ```

2. **Try the LLM Assistant**
   - Example: "Create a protocol for suspected myocarditis in a 45yo male"
   - The AI will generate a complete protocol with rationale

3. **Adjust Parameters**
   - Use sliders to modify acceleration factor, coil elements, etc.
   - Watch real-time updates of g-factor, SNR penalty, and scan time

4. **Export**
   - Click "Export PyPulseq" for executable Python code
   - Generate clinical reports for protocol documentation

## Clinical Use Cases

### Myocarditis Protocol
```
LLM Prompt: "45 year old male, suspected acute myocarditis, chest pain"
Generated: CINE + T2 mapping + T1 mapping + LGE
Rationale: T2 for edema, T1 for ECV, LGE for fibrosis
```

### Viability Assessment
```
LLM Prompt: "Optimize for ischemic viability assessment"
Generated: CINE + Rest Perfusion + LGE
Parameters: R=3.0x hybrid acceleration for breath-hold tolerance
```

### Function Assessment
```
LLM Prompt: "High temporal resolution for function only"
Generated: Multi-phase CINE with 30 cardiac phases
Parameters: 35ms temporal resolution, R=2.5x SENSE
```

## Technical Architecture

### Physics Engine
- `cardiac-physics.js` - SSFP, IR, T1/T2 fitting, perfusion kinetics
- `parallel-physics.js` - G-factor, SNR penalty, CS patterns, SMS
- `physics.js` - Base MRI calculations

### Sequence Management
- `parallel-imaging.js` - Acceleration configuration and metrics
- `cardiac-sequences.js` - CINE, perfusion, LGE, mapping, flow
- `cardiac-gating.js` - ECG synchronization

### LLM Integration
- `llm-engine.js` - Protocol templates, optimization, reasoning
- `llm-interface.js` - Chat UI, message handling, settings application
- `protocol-reasoning.js` - Clinical decision support

### Visualization
- `visualizer.js` - K-space patterns, ECG timing, Canvas rendering
- Supports high-DPI displays with automatic scaling

### Export
- `export.js` - Multi-format exportwith PyPulseq code generation
- `pypulseq-generator.js` - Sequence-specific Python templates

## Parallel Imaging Background

### G-Factor
Quantifies SNR penalty from parallel imaging geometry:
- **g = 1.0**: Perfect (no penalty)
- **g = 1.2-1.5**: Excellent clinical quality
- **g > 2.0**: Poor coil geometry, high noise

Formula: `SNR_parallel = SNR_full / (√R × g)`

### Acceleration Trade-offs
- **Higher R**: Faster scans, lower SNR, potential artifacts
- **Lower R**: Better quality, longer acquisition time
- **Optimal R**: Balance based on coil array, anatomy, and clinical needs

### GRAPPA vs SENSE
- **GRAPPA**: k-space domain, needs ACS lines, robust to motion
- **SENSE**: Image domain, requires coil sensitivity maps, lower noise
- **Hybrid**: Combines advantages, ideal for high acceleration

## Cardiac Imaging Guidelines

### Temporal Resolution
- **Standard CINE**: 40-50ms (~20-25 phases at 70 bpm)
- **High HR patients**: 30-35ms (~30 phases)
- **Stress imaging**: <50ms to avoid temporal blurring

### InversionTime (TI) for LGE
- **1.5T**: Typically 250-350ms for myocardial nulling
- **3T**: Typically 300-400ms
- **Always perform TI scout** to optimize per patient

### T1 Mapping Schemes
- **MOLLI 5(3)3**: Standard, 17 heartbeats
- **ShMOLLI 4(1)3(1)2**: Shortened, 9 heartbeats
- **SAPPHIRE**: Saturation recovery, HR independent

## Browser Compatibility

- **Recommended**: Chrome 90+, Firefox 88+, Edge 90+
- **Required**: Canvas API, ES6 JavaScript
- **No build process**: Runs directly in browser

## Future Enhancements

- [ ] 4D Flow visualization with particle tracing
- [ ] Real-time reconstruction simulation
- [ ] Integration with actual ML models (ONNX Runtime)
- [ ] DICOM export with embedded parameters
- [ ] Multi-user protocol sharing and collaboration
- [ ] Vendor-specific hardware limit validation

## License & Citation

Created for research and educational purposes. If using in publications, please cite:

```
Cardiac Parallel Imaging Generator with LLM Integration
Advanced MRI Protocol Design Tool, 2025
```

## Support

For questions or contributions:
- Review the inline code documentation
- Check browser console for debugging information
- LLM responses are rule-based demonstrations (easily upgraded to real API)

---

**Built with ❤️ for the cardiac MRI community**
