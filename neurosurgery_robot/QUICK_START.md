# Quick Start Guide - NeuroMorph Surgical Robotics

## 🚀 Launch the Application

### Prerequisites
- Python 3.8+
- Required packages: flask, numpy, scipy, matplotlib, scikit-image

### Start Server
```bash
cd /Users/cartiksharma/Downloads/neuromorph-main-10/neurosurgery_robot
FLASK_RUN_PORT=5001 python3 app.py
```

### Access Web Interface
Open browser and navigate to: **http://localhost:5001**

---

## 🎯 Using the System

### 1. Robot Control
The 6-DOF surgical robot can be controlled in real-time:

**Control Commands:**
```json
POST /api/control
{
    "target_pos": {"x": 0.5, "y": 0.0, "z": 0.5},
    "laser": true,        // Enable/disable laser ablation
    "cryo": false,        // Enable/disable cryoprobe
    "ablation": true,     // Start/stop ablation sequence
    "home": false         // Move to home position
}
```

**Expected Response:**
```json
{"status": "ok", "updated": true}
```

### 2. Temperature Monitoring
Real-time thermal imaging with accurate bioheat physics:

**Get Telemetry:**
```
GET /api/telemetry
```

Returns:
- Robot position and joint angles
- Temperature map (128×128 grid)
- Thermal damage (CEM43 dose)
- Visualization images (base64)
- Performance metrics

### 3. Tumor Segmentation
Automatic tumor detection with level set algorithm:

**Check Segmentation Quality:**
```
GET /api/segmentation/quality
```

Response includes:
- Quality rating (excellent/good/fair)
- Circularity metric (0-1)
- Solidity metric (0-1)
- Readiness for ablation

### 4. Ablation Planning

**Plan Therapy Trajectory:**
```json
POST /api/ablation/plan
{
    "method": "spiral"    // Options: sequential, concentric, spiral
}
```

**Available Methods:**
- **sequential**: Direct center targeting
- **concentric**: Circular paths around tumor
- **spiral**: Expanding spiral pattern

### 5. Trajectory Planning

**Plan Robot Movement:**
```json
POST /api/trajectory/plan
{
    "x": 0.5,
    "y": 0.0,
    "z": 0.5,
    "duration": 5.0,
    "path_type": "linear"  // Options: linear, circular
}
```

### 6. Thermal History

**Get Temperature Over Time:**
```
GET /api/thermal/history
```

Returns latest 100 temperature samples and simulation time.

---

## 🔧 Control Workflow Example

### Step-by-Step Surgical Procedure

```
1. START APPLICATION
   → APP starts at http://localhost:5001
   → Simulation loop begins (50Hz)
   → Tumor auto-detected and segmented

2. VERIFY SEGMENTATION
   → GET /api/segmentation/quality
   → Check circularity > 0.6
   → Confirm "tumor_ready_for_ablation": true

3. PLAN ABLATION
   → POST /api/ablation/plan with method="spiral"
   → Receive waypoint trajectory
   → Review trajectory safety

4. PREPARE ROBOT
   → POST /api/control with "home": true
   → Wait for robot to reach home position
   → Verify safety status

5. NAVIGATE TO TUMOR
   → POST /api/control with target_pos toward tumor_center
   → Monitor GET /api/telemetry for position
   → Wait for position_error < 0.01m

6. ACTIVATE ABLATION
   → POST /api/control with "laser": true, "ablation": true
   → Monitor temperature: GET /api/thermal/history
   → Watch for peak temp reaching 50-60°C

7. APPLY CRYOGENIC SHOCK (Optional)
   → POST /api/control with "laser": false, "cryo": true
   → Monitor ice ball formation
   → Observe freeze/thaw cycle

8. MONITOR THERMAL DOSE
   → GET /api/telemetry repeatedly
   → Check "thermal.metrics.peak_damage"
   → Abort when damage > 240 CEM43

9. COMPLETE PROCEDURE
   → POST /api/control with all systems: false
   → Document results
   → Get final telemetry snapshot

10. REVIEW RESULTS
    → Temperature map shows ablated region
    → Segmentation shows tumor coverage
    → Cryo map shows ice ball extent
```

---

## 📊 Understanding the Parameters

### Thermal Parameters
- **Max Temperature**: 20-100°C range
- **Normal**: 37°C (body temperature)
- **Therapeutic Range**: 45-60°C for heating ablation
- **CEM43 Dose**: Cumulative equivalent minutes at 43°C
  - < 60 CEM43: No effect
  - 60-240 CEM43: Partial damage
  - > 240 CEM43: Irreversible necrosis

### Cryo Parameters
- **Probe Temperature**: -150°C (Argon gas)
- **Ice Fraction**: 0.0 (liquid) to 1.0 (solid)
- **Freeze Point**: 0°C
- **Ice Ball Radius**: Grows exponentially with active cooling
- **Thawing Rate**: 0.15-0.2°C/second passive

### Robot Parameters
- **Workspace**: Normalized [0, 1]³ coordinates
- **Speed Limits**: 0.8-2.0 rad/s per joint
- **Accuracy**: ±5mm typical error
- **IK Convergence**: < 50ms for new target calculus
- **Update Rate**: 50Hz (20ms per cycle)

### Segmentation Parameters
- **Safety Margin**: 5mm automatic buffer
- **Circularity**: 1.0 = perfect circle, <0.5 = irregular
- **Solidity**: 1.0 = convex, <0.8 = irregular shape
- **Tumor Detection Threshold**: > 0.8 intensity

---

## 🔬 API Response Examples

### Telemetry Example
```json
{
  "robot": {
    "joints": [0.1, 0.2, -0.1, 0.5, -0.3, 0.2],
    "position": [0.45, -0.05, 0.52],
    "target_pos": [0.5, 0.0, 0.5],
    "position_error_m": 0.032,
    "safety": {
      "in_workspace": true,
      "joints_ok": true,
      "safe": true
    }
  },
  "thermal": {
    "max_temperature": 48.5,
    "mean_temperature": 37.2,
    "metrics": {
      "max_temperature": 48.5,
      "mean_temperature": 37.2,
      "peak_damage": 145.3,
      "necrotic_volume": 42,
      "temperature_gradient": 2.1
    },
    "visualization": "data:image/png;base64,..."
  },
  "cryo": {
    "metrics": {
      "total_ice_pixels": 234,
      "fully_frozen_pixels": 89,
      "transition_zone_pixels": 145,
      "ice_ball_radius_mm": 8.5,
      "min_temperature": -45.2,
      "probe_active": true
    }
  },
  "segmentation": {
    "tumor_center": [0.48, 0.52],
    "tumor_volume_pixels": 312,
    "quality_metrics": {
      "tumor_volume_pixels": 312,
      "boundary_length": 92,
      "circularity": 0.78,
      "solidity": 0.82
    }
  }
}
```

---

## ⚠️ Safety Considerations

1. **Workspace Limits**: Robot constrained to [0, 1]³
2. **Temperature Limits**: 
   - Normal: 37°C
   - Maximum safe: 80°C
   - Emergency: > 85°C
3. **Safety Margins**: Automatic 5mm buffer around tumor
4. **Joint Limits**: Enforced per DH parameters
5. **Thermal Dose**: Stop when CEM43 > 240

---

## 🐛 Troubleshooting

### Issue: App won't start
```bash
# Check if port is in use
lsof -i :5001

# Kill existing process if needed
kill -9 <PID>

# Try different port
FLASK_RUN_PORT=5002 python3 app.py
```

### Issue: Robot not moving
1. Check position_error in telemetry
2. Verify target_pos is within [0, 1]³
3. Check safety.safe == true
4. Verify joints_ok == true

### Issue: Temperature not rising
1. Check laser_enabled == true
2. Check ablation_active == true
3. Verify robot is over tumor target
4. Check thermal.max_temperature

### Issue: Ice ball not forming
1. Check cryo_enabled == true
2. Verify probe_active == true
3. Check ice_ball_radius_mm increasing
4. Monitor temperature dropping

---

## 📈 Performance Monitoring

### Key Metrics to Watch
```
Robot:
  - position_error: Should decrease over time
  - joints_ok: Should always be true
  - safe: Should always be true

Thermal:
  - max_temperature: Should increase with laser
  - peak_damage: Should increase toward target (240)

Cryo:
  - total_ice_pixels: Should increase when active
  - ice_ball_radius_mm: Should grow gradually

Segmentation:
  - circularity: Should be > 0.6
  - solidity: Should be > 0.7
```

---

## 📚 Additional Resources

- **Complete Implementation**: See `COMPLETE_IMPLEMENTATION.md`
- **Session Report**: See `SESSION_COMPLETION.md`
- **Code Documentation**: See docstrings in `*_kinematics.py`, `enhanced_thermometry.py`, etc.

---

**Last Updated**: March 13, 2026
**Status**: Production Ready
**Support**: Refer to code comments for detailed API information
