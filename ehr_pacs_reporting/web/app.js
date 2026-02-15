// EMR Platform - Application Logic with Gemini 3.0 Integration

// Application State
const appState = {
    currentSection: 'dashboard',
    patients: [],
    reports: [],
    currentReport: null,
    geminiMetrics: {
        totalInferences: 0,
        averageConfidence: 0,
        tokensPerSecond: 0,
        reasoningDepth: 0
    }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 EMR Platform with Gemini 3.0 Initializing...');

    initializeDemo();
    updateDashboardStats();

    // Set up event handlers
    document.getElementById('create-patient-form')?.addEventListener('submit', handleCreatePatient);

    console.log('✓ Application ready');
});

// Navigation
function showSection(sectionName) {
    // Update navigation
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    event.target.closest('.nav-btn')?.classList.add('active');

    // Update sections
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(`${sectionName}-section`)?.classList.add('active');

    appState.currentSection = sectionName;
}

// Patient Management
function showCreatePatientModal() {
    const modal = document.getElementById('create-patient-modal');
    modal.classList.add('active');
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.classList.remove('active');
}

function handleCreatePatient(e) {
    e.preventDefault();

    const patient = {
        id: generateId(),
        mrn: `MRN${Date.now().toString().slice(-8)}`,
        demographics: {
            first_name: document.getElementById('patient-first-name').value,
            last_name: document.getElementById('patient-last-name').value,
            date_of_birth: document.getElementById('patient-dob').value,
            gender: document.getElementById('patient-gender').value,
            email: document.getElementById('patient-email').value,
            phone: document.getElementById('patient-phone').value
        },
        medical_history: {
            allergies: [],
            conditions: [],
            medications: []
        },
        reports: [],
        created_at: new Date().toISOString()
    };

    appState.patients.push(patient);
    updatePatientsDisplay();
    updateDashboardStats();

    closeModal('create-patient-modal');
    document.getElementById('create-patient-form').reset();

    showNotification('Patient created successfully', 'success');
}

function updatePatientsDisplay() {
    const container = document.getElementById('patients-list');

    if (appState.patients.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">👥</span>
                <p>No patients yet. Create your first patient!</p>
            </div>
        `;
        return;
    }

    container.innerHTML = appState.patients.map(patient => `
        <div class="glass-card" style="padding: 1.5rem; cursor: pointer;" onclick="viewPatient('${patient.id}')">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                <div>
                    <h3 style="margin-bottom: 0.25rem;">${patient.demographics.first_name} ${patient.demographics.last_name}</h3>
                    <p style="color: var(--text-muted); font-size: 0.9rem;">MRN: ${patient.mrn}</p>
                </div>
                <span class="badge" style="background: var(--glass-bg); color: var(--text-secondary);">
                    ${patient.demographics.gender}
                </span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.9rem;">
                <div>
                    <span style="color: var(--text-muted);">DOB:</span>
                    <span style="color: var(--text-primary);">${formatDate(patient.demographics.date_of_birth)}</span>
                </div>
                <div>
                    <span style="color: var(--text-muted);">Age:</span>
                    <span style="color: var(--text-primary);">${calculateAge(patient.demographics.date_of_birth)} yrs</span>
                </div>
                <div>
                    <span style="color: var(--text-muted);">Reports:</span>
                    <span style="color: var(--text-primary);">${patient.reports.length}</span>
                </div>
                <div>
                    <span style="color: var(--text-muted);">Email:</span>
                    <span style="color: var(--text-primary); font-size: 0.85rem;">${patient.demographics.email || 'N/A'}</span>
                </div>
            </div>
        </div>
    `).join('');
}

function searchPatients() {
    const query = document.getElementById('patient-search').value.toLowerCase();

    if (!query) {
        updatePatientsDisplay();
        return;
    }

    const filtered = appState.patients.filter(p =>
        p.demographics.first_name.toLowerCase().includes(query) ||
        p.demographics.last_name.toLowerCase().includes(query) ||
        p.mrn.toLowerCase().includes(query) ||
        p.demographics.date_of_birth.includes(query)
    );

    const container = document.getElementById('patients-list');
    container.innerHTML = filtered.map(patient => `
        <div class="glass-card" style="padding: 1.5rem; cursor: pointer;" onclick="viewPatient('${patient.id}')">
            <h3>${patient.demographics.first_name} ${patient.demographics.last_name}</h3>
            <p style="color: var(--text-muted);">MRN: ${patient.mrn}</p>
        </div>
    `).join('');
}

function viewPatient(patientId) {
    const patient = appState.patients.find(p => p.id === patientId);
    if (!patient) return;

    alert(`Patient Details:\n\nName: ${patient.demographics.first_name} ${patient.demographics.last_name}\nMRN: ${patient.mrn}\nDOB: ${patient.demographics.date_of_birth}\nReports: ${patient.reports.length}`);
}

// Template Selection & Report Creation
function selectTemplate(templateId) {
    console.log(`Selected template: ${templateId}`);

    // Show report editor
    document.getElementById('report-editor').style.display = 'block';
    document.getElementById('editor-title').textContent = getTemplateDisplayName(templateId);

    // Simulate Gemini optimization
    simulateGeminiOptimization(templateId);

    // Generate form fields based on template
    generateReportForm(templateId);

    // Scroll to editor
    document.getElementById('report-editor').scrollIntoView({ behavior: 'smooth' });
}

function getTemplateDisplayName(templateId) {
    const names = {
        'radiology_ct_brain': 'CT Brain Report',
        'radiology_mri_spine': 'MRI Spine Report',
        'cardiology_echo': 'Echocardiogram Report',
        'cardiology_ecg': 'ECG Interpretation',
        'pathology_histology': 'Histopathology Report',
        'general_hp': 'History & Physical'
    };
    return names[templateId] || 'New Report';
}

function simulateGeminiOptimization(templateId) {
    const suggestionContainer = document.getElementById('gemini-suggestions');
    if (!suggestionContainer) return;

    // Show loading
    suggestionContainer.innerHTML = `
        <div class="loading-gemini">
            <div class="gemini-spinner"></div>
            <p>Gemini 3.0 is reasoning...</p>
        </div>
    `;

    // Simulate optimization delay
    setTimeout(() => {
        const suggestions = generateGeminiSuggestions(templateId);
        displayGeminiSuggestions(suggestions);

        // Update metrics
        appState.geminiMetrics.totalInferences++;
        appState.geminiMetrics.averageConfidence = (Math.random() * 0.1 + 0.9).toFixed(3); // Higher for Gemini
        appState.geminiMetrics.tokensPerSecond = Math.floor(10000 + Math.random() * 5000);
        updateDashboardStats();
    }, 1500);
}

function generateGeminiSuggestions(templateId) {
    // Simulate Gemini-generated suggestions based on template
    const suggestions = [
        {
            field: 'brain_parenchyma',
            suggested_value: 'No acute intracranial abnormality. Grey-white differentiation preserved.',
            confidence: 0.98,
            reasoning: 'Inferred from lack of acute findings in pixel data.'
        },
        {
            field: 'hemorrhage',
            suggested_value: 'None',
            confidence: 0.99,
            reasoning: 'Hounsfield units consistent with normal tissue.'
        },
        {
            field: 'ventricles',
            suggested_value: 'Normal size and configuration.',
            confidence: 0.95,
            reasoning: 'Volume analysis matches age-related norms.'
        },
        {
            field: 'mass_effect',
            suggested_value: false,
            confidence: 0.97,
            reasoning: 'Midline shift: 0mm.'
        },
        {
            field: 'impression',
            suggested_value: 'Unremarkable CT Head. No acute intracranial pathology.',
            confidence: 0.96,
            reasoning: 'Synthesis of all findings.'
        }
    ];

    return suggestions.sort((a, b) => b.confidence - a.confidence);
}

function displayGeminiSuggestions(suggestions) {
    const container = document.getElementById('gemini-suggestions');
    if (!container) return;

    container.innerHTML = `
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                <div class="status-indicator"></div>
                <span>Multimodal Analysis Complete</span>
            </div>
        </div>
        ${suggestions.map(s => `
            <div class="suggestion-item" style="padding: 0.75rem; background: rgba(66, 133, 244, 0.1); border-radius: 8px; margin-bottom: 0.75rem; cursor: pointer;" 
                 onclick="applySuggestion('${s.field}', '${s.suggested_value}')">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                    <span style="font-weight: 600; font-size: 0.9rem;">${formatFieldName(s.field)}</span>
                    <span style="font-size: 0.85rem; color: var(--gemini-glow);">${(s.confidence * 100).toFixed(0)}%</span>
                </div>
                <div style="font-size: 0.85rem; color: var(--text-secondary);">${s.suggested_value}</div>
                <div style="font-size: 0.75rem; color: #888; font-style: italic; margin-top: 4px;">Reasoning: ${s.reasoning}</div>
                <div style="margin-top: 0.5rem;">
                    <div style="height: 4px; background: rgba(0,0,0,0.3); border-radius: 2px; overflow: hidden;">
                        <div style="height: 100%; width: ${s.confidence * 100}%; background: linear-gradient(90deg, #4285F4, #AA00FF); border-radius: 2px;"></div>
                    </div>
                </div>
            </div>
        `).join('')}
    `;
}

function generateReportForm(templateId) {
    const formContainer = document.getElementById('report-form');

    // Template-specific fields (simplified demonstration)
    const fields = {
        'radiology_ct_brain': [
            { name: 'indication', label: 'Clinical Indication', type: 'text', required: true },
            { name: 'contrast', label: 'Contrast', type: 'select', options: ['Non-contrast', 'With contrast', 'With and without contrast'], required: true },
            { name: 'brain_parenchyma', label: 'Brain Parenchyma', type: 'textarea', required: true },
            { name: 'hemorrhage', label: 'Hemorrhage', type: 'select', options: ['None', 'Acute', 'Subacute', 'Chronic'], required: true },
            { name: 'ventricles', label: 'Ventricles', type: 'select', options: ['Normal', 'Dilated', 'Compressed'], required: true },
            { name: 'impression', label: 'Impression', type: 'textarea', required: true }
        ],
        'cardiology_echo': [
            { name: 'indication', label: 'Clinical Indication', type: 'text', required: true },
            { name: 'lvef', label: 'LVEF (%)', type: 'number', required: true },
            { name: 'lv_function', label: 'LV Function', type: 'select', options: ['Normal', 'Mildly reduced', 'Moderately reduced', 'Severely reduced'], required: true },
            { name: 'mitral_valve', label: 'Mitral Valve', type: 'select', options: ['Normal', 'Regurgitation - mild', 'Regurgitation - moderate', 'Stenosis'], required: true },
            { name: 'impression', label: 'Impression', type: 'textarea', required: true }
        ]
    };

    const templateFields = fields[templateId] || fields['radiology_ct_brain'];

    formContainer.innerHTML = templateFields.map(field => {
        if (field.type === 'textarea') {
            return `
                <div class="form-group">
                    <label for="${field.name}">${field.label}${field.required ? ' *' : ''}</label>
                    <textarea id="${field.name}" ${field.required ? 'required' : ''}></textarea>
                </div>
            `;
        } else if (field.type === 'select') {
            return `
                <div class="form-group">
                    <label for="${field.name}">${field.label}${field.required ? ' *' : ''}</label>
                    <select id="${field.name}" ${field.required ? 'required' : ''}>
                        <option value="">Select...</option>
                        ${field.options.map(opt => `<option value="${opt}">${opt}</option>`).join('')}
                    </select>
                </div>
            `;
        } else {
            return `
                <div class="form-group">
                    <label for="${field.name}">${field.label}${field.required ? ' *' : ''}</label>
                    <input type="${field.type}" id="${field.name}" ${field.required ? 'required' : ''}>
                </div>
            `;
        }
    }).join('');
}

function applySuggestion(fieldName, value) {
    const field = document.getElementById(fieldName);
    if (field) {
        field.value = value;
        field.style.background = 'rgba(66, 133, 244, 0.2)';
        setTimeout(() => {
            field.style.background = '';
        }, 1000);

        showNotification(`Applied Gemini suggestion to ${formatFieldName(fieldName)}`, 'success');
    }
}

function saveReport() {
    const report = {
        id: generateId(),
        template_id: 'current_template',
        status: 'draft',
        created_at: new Date().toISOString(),
        gemini_optimized: true
    };

    appState.reports.push(report);
    updateDashboardStats();
    showNotification('Report saved as draft', 'success');
}

function finalizeReport() {
    const report = {
        id: generateId(),
        template_id: 'current_template',
        status: 'finalized',
        created_at: new Date().toISOString(),
        finalized_at: new Date().toISOString(),
        gemini_optimized: true,
        quality_score: (Math.random() * 0.1 + 0.9).toFixed(3)
    };

    appState.reports.push(report);
    updateDashboardStats();
    updateRecentReports();

    showNotification('Report finalized successfully', 'success');

    setTimeout(() => {
        closeReportEditor();
    }, 1500);
}

function closeReportEditor() {
    document.getElementById('report-editor').style.display = 'none';
}

function showCreateReportModal() {
    // Scroll to templates section
    document.querySelector('.templates-grid').scrollIntoView({ behavior: 'smooth' });
}

// Ambra Gateway & Annotation Logic
let currentTool = null;
let isDrawing = false;
let annotations = [];
let currentStudy = null;

function initializeAmbraWorklist() {
    const worklist = [
        { id: 'ST-101', patient: 'John Doe', modality: 'CT Head', date: '2025-12-03', status: 'STAT', finding: 'Possible Hemorrhage' },
        { id: 'ST-102', patient: 'Jane Smith', modality: 'MRI Spine', date: '2025-12-02', status: 'Routine', finding: 'Degenerative Changes' },
        { id: 'ST-103', patient: 'Patient A', modality: 'CXR', date: '2025-12-03', status: 'Urgent', finding: 'Pneumonia' }
    ];

    const container = document.getElementById('ambra-worklist');
    if (!container) return;

    container.innerHTML = worklist.map(study => `
        <div class="worklist-item glass-card" style="padding: 10px; margin-bottom: 10px; cursor: pointer; border-left: 4px solid ${study.status === 'STAT' ? '#ef4444' : '#4285f4'};" onclick="loadAmbraStudy('${study.id}')">
            <div style="font-weight: bold; font-size: 0.9em;">${study.patient}</div>
            <div style="font-size: 0.8em; color: var(--text-secondary);">${study.modality} • ${study.date}</div>
            ${study.status === 'STAT' ? '<div style="color: #ef4444; font-size: 0.75em; font-weight: bold;">STAT</div>' : ''}
        </div>
    `).join('');
}

function loadAmbraStudy(studyId) {
    currentStudy = studyId;
    const canvas = document.getElementById('ambra-canvas');
    const ctx = canvas.getContext('2d');

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    annotations = [];

    // Simulate loading image
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw dummy medical image (brain-like)
    ctx.fillStyle = '#333';
    ctx.beginPath();
    ctx.arc(canvas.width / 2, canvas.height / 2, 150, 0, Math.PI * 2);
    ctx.fill();

    // Add some "details"
    ctx.fillStyle = '#444';
    ctx.beginPath();
    ctx.arc(canvas.width / 2 - 40, canvas.height / 2 - 20, 30, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(canvas.width / 2 + 40, canvas.height / 2 - 20, 30, 0, Math.PI * 2);
    ctx.fill();

    // Update overlay
    document.getElementById('patient-info-overlay').innerHTML = `
        ID: ${studyId}<br>
        Name: John Doe<br>
        Modality: CT<br>
        Slice: 24/60<br>
        <span style="color: #4CAF50">Neuromorph Cloud: Connected</span>
    `;

    // Stimulate Gemini Findings
    document.getElementById('ai-findings-list').innerHTML = `
        <div class="loading-gemini">
            <div class="gemini-spinner" style="width: 20px; height: 20px;"></div>
            Analyzing...
        </div>
    `;

    setTimeout(() => {
        document.getElementById('ai-findings-list').innerHTML = `
            <ul style="list-style: none; padding: 0;">
                <li style="margin-bottom: 5px;">• No acute hemorrhage detected (Confidence: 99%)</li>
                <li style="margin-bottom: 5px;">• Ventricles normal in size</li>
                <li style="margin-bottom: 5px; color: #f59e0b;">• Note: Mild white matter changes</li>
            </ul>
        `;
    }, 1500);
}

function toggleAnnotationToolbar() {
    const toolbar = document.getElementById('annotation-toolbar');
    toolbar.style.display = toolbar.style.display === 'none' ? 'flex' : 'none';
}

function setTool(tool) {
    currentTool = tool;
    showNotification(`Tool selected: ${tool.toUpperCase()}`);
}

function clearAnnotations() {
    annotations = [];
    if (currentStudy) loadAmbraStudy(currentStudy); // Reload base image
    else {
        const canvas = document.getElementById('ambra-canvas');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

// Canvas Interaction for Annotation
const canvas = document.getElementById('ambra-canvas');
if (canvas) {
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
}

let lastX = 0;
let lastY = 0;

function startDrawing(e) {
    if (!currentTool) return;
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function draw(e) {
    if (!isDrawing || !currentTool) return;
    const ctx = canvas.getContext('2d');

    ctx.strokeStyle = '#00E676';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';

    if (currentTool === 'measure') {
        // Simple line drawing for now
        // In a real app, we'd redraw the previous state + new line
        // For simulation, just draw
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    } else if (currentTool === 'roi') {
        ctx.beginPath();
        ctx.arc(e.offsetX, e.offsetY, 20, 0, Math.PI * 2);
        ctx.stroke();
        isDrawing = false; // Single click for ROI circle
    }
}

function stopDrawing() {
    isDrawing = false;
}

function shareStudy() {
    const email = prompt("Enter email to share study with:");
    if (email) {
        showNotification(`Study shared with ${email} via Neuromorph Cloud`);
    }
}

function generateReportFromImage() {
    showSection('reports');
    selectTemplate('radiology_ct_brain');

    // Auto-fill findings based on "AI Analysis"
    setTimeout(() => {
        applySuggestion('findings', 'No acute intracranial hemorrhage. Ventricles are normal in size. Mild chronic microvascular ischemic changes.');
        applySuggestion('impression', 'No acute intracranial abnormality.');
        showNotification("Auto-populated report from Neuromorph findings", "success");
    }, 1000);
}


// Dashboard Updates
function updateDashboardStats() {
    document.getElementById('stat-patients').textContent = appState.patients.length;
    document.getElementById('stat-reports').textContent = appState.reports.length;
    document.getElementById('stat-gemini-ops').textContent = appState.geminiMetrics.totalInferences;
    document.getElementById('stat-quality').textContent = appState.geminiMetrics.averageConfidence || '0.0';
    document.getElementById('avg-confidence').textContent = appState.geminiMetrics.averageConfidence || '0.00';
    document.getElementById('tokens-sec').textContent = appState.geminiMetrics.tokensPerSecond || '0';
}

function updateRecentReports() {
    const container = document.getElementById('recent-reports-list');

    if (appState.reports.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">📄</span>
                <p>No reports yet. Create your first report!</p>
            </div>
        `;
        return;
    }

    const recentReports = appState.reports.slice(-5).reverse();

    container.innerHTML = recentReports.map(report => `
        <div style="padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                <div>
                    <h4 style="font-size: 0.95rem; margin-bottom: 0.25rem;">${getTemplateDisplayName(report.template_id)}</h4>
                    <p style="font-size: 0.85rem; color: var(--text-muted);">${formatDateTime(report.created_at)}</p>
                </div>
                <span class="badge ${report.status === 'finalized' ? 'gemini-badge' : ''}" style="${report.status !== 'finalized' ? 'background: var(--warning); color: white;' : ''}">
                    ${report.status}
                </span>
            </div>
            ${report.quality_score ? `<div style="font-size: 0.85rem; color: var(--gemini-glow);">Quality: ${report.quality_score}</div>` : ''}
        </div>
    `).join('');
}

// Utility Functions
function generateId() {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

function formatDateTime(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function calculateAge(dobString) {
    if (!dobString) return 0;
    const today = new Date();
    const birthDate = new Date(dobString);
    let age = today.getFullYear() - birthDate.getFullYear();
    const monthDiff = today.getMonth() - birthDate.getMonth();
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
        age--;
    }
    return age;
}

function formatFieldName(fieldName) {
    return fieldName
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'success' ? 'var(--success)' : 'var(--accent-primary)'};
        color: white;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        z-index: 3000;
        font-weight: 600;
        animation: slideInRight 0.3s ease;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Initialize Demo Data
function initializeDemo() {
    // Initialize Ambra Worklist
    initializeAmbraWorklist();

    // Create sample patients
    const samplePatients = [
        {
            id: generateId(),
            mrn: 'MRN20251203001',
            demographics: {
                first_name: 'John',
                last_name: 'Doe',
                date_of_birth: '1975-05-15',
                gender: 'M',
                email: 'john.doe@example.com',
                phone: '555-0101'
            },
            medical_history: {
                allergies: ['Penicillin'],
                conditions: ['Hypertension'],
                medications: ['Lisinopril']
            },
            reports: [],
            created_at: new Date().toISOString()
        },
        {
            id: generateId(),
            mrn: 'MRN20251203002',
            demographics: {
                first_name: 'Jane',
                last_name: 'Smith',
                date_of_birth: '1982-11-23',
                gender: 'F',
                email: 'jane.smith@example.com',
                phone: '555-0102'
            },
            medical_history: {
                allergies: [],
                conditions: ['Migraine'],
                medications: ['Sumatriptan']
            },
            reports: [],
            created_at: new Date().toISOString()
        }
    ];

    appState.patients = samplePatients;
    appState.geminiMetrics = {
        totalInferences: 0,
        averageConfidence: 0.000,
        tokensPerSecond: 0
    };

    updatePatientsDisplay();
    updateRecentReports();
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

console.log('✓ EMR Platform Application Loaded');
