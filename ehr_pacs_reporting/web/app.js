// EMR Platform - Application Logic with Quantum Integration

// Application State
const appState = {
    currentSection: 'dashboard',
    patients: [],
    reports: [],
    currentReport: null,
    quantumMetrics: {
        totalOptimizations: 0,
        averageConfidence: 0,
        quantumAdvantage: 0
    }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 EMR Platform with NVQLink Initializing...');

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

    // Simulate quantum optimization
    simulateQuantumOptimization(templateId);

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

function simulateQuantumOptimization(templateId) {
    const suggestionContainer = document.getElementById('quantum-suggestions');

    // Show loading
    suggestionContainer.innerHTML = `
        <div class="loading-quantum">
            <div class="quantum-spinner"></div>
            <p>Running quantum optimization...</p>
        </div>
    `;

    // Simulate optimization delay
    setTimeout(() => {
        const suggestions = generateQuantumSuggestions(templateId);
        displayQuantumSuggestions(suggestions);

        // Update metrics
        appState.quantumMetrics.totalOptimizations++;
        appState.quantumMetrics.averageConfidence = (Math.random() * 0.3 + 0.7).toFixed(3);
        appState.quantumMetrics.quantumAdvantage = (Math.random() * 0.4 + 0.6).toFixed(3);
        updateDashboardStats();
    }, 2000);
}

function generateQuantumSuggestions(templateId) {
    // Simulate quantum-generated suggestions based on template
    const suggestions = [
        {
            field: 'brain_parenchyma',
            suggested_value: 'No acute intracranial abnormality',
            confidence: 0.92
        },
        {
            field: 'hemorrhage',
            suggested_value: 'None',
            confidence: 0.88
        },
        {
            field: 'ventricles',
            suggested_value: 'Normal',
            confidence: 0.85
        },
        {
            field: 'mass_effect',
            suggested_value: false,
            confidence: 0.91
        },
        {
            field: 'impression',
            suggested_value: 'No acute findings',
            confidence: 0.87
        }
    ];

    return suggestions.sort((a, b) => b.confidence - a.confidence);
}

function displayQuantumSuggestions(suggestions) {
    const container = document.getElementById('quantum-suggestions');

    container.innerHTML = `
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                <div class="status-indicator pulse"></div>
                <span>Quantum optimization complete</span>
            </div>
        </div>
        ${suggestions.map(s => `
            <div class="suggestion-item" style="padding: 0.75rem; background: rgba(99, 102, 241, 0.1); border-radius: 8px; margin-bottom: 0.75rem; cursor: pointer;" 
                 onclick="applySuggestion('${s.field}', '${s.suggested_value}')">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                    <span style="font-weight: 600; font-size: 0.9rem;">${formatFieldName(s.field)}</span>
                    <span style="font-size: 0.85rem; color: var(--quantum-glow);">${(s.confidence * 100).toFixed(0)}%</span>
                </div>
                <div style="font-size: 0.85rem; color: var(--text-secondary);">${s.suggested_value}</div>
                <div style="margin-top: 0.5rem;">
                    <div style="height: 4px; background: rgba(0,0,0,0.3); border-radius: 2px; overflow: hidden;">
                        <div style="height: 100%; width: ${s.confidence * 100}%; background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)); border-radius: 2px;"></div>
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
        field.style.background = 'rgba(99, 102, 241, 0.2)';
        setTimeout(() => {
            field.style.background = '';
        }, 1000);

        showNotification(`Applied quantum suggestion to ${formatFieldName(fieldName)}`, 'success');
    }
}

function saveReport() {
    const report = {
        id: generateId(),
        template_id: 'current_template',
        status: 'draft',
        created_at: new Date().toISOString(),
        quantum_optimized: true
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
        quantum_optimized: true,
        quality_score: (Math.random() * 0.3 + 0.7).toFixed(3)
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

// Dashboard Updates
function updateDashboardStats() {
    document.getElementById('stat-patients').textContent = appState.patients.length;
    document.getElementById('stat-reports').textContent = appState.reports.length;
    document.getElementById('stat-quantum-ops').textContent = appState.quantumMetrics.totalOptimizations;
    document.getElementById('stat-quality').textContent = appState.quantumMetrics.averageConfidence || '0.0';
    document.getElementById('avg-confidence').textContent = appState.quantumMetrics.averageConfidence || '0.00';
    document.getElementById('quantum-advantage').textContent = appState.quantumMetrics.quantumAdvantage || '0.00';
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
                <span class="badge ${report.status === 'finalized' ? 'quantum-badge' : ''}" style="${report.status !== 'finalized' ? 'background: var(--warning); color: white;' : ''}">
                    ${report.status}
                </span>
            </div>
            ${report.quality_score ? `<div style="font-size: 0.85rem; color: var(--quantum-glow);">Quality: ${report.quality_score}</div>` : ''}
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
    appState.quantumMetrics = {
        totalOptimizations: 0,
        averageConfidence: 0.000,
        quantumAdvantage: 0.000
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
