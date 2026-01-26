/**
 * Main Application Controller
 * Handles navigation, API communication, and module coordination
 */

const API_BASE = 'http://localhost:5002/api';

class DBSApplication {
    constructor() {
        this.currentModule = 'circuit';
        this.serverStatus = 'disconnected';
        this.init();
    }

    async init() {
        this.setupNavigation();
        await this.checkServerHealth();
        this.updateStatus();

        // Initialize modules
        if (typeof CircuitDesigner !== 'undefined') {
            window.circuitDesigner = new CircuitDesigner();
        }
        if (typeof BrainVisualizer !== 'undefined') {
            window.brainVisualizer = new BrainVisualizer();
        }
        if (typeof WaveformGenerator !== 'undefined') {
            window.waveformGenerator = new WaveformGenerator();
        }
        if (typeof AIOptimizer !== 'undefined') {
            window.aiOptimizer = new AIOptimizer();
        }
        if (typeof ClinicalDashboard !== 'undefined') {
            window.clinicalDashboard = new ClinicalDashboard();
        }
        if (typeof QuantumOptimizerUI !== 'undefined') {
            window.quantumOptimizerUI = new QuantumOptimizerUI();
        }
        if (typeof OCDDashboard !== 'undefined') {
            window.ocdDashboard = new OCDDashboard();
        }

        if (typeof SADDashboard !== 'undefined') {
            window.sadDashboard = new SADDashboard();
        }
    }

    setupNavigation() {
        const tabs = document.querySelectorAll('.nav-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const module = tab.dataset.module;
                this.switchModule(module);
            });
        });
    }

    switchModule(moduleName) {
        // Update tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-module="${moduleName}"]`).classList.add('active');

        // Update modules
        document.querySelectorAll('.module').forEach(module => {
            module.classList.remove('active');
        });
        document.getElementById(`${moduleName}Module`).classList.add('active');

        this.currentModule = moduleName;

        // Initialize module if needed
        this.initializeModule(moduleName);
    }

    initializeModule(moduleName) {
        switch (moduleName) {
            case 'circuit':
                if (window.circuitDesigner) window.circuitDesigner.loadSVG();
                break;
            case 'brain':
                if (window.brainVisualizer) window.brainVisualizer.init();
                break;
            case 'waveform':
                if (window.waveformGenerator) window.waveformGenerator.init();
                break;
            case 'ai':
                if (window.aiOptimizer) window.aiOptimizer.init();
                break;
            case 'clinical':
                if (window.clinicalDashboard) window.clinicalDashboard.init();
                break;

            case 'quantum':
                if (window.quantumOptimizerUI) window.quantumOptimizerUI.init();
                break;
            case 'ocd':
                if (window.ocdDashboard) window.ocdDashboard.init();
                break;
            case 'asd':
                if (window.asdDashboard) window.asdDashboard.init();
                break;
            case 'sad':
                if (window.sadDashboard) window.sadDashboard.init();
                break;

        }
    }

    async checkServerHealth() {
        try {
            const response = await fetch(`${API_BASE}/health`);
            const data = await response.json();
            this.serverStatus = data.status === 'healthy' ? 'connected' : 'error';
        } catch (error) {
            console.error('Server health check failed:', error);
            this.serverStatus = 'disconnected';
        }
    }

    updateStatus() {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');

        if (this.serverStatus === 'connected') {
            statusDot.style.background = '#00ff88';
            statusDot.style.boxShadow = '0 0 10px #00ff88';
            statusText.textContent = 'Connected';
        } else if (this.serverStatus === 'disconnected') {
            statusDot.style.background = '#ff3366';
            statusDot.style.boxShadow = '0 0 10px #ff3366';
            statusText.textContent = 'Server Offline';
        }
    }

    // Utility methods for API calls
    async get(endpoint) {
        try {
            const response = await fetch(`${API_BASE}${endpoint}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error(`GET ${endpoint} failed:`, error);
            throw error;
        }
    }

    async post(endpoint, data) {
        try {
            const response = await fetch(`${API_BASE}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error(`POST ${endpoint} failed:`, error);
            throw error;
        }
    }

    showLoading(element) {
        element.innerHTML = '<div class="loading"></div>';
    }

    showError(element, message) {
        element.innerHTML = `<div class="error-message">❌ ${message}</div>`;
    }

    formatJSON(obj) {
        return `<pre><code>${JSON.stringify(obj, null, 2)}</code></pre>`;
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DBSApplication();
});

// Export for use in other modules
window.API_BASE = API_BASE;
