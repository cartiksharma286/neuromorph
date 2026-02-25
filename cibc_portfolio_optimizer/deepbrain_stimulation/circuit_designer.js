/**
 * Circuit Designer Module
 * Handles circuit schematic generation and display
 */

class CircuitDesigner {
    constructor() {
        this.currentComponent = null;
        this.init();
    }

    init() {
        this.setupComponentButtons();
        this.loadSVG();
    }

    setupComponentButtons() {
        const buttons = document.querySelectorAll('.component-btn');
        buttons.forEach(button => {
            button.addEventListener('click', () => {
                const component = button.dataset.component;
                this.loadComponent(component);
            });
        });
    }

    async loadComponent(componentName) {
        this.currentComponent = componentName;
        const display = document.getElementById('circuitDisplay');

        // Show loading
        display.innerHTML = '<div class="loading"></div>';

        try {
            const endpoint = `/circuit/${componentName}`;
            const data = await window.app.get(endpoint);

            // Display the schematic data
            this.displaySchematic(data);
        } catch (error) {
            display.innerHTML = `<div class="error-message">Failed to load component: ${error.message}</div>`;
        }
    }

    displaySchematic(data) {
        const display = document.getElementById('circuitDisplay');

        let html = `<h3>${data.component || 'Circuit Component'}</h3>`;

        if (data.type) {
            html += `<p><strong>Type:</strong> ${data.type}</p>`;
        }

        if (data.architecture) {
            html += `<p><strong>Architecture:</strong> ${data.architecture}</p>`;
        }

        // Display specifications
        if (data.specifications) {
            html += '<h4>Specifications</h4><ul>';
            for (const [key, value] of Object.entries(data.specifications)) {
                html += `<li><strong>${this.formatKey(key)}:</strong> ${value}</li>`;
            }
            html += '</ul>';
        }

        // Display main components
        if (data.main_components) {
            html += '<h4>Main Components</h4>';
            html += this.renderComponents(data.main_components);
        }

        // Display connections
        if (data.connections) {
            html += '<h4>Connections</h4><ul>';
            data.connections.forEach(conn => {
                html += `<li>Contact ${conn.contact}: ${conn.wire} (${conn.color})</li>`;
            });
            html += '</ul>';
        }

        // Display design notes
        if (data.design_notes) {
            html += '<h4>Design Notes</h4><ul>';
            data.design_notes.forEach(note => {
                html += `<li>${note}</li>`;
            });
            html += '</ul>';
        }

        // Display target regions
        if (data.target_brain_regions) {
            html += '<h4>Target Brain Regions</h4>';
            html += `<p><strong>Primary:</strong> ${data.target_brain_regions.primary}</p>`;
            html += `<p><strong>Rationale:</strong> ${data.target_brain_regions.rationale}</p>`;
        }

        // Display system specifications
        if (data.system_specifications) {
            html += '<h4>System Specifications</h4><ul>';
            for (const [key, value] of Object.entries(data.system_specifications)) {
                html += `<li><strong>${this.formatKey(key)}:</strong> ${JSON.stringify(value)}</li>`;
            }
            html += '</ul>';
        }

        display.innerHTML = html;
    }

    renderComponents(components) {
        let html = '<div class="components-tree">';

        for (const [name, details] of Object.entries(components)) {
            html += `<div class="component-detail">`;
            html += `<h5>${this.formatKey(name)}</h5>`;

            if (typeof details === 'object') {
                html += '<ul>';
                for (const [key, value] of Object.entries(details)) {
                    if (typeof value === 'object') {
                        html += `<li><strong>${this.formatKey(key)}:</strong><ul>`;
                        for (const [k, v] of Object.entries(value)) {
                            html += `<li>${this.formatKey(k)}: ${v}</li>`;
                        }
                        html += '</ul></li>';
                    } else {
                        html += `<li><strong>${this.formatKey(key)}:</strong> ${value}</li>`;
                    }
                }
                html += '</ul>';
            }

            html += '</div>';
        }

        html += '</div>';
        return html;
    }

    async loadSVG() {
        const svgDisplay = document.getElementById('svgDisplay');

        try {
            const response = await fetch(`${window.API_BASE}/circuit/svg`);
            const svgText = await response.text();
            svgDisplay.innerHTML = svgText;
        } catch (error) {
            console.error('Failed to load SVG:', error);
            svgDisplay.innerHTML = '<div class="empty-state"><p>SVG diagram not available</p></div>';
        }
    }

    formatKey(key) {
        return key
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }
}

// Make available globally
window.CircuitDesigner = CircuitDesigner;
