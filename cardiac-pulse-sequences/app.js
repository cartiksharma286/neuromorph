// Main Application Controller

const App = {
    /**
     * Initialize the application
     */
    init() {
        console.log('Cardiac Parallel Imaging Generator starting...');

        // Initialize all modules
        this.initializeModules();

        // Setup tab navigation
        this.setupTabs();

        // Initial render
        this.initialRender();

        console.log('Application initialized successfully!');
    },

    /**
     * Initialize all application modules
     */
    initializeModules() {
        // Initialize in dependency order
        if (window.Visualizer) {
            Visualizer.init();
        }

        if (window.ParallelImaging) {
            ParallelImaging.init();
        }

        if (window.CardiacSequences) {
            CardiacSequences.init();
        }

        if (window.LLMInterface) {
            LLMInterface.init();
        }

        if (window.ExportModule) {
            ExportModule.init();
        }
    },

    /**
     * Setup tab navigation
     */
    setupTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabId = button.getAttribute('data-tab');

                // Remove active class from all tabs
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                // Add active class to clicked tab
                button.classList.add('active');
                const targetContent = document.getElementById(tabId);
                if (targetContent) {
                    targetContent.classList.add('active');
                }

                // Trigger visualization update for the active tab
                this.updateVisualizationForTab(tabId);
            });
        });
    },

    /**
     * Update visualization when tab changes
     */
    updateVisualizationForTab(tabId) {
        if (!window.Visualizer) return;

        switch (tabId) {
            case 'parallel-imaging':
                if (window.ParallelImaging) {
                    ParallelImaging.updateCalculations();
                }
                break;
            case 'cine':
                if (window.CardiacSequences) {
                    CardiacSequences.updateCINECalculations();
                }
                break;
            // Other tabs would trigger their specific visualizations
        }
    },

    /**
     * Initial render of all visualizations
     */
    initialRender() {
        // Render parallel imaging k-space
        if (window.ParallelImaging && window.Visualizer) {
            ParallelImaging.updateCalculations();
        }

        // Render CINE timing
        if (window.CardiacSequences && window.Visualizer) {
            CardiacSequences.updateCINECalculations();
        }
    }
};

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => App.init());
} else {
    App.init();
}

// Export for global access
window.App = App;
