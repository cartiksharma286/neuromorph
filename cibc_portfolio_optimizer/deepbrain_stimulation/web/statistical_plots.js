// ===== Statistical Plots Visualization =====

class StatisticalPlots {
    constructor() {
        this.charts = {};
        this.initCharts();
    }

    initCharts() {
        // Weight Distribution Chart
        const weightCtx = document.getElementById('weight-distribution-chart');
        if (weightCtx) {
            this.charts.weightDist = new Chart(weightCtx, {
                type: 'bar',
                data: {
                    labels: Array.from({ length: 20 }, (_, i) => (i * 0.05).toFixed(2)),
                    datasets: [{
                        label: 'Synaptic Weights',
                        data: Array(20).fill(0),
                        backgroundColor: 'rgba(99, 102, 241, 0.6)',
                        borderColor: '#6366f1',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            title: { display: true, text: 'Count', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' }
                        },
                        x: {
                            title: { display: true, text: 'Weight Value', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' }
                        }
                    }
                }
            });
        }

        // Spike Raster Plot
        const rasterCtx = document.getElementById('spike-raster-chart');
        if (rasterCtx) {
            this.charts.spikeRaster = new Chart(rasterCtx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Spikes',
                        data: [],
                        backgroundColor: '#ec4899',
                        pointRadius: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            title: { display: true, text: 'Neuron ID', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' }
                        },
                        x: {
                            title: { display: true, text: 'Time (ms)', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' }
                        }
                    }
                }
            });
        }

        // ISI Histogram
        const isiCtx = document.getElementById('isi-histogram-chart');
        if (isiCtx) {
            this.charts.isiHistogram = new Chart(isiCtx, {
                type: 'bar',
                data: {
                    labels: Array.from({ length: 30 }, (_, i) => i * 2),
                    datasets: [{
                        label: 'Inter-Spike Intervals',
                        data: Array(30).fill(0),
                        backgroundColor: 'rgba(139, 92, 246, 0.6)',
                        borderColor: '#8b5cf6',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            title: { display: true, text: 'Frequency', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' }
                        },
                        x: {
                            title: { display: true, text: 'Interval (ms)', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' }
                        }
                    }
                }
            });
        }

        // Learning Curve
        const learningCtx = document.getElementById('learning-curve-chart');
        if (learningCtx) {
            this.charts.learningCurve = new Chart(learningCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Network Performance',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            title: { display: true, text: 'Performance', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' },
                            min: 0,
                            max: 1
                        },
                        x: {
                            title: { display: true, text: 'Epoch', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' }
                        }
                    }
                }
            });
        }
    }

    update(statsData) {
        if (!statsData) return;

        // Update weight distribution
        if (this.charts.weightDist && statsData.weightDistribution) {
            const histogram = this.createHistogram(statsData.weightDistribution, 20);
            this.charts.weightDist.data.datasets[0].data = histogram;
            this.charts.weightDist.update('none');
        }

        // Update spike raster
        if (this.charts.spikeRaster && statsData.spikeTrains) {
            const spikes = [];
            statsData.spikeTrains.forEach((train, neuronId) => {
                train.forEach((spike, time) => {
                    if (spike === 1) {
                        spikes.push({ x: time, y: neuronId });
                    }
                });
            });
            this.charts.spikeRaster.data.datasets[0].data = spikes.slice(-500);
            this.charts.spikeRaster.update('none');
        }

        // Update ISI histogram
        if (this.charts.isiHistogram && statsData.isiHistogram) {
            this.charts.isiHistogram.data.datasets[0].data = statsData.isiHistogram;
            this.charts.isiHistogram.update('none');
        }

        // Update learning curve
        if (this.charts.learningCurve && statsData.learningCurve) {
            const epoch = this.charts.learningCurve.data.labels.length;
            this.charts.learningCurve.data.labels.push(epoch);
            this.charts.learningCurve.data.datasets[0].data.push(
                statsData.learningCurve[statsData.learningCurve.length - 1]
            );

            if (this.charts.learningCurve.data.labels.length > 100) {
                this.charts.learningCurve.data.labels.shift();
                this.charts.learningCurve.data.datasets[0].data.shift();
            }

            this.charts.learningCurve.update('none');
        }
    }

    createHistogram(data, bins) {
        const histogram = Array(bins).fill(0);
        data.forEach(value => {
            const bin = Math.min(Math.floor(value * bins), bins - 1);
            histogram[bin]++;
        });
        return histogram;
    }

    reset() {
        Object.values(this.charts).forEach(chart => {
            if (chart.data.labels) chart.data.labels = [];
            if (chart.data.datasets) {
                chart.data.datasets.forEach(dataset => {
                    dataset.data = [];
                });
            }
            chart.update();
        });
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.statsViz = new StatisticalPlots();
    console.log('✓ Statistical Plots initialized');
});
