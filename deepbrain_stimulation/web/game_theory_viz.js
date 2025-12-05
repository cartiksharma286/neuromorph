// ===== Game Theory Visualization =====

class GameTheoryVisualization {
    constructor() {
        this.gameBoardContainer = document.getElementById('game-board-viz');
        this.payoffMatrixContainer = document.getElementById('payoff-matrix-viz');
        this.nashContainer = document.getElementById('nash-equilibrium-viz');
        this.nimContainer = document.getElementById('nim-heaps-viz');

        this.init();
    }

    init() {
        // Initialize visualizations
        this.drawGameBoard();
        this.drawPayoffMatrix();
        this.drawNashEquilibrium();
        this.drawNimHeaps();
    }

    drawGameBoard() {
        const svg = d3.select(this.gameBoardContainer)
            .append('svg')
            .attr('width', '100%')
            .attr('height', 400);

        // Create game board grid
        const boardSize = 8;
        const cellSize = 40;

        const board = svg.append('g')
            .attr('transform', 'translate(50, 50)');

        // Draw grid
        for (let i = 0; i < boardSize; i++) {
            for (let j = 0; j < boardSize; j++) {
                board.append('rect')
                    .attr('x', i * cellSize)
                    .attr('y', j * cellSize)
                    .attr('width', cellSize)
                    .attr('height', cellSize)
                    .attr('fill', (i + j) % 2 === 0 ? '#1a1a24' : '#121218')
                    .attr('stroke', '#6c757d')
                    .attr('stroke-width', 1);
            }
        }

        // Add player pieces
        board.append('circle')
            .attr('cx', cellSize * 1.5)
            .attr('cy', cellSize * 1.5)
            .attr('r', 15)
            .attr('fill', '#ef4444')
            .attr('class', 'damage-player');

        board.append('circle')
            .attr('cx', cellSize * 6.5)
            .attr('cy', cellSize * 6.5)
            .attr('r', 15)
            .attr('fill', '#10b981')
            .attr('class', 'plasticity-player');

        // Add labels
        svg.append('text')
            .attr('x', 50)
            .attr('y', 30)
            .attr('fill', '#ef4444')
            .attr('font-size', '14px')
            .attr('font-weight', 'bold')
            .text('Damage (Red)');

        svg.append('text')
            .attr('x', 200)
            .attr('y', 30)
            .attr('fill', '#10b981')
            .attr('font-size', '14px')
            .attr('font-weight', 'bold')
            .text('Plasticity (Green)');

        this.gameBoard = { svg, board, cellSize };
    }

    drawPayoffMatrix() {
        const data = [
            { strategy1: 'High Damage', strategy2: 'High Plasticity', payoff1: -2, payoff2: 3 },
            { strategy1: 'High Damage', strategy2: 'Low Plasticity', payoff1: 5, payoff2: -3 },
            { strategy1: 'Low Damage', strategy2: 'High Plasticity', payoff1: -1, payoff2: 2 },
            { strategy1: 'Low Damage', strategy2: 'Low Plasticity', payoff1: 1, payoff2: 1 }
        ];

        const table = d3.select(this.payoffMatrixContainer)
            .append('table')
            .style('width', '100%')
            .style('border-collapse', 'collapse');

        // Header
        const header = table.append('thead').append('tr');
        header.append('th').text('Damage \\ Plasticity').style('padding', '8px').style('border', '1px solid #6c757d');
        header.append('th').text('High Plasticity').style('padding', '8px').style('border', '1px solid #6c757d');
        header.append('th').text('Low Plasticity').style('padding', '8px').style('border', '1px solid #6c757d');

        // Rows
        const rows = table.append('tbody').selectAll('tr')
            .data([
                { label: 'High Damage', values: ['-2, 3', '5, -3'] },
                { label: 'Low Damage', values: ['-1, 2', '1, 1'] }
            ])
            .enter()
            .append('tr');

        rows.append('td')
            .text(d => d.label)
            .style('padding', '8px')
            .style('border', '1px solid #6c757d')
            .style('font-weight', 'bold');

        rows.selectAll('td.value')
            .data(d => d.values)
            .enter()
            .append('td')
            .attr('class', 'value')
            .text(d => d)
            .style('padding', '8px')
            .style('border', '1px solid #6c757d')
            .style('text-align', 'center')
            .style('font-family', 'JetBrains Mono, monospace');
    }

    drawNashEquilibrium() {
        const width = this.nashContainer.clientWidth || 300;
        const height = 250;

        const svg = d3.select(this.nashContainer)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        // Create strategy space
        const margin = { top: 20, right: 20, bottom: 40, left: 50 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left}, ${margin.top})`);

        // Axes
        const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerWidth]);
        const yScale = d3.scaleLinear().domain([0, 1]).range([innerHeight, 0]);

        g.append('g')
            .attr('transform', `translate(0, ${innerHeight})`)
            .call(d3.axisBottom(xScale))
            .attr('color', '#adb5bd');

        g.append('g')
            .call(d3.axisLeft(yScale))
            .attr('color', '#adb5bd');

        // Labels
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', height - 5)
            .attr('text-anchor', 'middle')
            .attr('fill', '#adb5bd')
            .attr('font-size', '12px')
            .text('Player 1 Strategy');

        svg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -height / 2)
            .attr('y', 15)
            .attr('text-anchor', 'middle')
            .attr('fill', '#adb5bd')
            .attr('font-size', '12px')
            .text('Player 2 Strategy');

        // Nash equilibrium point
        g.append('circle')
            .attr('cx', xScale(0.6))
            .attr('cy', yScale(0.4))
            .attr('r', 6)
            .attr('fill', '#ec4899')
            .attr('stroke', '#f43f5e')
            .attr('stroke-width', 2);

        g.append('text')
            .attr('x', xScale(0.6) + 10)
            .attr('y', yScale(0.4))
            .attr('fill', '#ec4899')
            .attr('font-size', '12px')
            .text('Nash Eq.');

        this.nashEquilibrium = { svg, g, xScale, yScale };
    }

    drawNimHeaps() {
        const svg = d3.select(this.nimContainer)
            .append('svg')
            .attr('width', '100%')
            .attr('height', 200);

        const heaps = [3, 5, 7];
        const heapWidth = 60;
        const heapSpacing = 80;
        const startX = 30;

        heaps.forEach((count, i) => {
            const x = startX + i * heapSpacing;

            // Draw heap label
            svg.append('text')
                .attr('x', x + heapWidth / 2)
                .attr('y', 20)
                .attr('text-anchor', 'middle')
                .attr('fill', '#adb5bd')
                .attr('font-size', '12px')
                .text(`Heap ${i + 1}`);

            // Draw objects in heap
            for (let j = 0; j < count; j++) {
                svg.append('rect')
                    .attr('x', x)
                    .attr('y', 40 + j * 20)
                    .attr('width', heapWidth)
                    .attr('height', 15)
                    .attr('fill', '#6366f1')
                    .attr('stroke', '#8b5cf6')
                    .attr('stroke-width', 2)
                    .attr('rx', 3);
            }

            // Display count
            svg.append('text')
                .attr('x', x + heapWidth / 2)
                .attr('y', 180)
                .attr('text-anchor', 'middle')
                .attr('fill', '#ec4899')
                .attr('font-size', '16px')
                .attr('font-weight', 'bold')
                .text(count);
        });

        // Display Grundy value
        const grundy = heaps.reduce((a, b) => a ^ b, 0);
        svg.append('text')
            .attr('x', startX + heaps.length * heapSpacing)
            .attr('y', 100)
            .attr('fill', '#10b981')
            .attr('font-size', '14px')
            .attr('font-weight', 'bold')
            .text(`Grundy: ${grundy}`);

        this.nimHeaps = { svg, heaps };
    }

    update(gameData) {
        if (!gameData) return;

        // Update game board positions
        if (gameData.positions && this.gameBoard) {
            // Animate player movements
            const { damage, plasticity } = gameData.positions;

            d3.select('.damage-player')
                .transition()
                .duration(500)
                .attr('cx', damage.x * this.gameBoard.cellSize + this.gameBoard.cellSize / 2)
                .attr('cy', damage.y * this.gameBoard.cellSize + this.gameBoard.cellSize / 2);

            d3.select('.plasticity-player')
                .transition()
                .duration(500)
                .attr('cx', plasticity.x * this.gameBoard.cellSize + this.gameBoard.cellSize / 2)
                .attr('cy', plasticity.y * this.gameBoard.cellSize + this.gameBoard.cellSize / 2);
        }

        // Update Nash equilibrium
        if (gameData.nashEquilibrium && this.nashEquilibrium) {
            const { player1, player2 } = gameData.nashEquilibrium;

            this.nashEquilibrium.g.select('circle')
                .transition()
                .duration(500)
                .attr('cx', this.nashEquilibrium.xScale(player1))
                .attr('cy', this.nashEquilibrium.yScale(player2));
        }

        // Update Nim heaps
        if (gameData.nimHeaps && this.nimHeaps) {
            // Redraw heaps with new values
            this.nimContainer.innerHTML = '';
            this.drawNimHeaps();
        }
    }

    reset() {
        // Reset to initial state
        this.gameBoardContainer.innerHTML = '';
        this.payoffMatrixContainer.innerHTML = '';
        this.nashContainer.innerHTML = '';
        this.nimContainer.innerHTML = '';
        this.init();
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.gameTheoryViz = new GameTheoryVisualization();
    console.log('✓ Game Theory Visualization initialized');
});
