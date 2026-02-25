// Circuit Schematic Generator for RF Coils
class SchematicGenerator {
    constructor(svgElement) {
        this.svg = svgElement;
        this.width = 1000;
        this.height = 800;
        this.components = [];
    }

    // Clear SVG
    clear() {
        while (this.svg.firstChild) {
            this.svg.removeChild(this.svg.firstChild);
        }
        this.components = [];
    }

    // Draw capacitor symbol
    drawCapacitor(x, y, value, vertical = false) {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        if (vertical) {
            // Vertical capacitor
            const plate1 = this.createLine(x - 15, y, x + 15, y, 'black', 2);
            const plate2 = this.createLine(x - 15, y + 20, x + 15, y + 20, 'black', 2);
            const wire1 = this.createLine(x, y - 10, x, y, 'black', 2);
            const wire2 = this.createLine(x, y + 20, x, y + 30, 'black', 2);
            g.appendChild(plate1);
            g.appendChild(plate2);
            g.appendChild(wire1);
            g.appendChild(wire2);
        } else {
            // Horizontal capacitor
            const plate1 = this.createLine(x, y - 15, x, y + 15, 'black', 2);
            const plate2 = this.createLine(x + 20, y - 15, x + 20, y + 15, 'black', 2);
            const wire1 = this.createLine(x - 10, y, x, y, 'black', 2);
            const wire2 = this.createLine(x + 20, y, x + 30, y, 'black', 2);
            g.appendChild(plate1);
            g.appendChild(plate2);
            g.appendChild(wire1);
            g.appendChild(wire2);
        }

        // Label
        const text = this.createText(x + (vertical ? 25 : 10), y + (vertical ? 10 : -20), value, '12px');
        g.appendChild(text);

        this.svg.appendChild(g);
        return g;
    }

    // Draw inductor (coil) symbol
    drawInductor(x, y, value, vertical = false) {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        if (vertical) {
            // Vertical inductor (coil arcs)
            for (let i = 0; i < 4; i++) {
                const arc = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                const yPos = y + i * 10;
                arc.setAttribute('d', `M ${x} ${yPos} Q ${x + 8} ${yPos + 5}, ${x} ${yPos + 10}`);
                arc.setAttribute('stroke', 'black');
                arc.setAttribute('stroke-width', '2');
                arc.setAttribute('fill', 'none');
                g.appendChild(arc);
            }
            const wire1 = this.createLine(x, y - 10, x, y, 'black', 2);
            const wire2 = this.createLine(x, y + 40, x, y + 50, 'black', 2);
            g.appendChild(wire1);
            g.appendChild(wire2);
        } else {
            // Horizontal inductor
            for (let i = 0; i < 4; i++) {
                const arc = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                const xPos = x + i * 10;
                arc.setAttribute('d', `M ${xPos} ${y} Q ${xPos + 5} ${y - 8}, ${xPos + 10} ${y}`);
                arc.setAttribute('stroke', 'black');
                arc.setAttribute('stroke-width', '2');
                arc.setAttribute('fill', 'none');
                g.appendChild(arc);
            }
            const wire1 = this.createLine(x - 10, y, x, y, 'black', 2);
            const wire2 = this.createLine(x + 40, y, x + 50, y, 'black', 2);
            g.appendChild(wire1);
            g.appendChild(wire2);
        }

        // Label
        const text = this.createText(x + (vertical ? 20 : 20), y + (vertical ? 20 : -15), value, '12px');
        g.appendChild(text);

        this.svg.appendChild(g);
        return g;
    }

    // Draw resistor symbol
    drawResistor(x, y, value, vertical = false) {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        if (vertical) {
            // Vertical resistor (zigzag)
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', `M ${x} ${y} l 5 5 l -10 10 l 10 10 l -10 10 l 10 10 l -5 5`);
            path.setAttribute('stroke', 'black');
            path.setAttribute('stroke-width', '2');
            path.setAttribute('fill', 'none');
            g.appendChild(path);
            const wire1 = this.createLine(x, y - 10, x, y, 'black', 2);
            const wire2 = this.createLine(x, y + 50, x, y + 60, 'black', 2);
            g.appendChild(wire1);
            g.appendChild(wire2);
        } else {
            // Horizontal resistor
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', `M ${x} ${y} l 5 -5 l 10 10 l 10 -10 l 10 10 l 10 -10 l 5 5`);
            path.setAttribute('stroke', 'black');
            path.setAttribute('stroke-width', '2');
            path.setAttribute('fill', 'none');
            g.appendChild(path);
            const wire1 = this.createLine(x - 10, y, x, y, 'black', 2);
            const wire2 = this.createLine(x + 50, y, x + 60, y, 'black', 2);
            g.appendChild(wire1);
            g.appendChild(wire2);
        }

        // Label
        const text = this.createText(x + (vertical ? 20 : 25), y + (vertical ? 25 : -15), value, '12px');
        g.appendChild(text);

        this.svg.appendChild(g);
        return g;
    }

    // Draw circular loop (coil)
    drawLoop(x, y, radius, label) {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x);
        circle.setAttribute('cy', y);
        circle.setAttribute('r', radius);
        circle.setAttribute('stroke', '#6366f1');
        circle.setAttribute('stroke-width', '3');
        circle.setAttribute('fill', 'none');
        g.appendChild(circle);

        // Label
        const text = this.createText(x, y, label, '14px', 'bold');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', '#6366f1');
        g.appendChild(text);

        this.svg.appendChild(g);
        return g;
    }

    // Draw ground symbol
    drawGround(x, y) {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        const line1 = this.createLine(x, y, x, y + 10, 'black', 2);
        const line2 = this.createLine(x - 15, y + 10, x + 15, y + 10, 'black', 2);
        const line3 = this.createLine(x - 10, y + 15, x + 10, y + 15, 'black', 2);
        const line4 = this.createLine(x - 5, y + 20, x + 5, y + 20, 'black', 2);

        g.appendChild(line1);
        g.appendChild(line2);
        g.appendChild(line3);
        g.appendChild(line4);

        this.svg.appendChild(g);
        return g;
    }

    // Helper: create line
    createLine(x1, y1, x2, y2, color, width) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', color);
        line.setAttribute('stroke-width', width);
        return line;
    }

    // Helper: create text
    createText(x, y, content, fontSize = '12px', fontWeight = 'normal') {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x);
        text.setAttribute('y', y);
        text.setAttribute('font-family', 'Inter, sans-serif');
        text.setAttribute('font-size', fontSize);
        text.setAttribute('font-weight', fontWeight);
        text.textContent = content;
        return text;
    }

    // Generate complete coil circuit
    generateCoilCircuit(config) {
        this.clear();

        const {
            loopDiameter = 80,
            loopInductance = 180,
            tuningCap = 8.2,
            matchingCap = 22,
            includeMatching = true,
            includeDecoupling = true
        } = config;

        // Title
        const title = this.createText(500, 40, 'RF Coil Element Circuit', '20px', 'bold');
        title.setAttribute('text-anchor', 'middle');
        this.svg.appendChild(title);

        // Draw main loop
        this.drawLoop(200, 200, 80, 'L');

        // Tuning capacitor (series)
        this.drawLine(280, 200, 350, 200, 'black', 2);
        this.drawCapacitor(350, 200, `${tuningCap} pF`);

        // Connection to matching network
        this.drawLine(380, 200, 450, 200, 'black', 2);

        if (includeMatching) {
            // Matching capacitor (parallel to ground)
            this.drawLine(450, 200, 450, 250, 'black', 2);
            this.drawCapacitor(450, 270, `${matchingCap} pF`, true);
            this.drawLine(450, 300, 450, 320, 'black', 2);
            this.drawGround(450, 320);

            // Output to 50Ω
            this.drawLine(450, 200, 600, 200, 'black', 2);
            const output = this.createText(620, 205, '50Ω Output', '14px', 'bold');
            output.setAttribute('fill', '#6366f1');
            this.svg.appendChild(output);
        }

        // Return path
        this.drawLine(200, 280, 200, 350, 'black', 2);
        this.drawLine(200, 350, 120, 350, 'black', 2);
        this.drawLine(120, 350, 120, 200, 'black', 2);
        this.drawLine(120, 200, 120, 200, 'black', 2);

        // Component annotations
        const annotations = [
            { x: 200, y: 120, text: `Loop Inductance: ${loopInductance} nH` },
            { x: 200, y: 140, text: `Diameter: ${loopDiameter} mm` },
            { x: 200, y: 450, text: 'Tuning: Series capacitor for resonance' },
            { x: 200, y: 470, text: 'Matching: Parallel capacitor for 50Ω impedance' }
        ];

        annotations.forEach(ann => {
            const text = this.createText(ann.x, ann.y, ann.text, '12px');
            text.setAttribute('fill', '#64748b');
            this.svg.appendChild(text);
        });

        if (includeDecoupling) {
            // Add decoupling network diagram
            const decoupleY = 550;
            const decoupleTitle = this.createText(500, decoupleY, 'Decoupling Strategy', '16px', 'bold');
            decoupleTitle.setAttribute('text-anchor', 'middle');
            this.svg.appendChild(decoupleTitle);

            // Draw two overlapping loops
            this.drawLoop(350, decoupleY + 80, 50, 'L1');
            this.drawLoop(450, decoupleY + 80, 50, 'L2');

            const overlap = this.createText(500, decoupleY + 130, 'Geometric Overlap', '12px');
            overlap.setAttribute('text-anchor', 'middle');
            overlap.setAttribute('fill', '#64748b');
            this.svg.appendChild(overlap);
        }
    }

    // Helper: draw line wrapper
    drawLine(x1, y1, x2, y2, color, width) {
        const line = this.createLine(x1, y1, x2, y2, color, width);
        this.svg.appendChild(line);
        return line;
    }
}

// Initialize schematic generator
let schematicGenerator = null;

function initSchematicGenerator() {
    const svg = document.getElementById('schematicSvg');
    if (!svg) return;

    schematicGenerator = new SchematicGenerator(svg);
    schematicGenerator.generateCoilCircuit({
        loopDiameter: 80,
        loopInductance: 180,
        tuningCap: 8.2,
        matchingCap: 22,
        includeMatching: true,
        includeDecoupling: true
    });
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SchematicGenerator };
}
