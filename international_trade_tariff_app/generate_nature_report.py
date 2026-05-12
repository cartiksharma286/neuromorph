from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_report():
    output_path = "international_trade_tariff_app/Nature_Report_International_Trade.pdf"
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Nature-style styles
    title_style = ParagraphStyle(
        name='TitleStyle', 
        parent=styles['Heading1'], 
        fontName='Helvetica-Bold', 
        fontSize=16, 
        spaceAfter=20,
        alignment=1
    )
    abstract_style = ParagraphStyle(
        name='Abstract', 
        parent=styles['Normal'], 
        fontName='Helvetica-Oblique', 
        fontSize=10, 
        leftIndent=30, 
        rightIndent=30, 
        spaceAfter=20
    )
    heading_style = ParagraphStyle(
        name='Heading', 
        parent=styles['Heading2'], 
        fontName='Helvetica-Bold', 
        fontSize=12, 
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        name='Body', 
        parent=styles['Normal'], 
        fontName='Helvetica', 
        fontSize=10, 
        spaceAfter=12,
        leading=14
    )
    
    story = []
    
    # Title
    story.append(Paragraph("A Comprehensive Multidisciplinary Mathematical Framework for Global Trade and NACETA Tariffs", title_style))
    
    # Abstract
    abstract_text = ("<b>Abstract:</b> This report presents a unified theoretical approach to modeling international trade tariffs and subsidies, particularly within the NACETA framework. By integrating cooperative game theory, finite mathematics for portfolio optimization, continued fractions for verifiability, kinetic econophysics for socio-economic stability, and quantum-inspired Feynman path integrals for predictive route modeling, we establish an exact, mathematically optimal blueprint for global resource exchange.")
    story.append(Paragraph(abstract_text, abstract_style))
    
    # 1. Cooperative Game Theory
    story.append(Paragraph("1. Cooperative Game Theory: Shapley Value Derivations", heading_style))
    story.append(Paragraph("To explicitly derive the tariff subsidy constraints under cooperative mechanics, we assume a finite player set <i>N</i> representing trade blocs. Let <i>v(S)</i> be the characteristic function measuring surplus for coalition <i>S</i>.", body_style))
    story.append(Paragraph("For any ordering (permutation) <i>π</i> of <i>N</i>, the marginal contribution of bloc <i>i</i> is derived as <i>v(S ∪ {i}) - v(S)</i>, where <i>S</i> represents blocs preceding <i>i</i>. Assuming a uniform probability <i>1 / |N|!</i> for each sequence, we derive the Shapley expectation:", body_style))
    story.append(Paragraph("<i>E[φ] = Σ<sub>π ∈ N!</sub> (1 / |N|!) [v(S<sub>π</sub> ∪ {i}) - v(S<sub>π</sub>)]</i>", body_style))
    story.append(Paragraph("Since the number of permutations where <i>S</i> is of size <i>|S|</i> represents exactly <i>|S|! (|N| - |S| - 1)!</i> orderings, substituting this translates directly to the finite exact discrete sum:", body_style))
    story.append(Paragraph("<i>φ<sub>i</sub>(v) = ∑<sub>S ⊆ N ∖ {i}</sub> [|S|!(n - |S| - 1)! / n!] [v(S ∪ {i}) - v(S)]</i>", body_style))
    
    # 2. Portfolio Optimization
    story.append(Paragraph("2. Finite Constrained Dividend Trade Optimizations", heading_style))
    story.append(Paragraph("The allocation matrix between Grain, Beef, and Mineral Ore requires establishing a finite Lagrangian minimization to isolate maximum dividend yield against minimal risk <i>σ<sub>p</sub><sup>2</sup></i>. We structure the portfolio variance derived linearly as:", body_style))
    story.append(Paragraph("<i>σ<sub>p</sub><sup>2</sup> = w<sup>T</sup> Σ w</i>", body_style))
    story.append(Paragraph("Subject to the finite deterministic constraint that sum of portfolio weights equals 1: <i>w<sup>T</sup> <b>1</b> = 1</i>. To derive the closed-form optimal allocation, we introduce Lagrange multiplier <i>γ</i>:", body_style))
    story.append(Paragraph("<i>L(w, γ) = (1/2) w<sup>T</sup> Σ w - γ(w<sup>T</sup> <b>1</b> - 1)</i>", body_style))
    story.append(Paragraph("Taking the partial derivative with respect to the weight vector <i>w</i> and setting to 0 yields <i>Σ w = γ <b>1</b></i>, solving explicitly to <i>w = γ Σ<sup>-1</sup> <b>1</b></i>. Multiplying by <i><b>1</b><sup>T</sup></i> provides the final derived finite matrix allocation solving precisely for our optimal international trade asset weighting.", body_style))
    
    # 3. Continued Fractions
    story.append(Paragraph("3. Algorithm Derivation of Verifiable Continued Fractions", heading_style))
    story.append(Paragraph("For transparent verification of fractional tariffs <i>p/q</i>, we apply repeated applications of the finite Euclidean division algorithm. Setting <i>r<sub>0</sub> = p</i> and <i>r<sub>1</sub> = q</i>, we iterate:", body_style))
    story.append(Paragraph("<i>r<sub>k-1</sub> = a<sub>k</sub> r<sub>k</sub> + r<sub>k+1</sub></i>", body_style))
    story.append(Paragraph("Rearranging isolates the fraction <i>r<sub>k-1</sub> / r<sub>k</sub> = a<sub>k</sub> + r<sub>k+1</sub> / r<sub>k</sub> = a<sub>k</sub> + 1 / (r<sub>k</sub> / r<sub>k+1</sub>)</i>. Inductively substituting this finite recurrence backwards precisely generates the continued fraction expansion:", body_style))
    story.append(Paragraph("<i>x = a<sub>0</sub> + 1 / (a<sub>1</sub> + 1 / (a<sub>2</sub> + ...))</i>", body_style))
    
    # 4. Econophysics
    story.append(Paragraph("4. Kinetic Exchange and Fokker-Planck Econophysics Derivation", heading_style))
    story.append(Paragraph("We simulate individual trade states iteratively as physical kinetic gas collisions where energy equates to capital. Let actors have wealth <i>w<sub>i</sub>, w<sub>j</sub></i> tracking saving propensity <i>λ</i> and random exchange proportion <i>ε</i>. The conservation transaction derivations equate to:", body_style))
    story.append(Paragraph("<i>w<sub>i</sub>' = λ w<sub>i</sub> + ε(1 - λ)(w<sub>i</sub> + w<sub>j</sub>)</i>", body_style))
    story.append(Paragraph("<i>w<sub>j</sub>' = λ w<sub>j</sub> + (1 - ε)(1 - λ)(w<sub>i</sub> + w<sub>j</sub>)</i>", body_style))
    story.append(Paragraph("Under iterative expansion, setting the steady-state wealth moments across agents matches solving a stationary diffusion Fokker-Planck derivation. Analytically integrating the equilibrium continuous limit leads exactly to the generalized Gamma distribution preventing monopole exponential collapses:", body_style))
    story.append(Paragraph("<i>P(w) = C w<sup>3λ/(1-λ)</sup> exp(-w/T)</i>", body_style))
    
    # 5. Feynman Path Integrals
    story.append(Paragraph("5. Path Integrals Logistical Integrations via Functional Calculus", heading_style))
    story.append(Paragraph("For predictive neural nets in trade routing (NACETA logistics), we transition classical supply chain logistics to quantum probability amplitudes. Trade paths are discretized over <i>N</i> temporal hops with action <i>S = ∫ L(x, ẋ) dt</i> representing cumulative logistic friction.", body_style))
    story.append(Paragraph("To derive the optimal distribution, we integrate over all logistical paths limits:", body_style))
    story.append(Paragraph("<i>K(x<sub>b</sub>, t<sub>b</sub>; x<sub>a</sub>, t<sub>a</sub>) = lim<sub>N→∞</sub> ∫ ... ∫ exp((i/ℏ) ∑ L(x<sub>k</sub>, ẋ<sub>k</sub>) Δt) dx<sub>1</sub>...dx<sub>N-1</sub></i>", body_style))
    story.append(Paragraph("In the discrete neural network framework, the infinite-dimensional functional measure <i>Dx(t)</i> collapses into the normalized softmax limits of simulated routing weights <i>W<sub>i</sub> = exp(-Cost<sub>i</sub> / κ) / Z</i>, algorithmically isolating the definitive 'classical path of least action'.", body_style))
    
    # Conclusion
    story.append(Paragraph("6. Conclusion", heading_style))
    story.append(Paragraph("By spanning cooperative coalitions, exact finite linear algebra, rational sequence mathematics, thermodynamic econophysics, and quantum routing algorithms, this synthesized model ensures that all multi-agent trade policies under NACETA remain transparent, mathematically fair, and strictly optimal.", body_style))
    
    doc.build(story)

if __name__ == '__main__':
    generate_report()
    print("Nature Report generated successfully.")
