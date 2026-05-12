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
    story.append(Paragraph("1. Cooperative Game Theory: Surplus Division", heading_style))
    story.append(Paragraph("To model tariff subsidies, we assume a cooperative game where stakeholders (e.g., Canada and the EU) form coalitions. The Shapley value defines a unique, fair distribution of trade surplus by assigning to each participant their marginal average contribution over all possible permutations:", body_style))
    story.append(Paragraph("<i>φ<sub>i</sub>(v) = ∑ [|S|!(n - |S| - 1)! / n!] [v(S ∪ {i}) - v(S)]</i>", body_style))
    story.append(Paragraph("This finite summation guarantees Pareto efficiency and symmetry in resource subsidies.", body_style))
    
    # 2. Portfolio Optimization
    story.append(Paragraph("2. Finite Mathematics in Dividend Trade Portfolios", heading_style))
    story.append(Paragraph("The allocation of capital between Grain, Beef, and Mineral Ore trade states relies on mean-variance optimizations. Given a weight vector <i>w</i> and a finite state covariance matrix <i>Σ</i>, the portfolio volatility risk is defined as:", body_style))
    story.append(Paragraph("<i>σ<sub>p</sub> = √(w<sup>T</sup> Σ w)</i>", body_style))
    story.append(Paragraph("This quadratic form allows states to perfectly hedge their tariff-adjusted dividend portfolios across substitutable sectors.", body_style))
    
    # 3. Continued Fractions
    story.append(Paragraph("3. Trade Verifiability via Continued Fractions", heading_style))
    story.append(Paragraph("Exchange rates and tariff ratios form complex rational numbers that must be insulated against floating-point asymmetries. We utilize the continued fraction expansion:", body_style))
    story.append(Paragraph("<i>x = a<sub>0</sub> + 1 / (a<sub>1</sub> + 1 / (a<sub>2</sub> + ...))</i>", body_style))
    story.append(Paragraph("The convergents of these finite sequences algorithmically map the negotiated quantum of tariffs, allowing immediate verifiability of declared exchange ledgers.", body_style))
    
    # 4. Econophysics
    story.append(Paragraph("4. Socio-Economic Kinetic Exchange Models", heading_style))
    story.append(Paragraph("Expanding statistical mechanics via the Chakraborti-Chakrabarti model, we account for saving propensity <i>λ</i> during pairwise trade collisions. Rather than a pure Monopolistic Boltzmann-Gibbs collapse, capital adheres to a target Gamma distribution:", body_style))
    story.append(Paragraph("<i>P(w) = C w<sup>3λ/(1-λ)</sup> exp(-w/T)</i>", body_style))
    story.append(Paragraph("This demonstrates that parameterized partial wealth retention mathematically ensures a stable mid-class trade equilibrium.", body_style))
    
    # 5. Feynman Path Integrals
    story.append(Paragraph("5. Path Integrals & Optimal Neural Net Routing", heading_style))
    story.append(Paragraph("Evaluating global supply chains requires optimizing among an infinite set of logistical routes. Applying Feynman's path integral formulation:", body_style))
    story.append(Paragraph("<i>K(x<sub>b</sub>, t<sub>b</sub>; x<sub>a</sub>, t<sub>a</sub>) = ∫ exp((i/ℏ) S[x(t)]) Dx(t)</i>", body_style))
    story.append(Paragraph("we define the classical action <i>S[x(t)]</i> as trade friction (cost). A predictive neural network layer translates these lowest-friction paths into maximal probability weightings, dictating the ultimate asset portfolio distribution (e.g., billions in Beef vs Mineral Ore) for the NACETA block.", body_style))
    
    # Conclusion
    story.append(Paragraph("6. Conclusion", heading_style))
    story.append(Paragraph("By spanning cooperative coalitions, exact finite linear algebra, rational sequence mathematics, thermodynamic econophysics, and quantum routing algorithms, this synthesized model ensures that all multi-agent trade policies under NACETA remain transparent, mathematically fair, and strictly optimal.", body_style))
    
    doc.build(story)

if __name__ == '__main__':
    generate_report()
    print("Nature Report generated successfully.")
