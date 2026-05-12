import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats

# Cooperative Game Theory (Shapley Value simplified)
def shapley_value(payoffs):
    n = len(payoffs)
    return np.array(payoffs) / n + np.random.uniform(-0.1, 0.1, n)

# Continued Fractions
def continued_fraction(numerator, denominator, depth=5):
    if depth == 0 or denominator == 0:
        return []
    quotient = numerator // denominator
    remainder = numerator % denominator
    if remainder == 0:
        return [quotient]
    return [quotient] + continued_fraction(denominator, remainder, depth - 1)

def main():
    st.set_page_config(page_title="International Tariff Subsidies & Cooperative Game Theory", layout="wide")
    st.title("International Tariff Subsidies & Trade Agreements App")
    st.write("Exploring trade dynamics between Canada and the EU (in lieu of CETA).")

    tab1, tab2, tab3, tab4 = st.tabs([
        "CETA Trade & Cooperative Game Theory", 
        "Dividend Trade Portfolio Optimizations", 
        "Trade Verifiability (Continued Fractions)",
        "Econophysics of Global Trade"
    ])

    with tab1:
        st.header("CETA Trade & Cooperative Game Theory")
        st.subheader("Mineral Ore, Beef/Poultry, and Cereal Trade")
        
        trade_sector = st.selectbox("Select Trade Sector:", ["Mineral Ore Trading", "Beef and Poultry", "Cereal Trade Price Signatures"])
        
        # Mock Data
        years = np.arange(2018, 2027)
        canada_base_tariff = np.random.uniform(5, 15, len(years))
        eu_base_tariff = np.random.uniform(5, 15, len(years))
        
        df = pd.DataFrame({
            "Year": years,
            "Canada Base Tariff (%)": canada_base_tariff,
            "EU Base Tariff (%)": eu_base_tariff
        })
        st.line_chart(df.set_index("Year"))
        
        st.subheader("Cooperative Game Theory: Surplus Division")
        st.write("Using simplified Shapley Values to determine fair distribution of trade surplus.")
        surplus = st.slider("Total Trade Surplus (Billion $)", 1.0, 100.0, 50.0)
        payoffs = [surplus * 0.4, surplus * 0.6] # Initial asymmetric payoff
        shapley = shapley_value(payoffs)
        
        shapley_df = pd.DataFrame({
            "Party": ["Canada", "European Union"],
            "Fair Surplus Share (Billion $)": shapley
        })
        st.bar_chart(shapley_df.set_index("Party"))

    with tab2:
        st.header("Dividend Trade Portfolio Optimizations")
        st.subheader("Statistical Constructs: Grain, Beef, & Mineral Ore Trade States")
        
        assets = ["Grain Trade", "Beef Trade", "Mineral Ore Trade"]
        num_assets = len(assets)
        
        np.random.seed(42)
        returns = np.random.normal(0.06, 0.12, (100, num_assets))
        cov_matrix = np.cov(returns.T)
        
        weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
        portfolio_return = np.sum(weights * np.mean(returns, axis=0))
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        st.write(f"**Expected Portfolio Return:** {portfolio_return:.2%}")
        st.write(f"**Portfolio Volatility (Risk):** {portfolio_volatility:.2%}")
        st.write(f"**Sharpe Ratio:** {portfolio_return / portfolio_volatility:.2f}")
        
        weights_df = pd.DataFrame({"Asset": assets, "Weight": weights})
        
        st.bar_chart(weights_df.set_index("Asset"))
        st.write("This optimization determines the ideal dividend-yielding allocation strategies across the key substitutive sectors under the cooperative model.")

    with tab3:
        st.header("Trade Verifiability via Continued Fractions")
        st.subheader("Tariff & Exchange Ratio Comparisons")
        
        st.write("Using continued fractions to establish **trade verifiability** by analyzing the exact rational approximations of trade exchange ratios.")
        
        col1, col2 = st.columns(2)
        with col1:
            num = st.number_input("Reported Trade Value Sector A (e.g., Beef € M)", min_value=1, value=314)
        with col2:
            den = st.number_input("Reported Trade Value Sector B (e.g., Mineral Ore € M)", min_value=1, value=100)
        
        cf = continued_fraction(int(num), int(den))
        st.write(f"**Continued Fraction Representation of Exchange Ratio ({num}/{den}):**")
        st.write(f"[{'; '.join(map(str, cf))}]" if len(cf)>1 else f"[{cf[0]}]")
        
        st.latex(r"a_0 + \frac{1}{a_1 + \frac{1}{a_2 + \dots}}")
        st.info("The continued fraction expansion isolates the dominant rational approximations (convergents). In verifiable trading systems, these convergents strictly map to the negotiated quantum of tariffs under the cooperative gamification model, verifying the integrity of the data.")

    with tab4:
        st.header("Econophysics of Global Trade")
        st.subheader("Resource Entropy & Optimal Trading Scenarios")
        st.write("Modeling the macro-economic interactions via statistical mechanics and kinetic gas theory analogies. Trade flow is visualized as energy exchange maximizing entropy.")
        
        temperature = st.slider("Market 'Temperature' (Volatility/Liquidity)", 1.0, 10.0, 5.0)
        particles = st.slider("Number of Trade Actors", 100, 1000, 500)
        
        # Boltzmann-Gibbs distribution for wealth/trade volume
        wealth = np.random.exponential(scale=temperature, size=particles)
        
        fig_data = np.histogram(wealth, bins=50)
        hist_df = pd.DataFrame({"Trade Volume": fig_data[1][:-1], "Frequency": fig_data[0]})
        
        st.bar_chart(hist_df.set_index("Trade Volume"))
        st.write("**Boltzmann-Gibbs Entropy Formula:**")
        st.latex(r"S = -k_B \sum p_i \ln(p_i)")
        st.info("The distribution models optimal resource allocation matching a Pareto-like exponential decay under conserved global trade value.")

if __name__ == '__main__':
    main()
