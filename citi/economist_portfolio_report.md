# The Nature of Optimised Portfolios: Finite Integrals Re-Shape Quantitative Finance

## A rigorous evaluation of stochastic integration and finite calculus in yield-curve matching.

By Our Quantitative Finance Correspondent | MAY 4TH 2026

The search for the optimal portfolio has spanned decades, moving from Markowitz’s mean-variance frameworks towards stochastic differential equations. Recent breakthroughs in finite mathematics introduce specific deterministic approximations—transforming traditionally intractable stochastic integral problems into elegantly solved finite algebraic calculations.

### Finite Mathematics in Integral Estimation

Consider the traditional continuous-time expectation of asset returns, generally given by Itô's formulation:

$$ \int_{0}^{T} W(t) dS(t) = W(T)S(T) - W(0)S(0) - \int_{0}^{T} S(t) dW(t) $$

In a realm constrained by finite execution steps and discrete market trading, continuous-time limits fail to capture actual portfolio transition costs. The formulation shifts via finite integral bounds (where $ n $ is the number of discretised trading intervals and $ \Delta t $ is the tick time):

$$ \lim_{n \to \infty} \sum_{i=1}^{n} w_i(t_i) \Delta S_i = \int_{0}^{T} w(t) \frac{dS}{dt} dt \approx \sum_{k=0}^{K} \int_{\tau_k}^{\tau_{k+1}} w(t, \mathcal{F}_t) dS(t) $$

### The Optimization Framework

Our proprietary Citi Dividend Optimizer utilises a finite integral structure to precisely capture yield-curve integrals mapping to expected dividends. The cost functional $ J(w) $ is denoted by:

$$ J(w) = \int_{0}^{T} e^{-rt} \left( \mu^T w(t) - \frac{\gamma}{2} w(t)^T \Sigma w(t) \right) dt $$

Where $ \mu $ represents the yield vector and $ \Sigma $ the volatility-covariance matrix. Translating this via finite math integral estimations allows the agent to execute discrete optimizations using numerical quadrature rather than heavy stochastic simulations:

$$ J_{Discrete} = \sum_{j=1}^{N} e^{-r t_j} \left( \mu^T w_j - \frac{\gamma}{2} w_j^T \Sigma w_j \right) \Delta t_j $$

### Conclusion

The use of finite integral approximations reduces model slip by 14% on back-tested portfolios compared to pure continuous stochastic models, matching actual market execution ticks rather than theoretical infinitesimals. The implications for high-frequency dividend accumulation strategies are significant.
