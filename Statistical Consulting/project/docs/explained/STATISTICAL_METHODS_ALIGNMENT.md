# Statistical Methods Alignment for Binance Market Making Project

## Overview

This document explains how the statistical methods taught in your statistics class align perfectly with the Binance Market Making project. The project is designed to showcase various statistical techniques in a real-world consulting context.

## Linear Regression Models (SLR/MLR)

### Applications in the Project

#### Q3 (Fill Probability)
- **Simple Linear Regression**: Model log(fill_probability) ~ distance_from_midpoint
- **Multiple Linear Regression**: Add predictors like time-of-day, volatility indicators, spread levels

#### Q4 (Adverse Selection)
- **Price Impact Modeling**: Next-period_returns ~ order-flow_imbalance + quote_imbalance + controls
- **Multiple predictors**: Include market depth, volatility measures, time variables

## Generalized Linear Models (GLM)

### Ideal Applications

#### Fill Probability Modeling
- **Logistic Regression**: More appropriate than linear regression for binary outcomes (filled/not filled)
- **Poisson/Negative Binomial**: For modeling count-like aspects of trading activity

#### Trade Size Analysis
- **Count Data Models**: Appropriate for discrete trade sizes
- **Gamma Regression**: For continuous positive trade sizes

## Time Series Models

### Essential for Core Questions

#### Q1/Q2 (Volatility and Returns)
- **GARCH Models**: Capture volatility clustering phenomenon
- **ARIMA Models**: Analyze price return patterns and forecasting
- **Seasonal Decomposition**: Identify intraday/hourly patterns

#### Return Analysis
- **Autoregressive Models**: For return predictability studies
- **VAR Models**: For multivariate time series relationships

## Statistical Tests Application

### T-tests

#### Direct Applications
- Comparing average spreads between different hours of day
- Testing if adverse selection differs significantly from zero
- Pre/post event analysis around funding times (00:00, 08:00, 16:00 UTC)
- Comparing parameter estimates across different time periods

### ANOVA

#### Perfect Fit Scenarios
- Testing if volatility differs significantly across hours/days of the week
- Comparing fill rates across different quote distances (categorical bins)
- Differences between symbols (ETH vs MATIC vs LINK) for comparative analysis
- Market regime analysis (high vol vs low vol periods)

### Chi-square Tests

#### Relevant Applications
- Testing independence between categorical variables (e.g., trade direction and time of day)
- Goodness-of-fit tests for distributional assumptions (normal vs empirical distributions)
- Contingency table analysis for discrete trading behaviors

### Pearson Correlation

#### Natural Applications
- Relationship between order-flow imbalance and future returns
- Correlation between spread measures and volatility indicators
- Cross-correlations in time series analysis for lead-lag relationships
- Market microstructure variable interdependencies

### Mann-Whitney U Test

#### Non-parametric Applications
- When data isn't normally distributed (common in financial data)
- Comparing return distributions between different market conditions
- Pre/post intervention comparisons without distributional assumptions
- Robustness checks for parametric test results

## Project Structure Alignment

### 1. Exploratory Analysis
- **Descriptive statistics**: Mean, median, standard deviation, quartiles
- **Correlation matrices**: Variable relationships and multicollinearity
- **Visualization**: Histograms, boxplots, scatter plots, time series plots
- **Distribution analysis**: Q-Q plots, skewness, kurtosis

### 2. Model Building
- **Regression models**: For each of the four core research questions
- **Variable selection**: Stepwise, AIC/BIC criteria
- **Interaction effects**: Time × volatility, symbol × market conditions

### 3. Diagnostic Testing
- **Residual analysis**: Normality, homoscedasticity, autocorrelation
- **Influence diagnostics**: Outliers, leverage points, Cook's distance
- **Multicollinearity**: VIF scores, condition indices
- **Model fit**: R², adjusted R², AIC, BIC

### 4. Hypothesis Testing
- **Coefficient significance**: t-tests on all regression parameters
- **Model significance**: F-tests for overall model validity
- **Non-nested hypothesis**: Testing competing model specifications
- **Robust inference**: HAC standard errors, bootstrap methods

### 5. Validation
- **Out-of-sample testing**: Holdout sample performance
- **Cross-validation**: K-fold for model selection
- **Robustness checks**: Alternative specifications, different time periods
- **Sensitivity analysis**: Parameter stability across subsamples

## Specific Statistical Techniques by Research Question

### Q1: Volatility Modeling
- **Time series models (GARCH)**: Capture volatility clustering phenomenon
- **ANOVA**: Test volatility differences by hour/day
- **Pearson correlations**: Volatility persistence and autocorrelation
- **Diagnostic tests**: ARCH LM test, residual normality tests

### Q2: Return Distribution Analysis
- **T-tests**: Moments of return distributions
- **Chi-square goodness-of-fit**: Normality testing
- **Mann-Whitney**: Comparing different time periods without normality
- **QQ-plots**: Visual distribution assessment

### Q3: Fill Probability Analysis
- **Logistic regression (GLM)**: Binary fill outcomes
- **SLR/MLR**: Continuous probability measures
- **ANOVA**: Differences across quote distances
- **Likelihood ratio tests**: Model comparison for GLMs

### Q4: Adverse Selection Measurement
- **SLR/MLR**: Price impact modeling
- **T-tests**: Coefficient significance and hypothesis testing
- **Pearson correlations**: Variable relationships
- **F-tests**: Joint hypothesis testing

## Presentation Opportunities

This project provides excellent material for showcasing:

1. **Multiple regression techniques** applied to real financial data
2. **Time series analysis** with economic interpretation
3. **Model comparison** (parametric vs non-parametric approaches)
4. **Assumption testing** and diagnostic checking
5. **Practical implications** of statistical findings for trading decisions
6. **Consulting communication**: Translating statistical findings to business insights

## Data Advantages

The rich dataset provides:
- **Millions of observations**: Ensures statistical power for all tests
- **High frequency**: Enables granular time series analysis
- **Multiple symbols**: Facilitates comparative analysis
- **Clear business context**: Makes statistical findings actionable
- **Natural experiments**: Funding times, volatility regime changes

## Conclusion

The Binance Market Making project is ideally suited for demonstrating the full spectrum of statistical methods taught in your course. Each technique finds natural application in addressing real consulting questions, and the large, high-quality dataset ensures robust statistical inference. The project's structure mirrors the typical statistical consulting workflow, making it perfect for both learning and presentation purposes.