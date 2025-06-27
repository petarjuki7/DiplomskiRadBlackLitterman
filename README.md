# Black-Litterman Portfolio Optimization - Master Thesis Implementation

## Repository Overview

This repository contains multiple implementations and approaches for the Black-Litterman portfolio optimization model with machine learning-generated views, developed as part of a master thesis project.

## Files Description

### 1. `BlackLittermanNotebook.ipynb`

**Main implementation** - A clean, notebook-style approach using PyPortfolioOpt throughout:

- Uses Dow Jones 30 stocks as investment universe
- Custom set views on stock performance
- Compares Black-Litterman vs Market-Cap vs Equal-Weight portfolios
- Comapres Black-Litterman portfolio with different confidences in views using the Idzorek method for setting confidences
- Includes statistical significance testing and turnover analysis
- More straightforward than the hybrid approach


### 2. `Hybrid_view_generation.py`

**Idea based on**: https://www.researchgate.net/publication/332174219_A_hybrid_approach_for_generating_investor_views_in_Black-Litterman_model

**Alternative implementation** - A complete hybrid approach that replaces subjective investor views with ML predictions using a three-stage pipeline:
- ARMA-GARCH forecasting of technical indicators
- SVR (Support Vector Regression) for return prediction  
- Black-Litterman optimization with PyPortfolioOpt
- Comprehensive backtesting framework with visualization
  
**Results**: 3-5% annual outperformance vs market-cap benchmark

### 3. `Comparison.ipynb`

**Performance comparison notebook** - Visualizes and compares results:
- Side-by-side comparison of different portfolio optimization approaches
- Performance metrics comparison tables
- Visualization of returns, Sharpe ratios, and other key metrics
- Statistical tests for significance of outperformance

## Testing & Results

Comparison was tested on:
- **Period**: 2019-2024 (5 years)
- **Assets**: Major US equities (tech, finance, healthcare, energy)
- **Rebalancing**: Quarterly

## Main Conclusion

The project successfully demonstrates that machine learning can effectively aid subjective human views in the Black-Litterman model, achieving superior risk-adjusted returns through ML, data-driven view generation.