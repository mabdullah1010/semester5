# Quick Reference: Binance Market Making Project

## What is this project?
Testing if academic market making models match real crypto market behavior.

## Who benefits?
- **Researchers**: Validate theoretical models against real data
- **Trading firms**: Optimize market making strategies
- **Fund traders**: Time executions for lower costs
- **Students**: Learn practical market microstructure

## Key Dataset Components

### bookTicker Data (~21M rows/day for ETH)
- Best bid/ask prices and quantities
- Real-time limit order book snapshots
- Shows what people are willing to trade at any moment

### trades Data (~2.8M rows/day for ETH)
- Actual transaction records
- Price, quantity, and buyer/seller identification
- Historical record of what actually traded

## 4 Core Research Questions

1. **Is volatility constant?** 
   - Theory: Steady price fluctuations
   - Reality check: Do volatility "clusters" exist?

2. **Are price changes normal?**
   - Theory: Bell curve distribution
   - Reality check: How often do extreme moves happen?

3. **Do fills decay exponentially?**
   - Theory: Smooth exponential decay with distance
   - Reality check: Do distant quotes fill predictably less?

4. **Is there adverse selection?**
   - Theory: Trades don't predict future prices
   - Reality check: Do prices move after trades?

## Why This Matters

Market makers provide essential liquidity services:
- Enable smooth trading for all participants
- Profit from bid-ask spreads while managing risk
- Need accurate models to avoid losses

If models don't match reality:
- Strategies may lose money
- Risk management fails
- Capital gets misallocated

## Practical Applications

- **Strategy optimization**: Better quote placement
- **Risk management**: Realistic parameter settings
- **Cost reduction**: Cheaper trading for large orders
- **Performance improvement**: Higher profitability

## Technical Approach

1. **Data acquisition**: Automated downloading with verification
2. **Processing**: Efficient Parquet format for analysis
3. **Exploration**: Visualizations and summary statistics
4. **Modeling**: Statistical methods (GARCH, regression, bootstrap)
5. **Validation**: Empirical testing of theoretical assumptions

## Expected Deliverables

- Empirical validation of market making model assumptions
- Calibrated parameters from real market data
- Actionable recommendations for model improvements
- Documentation of limitations and edge cases