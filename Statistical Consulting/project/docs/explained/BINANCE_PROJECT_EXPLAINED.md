# Understanding the Binance Market Making Project
*A Simple Guide to the Dataset and Project Goals*

## What is this project about?

This is a statistical consulting project that tests mathematical models used in **crypto market making** against real market data from Binance (a major cryptocurrency exchange). Specifically, it examines whether the assumptions in a famous academic model called the **Avellaneda-Stoikov (A-S) model** actually match how markets behave in reality.

Think of it like testing a physics formula against real-world experiments - the formula might work perfectly in theory, but when applied to the messy real world, it might need adjustments.

## Who is this for?

This project serves several potential "clients":

1. **Jim O'Connor's research lab** - They're developing automated trading systems and need to know which parts of market models to focus on first
2. **Small crypto trading firms** - They need to decide how far from the current price to place their buy/sell orders
3. **Fund execution traders** - They want to know when trading crypto is cheapest (least costly)
4. **Abdullah himself** - As a student researcher working on market making algorithms

## What exactly is in the dataset?

The project uses **free historical market data** from Binance, specifically for **perpetual futures contracts**. There are two main types of data files:

### 1. bookTicker Data (~21 million rows per day for ETH)
This captures the "order book" - what people are willing to buy and sell at any moment:

- **best_bid_price**: Highest price someone wants to BUY at ($2,474.30)
- **best_bid_qty**: How much they want to buy (6.009 ETH)
- **best_ask_price**: Lowest price someone wants to SELL at ($2,474.31)
- **best_ask_qty**: How much they want to sell (44.757 ETH)
- **event_time**: When this quote was published (timestamp)

Think of this like a constantly updating bulletin board where buyers post "I want to buy X amount at $Y" and sellers post "I want to sell X amount at $Z".

### 2. trades Data (~2.8 million rows per day for ETH)
This records actual transactions that occurred:

- **price**: What price the trade executed at ($2,474.30)
- **qty**: How much was traded (0.009 ETH)
- **is_buyer_maker**: Whether the buyer was the "resting" order (TRUE) or the aggressive buyer (FALSE)
- **time**: When the trade happened (timestamp)

This is like a sales ledger showing "At time T, Person A bought B amount at price P from Person B".

## Why is this data valuable?

1. **High frequency**: Updates happen multiple times per second
2. **Real market behavior**: Shows actual buying/selling patterns, not surveys or estimates
3. **Free and comprehensive**: Covers months of continuous trading
4. **Multiple cryptocurrencies**: ETH (highly liquid), MATIC, LINK (mid-cap coins) for comparison

## What are the key research questions?

The project tests three core assumptions of the A-S model:

### Q1: Is market volatility constant?
**Theory says**: Prices move randomly with steady volatility
**Reality check**: Do prices move more during certain hours? Do volatility "shocks" last a long time?

*This affects how market makers set their risk parameters.*

### Q2: Do price returns follow a normal distribution?
**Theory says**: Price changes follow a bell curve
**Reality check**: Are there more extreme price moves than expected? Does this change with time horizon?

*This determines how often market makers get "surprised" by big price moves.*

### Q3: Does fill probability decay exponentially with distance?
**Theory says**: Quotes further from the current price get filled less often, following a smooth exponential curve
**Reality check**: Do distant quotes really get filled predictably less often?

*This helps market makers decide how far from the current price to place their orders.*

### Q4: Is there adverse selection (informed trading)?
**Theory says**: Trades don't predict future price movements
**Reality check**: After a trade happens, does the price tend to keep moving in the same direction?

*This measures how much smart traders "pick off" market makers by trading just before prices move.*

## Why does this matter?

Market making is crucial for financial markets because:

1. **Liquidity provision**: Market makers provide the "grease" that allows buyers and sellers to transact quickly
2. **Price discovery**: Their activity helps establish fair market prices
3. **Risk management**: They profit from the spread between buy/sell prices while managing inventory risk

If the A-S model accurately describes real markets, then trading firms can:
- Use the model's formulas to optimize their quoting strategies
- Set better risk parameters
- Predict profitability more accurately

If the model doesn't match reality, then:
- The formulas might lead to losses
- Market makers might quote at wrong distances from the price
- Risk management might be inadequate

## How does this project help clients?

### For Jim's Lab:
- **Focus efforts**: Tells them which parts of the A-S model need refinement first
- **Parameter calibration**: Provides realistic ranges for volatility, fill rates, etc.
- **Validation**: Confirms their simulation models against real data before investing in expensive commercial data

### For Small Trading Firms:
- **Quoting strategy**: Helps decide how far from the mid-price to place orders
- **Timing**: Identifies when markets are cheaper/expensive to trade
- **Risk setting**: Provides empirical estimates for volatility clustering

### For Fund Traders:
- **Execution timing**: Knows when spreads are narrow vs. wide
- **Cost estimation**: Understands price impact of their trades

## Technical Details Made Simple

### Data Processing:
1. **Download**: Automatically fetch files from Binance's data repository
2. **Verify**: Check file integrity with checksums
3. **Convert**: Transform compressed CSV files to efficient Parquet format
4. **Clean**: Sort timestamps, remove duplicates, fix data types

### Analysis Approach:
1. **Exploratory Analysis**: Generate plots and summary statistics
2. **Statistical Modeling**: Use time-series methods (GARCH models) for volatility
3. **Curve Fitting**: Find best-fit parameters for exponential decay in fill rates
4. **Regression Analysis**: Measure relationships between variables with proper standard errors

## Expected Outcomes

This project will produce:
1. **Empirical evidence** on whether A-S assumptions hold in real markets
2. **Calibrated parameters** (volatility σ, fill rate A & k) from actual data
3. **Recommendations** for adjusting the model based on findings
4. **Limitations documentation** explaining when the model works vs. doesn't work

## Why This Matters for Crypto Markets

Unlike traditional stock markets, crypto markets:
- Operate 24/7 with no closing breaks
- Have different participant behaviors (more retail trading)
- Experience higher volatility periods
- Show unique patterns like funding rate effects

Testing market making models in this environment validates whether academic finance theories apply to this newer, different market structure.

---

*This project bridges academic finance theory with real-world crypto market practice, providing actionable insights for anyone using algorithmic market making strategies.*