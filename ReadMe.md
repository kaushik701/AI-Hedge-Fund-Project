# AI Hedge Fund

## Overview
The AI Hedge Fund is an advanced proof-of-concept system that leverages artificial intelligence to simulate trading decisions in financial markets. This project explores how multiple AI agents, each modeled after legendary investors, can collaborate to analyze market data and make informed investment decisions.

**IMPORTANT DISCLAIMER:** This project is strictly for **educational purposes only** and is not intended for real trading or investment activities. The system simulates trading decisions but does not execute actual trades. Use at your own risk.

## System Architecture

The system employs a multi-agent framework where specialized AI agents work together to analyze different aspects of potential investments:

### Investment Philosophy Agents
Each of these agents embodies the investment philosophy of a legendary investor:

1. **Ben Graham Agent** - Focuses on value investing principles, seeking undervalued stocks with a significant margin of safety.
2. **Bill Ackman Agent** - Emulates activist investing, taking bold positions and advocating for corporate change.
3. **Cathie Wood Agent** - Specializes in growth investing, prioritizing innovation and disruptive technologies.
4. **Charlie Munger Agent** - Follows the principle of buying wonderful businesses at fair prices, emphasizing quality.
5. **Peter Lynch Agent** - Seeks high-growth opportunities ("ten-baggers") and invests in familiar industries.
6. **Phil Fisher Agent** - Implements scuttlebutt analysis methodology to uncover high-quality growth stocks.
7. **Stanley Druckenmiller Agent** - Focuses on macroeconomic trends to identify asymmetric opportunities.
8. **Warren Buffett Agent** - Searches for wonderful companies at fair prices with sustainable competitive advantages.

### Analysis Agents
These agents focus on specific analytical approaches:

9. **Valuation Agent** - Calculates intrinsic value and generates trading signals.
10. **Sentiment Agent** - Analyzes market sentiment from various sources.
11. **Fundamentals Agent** - Examines company financial data and performance metrics.
12. **Technicals Agent** - Evaluates price patterns and technical indicators.

### Management Agents
These agents make the final decisions:

13. **Risk Manager** - Assesses risk metrics and establishes position limits.
14. **Portfolio Manager** - Integrates all signals to make final trading decisions.

## Technical Analysis Formulas

The system employs various technical formulas across its agents:

### Valuation Agent
- **Discounted Cash Flow (DCF)**:
  ```
  Intrinsic Value = ∑(FCF_t / (1 + r)^t) + Terminal Value
  ```
  Where:
  - FCF_t = Free Cash Flow in year t
  - r = Discount rate
  - Terminal Value = FCF_n × (1 + g) / (r - g)
  - g = Long-term growth rate

- **Price-to-Earnings (P/E) Ratio**:
  ```
  P/E Ratio = Market Price per Share / Earnings per Share
  ```

- **Enterprise Value-to-EBITDA Ratio**:
  ```
  EV/EBITDA = Enterprise Value / EBITDA
  ```

### Fundamentals Agent
- **Return on Equity (ROE)**:
  ```
  ROE = Net Income / Shareholder's Equity
  ```

- **Debt-to-Equity Ratio**:
  ```
  D/E Ratio = Total Liabilities / Shareholder's Equity
  ```

- **Gross Margin**:
  ```
  Gross Margin = (Revenue - COGS) / Revenue
  ```

### Technicals Agent
- **Moving Average Convergence Divergence (MACD)**:
  ```
  MACD Line = 12-period EMA - 26-period EMA
  Signal Line = 9-period EMA of MACD Line
  ```

- **Relative Strength Index (RSI)**:
  ```
  RSI = 100 - (100 / (1 + RS))
  RS = Average Gain / Average Loss
  ```

### Risk Manager
- **Value at Risk (VaR)**:
  ```
  VaR = Portfolio Value × (Z-score × Portfolio Standard Deviation) × √Time Horizon
  ```

- **Sharpe Ratio**:
  ```
  Sharpe Ratio = (Expected Return - Risk-Free Rate) / Portfolio Standard Deviation
  ```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Poetry (dependency management)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kaushik701/AI-Hedge-Fund-Project.git
   cd ai-hedge-fund
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Set up environment variables**:
   ```bash
   # Create .env file for your API keys
   cp .env.example .env
   ```

5. **Configure API keys** in the `.env` file:
   ```
   # For running LLMs hosted by groq (deepseek, llama3, etc.)
   # Get your Groq API key from https://groq.com/
   GROQ_API_KEY=your-groq-api-key

   # For getting financial data to power the hedge fund
   # Get your Financial Datasets API key from https://financialdatasets.ai/
   FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
   ```

### API Information

The system requires the following APIs:

1. **Groq API**
   - Purpose: Powers the large language models for agent reasoning
   - Website: [https://groq.com/](https://groq.com/)
   - Usage: Set as `GROQ_API_KEY` in your .env file
   - Implementation: Used in the agents' reasoning modules to generate investment theses

2. **Financial Datasets API**
   - Purpose: Provides real-time and historical financial data
   - Website: [https://financialdatasets.ai/](https://financialdatasets.ai/)
   - Usage: Set as `FINANCIAL_DATASETS_API_KEY` in your .env file
   - Implementation: Used to fetch company fundamentals, price data, and financial metrics
   - Note: Data for AAPL, GOOGL, MSFT, NVDA, and TSLA is available without an API key

## Usage

### Running the Hedge Fund

Basic usage with default parameters:
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

Show detailed agent reasoning:
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --show-reasoning
```

Specify a custom date range:
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

### Running the Backtester

Basic backtesting:
```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

Backtest with custom date range:
```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

## Project Structure

```
ai-hedge-fund/
├── src/
│   ├── agents/                   # Agent definitions and workflow
│   │   ├── bill_ackman.py        # Bill Ackman agent
│   │   ├── ben_graham.py         # Ben Graham agent
│   │   ├── cathie_wood.py        # Cathie Wood agent
│   │   ├── charlie_munger.py     # Charlie Munger agent
│   │   ├── peter_lynch.py        # Peter Lynch agent
│   │   ├── phil_fisher.py        # Phil Fisher agent
│   │   ├── stanley_druckenmiller.py # Stanley Druckenmiller agent
│   │   ├── warren_buffett.py     # Warren Buffett agent
│   │   ├── fundamentals.py       # Fundamental analysis agent
│   │   ├── portfolio_manager.py  # Portfolio management agent
│   │   ├── risk_manager.py       # Risk management agent
│   │   ├── sentiment.py          # Sentiment analysis agent
│   │   ├── technicals.py         # Technical analysis agent
│   │   ├── valuation.py          # Valuation analysis agent
│   ├── tools/                    # Agent tools
│   │   ├── api.py                # API tools
│   │   ├── data_parser.py        # Data processing utilities
│   │   ├── market_data.py        # Market data fetching
│   ├── models/                   # Model definitions
│   │   ├── stock.py              # Stock data model
│   │   ├── trade.py              # Trade data model
│   ├── backtester.py             # Backtesting tools
│   ├── main.py                   # Main entry point
├── pyproject.toml                # Poetry configuration
├── .env.example                  # Example environment variables
├── README.md                     # This file
```

## Development Status

This project is currently **under active development**. The core functionality is implemented, but we are continuously refining the algorithms, improving agent reasoning, and enhancing the backtesting capabilities.

Current development priorities:
1. Enhancing agent reasoning with more sophisticated market models
2. Implementing additional technical indicators
3. Improving the backtesting engine with more realistic transaction costs
4. Adding support for additional asset classes beyond equities

## Contributing

We welcome contributions to improve the AI Hedge Fund project:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Best practices**: Please keep your pull requests small and focused to facilitate easier review and integration.

## Feature Requests and Feedback

We are open to suggestions and improvements:

- For feature requests, please open an [issue](https://github.com/kaushik701/AI-Hedge-Fund-Project/issues) tagged with `enhancement`
- For bug reports, use the `bug` tag
- For general feedback, feel free to contact the repository owner

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

Special thanks to the open-source community and the legendary investors whose philosophies have inspired this project.