import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import questionary
import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore, Style, init
import numpy as np
import itertools

from llm.models import LLM_ORDER, get_model_info
from util.analysts import ANALYST_ORDER
from main import run_hedge_fund
from tools.api import (
    get_company_news,
    get_price_data,
    get_prices,
    get_financial_metrics,
    get_insider_trades,
)
from util.display import print_backtest_results, format_backtest_row

init(autoreset=True)


class Backtester:
    def __init__(
        self,
        agent,
        tickers,
        start_date,
        end_date,
        initial_capital,
        model_name="gpt-4o",
        model_provider="OpenAI",
        selected_analysts=[],
        initial_margin_requirement=0.0,
    ):
        """Initialize the backtester with trading parameters."""
        self.agent = agent
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.model_name = model_name
        self.model_provider = model_provider
        self.selected_analysts = selected_analysts
        self.margin_ratio = initial_margin_requirement
        self.portfolio_values = []
        
        # Initialize portfolio
        self.portfolio = {
            "cash": initial_capital,
            "margin_used": 0.0,
            "positions": {},
            "realized_gains": {}
        }
        
        # Initialize position data for each ticker
        for ticker in tickers:
            self.portfolio["positions"][ticker] = {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0
            }
            
            self.portfolio["realized_gains"][ticker] = {
                "long": 0.0,
                "short": 0.0
            }

    def execute_trade(self, ticker, action, quantity, current_price):
        """Execute a trade and update the portfolio."""
        if quantity <= 0:
            return 0

        quantity = int(quantity)  # Use integer shares
        position = self.portfolio["positions"][ticker]
        
        # Handle different trade actions
        if action == "buy":
            return self._execute_buy(ticker, quantity, current_price, position)
        elif action == "sell":
            return self._execute_sell(ticker, quantity, current_price, position)
        elif action == "short":
            return self._execute_short(ticker, quantity, current_price, position)
        elif action == "cover":
            return self._execute_cover(ticker, quantity, current_price, position)
            
        return 0

    def _execute_buy(self, ticker, quantity, price, position):
        """Execute a buy order."""
        cost = quantity * price
        
        # Check if we have enough cash
        if cost > self.portfolio["cash"]:
            # Calculate maximum affordable quantity
            quantity = int(self.portfolio["cash"] / price)
            if quantity <= 0:
                return 0
            cost = quantity * price
        
        # Update position and cash
        old_shares = position["long"]
        old_cost_basis = position["long_cost_basis"]
        
        # Calculate new weighted average cost basis
        if old_shares + quantity > 0:
            old_cost = old_cost_basis * old_shares
            new_cost = cost
            position["long_cost_basis"] = (old_cost + new_cost) / (old_shares + quantity)
        
        position["long"] += quantity
        self.portfolio["cash"] -= cost
        
        return quantity

    def _execute_sell(self, ticker, quantity, price, position):
        """Execute a sell order."""
        # Can only sell shares you own
        quantity = min(quantity, position["long"])
        if quantity <= 0:
            return 0
        
        # Calculate realized gain/loss
        cost_basis = position["long_cost_basis"]
        realized_gain = (price - cost_basis) * quantity
        self.portfolio["realized_gains"][ticker]["long"] += realized_gain
        
        # Update position and cash
        position["long"] -= quantity
        self.portfolio["cash"] += quantity * price
        
        # Reset cost basis if no shares left
        if position["long"] == 0:
            position["long_cost_basis"] = 0.0
            
        return quantity

    def _execute_short(self, ticker, quantity, price, position):
        """Execute a short sell order."""
        # Calculate margin required
        proceeds = price * quantity
        margin_required = proceeds * self.margin_ratio
        
        # Check if we have enough cash for margin
        if margin_required > self.portfolio["cash"]:
            # Calculate maximum shortable quantity
            if self.margin_ratio > 0:
                quantity = int(self.portfolio["cash"] / (price * self.margin_ratio))
            else:
                quantity = 0
                
            if quantity <= 0:
                return 0
                
            proceeds = price * quantity
            margin_required = proceeds * self.margin_ratio
        
        # Calculate new weighted average cost basis
        old_shares = position["short"]
        old_cost_basis = position["short_cost_basis"]
        
        if old_shares + quantity > 0:
            old_cost = old_cost_basis * old_shares
            new_cost = price * quantity
            position["short_cost_basis"] = (old_cost + new_cost) / (old_shares + quantity)
        
        # Update position and cash
        position["short"] += quantity
        position["short_margin_used"] += margin_required
        self.portfolio["margin_used"] += margin_required
        
        # Add proceeds and subtract margin
        self.portfolio["cash"] += proceeds - margin_required
        
        return quantity

    def _execute_cover(self, ticker, quantity, price, position):
        """Execute a cover order to close short positions."""
        # Can only cover shares you've shorted
        quantity = min(quantity, position["short"])
        if quantity <= 0:
            return 0
        
        # Calculate realized gain/loss
        cost_basis = position["short_cost_basis"]
        realized_gain = (cost_basis - price) * quantity
        self.portfolio["realized_gains"][ticker]["short"] += realized_gain
        
        # Calculate margin to release
        if position["short"] > 0:
            portion = quantity / position["short"]
        else:
            portion = 1.0
            
        margin_to_release = portion * position["short_margin_used"]
        
        # Update position and margin
        position["short"] -= quantity
        position["short_margin_used"] -= margin_to_release
        self.portfolio["margin_used"] -= margin_to_release
        
        # Update cash: pay cover cost but get back margin
        self.portfolio["cash"] += margin_to_release - (price * quantity)
        
        # Reset if no short position left
        if position["short"] == 0:
            position["short_cost_basis"] = 0.0
            position["short_margin_used"] = 0.0
            
        return quantity

    def calculate_portfolio_value(self, current_prices):
        """Calculate total portfolio value including cash and positions."""
        total_value = self.portfolio["cash"]
        
        for ticker in self.tickers:
            position = self.portfolio["positions"][ticker]
            price = current_prices[ticker]
            
            # Add long position value
            long_value = position["long"] * price
            total_value += long_value
            
            # Add short position P&L
            if position["short"] > 0:
                short_pnl = position["short"] * (position["short_cost_basis"] - price)
                total_value += short_pnl
                
        return total_value

    def prefetch_data(self):
        """Pre-fetch all data needed for the backtest period."""
        print("\nPre-fetching data for the entire backtest period...")
        
        # Fetch data from up to 1 year before end date
        end_date_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        start_date_dt = end_date_dt - relativedelta(years=1)
        extended_start = start_date_dt.strftime("%Y-%m-%d")
        
        for ticker in self.tickers:
            # Fetch all required data types
            get_prices(ticker, extended_start, self.end_date)
            get_financial_metrics(ticker, self.end_date, limit=10)
            get_insider_trades(ticker, self.end_date, start_date=self.start_date, limit=1000)
            get_company_news(ticker, self.end_date, start_date=self.start_date, limit=1000)
            
        print("Data pre-fetch complete.")

    def parse_agent_response(self, agent_output):
        """Parse JSON output from the agent."""
        try:
            return json.loads(agent_output)
        except Exception:
            print(f"Error parsing action: {agent_output}")
            return {"action": "hold", "quantity": 0}

    def run_backtest(self):
        """Run the backtest over the specified time period."""
        # Prefetch data
        self.prefetch_data()
        
        # Setup
        dates = pd.date_range(self.start_date, self.end_date, freq="B")
        table_rows = []
        performance_metrics = {
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'max_drawdown': None,
            'long_short_ratio': None,
            'gross_exposure': None,
            'net_exposure': None
        }
        
        print("\nStarting backtest...")
        
        # Initialize portfolio values with initial capital
        if len(dates) > 0:
            self.portfolio_values = [{"Date": dates[0], "Portfolio Value": self.initial_capital}]
        else:
            self.portfolio_values = []
            
        # Iterate through each trading day
        for current_date in dates:
            # Setup date strings
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")
            previous_date_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Skip first day (no lookback)
            if lookback_start == current_date_str:
                continue
                
            # Get current prices
            try:
                current_prices = {}
                for ticker in self.tickers:
                    price_df = get_price_data(ticker, previous_date_str, current_date_str)
                    current_prices[ticker] = price_df.iloc[-1]["close"]
            except Exception:
                print(f"Error fetching prices for {current_date_str}")
                continue
                
            # Run agent to get trading decisions
            output = self.agent(
                tickers=self.tickers,
                start_date=lookback_start,
                end_date=current_date_str,
                portfolio=self.portfolio,
                model_name=self.model_name,
                model_provider=self.model_provider,
                selected_analysts=self.selected_analysts,
            )
            
            decisions = output["decisions"]
            analyst_signals = output["analyst_signals"]
            
            # Execute trades
            executed_trades = {}
            for ticker in self.tickers:
                decision = decisions.get(ticker, {"action": "hold", "quantity": 0})
                action = decision.get("action", "hold")
                quantity = decision.get("quantity", 0)
                
                executed_qty = self.execute_trade(ticker, action, quantity, current_prices[ticker])
                executed_trades[ticker] = executed_qty
                
            # Calculate portfolio value after trades
            total_value = self.calculate_portfolio_value(current_prices)
            
            # Calculate exposures
            long_exposure = sum(
                self.portfolio["positions"][t]["long"] * current_prices[t]
                for t in self.tickers
            )
            
            short_exposure = sum(
                self.portfolio["positions"][t]["short"] * current_prices[t]
                for t in self.tickers
            )
            
            gross_exposure = long_exposure + short_exposure
            net_exposure = long_exposure - short_exposure
            long_short_ratio = long_exposure / short_exposure if short_exposure > 1e-9 else float('inf')
            
            # Track portfolio values
            self.portfolio_values.append({
                "Date": current_date,
                "Portfolio Value": total_value,
                "Long Exposure": long_exposure,
                "Short Exposure": short_exposure,
                "Gross Exposure": gross_exposure,
                "Net Exposure": net_exposure,
                "Long/Short Ratio": long_short_ratio
            })
            
            # Add rows for each ticker
            date_rows = []
            for ticker in self.tickers:
                # Count signals
                ticker_signals = {
                    agent: signals[ticker]
                    for agent, signals in analyst_signals.items()
                    if ticker in signals
                }
                
                bullish_count = sum(1 for s in ticker_signals.values() if s.get("signal", "").lower() == "bullish")
                bearish_count = sum(1 for s in ticker_signals.values() if s.get("signal", "").lower() == "bearish")
                neutral_count = sum(1 for s in ticker_signals.values() if s.get("signal", "").lower() == "neutral")
                
                # Calculate position value
                pos = self.portfolio["positions"][ticker]
                long_val = pos["long"] * current_prices[ticker]
                short_val = pos["short"] * current_prices[ticker]
                net_position = long_val - short_val
                
                # Get action and quantity
                action = decisions.get(ticker, {}).get("action", "hold")
                quantity = executed_trades.get(ticker, 0)
                
                # Add row
                date_rows.append(
                    format_backtest_row(
                        date=current_date_str,
                        ticker=ticker,
                        action=action,
                        quantity=quantity,
                        price=current_prices[ticker],
                        shares_owned=pos["long"] - pos["short"],
                        position_value=net_position,
                        bullish_count=bullish_count,
                        bearish_count=bearish_count,
                        neutral_count=neutral_count,
                    )
                )
                
            # Calculate portfolio return
            total_realized_gains = sum(
                sum(self.portfolio["realized_gains"][t].values())
                for t in self.tickers
            )
            
            portfolio_return = ((total_value + total_realized_gains) / self.initial_capital - 1) * 100
            
            # Add summary row
            date_rows.append(
                format_backtest_row(
                    date=current_date_str,
                    ticker="",
                    action="",
                    quantity=0,
                    price=0,
                    shares_owned=0,
                    position_value=0,
                    bullish_count=0,
                    bearish_count=0,
                    neutral_count=0,
                    is_summary=True,
                    total_value=total_value,
                    return_pct=portfolio_return,
                    cash_balance=self.portfolio["cash"],
                    total_position_value=total_value - self.portfolio["cash"],
                    sharpe_ratio=performance_metrics["sharpe_ratio"],
                    sortino_ratio=performance_metrics["sortino_ratio"],
                    max_drawdown=performance_metrics["max_drawdown"],
                )
            )
            
            # Update display
            table_rows.extend(date_rows)
            print_backtest_results(table_rows)
            
            # Update performance metrics
            if len(self.portfolio_values) > 3:
                self._update_performance_metrics(performance_metrics)
                
        return performance_metrics

    def _update_performance_metrics(self, performance_metrics):
        """Update performance metrics based on portfolio history."""
        # Create dataframe from portfolio values
        values_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        values_df["Daily Return"] = values_df["Portfolio Value"].pct_change()
        returns = values_df["Daily Return"].dropna()
        
        if len(returns) < 2:
            return  # Not enough data
            
        # Constants
        daily_rf_rate = 0.0434 / 252  # Risk-free rate
        
        # Calculate excess returns
        excess_returns = returns - daily_rf_rate
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()
        
        # Sharpe ratio
        if std_excess > 1e-12:
            performance_metrics["sharpe_ratio"] = np.sqrt(252) * (mean_excess / std_excess)
        else:
            performance_metrics["sharpe_ratio"] = 0.0
            
        # Sortino ratio
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std()
            if downside_std > 1e-12:
                performance_metrics["sortino_ratio"] = np.sqrt(252) * (mean_excess / downside_std)
            else:
                performance_metrics["sortino_ratio"] = float('inf') if mean_excess > 0 else 0
        else:
            performance_metrics["sortino_ratio"] = float('inf') if mean_excess > 0 else 0
            
        # Maximum drawdown
        rolling_max = values_df["Portfolio Value"].cummax()
        drawdown = (values_df["Portfolio Value"] - rolling_max) / rolling_max
        performance_metrics["max_drawdown"] = drawdown.min() * 100

    def analyze_performance(self):
        """Analyze and display backtest performance."""
        if not self.portfolio_values:
            print("No portfolio data found. Please run the backtest first.")
            return pd.DataFrame()
            
        # Create performance dataframe
        perf_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        if perf_df.empty:
            print("No valid performance data to analyze.")
            return perf_df
            
        # Calculate overall performance
        final_value = perf_df["Portfolio Value"].iloc[-1]
        total_gains = sum(
            sum(self.portfolio["realized_gains"][ticker].values())
            for ticker in self.tickers
        )
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Print summary
        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO PERFORMANCE SUMMARY:{Style.RESET_ALL}")
        print(f"Total Return: {Fore.GREEN if total_return >= 0 else Fore.RED}{total_return:.2f}%{Style.RESET_ALL}")
        print(f"Total Realized Gains/Losses: {Fore.GREEN if total_gains >= 0 else Fore.RED}${total_gains:,.2f}{Style.RESET_ALL}")
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(perf_df.index, perf_df["Portfolio Value"], color="blue")
        plt.title("Portfolio Value Over Time")
        plt.ylabel("Portfolio Value ($)")
        plt.xlabel("Date")
        plt.grid(True)
        plt.show()
        
        # Calculate performance metrics
        perf_df["Daily Return"] = perf_df["Portfolio Value"].pct_change().fillna(0)
        daily_rf = 0.0434 / 252
        mean_daily_return = perf_df["Daily Return"].mean()
        std_daily_return = perf_df["Daily Return"].std()
        
        # Sharpe ratio
        if std_daily_return != 0:
            sharpe = np.sqrt(252) * ((mean_daily_return - daily_rf) / std_daily_return)
        else:
            sharpe = 0
            
        print(f"\nSharpe Ratio: {Fore.YELLOW}{sharpe:.2f}{Style.RESET_ALL}")
        
        # Max drawdown
        rolling_max = perf_df["Portfolio Value"].cummax()
        drawdown = (perf_df["Portfolio Value"] - rolling_max) / rolling_max
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        if pd.notnull(max_dd_date):
            print(f"Maximum Drawdown: {Fore.RED}{max_dd * 100:.2f}%{Style.RESET_ALL} (on {max_dd_date.strftime('%Y-%m-%d')})")
        else:
            print(f"Maximum Drawdown: {Fore.RED}0.00%{Style.RESET_ALL}")
            
        # Win rate
        winning_days = len(perf_df[perf_df["Daily Return"] > 0])
        total_days = max(len(perf_df) - 1, 1)
        win_rate = (winning_days / total_days) * 100
        print(f"Win Rate: {Fore.GREEN}{win_rate:.2f}%{Style.RESET_ALL}")
        
        # Win/Loss ratio
        positive_returns = perf_df[perf_df["Daily Return"] > 0]["Daily Return"]
        negative_returns = perf_df[perf_df["Daily Return"] < 0]["Daily Return"]
        avg_win = positive_returns.mean() if not positive_returns.empty else 0
        avg_loss = abs(negative_returns.mean()) if not negative_returns.empty else 0
        
        win_loss = avg_win / avg_loss if avg_loss != 0 else (float('inf') if avg_win > 0 else 0)
        print(f"Win/Loss Ratio: {Fore.GREEN}{win_loss:.2f}{Style.RESET_ALL}")
        
        # Consecutive wins/losses
        returns_binary = (perf_df["Daily Return"] > 0).astype(int)
        if len(returns_binary) > 0:
            max_cons_wins = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 1), default=0)
            max_cons_losses = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 0), default=0)
        else:
            max_cons_wins = 0
            max_cons_losses = 0
            
        print(f"Max Consecutive Wins: {Fore.GREEN}{max_cons_wins}{Style.RESET_ALL}")
        print(f"Max Consecutive Losses: {Fore.RED}{max_cons_losses}{Style.RESET_ALL}")
        
        return perf_df


# Main execution
if __name__ == "__main__":
    import argparse
    import json
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run backtesting simulation")
    parser.add_argument(
        "--tickers",
        type=str,
        required=False,
        help="Comma-separated list of stock ticker symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - relativedelta(months=1)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000,
        help="Initial capital amount (default: 100000)",
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help="Margin ratio for short positions, e.g. 0.5 for 50% (default: 0.0)",
    )
    
    args = parser.parse_args()
    
    # Parse tickers
    tickers = [ticker.strip() for ticker in args.tickers.split(",")] if args.tickers else []
    
    # Select analysts
    selected_analysts = questionary.checkbox(
        "Use the Space bar to select/unselect analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\nPress 'a' to toggle all.\n\nPress Enter when done to run the hedge fund.",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style([
            ("checkbox-selected", "fg:green"),
            ("selected", "fg:green noinherit"),
            ("highlighted", "noinherit"),
            ("pointer", "noinherit"),
        ]),
    ).ask()
    
    if not selected_analysts:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
        
    print(f"\nSelected analysts: "
        f"{', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in selected_analysts)}")
    
    # Select LLM model
    model_choice = questionary.select(
        "Select your LLM model:",
        choices=[questionary.Choice(display, value=value) for display, value, _ in LLM_ORDER],
        style=questionary.Style([
            ("selected", "fg:green bold"),
            ("pointer", "fg:green bold"),
            ("highlighted", "fg:green"),
            ("answer", "fg:green bold"),
        ])
    ).ask()
    
    if not model_choice:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
        
    model_info = get_model_info(model_choice)
    if model_info:
        model_provider = model_info.provider.value
        print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
    else:
        model_provider = "Unknown"
        print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
    
    # Run backtest
    backtester = Backtester(
        agent=run_hedge_fund,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        model_name=model_choice,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
        initial_margin_requirement=args.margin_requirement,
    )
    
    performance_metrics = backtester.run_backtest()
    performance_df = backtester.analyze_performance()