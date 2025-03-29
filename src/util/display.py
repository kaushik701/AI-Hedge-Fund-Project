from colorama import Fore, Style
from tabulate import tabulate
from .analysts import ANALYST_ORDER
import os
import json


def sort_agent_signals(signals):
    """Sort agent signals in a consistent order."""
    # Create order mapping from ANALYST_ORDER
    analyst_order = {display: idx for idx, (display, _) in enumerate(ANALYST_ORDER)}
    analyst_order["Risk Management"] = len(ANALYST_ORDER)  # Add Risk Management at the end
    
    return sorted(signals, key=lambda x: analyst_order.get(x[0], 999))


def wrap_text(text, max_length=60):
    """Wrap text to a maximum line length."""
    if not text:
        return ""
        
    wrapped = []
    current_line = ""
    
    for word in str(text).split():
        if len(current_line) + len(word) + 1 > max_length:
            wrapped.append(current_line)
            current_line = word
        else:
            current_line = word if not current_line else current_line + " " + word
            
    if current_line:
        wrapped.append(current_line)
        
    return "\n".join(wrapped)


def format_reasoning(reasoning):
    """Format reasoning data into a readable string."""
    if isinstance(reasoning, str):
        return reasoning
    elif isinstance(reasoning, dict):
        return json.dumps(reasoning, indent=2)
    else:
        return str(reasoning)


def print_trading_output(result):
    """Print formatted trading results for multiple tickers."""
    decisions = result.get("decisions", {})
    if not decisions:
        print(f"{Fore.RED}No trading decisions available{Style.RESET_ALL}")
        return

    for ticker, decision in decisions.items():
        # Print header
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Analysis for {Fore.CYAN}{ticker}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{Style.BRIGHT}{'=' * 50}{Style.RESET_ALL}")
        
        # Prepare analyst signals
        table_data = []
        for agent, signals in result.get("analyst_signals", {}).items():
            if ticker not in signals or agent == "risk_management_agent":
                continue
                
            signal = signals[ticker]
            agent_name = agent.replace("_agent", "").replace("_", " ").title()
            signal_type = signal.get("signal", "").upper()
            confidence = signal.get("confidence", 0)
            
            # Set signal color
            signal_color = {
                "BULLISH": Fore.GREEN,
                "BEARISH": Fore.RED,
                "NEUTRAL": Fore.YELLOW
            }.get(signal_type, Fore.WHITE)
            
            # Format reasoning
            if "reasoning" in signal and signal["reasoning"]:
                reasoning_text = format_reasoning(signal["reasoning"])
                wrapped_reasoning = wrap_text(reasoning_text)
            else:
                wrapped_reasoning = ""
            
            table_data.append([
                f"{Fore.CYAN}{agent_name}{Style.RESET_ALL}",
                f"{signal_color}{signal_type}{Style.RESET_ALL}",
                f"{Fore.WHITE}{confidence}%{Style.RESET_ALL}",
                f"{Fore.WHITE}{wrapped_reasoning}{Style.RESET_ALL}"
            ])
        
        # Sort and print agent analysis
        table_data = sort_agent_signals(table_data)
        print(f"\n{Fore.WHITE}{Style.BRIGHT}AGENT ANALYSIS:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]")
        print(tabulate(
            table_data,
            headers=[f"{Fore.WHITE}Agent", "Signal", "Confidence", "Reasoning"],
            tablefmt="grid",
            colalign=("left", "center", "right", "left")
        ))
        
        # Format trading decision
        action = decision.get("action", "").upper()
        action_color = {
            "BUY": Fore.GREEN,
            "SELL": Fore.RED,
            "HOLD": Fore.YELLOW,
            "COVER": Fore.GREEN,
            "SHORT": Fore.RED
        }.get(action, Fore.WHITE)
        
        wrapped_reasoning = wrap_text(decision.get("reasoning", ""))
        
        decision_data = [
            ["Action", f"{action_color}{action}{Style.RESET_ALL}"],
            ["Quantity", f"{action_color}{decision.get('quantity')}{Style.RESET_ALL}"],
            ["Confidence", f"{Fore.WHITE}{decision.get('confidence'):.1f}%{Style.RESET_ALL}"],
            ["Reasoning", f"{Fore.WHITE}{wrapped_reasoning}{Style.RESET_ALL}"]
        ]
        
        # Print trading decision
        print(f"\n{Fore.WHITE}{Style.BRIGHT}TRADING DECISION:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]")
        print(tabulate(decision_data, tablefmt="grid", colalign=("left", "left")))
    
    # Print portfolio summary
    print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY:{Style.RESET_ALL}")
    
    # Extract portfolio manager reasoning from any ticker
    portfolio_reasoning = None
    for decision in decisions.values():
        if decision.get("reasoning"):
            portfolio_reasoning = decision.get("reasoning")
            break
    
    # Format portfolio data
    portfolio_data = []
    for ticker, decision in decisions.items():
        action = decision.get("action", "").upper()
        action_color = {
            "BUY": Fore.GREEN,
            "SELL": Fore.RED,
            "HOLD": Fore.YELLOW,
            "COVER": Fore.GREEN,
            "SHORT": Fore.RED
        }.get(action, Fore.WHITE)
        
        portfolio_data.append([
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{action_color}{action}{Style.RESET_ALL}",
            f"{action_color}{decision.get('quantity')}{Style.RESET_ALL}",
            f"{Fore.WHITE}{decision.get('confidence'):.1f}%{Style.RESET_ALL}"
        ])
    
    # Print portfolio table
    print(tabulate(
        portfolio_data,
        headers=[f"{Fore.WHITE}Ticker", "Action", "Quantity", "Confidence"],
        tablefmt="grid",
        colalign=("left", "center", "right", "right")
    ))
    
    # Print portfolio strategy if available
    if portfolio_reasoning:
        wrapped_reasoning = wrap_text(format_reasoning(portfolio_reasoning))
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Portfolio Strategy:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{wrapped_reasoning}{Style.RESET_ALL}")


def print_backtest_results(table_rows):
    """Print backtest results in a formatted table."""
    # Clear screen
    os.system("cls" if os.name == "nt" else "clear")
    
    # Split rows into ticker and summary rows
    ticker_rows = []
    summary_rows = []
    
    for row in table_rows:
        if isinstance(row[1], str) and "PORTFOLIO SUMMARY" in row[1]:
            summary_rows.append(row)
        else:
            ticker_rows.append(row)
    
    # Display latest portfolio summary
    if summary_rows:
        latest = summary_rows[-1]
        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY:{Style.RESET_ALL}")
        
        # Extract values 
        cash_str = latest[7].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        position_str = latest[6].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        total_str = latest[8].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        
        # Print portfolio metrics
        print(f"Cash Balance: {Fore.CYAN}${float(cash_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Position Value: {Fore.YELLOW}${float(position_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Value: {Fore.WHITE}${float(total_str):,.2f}{Style.RESET_ALL}")
        print(f"Return: {latest[9]}")
        
        # Print performance metrics if available
        if latest[10]:  # Sharpe ratio
            print(f"Sharpe Ratio: {latest[10]}")
        if latest[11]:  # Sortino ratio
            print(f"Sortino Ratio: {latest[11]}")
        if latest[12]:  # Max drawdown
            print(f"Max Drawdown: {latest[12]}")
    
    # Print ticker table
    print("\n\n")
    print(tabulate(
        ticker_rows,
        headers=[
            "Date", "Ticker", "Action", "Quantity", "Price", "Shares", 
            "Position Value", "Bullish", "Bearish", "Neutral"
        ],
        tablefmt="grid",
        colalign=("left", "left", "center", "right", "right", "right", 
                "right", "right", "right", "right")
    ))
    print("\n\n\n\n")


def format_backtest_row(
    date, ticker, action, quantity, price, shares_owned, position_value,
    bullish_count, bearish_count, neutral_count, is_summary=False,
    total_value=None, return_pct=None, cash_balance=None, 
    total_position_value=None, sharpe_ratio=None, sortino_ratio=None, 
    max_drawdown=None
):
    """Format a row for the backtest results table."""
    # Set action color
    action_color = {
        "BUY": Fore.GREEN,
        "COVER": Fore.GREEN,
        "SELL": Fore.RED,
        "SHORT": Fore.RED,
        "HOLD": Fore.WHITE
    }.get(action.upper(), Fore.WHITE)
    
    if is_summary:
        # Format summary row
        return_color = Fore.GREEN if return_pct >= 0 else Fore.RED
        
        return [
            date,
            f"{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY{Style.RESET_ALL}",
            "",  # Action
            "",  # Quantity
            "",  # Price
            "",  # Shares
            f"{Fore.YELLOW}${total_position_value:,.2f}{Style.RESET_ALL}",
            f"{Fore.CYAN}${cash_balance:,.2f}{Style.RESET_ALL}",
            f"{Fore.WHITE}${total_value:,.2f}{Style.RESET_ALL}",
            f"{return_color}{return_pct:+.2f}%{Style.RESET_ALL}",
            f"{Fore.YELLOW}{sharpe_ratio:.2f}{Style.RESET_ALL}" if sharpe_ratio is not None else "",
            f"{Fore.YELLOW}{sortino_ratio:.2f}{Style.RESET_ALL}" if sortino_ratio is not None else "",
            f"{Fore.RED}{max_drawdown:.2f}%{Style.RESET_ALL}" if max_drawdown is not None else ""
        ]
    else:
        # Format ticker row
        return [
            date,
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{action_color}{action.upper()}{Style.RESET_ALL}",
            f"{action_color}{quantity:,.0f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{price:,.2f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{shares_owned:,.0f}{Style.RESET_ALL}",
            f"{Fore.YELLOW}{position_value:,.2f}{Style.RESET_ALL}",
            f"{Fore.GREEN}{bullish_count}{Style.RESET_ALL}",
            f"{Fore.RED}{bearish_count}{Style.RESET_ALL}",
            f"{Fore.BLUE}{neutral_count}{Style.RESET_ALL}"
        ]