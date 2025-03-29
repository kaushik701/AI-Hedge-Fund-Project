import math
from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
import json
import pandas as pd
import numpy as np
from tools.api import get_prices, prices_to_df
from util.progress import progress


def technical_analyst_agent(state: AgentState):
    """Analyzes price data using multiple technical strategies."""
    data = state["data"]
    tickers = data["tickers"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    
    # Initialize results dictionary
    technical_analysis = {}
    
    for ticker in tickers:
        progress.update_status("technical_analyst_agent", ticker, "Fetching price data")
        
        # Get price data
        prices = get_prices(ticker, start_date, end_date)
        
        if not prices:
            progress.update_status("technical_analyst_agent", ticker, "Failed: No price data found")
            continue
        
        # Convert to DataFrame
        prices_df = prices_to_df(prices)
        
        # Run analysis strategies
        progress.update_status("technical_analyst_agent", ticker, "Running technical analysis")
        trend_signals = analyze_trend(prices_df)
        mean_reversion_signals = analyze_mean_reversion(prices_df)
        momentum_signals = analyze_momentum(prices_df)
        volatility_signals = analyze_volatility(prices_df)
        stat_arb_signals = analyze_statistical(prices_df)
        
        # Combine signals with weights
        progress.update_status("technical_analyst_agent", ticker, "Combining signals")
        strategy_weights = {
            "trend": 0.25,
            "mean_reversion": 0.20,
            "momentum": 0.25,
            "volatility": 0.15,
            "stat_arb": 0.15
        }
        
        combined_signal = combine_signals(
            {
                "trend": trend_signals,
                "mean_reversion": mean_reversion_signals,
                "momentum": momentum_signals,
                "volatility": volatility_signals,
                "stat_arb": stat_arb_signals
            },
            strategy_weights
        )
        
        # Store results
        technical_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "strategy_signals": {
                "trend_following": {
                    "signal": trend_signals["signal"],
                    "confidence": round(trend_signals["confidence"] * 100),
                    "metrics": convert_to_json(trend_signals["metrics"])
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": round(mean_reversion_signals["confidence"] * 100),
                    "metrics": convert_to_json(mean_reversion_signals["metrics"])
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": round(momentum_signals["confidence"] * 100),
                    "metrics": convert_to_json(momentum_signals["metrics"])
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": round(volatility_signals["confidence"] * 100),
                    "metrics": convert_to_json(volatility_signals["metrics"])
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": round(stat_arb_signals["confidence"] * 100),
                    "metrics": convert_to_json(stat_arb_signals["metrics"])
                }
            }
        }
        
        progress.update_status("technical_analyst_agent", ticker, "Analysis complete")
    
    # Create message with results
    message = HumanMessage(content=json.dumps(technical_analysis), name="technical_analyst_agent")
    
    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(technical_analysis, "Technical Analyst")
    
    # Store analysis in state
    state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis
    
    return {
        "messages": state["messages"] + [message],
        "data": data
    }


def analyze_trend(prices_df):
    """Analyzes trend using EMAs and ADX."""
    # Calculate EMAs
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)
    
    # Calculate ADX
    adx = calculate_adx(prices_df, 14)
    
    # Determine trend direction
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55
    
    # Get trend strength
    trend_strength = adx["adx"].iloc[-1] / 100.0
    
    # Generate signal
    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "bullish"
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5
    
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": float(adx["adx"].iloc[-1]),
            "trend_strength": float(trend_strength)
        }
    }


def analyze_mean_reversion(prices_df):
    """Analyzes mean reversion using Bollinger Bands and RSI."""
    # Calculate z-score
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50
    
    # Calculate Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)
    
    # Calculate RSI
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)
    
    # Calculate position relative to bands
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
    
    # Generate signal
    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = "bullish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = "bearish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5
    
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": float(z_score.iloc[-1]),
            "price_vs_bb": float(price_vs_bb),
            "rsi_14": float(rsi_14.iloc[-1]),
            "rsi_28": float(rsi_28.iloc[-1])
        }
    }


def analyze_momentum(prices_df):
    """Analyzes price and volume momentum."""
    # Calculate returns
    returns = prices_df["close"].pct_change()
    
    # Calculate momentum over different timeframes
    mom_1m = returns.rolling(21).sum()  # ~1 month
    mom_3m = returns.rolling(63).sum()  # ~3 months
    mom_6m = returns.rolling(126).sum() # ~6 months
    
    # Calculate volume momentum
    volume_ma = prices_df["volume"].rolling(21).mean()
    volume_momentum = prices_df["volume"] / volume_ma
    
    # Calculate combined momentum score
    momentum_score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m).iloc[-1]
    
    # Check volume confirmation
    volume_confirmation = volume_momentum.iloc[-1] > 1.0
    
    # Generate signal
    if momentum_score > 0.05 and volume_confirmation:
        signal = "bullish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal = "bearish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5
    
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": float(mom_1m.iloc[-1]),
            "momentum_3m": float(mom_3m.iloc[-1]),
            "momentum_6m": float(mom_6m.iloc[-1]),
            "volume_momentum": float(volume_momentum.iloc[-1])
        }
    }


def analyze_volatility(prices_df):
    """Analyzes volatility patterns for trading signals."""
    # Calculate returns
    returns = prices_df["close"].pct_change()
    
    # Calculate historical volatility (annualized)
    hist_vol = returns.rolling(21).std() * math.sqrt(252)
    
    # Calculate volatility moving average
    vol_ma = hist_vol.rolling(63).mean()
    
    # Calculate volatility regime
    vol_regime = hist_vol / vol_ma
    
    # Calculate volatility z-score
    vol_std = hist_vol.rolling(63).std()
    vol_z_score = (hist_vol - vol_ma) / vol_std
    
    # Calculate ATR ratio
    atr = calculate_atr(prices_df)
    atr_ratio = atr / prices_df["close"]
    
    # Generate signal
    current_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]
    
    if current_regime < 0.8 and vol_z < -1:
        signal = "bullish"  # Low volatility regime
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_regime > 1.2 and vol_z > 1:
        signal = "bearish"  # High volatility regime
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5
    
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": float(hist_vol.iloc[-1]),
            "volatility_regime": float(current_regime),
            "volatility_z_score": float(vol_z),
            "atr_ratio": float(atr_ratio.iloc[-1])
        }
    }


def analyze_statistical(prices_df):
    """Analyzes statistical properties of price series."""
    # Calculate returns
    returns = prices_df["close"].pct_change()
    
    # Calculate skewness and kurtosis
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()
    
    # Calculate Hurst exponent
    hurst = calculate_hurst_exponent(prices_df["close"])
    
    # Generate signal
    if hurst < 0.4 and skew.iloc[-1] > 1:
        signal = "bullish"
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew.iloc[-1] < -1:
        signal = "bearish"
        confidence = (0.5 - hurst) * 2
    else:
        signal = "neutral"
        confidence = 0.5
    
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": float(hurst),
            "skewness": float(skew.iloc[-1]),
            "kurtosis": float(kurt.iloc[-1])
        }
    }


def combine_signals(signals, weights):
    """Combines multiple signals with weighted approach."""
    # Convert signals to numeric values
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}
    
    weighted_sum = 0
    total_confidence = 0
    
    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        weight = weights[strategy]
        confidence = signal["confidence"]
        
        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence
    
    # Normalize the weighted sum
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0
    
    # Convert back to signal
    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"
    
    return {"signal": signal, "confidence": abs(final_score)}


def convert_to_json(obj):
    """Convert pandas objects to JSON-serializable types."""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: convert_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json(item) for item in obj]
    return obj


def calculate_rsi(prices_df, period=14):
    """Calculate Relative Strength Index."""
    delta = prices_df["close"].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(prices_df, window=20):
    """Calculate Bollinger Bands."""
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    
    return upper_band, lower_band


def calculate_ema(prices_df, window):
    """Calculate Exponential Moving Average."""
    return prices_df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(prices_df, period=14):
    """Calculate Average Directional Index."""
    # Calculate True Range
    prices_df["high_low"] = prices_df["high"] - prices_df["low"]
    prices_df["high_close"] = abs(prices_df["high"] - prices_df["close"].shift())
    prices_df["low_close"] = abs(prices_df["low"] - prices_df["close"].shift())
    prices_df["tr"] = prices_df[["high_low", "high_close", "low_close"]].max(axis=1)
    
    # Calculate Directional Movement
    prices_df["up_move"] = prices_df["high"] - prices_df["high"].shift()
    prices_df["down_move"] = prices_df["low"].shift() - prices_df["low"]
    
    prices_df["plus_dm"] = np.where(
        (prices_df["up_move"] > prices_df["down_move"]) & (prices_df["up_move"] > 0),
        prices_df["up_move"],
        0
    )
    prices_df["minus_dm"] = np.where(
        (prices_df["down_move"] > prices_df["up_move"]) & (prices_df["down_move"] > 0),
        prices_df["down_move"],
        0
    )
    
    # Calculate ADX components
    prices_df["+di"] = 100 * (prices_df["plus_dm"].ewm(span=period).mean() / 
                            prices_df["tr"].ewm(span=period).mean())
    prices_df["-di"] = 100 * (prices_df["minus_dm"].ewm(span=period).mean() / 
                            prices_df["tr"].ewm(span=period).mean())
    
    # Calculate DX and ADX
    prices_df["dx"] = 100 * abs(prices_df["+di"] - prices_df["-di"]) / (prices_df["+di"] + prices_df["-di"])
    prices_df["adx"] = prices_df["dx"].ewm(span=period).mean()
    
    return prices_df[["adx", "+di", "-di"]]


def calculate_atr(prices_df, period=14):
    """Calculate Average True Range."""
    high_low = prices_df["high"] - prices_df["low"]
    high_close = abs(prices_df["high"] - prices_df["close"].shift())
    low_close = abs(prices_df["low"] - prices_df["close"].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    return true_range.rolling(period).mean()


def calculate_hurst_exponent(price_series, max_lag=20):
    """Calculate Hurst Exponent for time series memory."""
    lags = range(2, max_lag)
    
    # Calculate tau (to avoid log(0), add small epsilon)
    tau = [
        max(1e-8, np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag]))))
        for lag in lags
    ]
    
    # Calculate Hurst exponent through linear regression
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]  # Slope is the Hurst exponent
    except (ValueError, RuntimeWarning):
        return 0.5  # Return random walk value if calculation fails