from pydantic import BaseModel, Field

# Price data models
class Price(BaseModel):
    open: float
    close: float
    high: float
    low: float
    volume: int
    time: str

class PriceResponse(BaseModel):
    ticker: str
    prices: list[Price]

# Financial metrics models
class FinancialMetrics(BaseModel):
    # Basic information
    ticker: str
    report_period: str
    period: str
    currency: str
    
    # Valuation metrics
    market_cap: float = None
    enterprise_value: float = None
    price_to_earnings_ratio: float = None
    price_to_book_ratio: float = None
    price_to_sales_ratio: float = None
    enterprise_value_to_ebitda_ratio: float = None
    enterprise_value_to_revenue_ratio: float = None
    free_cash_flow_yield: float = None
    peg_ratio: float = None
    
    # Profitability metrics
    gross_margin: float = None
    operating_margin: float = None
    net_margin: float = None
    return_on_equity: float = None
    return_on_assets: float = None
    return_on_invested_capital: float = None
    
    # Efficiency metrics
    asset_turnover: float = None
    inventory_turnover: float = None
    receivables_turnover: float = None
    days_sales_outstanding: float = None
    operating_cycle: float = None
    working_capital_turnover: float = None
    
    # Liquidity metrics
    current_ratio: float = None
    quick_ratio: float = None
    cash_ratio: float = None
    operating_cash_flow_ratio: float = None
    
    # Leverage metrics
    debt_to_equity: float = None
    debt_to_assets: float = None
    interest_coverage: float = None
    
    # Growth metrics
    revenue_growth: float = None
    earnings_growth: float = None
    book_value_growth: float = None
    earnings_per_share_growth: float = None
    free_cash_flow_growth: float = None
    operating_income_growth: float = None
    ebitda_growth: float = None
    
    # Per-share metrics
    payout_ratio: float = None
    earnings_per_share: float = None
    book_value_per_share: float = None
    free_cash_flow_per_share: float = None

class FinancialMetricsResponse(BaseModel):
    financial_metrics: list[FinancialMetrics]

# Financial line item models
class LineItem(BaseModel):
    ticker: str
    report_period: str
    period: str
    currency: str
    
    model_config = {"extra": "allow"}

class LineItemResponse(BaseModel):
    search_results: list[LineItem]

# Insider trading models
class InsiderTrade(BaseModel):
    ticker: str
    filing_date: str
    
    # Person information
    issuer: str = None
    name: str = None
    title: str = None
    is_board_director: bool = None
    security_title: str = None
    
    # Transaction details
    transaction_date: str = None
    transaction_shares: float = None
    transaction_price_per_share: float = None
    transaction_value: float = None
    shares_owned_before_transaction: float = None
    shares_owned_after_transaction: float = None

class InsiderTradeResponse(BaseModel):
    insider_trades: list[InsiderTrade]

# News models
class CompanyNews(BaseModel):
    ticker: str
    title: str
    author: str
    source: str
    date: str
    url: str
    sentiment: str = None

class CompanyNewsResponse(BaseModel):
    news: list[CompanyNews]

# Portfolio models
class Position(BaseModel):
    ticker: str
    cash: float = 0.0
    shares: int = 0

class Portfolio(BaseModel):
    positions: dict[str, Position]
    total_cash: float = 0.0

# Analysis models
class AnalystSignal(BaseModel):
    signal: str = None
    confidence: float = None
    reasoning: dict | str = None
    max_position_size: float = None

class TickerAnalysis(BaseModel):
    ticker: str
    analyst_signals: dict[str, AnalystSignal]

class AgentStateData(BaseModel):
    tickers: list[str]
    portfolio: Portfolio
    start_date: str
    end_date: str
    ticker_analyses: dict[str, TickerAnalysis]

class AgentStateMetadata(BaseModel):
    show_reasoning: bool = False
    model_config = {"extra": "allow"}