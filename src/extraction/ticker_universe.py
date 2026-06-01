"""Bundled US-equity ticker universe — symbol → headline company name.

This is the single source of truth for which companies MarketPulse can recognize
in news text. `TickerExtractor` builds its cashtag/bare-ticker map and its
company-name aliases from here, and `TickerSentimentAnalyzer` inverts it to map a
company back to its symbol. Add a row here and the whole pipeline picks it up.

Names are the form that actually appears in headlines (e.g. "Nvidia", "Eli Lilly",
"JPMorgan") so company-name matching — the dominant signal in news — works well.
"""

# symbol -> canonical company name (as written in headlines)
TICKER_UNIVERSE: dict[str, str] = {
    # Mega-cap tech
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOG": "Google", "GOOGL": "Google",
    "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta", "TSLA": "Tesla",
    "AVGO": "Broadcom", "ORCL": "Oracle", "CRM": "Salesforce", "ADBE": "Adobe",
    "AMD": "AMD", "INTC": "Intel", "QCOM": "Qualcomm", "TXN": "Texas Instruments",
    "CSCO": "Cisco", "IBM": "IBM", "NOW": "ServiceNow", "INTU": "Intuit",
    "AMAT": "Applied Materials", "MU": "Micron", "LRCX": "Lam Research",
    "ADI": "Analog Devices", "PANW": "Palo Alto Networks", "SNPS": "Synopsys",
    "CDNS": "Cadence", "KLAC": "KLA", "MRVL": "Marvell", "FTNT": "Fortinet",
    "DELL": "Dell", "HPQ": "HP", "SMCI": "Super Micro", "ARM": "Arm",
    # Communications / media / internet
    "NFLX": "Netflix", "DIS": "Disney", "CMCSA": "Comcast", "T": "AT&T",
    "VZ": "Verizon", "TMUS": "T-Mobile", "SNAP": "Snap", "PINS": "Pinterest",
    "SPOT": "Spotify", "RBLX": "Roblox", "UBER": "Uber", "LYFT": "Lyft",
    "ABNB": "Airbnb", "DASH": "DoorDash", "SHOP": "Shopify", "SQ": "Block",
    "PYPL": "PayPal", "COIN": "Coinbase", "HOOD": "Robinhood", "SOFI": "SoFi",
    "PLTR": "Palantir", "SNOW": "Snowflake", "DDOG": "Datadog", "NET": "Cloudflare",
    "CRWD": "CrowdStrike", "ZS": "Zscaler", "MDB": "MongoDB", "U": "Unity",
    # Financials
    "JPM": "JPMorgan", "BAC": "Bank of America", "WFC": "Wells Fargo",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "C": "Citigroup",
    "SCHW": "Charles Schwab", "BLK": "BlackRock", "AXP": "American Express",
    "V": "Visa", "MA": "Mastercard", "BX": "Blackstone", "KKR": "KKR",
    # Healthcare / pharma
    "UNH": "UnitedHealth", "LLY": "Eli Lilly", "JNJ": "Johnson & Johnson",
    "MRK": "Merck", "ABBV": "AbbVie", "PFE": "Pfizer", "TMO": "Thermo Fisher",
    "ABT": "Abbott", "DHR": "Danaher", "BMY": "Bristol Myers", "AMGN": "Amgen",
    "GILD": "Gilead", "CVS": "CVS Health", "MRNA": "Moderna", "ISRG": "Intuitive Surgical",
    # Consumer
    "WMT": "Walmart", "COST": "Costco", "HD": "Home Depot", "LOW": "Lowe's",
    "TGT": "Target", "NKE": "Nike", "SBUX": "Starbucks", "MCD": "McDonald's",
    "KO": "Coca-Cola", "PEP": "PepsiCo", "PG": "Procter & Gamble", "PM": "Philip Morris",
    "MDLZ": "Mondelez", "CL": "Colgate", "EL": "Estee Lauder", "LULU": "Lululemon",
    "CMG": "Chipotle", "MAR": "Marriott", "BKNG": "Booking", "F": "Ford",
    "GM": "General Motors", "RIVN": "Rivian", "LCID": "Lucid", "NIO": "NIO",
    # Industrials / energy / materials
    "XOM": "Exxon", "CVX": "Chevron", "COP": "ConocoPhillips", "SLB": "Schlumberger",
    "OXY": "Occidental", "BA": "Boeing", "CAT": "Caterpillar", "DE": "Deere",
    "GE": "GE", "HON": "Honeywell", "RTX": "RTX", "LMT": "Lockheed Martin",
    "UPS": "UPS", "FDX": "FedEx", "MMM": "3M", "EMR": "Emerson",
    "NEE": "NextEra", "DUK": "Duke Energy", "LIN": "Linde", "FCX": "Freeport",
    # Meme / retail / high-beta
    "GME": "GameStop", "AMC": "AMC", "BB": "BlackBerry", "BBBY": "Bed Bath & Beyond",
    "TLRY": "Tilray", "DKNG": "DraftKings", "CHWY": "Chewy", "CVNA": "Carvana",
    "AFRM": "Affirm", "UPST": "Upstart", "FUBO": "fuboTV", "WISH": "ContextLogic",
    "BABA": "Alibaba", "TSM": "TSMC", "NU": "Nu Holdings", "MARA": "Marathon Digital",
    "RIOT": "Riot Platforms", "MSTR": "MicroStrategy",
    # Additional large/mid-caps frequently in the news
    "PNC": "PNC", "USB": "US Bancorp", "TFC": "Truist",
    "SPGI": "S&P Global", "ICE": "Intercontinental Exchange", "CME": "CME Group",
    "PGR": "Progressive", "ALL": "Allstate", "MET": "MetLife", "AIG": "AIG",
    "CB": "Chubb", "MMC": "Marsh McLennan", "AON": "Aon",
    "ACN": "Accenture", "WDAY": "Workday", "TEAM": "Atlassian",
    "ZM": "Zoom", "DOCU": "DocuSign", "OKTA": "Okta", "TWLO": "Twilio",
    "ROKU": "Roku", "PARA": "Paramount", "WBD": "Warner Bros Discovery",
    "EA": "Electronic Arts", "TTWO": "Take-Two",
    "MGM": "MGM Resorts", "WYNN": "Wynn", "LVS": "Las Vegas Sands", "CCL": "Carnival",
    "RCL": "Royal Caribbean", "DAL": "Delta", "UAL": "United Airlines",
    "AAL": "American Airlines", "LUV": "Southwest",
    "GD": "General Dynamics", "NOC": "Northrop Grumman", "TXT": "Textron",
    "KMB": "Kimberly-Clark", "GIS": "General Mills", "K": "Kellanova",
    "HSY": "Hershey", "KHC": "Kraft Heinz", "STZ": "Constellation Brands",
    "MNST": "Monster Beverage", "KDP": "Keurig Dr Pepper", "SYY": "Sysco",
    "KR": "Kroger", "DG": "Dollar General", "DLTR": "Dollar Tree", "BBY": "Best Buy",
    "ROST": "Ross Stores", "TJX": "TJX", "ULTA": "Ulta", "DECK": "Deckers",
    "VLO": "Valero", "MPC": "Marathon Petroleum", "PSX": "Phillips 66",
    "KMI": "Kinder Morgan", "WMB": "Williams", "HAL": "Halliburton", "DVN": "Devon",
    "NEM": "Newmont", "NUE": "Nucor", "DOW": "Dow", "DD": "DuPont", "PPG": "PPG",
    "ETN": "Eaton", "ITW": "Illinois Tool Works",
    "PH": "Parker Hannifin", "ROK": "Rockwell", "CMI": "Cummins", "PCAR": "Paccar",
    "ZTS": "Zoetis", "BSX": "Boston Scientific", "MDT": "Medtronic", "SYK": "Stryker",
    "BDX": "Becton Dickinson", "HCA": "HCA Healthcare", "CI": "Cigna", "ELV": "Elevance",
    "VRTX": "Vertex", "REGN": "Regeneron", "BIIB": "Biogen", "ILMN": "Illumina",
    "DXCM": "Dexcom", "IDXX": "IDEXX", "ENPH": "Enphase", "FSLR": "First Solar",
    "PLUG": "Plug Power", "CCJ": "Cameco", "ALB": "Albemarle", "CLF": "Cleveland-Cliffs",
    "X": "US Steel", "LYV": "Live Nation",
    # Index / sector ETFs
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq ETF", "IWM": "Russell 2000 ETF",
    "DIA": "Dow ETF", "VOO": "Vanguard S&P 500", "VTI": "Vanguard Total Market",
}

# Tickers that are also common English words / initials — only match WITH a `$`
# prefix (never as a bare all-caps word) to avoid false positives.
AMBIGUOUS_TICKERS: set[str] = {
    "F", "T", "V", "A", "C", "U", "GE", "GM", "HP", "IT", "ON", "SO", "OR", "BE",
    "GO", "AI", "ALL", "ANY", "ARE", "NOW", "KEY", "PM", "MO", "RE", "DD", "EL",
    "NU", "MA", "BB", "MU", "NET", "ARM",
}

# Company names that are also common words — skip plain company-name matching
# (rely on the `$` cashtag instead) so "price target", "a block", "snap decision"
# don't get mislabeled.
COMMON_WORD_NAMES: set[str] = {
    "block", "snap", "gap", "target", "visa", "key", "now", "all", "unity",
    "arm", "net", "ups",
}
