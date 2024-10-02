import streamlit as st
import sqlite3
import bcrypt
import random
from datetime import datetime, timedelta
import base64
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from math import sqrt
from typing import List, Callable
import logging
import pytz  # Added for timezone handling

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger()

# ---------------------------
# 1. Setup and Configuration
# ---------------------------

# Set page configuration at the very top
st.set_page_config(
    page_title="📈 Stock Machine!",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# 2. Initialize Session State
# ---------------------------

# Initialize session state variables at the very top
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ''

# ---------------------------
# 3. Helper Functions
# ---------------------------

@st.cache_data
def get_sp500_tickers() -> List[str]:
    """Fetches the list of S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]  # For yfinance compatibility
        logger.debug("Fetched S&P 500 tickers successfully.")
        return tickers
    except Exception as e:
        st.error("Failed to fetch S&P 500 tickers. Please check your internet connection or try again later.")
        logger.error(f"Error fetching S&P 500 tickers: {e}")
        return []

def fetch_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """Fetches historical stock data for the given ticker and period."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            st.warning(f"No data found for ticker `{ticker}`.")
            logger.warning(f"No data found for ticker: {ticker}")
            return pd.DataFrame()
        logger.debug(f"Fetched stock data for {ticker} over period {period}.")
        return hist
    except Exception as e:
        st.error(f"An error occurred while fetching data for `{ticker}`.")
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def get_user_balance(username: str) -> float:
    """Retrieves the user's current balance in beans."""
    try:
        conn = sqlite3.connect('stock_machine.db')
        c = conn.cursor()
        c.execute("SELECT balance FROM users WHERE username=?", (username,))
        result = c.fetchone()
        conn.close()
        balance = result[0] if result else 0.0
        logger.debug(f"Retrieved balance for user {username}: {balance} Beans")
        return balance
    except Exception as e:
        st.error("An error occurred while retrieving your balance.")
        logger.error(f"Error retrieving balance for user {username}: {e}")
        return 0.0

def update_user_balance(username: str, amount: float):
    """Updates the user's balance by the specified amount."""
    try:
        conn = sqlite3.connect('stock_machine.db')
        c = conn.cursor()
        c.execute("UPDATE users SET balance = balance + ? WHERE username=?", (amount, username))
        conn.commit()
        conn.close()
        logger.debug(f"Updated balance for user {username} by {amount} Beans.")
    except Exception as e:
        st.error("An error occurred while updating your balance.")
        logger.error(f"Error updating balance for user {username}: {e}")

def record_purchase(username: str, stock: str, purchase_price: float, quantity: int, purchase_time: datetime):
    """Records a stock purchase in the database."""
    try:
        conn = sqlite3.connect('stock_machine.db')
        c = conn.cursor()
        
        # Insert purchase record with quantity
        c.execute("""
            INSERT INTO holdings (username, stock_symbol, purchase_price, quantity, purchase_time)
            VALUES (?, ?, ?, ?, ?)
        """, (username, stock, purchase_price, quantity, purchase_time))
        
        conn.commit()
        conn.close()
        logger.info(f"Recorded purchase for user {username} on stock {stock} at {purchase_price} Beans for {quantity} shares.")
    except Exception as e:
        st.error("An error occurred while recording your purchase.")
        logger.error(f"Error recording purchase for user {username}: {e}")

def get_user_holdings(username: str) -> pd.DataFrame:
    """Retrieves the user's current holdings."""
    try:
        conn = sqlite3.connect('stock_machine.db')
        c = conn.cursor()
        c.execute("""
            SELECT stock_symbol, purchase_price, quantity, purchase_time
            FROM holdings
            WHERE username=?
        """, (username,))
        holdings = c.fetchall()
        conn.close()
        df_holdings = pd.DataFrame(holdings, columns=["Stock", "Purchase Price (Beans)", "Quantity", "Purchase Time"])
        logger.debug(f"Retrieved holdings for user {username}.")
        return df_holdings
    except Exception as e:
        st.error("An error occurred while retrieving your holdings.")
        logger.error(f"Error retrieving holdings for user {username}: {e}")
        return pd.DataFrame()

def set_custom_frame(png_file: str):
    """Overlays a custom PNG frame over the app."""
    try:
        # Encode image to base64
        with open(png_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        
        # Define CSS for overlay
        frame_css = f"""
        <style>
        .custom-frame {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            pointer-events: none;  /* Allows interactions with underlying app */
            z-index: 9999;  /* Ensure it's on top */
        }}
        </style>
        <div class="custom-frame"></div>
        """
        
        st.markdown(frame_css, unsafe_allow_html=True)
        logger.debug(f"Applied custom frame from {png_file}.")
    except Exception as e:
        st.error("An error occurred while setting the custom frame.")
        logger.error(f"Error setting custom frame from {png_file}: {e}")

def init_db():
    """Initializes the SQLite database with necessary tables and columns."""
    try:
        conn = sqlite3.connect('stock_machine.db')
        c = conn.cursor()
        
        # Create users table with CHECK constraint
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT,
                balance REAL DEFAULT 1000.0 CHECK(balance >= 0)
            )
        ''')
        
        # Create holdings table
        c.execute('''
            CREATE TABLE IF NOT EXISTS holdings (
                holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                stock_symbol TEXT,
                purchase_price REAL,
                quantity INTEGER,
                purchase_time TIMESTAMP,
                FOREIGN KEY(username) REFERENCES users(username)
            )
        ''')
        
        # Check for missing columns and add them if necessary
        c.execute("PRAGMA table_info(holdings);")
        columns = [info[1] for info in c.fetchall()]
        
        required_columns = {
            "purchase_price": "REAL",
            "purchase_time": "TIMESTAMP",
            "quantity": "INTEGER"  # Added quantity
        }
        
        for column, dtype in required_columns.items():
            if column not in columns:
                c.execute(f"ALTER TABLE holdings ADD COLUMN {column} {dtype};")
                st.warning(f"Added missing column `{column}` to `holdings` table.")
                logger.warning(f"Added missing column `{column}` to `holdings` table.")
        
        conn.commit()
        conn.close()
        logger.debug("Database initialized successfully.")
    except Exception as e:
        st.error("An error occurred while initializing the database.")
        logger.error(f"Error initializing database: {e}")

def hash_password(password: str) -> bytes:
    """Hashes a password using bcrypt."""
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    logger.debug("Password hashed successfully.")
    return hashed

def check_password(hashed_password: bytes, user_password: str) -> bool:
    """Checks a hashed password against a user-provided password."""
    result = bcrypt.checkpw(user_password.encode(), hashed_password)
    logger.debug(f"Password verification result: {result}")
    return result

def register_user(username: str, password: str) -> bool:
    """Registers a new user. Returns True if successful, False if user exists."""
    try:
        conn = sqlite3.connect('stock_machine.db')
        c = conn.cursor()
        
        # Check if user exists
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        if c.fetchone():
            conn.close()
            logger.warning(f"Registration failed: Username {username} already exists.")
            return False  # User exists
        
        # Insert new user with hashed password
        hashed_pw = hash_password(password)
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        conn.close()
        logger.info(f"User {username} registered successfully.")
        return True
    except Exception as e:
        st.error("An error occurred during registration.")
        logger.error(f"Error registering user {username}: {e}")
        return False

def login_user(username: str, password: str) -> bool:
    """Logs in a user. Returns True if credentials are correct."""
    try:
        conn = sqlite3.connect('stock_machine.db')
        c = conn.cursor()
        
        # Fetch user
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        result = c.fetchone()
        conn.close()
        
        if result and check_password(result[0], password):
            logger.info(f"User {username} logged in successfully.")
            return True
        else:
            logger.warning(f"Failed login attempt for user {username}.")
            return False
    except Exception as e:
        st.error("An error occurred during login.")
        logger.error(f"Error logging in user {username}: {e}")
        return False

def display_outcome(outcome: str, profit: float):
    """Displays the outcome of the transaction to the user."""
    if outcome == "Success":
        st.success(f"🎉 Transaction successful! Profit: `{profit}` Beans.")
    elif outcome == "Failure":
        st.error(f"💸 Transaction failed.")
    else:
        st.warning("🔄 Transaction is pending.")
    logger.debug(f"Displayed outcome: {outcome} with profit {profit}.")

def admin_panel():
    """Admin panel to manage users and holdings."""
    st.sidebar.header("🔧 Admin Panel")
    
    admin_password_input = st.sidebar.text_input("Enter Admin Password", type='password')
    if st.sidebar.button("🔍 View All Users"):
        hashed_admin_password = hash_password("admin123")  # Replace with your secure admin password
        if check_password(hashed_admin_password, admin_password_input):
            conn = sqlite3.connect('stock_machine.db')
            c = conn.cursor()
            c.execute("SELECT username, balance FROM users")
            users = c.fetchall()
            conn.close()
            df_users = pd.DataFrame(users, columns=["Username", "Balance (Beans)"])
            st.sidebar.subheader("All Users")
            st.sidebar.dataframe(df_users)
            logger.info("Admin viewed all users.")
        else:
            st.sidebar.error("❌ Incorrect password.")
            logger.warning("Admin attempted to view users with incorrect password.")
    
    if st.sidebar.button("🔍 View All Holdings"):
        hashed_admin_password = hash_password("admin123")  # Replace with your secure admin password
        if check_password(hashed_admin_password, admin_password_input):
            conn = sqlite3.connect('stock_machine.db')
            c = conn.cursor()
            c.execute("SELECT username, stock_symbol, purchase_price, quantity, purchase_time FROM holdings")
            holdings = c.fetchall()
            conn.close()
            df_holdings = pd.DataFrame(holdings, columns=["Username", "Stock", "Purchase Price (Beans)", "Quantity", "Purchase Time"])
            st.sidebar.subheader("All Holdings")
            st.sidebar.dataframe(df_holdings)
            logger.info("Admin viewed all holdings.")
        else:
            st.sidebar.error("❌ Incorrect password.")
            logger.warning("Admin attempted to view holdings with incorrect password.")

def display_leaderboard():
    """Displays the top 5 users based on their bean balances."""
    try:
        conn = sqlite3.connect('stock_machine.db')
        c = conn.cursor()
        c.execute("SELECT username, balance FROM users ORDER BY balance DESC LIMIT 5")
        leaderboard = c.fetchall()
        conn.close()
        
        if leaderboard:
            df_leaderboard = pd.DataFrame(leaderboard, columns=["Username", "Balance (Beans)"])
            st.subheader("🏆 Leaderboard")
            st.table(df_leaderboard)
            logger.debug("Displayed leaderboard.")
        else:
            st.write("No users to display on the leaderboard yet.")
            logger.debug("Leaderboard is empty.")
    except Exception as e:
        st.error("An error occurred while displaying the leaderboard.")
        logger.error(f"Error displaying leaderboard: {e}")

def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
    """Calculate Bollinger Bands for given data."""
    data['SMA'] = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data['Upper_BB'] = data['SMA'] + (rolling_std * num_std)
    data['Lower_BB'] = data['SMA'] - (rolling_std * num_std)
    logger.debug("Calculated Bollinger Bands.")
    return data

def calculate_stochastic_oscillator(data: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator for given data."""
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    data['%K'] = (data['Close'] - low_min) / (high_max - low_min) * 100
    data['%D'] = data['%K'].rolling(window=d_window).mean()
    logger.debug("Calculated Stochastic Oscillator.")
    return data

def calculate_macd(data: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
    """Calculate MACD for given data."""
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    logger.debug("Calculated MACD.")
    return data

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate RSI for given data."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    logger.debug("Calculated RSI.")
    return data

def predict_macd(hist: pd.DataFrame, days_ahead: int = 30, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.Series:
    """Predict future prices using MACD with customizable parameters."""
    hist = calculate_macd(hist.copy(), short_window, long_window, signal_window)
    last_macd = hist['MACD'].iloc[-1]
    last_signal = hist['Signal_Line'].iloc[-1]
    
    predictions = []
    current_price = hist['Close'].iloc[-1]
    
    for _ in range(days_ahead):
        if last_macd > last_signal:
            current_price *= 1.001  # Slight increase
        else:
            current_price *= 0.999  # Slight decrease
        predictions.append(current_price)
    
    future_dates = pd.date_range(start=hist.index[-1] + timedelta(days=1), periods=days_ahead)
    logger.debug("Predicted future prices using MACD.")
    return pd.Series(predictions, index=future_dates)

def predict_rsi(hist: pd.DataFrame, days_ahead: int = 30, window: int = 14) -> pd.Series:
    """Predict future prices using RSI with customizable parameters."""
    hist = calculate_rsi(hist.copy(), window)
    last_rsi = hist['RSI'].iloc[-1]
    current_price = hist['Close'].iloc[-1]
    
    predictions = []
    
    for _ in range(days_ahead):
        if last_rsi > 70:
            current_price *= 0.998  # Overbought, expect decrease
        elif last_rsi < 30:
            current_price *= 1.002  # Oversold, expect increase
        else:
            current_price *= 1.0005  # Neutral, slight increase
        predictions.append(current_price)
    
    future_dates = pd.date_range(start=hist.index[-1] + timedelta(days=1), periods=days_ahead)
    logger.debug("Predicted future prices using RSI.")
    return pd.Series(predictions, index=future_dates)

def predict_momentum(hist: pd.DataFrame, days_ahead: int = 30) -> pd.Series:
    """Predict future prices using a momentum strategy."""
    if hist.empty:
        logger.warning("Empty historical data provided for momentum prediction.")
        return pd.Series(dtype=float)

    momentum_period = 5  # Lookback period for momentum
    hist['Momentum'] = hist['Close'].diff(momentum_period)
    last_momentum = hist['Momentum'].iloc[-1]
    current_price = hist['Close'].iloc[-1]
    
    predictions = []
    
    for _ in range(days_ahead):
        if last_momentum > 0:
            current_price *= 1.001  # Slight increase
        else:
            current_price *= 0.999  # Slight decrease
        predictions.append(current_price)
    
    future_dates = pd.date_range(start=hist.index[-1] + timedelta(days=1), periods=days_ahead)
    logger.debug("Predicted future prices using Momentum.")
    return pd.Series(predictions, index=future_dates)

def predict_mean_reversion(hist: pd.DataFrame, days_ahead: int = 30) -> pd.Series:
    """Predict future prices using a mean reversion strategy."""
    if hist.empty:
        logger.warning("Empty historical data provided for mean reversion prediction.")
        return pd.Series(dtype=float)
    
    mean_price = hist['Close'].mean()
    predictions = [mean_price for _ in range(days_ahead)]
    
    future_dates = pd.date_range(start=hist.index[-1] + timedelta(days=1), periods=days_ahead)
    logger.debug("Predicted future prices using Mean Reversion.")
    return pd.Series(predictions, index=future_dates)

def backtest_model(hist: pd.DataFrame, prediction_func: Callable, window: int = 30, **kwargs) -> float:
    """Backtest the prediction model and return RMSE."""
    try:
        actual_prices = hist['Close']
        predictions = []
        
        for i in range(len(hist) - window):
            train_data = hist.iloc[:i+window]
            pred = prediction_func(train_data, days_ahead=1, **kwargs).iloc[0]
            predictions.append(pred)
        
        if not predictions:
            logger.warning("No predictions were made for backtesting.")
            return float('nan')  # Return NaN if no predictions were made
        
        predictions = pd.Series(predictions, index=actual_prices.index[window:])
        rmse = sqrt(mean_squared_error(actual_prices[window:], predictions))
        logger.debug(f"Backtested model with RMSE: {rmse}")
        return rmse
    except Exception as e:
        logger.error(f"Error during backtesting: {e}")
        return float('nan')

def plot_candlestick(hist: pd.DataFrame, ticker: str):
    """Plots an interactive candlestick chart for the given historical data."""
    try:
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            name='Price'
        )])
        fig.update_layout(
            title=f"{ticker} Candlestick Chart",
            yaxis_title='Price (Beans)',
            xaxis_title='Date',
            template='plotly_dark',
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
        logger.debug(f"Plotted candlestick chart for {ticker}.")
    except Exception as e:
        st.error("An error occurred while plotting the candlestick chart.")
        logger.error(f"Error plotting candlestick chart for {ticker}: {e}")

def plot_bar_graph(hist: pd.DataFrame, ticker: str):
    """Plots a bar graph with green for gains and red for losses."""
    try:
        hist['Change'] = hist['Close'].pct_change() * 100  # Percentage change
        colors = ['green' if val > 0 else 'red' for val in hist['Change']]
        
        fig = go.Figure(data=[go.Bar(
            x=hist.index,
            y=hist['Change'],
            marker_color=colors
        )])
        fig.update_layout(
            title=f"{ticker} Daily Percentage Change",
            xaxis_title='Date',
            yaxis_title='Change (%)',
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
        logger.debug(f"Plotted bar graph for {ticker}.")
    except Exception as e:
        st.error("An error occurred while plotting the bar graph.")
        logger.error(f"Error plotting bar graph for {ticker}: {e}")

def display_performance_metrics(hist: pd.DataFrame):
    """Displays key performance metrics for the stock."""
    try:
        latest_close = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        pct_change = ((latest_close - previous_close) / previous_close) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Latest Close Price", value=f"{latest_close:,.2f} Beans")
        with col2:
            delta_color = "normal"
            if pct_change > 0:
                delta_color = "normal"
            elif pct_change < 0:
                delta_color = "inverse"
            st.metric(label="Change (%)", value=f"{pct_change:.2f}%", delta=f"{latest_close - previous_close:.2f}", delta_color=delta_color)
        logger.debug("Displayed performance metrics.")
    except Exception as e:
        st.error("An error occurred while calculating performance metrics.")
        logger.error(f"Error displaying performance metrics: {e}")

def display_company_overview(ticker: str):
    """Displays a brief overview of the company."""
    try:
        stock = yf.Ticker(ticker)
        company_info = stock.info
        st.subheader("🏢 Company Overview")
        st.write(f"**Name:** {company_info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
        st.write(f"**Description:** {company_info.get('longBusinessSummary', 'N/A')}")
        logger.debug(f"Displayed company overview for {ticker}.")
    except Exception as e:
        st.error("An error occurred while fetching company information.")
        logger.error(f"Error fetching company overview for {ticker}: {e}")

def check_market_status():
    """Checks and returns the current market status."""
    try:
        # Define Eastern Time timezone
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        current_time = now.time()
        today = now.weekday()  # Monday is 0 and Sunday is 6

        # Define market hours
        market_open = datetime(now.year, now.month, now.day, 9, 30, tzinfo=eastern).time()
        market_close = datetime(now.year, now.month, now.day, 16, 0, tzinfo=eastern).time()
        pre_market_open = datetime(now.year, now.month, now.day, 4, 0, tzinfo=eastern).time()
        pre_market_close = market_open
        after_market_open = market_close
        after_market_close = datetime(now.year, now.month, now.day, 20, 0, tzinfo=eastern).time()

        if today >= 5:  # Saturday or Sunday
            status = "Closed (Weekend)"
        elif pre_market_open <= current_time < pre_market_close:
            status = "Pre-Market"
        elif market_open <= current_time < market_close:
            status = "Open"
        elif after_market_open <= current_time < after_market_close:
            status = "After-Market"
        else:
            status = "Closed"

        st.sidebar.subheader("🕒 Market Status")
        st.sidebar.write(f"**Current Time (ET):** {now.strftime('%Y-%m-%d %H:%M:%S')}")
        st.sidebar.write(f"**Market Status:** {status}")
        st.sidebar.write("**Market Hours (ET):**")
        st.sidebar.write("- Pre-Market: 4:00 AM - 9:30 AM")
        st.sidebar.write("- Regular: 9:30 AM - 4:00 PM")
        st.sidebar.write("- After-Market: 4:00 PM - 8:00 PM")

        logger.debug(f"Market status checked: {status}")
        return status
    except Exception as e:
        st.error("An error occurred while checking market status.")
        logger.error(f"Error checking market status: {e}")
        return "Unknown"

def init_db():
    """Initializes the SQLite database with necessary tables and columns."""
    try:
        conn = sqlite3.connect('stock_machine.db')
        c = conn.cursor()
        
        # Create users table with CHECK constraint
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT,
                balance REAL DEFAULT 1000.0 CHECK(balance >= 0)
            )
        ''')
        
        # Create holdings table with quantity
        c.execute('''
            CREATE TABLE IF NOT EXISTS holdings (
                holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                stock_symbol TEXT,
                purchase_price REAL,
                quantity INTEGER,
                purchase_time TIMESTAMP,
                FOREIGN KEY(username) REFERENCES users(username)
            )
        ''')
        
        # Check for missing columns and add them if necessary
        c.execute("PRAGMA table_info(holdings);")
        columns = [info[1] for info in c.fetchall()]
        
        required_columns = {
            "purchase_price": "REAL",
            "purchase_time": "TIMESTAMP",
            "quantity": "INTEGER"  # Added quantity
        }
        
        for column, dtype in required_columns.items():
            if column not in columns:
                c.execute(f"ALTER TABLE holdings ADD COLUMN {column} {dtype};")
                st.warning(f"Added missing column `{column}` to `holdings` table.")
                logger.warning(f"Added missing column `{column}` to `holdings` table.")
        
        conn.commit()
        conn.close()
        logger.debug("Database initialized successfully.")
    except Exception as e:
        st.error("An error occurred while initializing the database.")
        logger.error(f"Error initializing database: {e}")

# Hashing and authentication functions remain unchanged

def display_outcome(outcome: str, profit: float):
    """Displays the outcome of the transaction to the user."""
    if outcome == "Success":
        st.success(f"🎉 Transaction successful! Profit: `{profit}` Beans.")
    elif outcome == "Failure":
        st.error(f"💸 Transaction failed.")
    else:
        st.warning("🔄 Transaction is pending.")
    logger.debug(f"Displayed outcome: {outcome} with profit {profit}.")

# ---------------------------
# 4. User Authentication UI
# ---------------------------

def authenticate():
    """Handles user authentication (Login/Register) using st.form."""
    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)
    logger.debug(f"Selected menu option: {choice}")
    
    if choice == "Register":
        st.subheader("📝 Create New Account")
        with st.form(key='register_form'):
            new_user = st.text_input("Username", key='register_username')
            new_password = st.text_input("Password", type='password', key='register_password')
            submit_button = st.form_submit_button(label='Register')
            
            if submit_button:
                if new_user and new_password:
                    if register_user(new_user, new_password):
                        st.success("✅ You have successfully created an account! Please login.")
                        logger.info(f"New user registered: {new_user}")
                    else:
                        st.error("❌ Username already exists. Please choose a different one.")
                        logger.warning(f"Registration failed for user: {new_user} (Username exists)")
                else:
                    st.warning("⚠️ Please enter both username and password.")
                    logger.warning("Registration attempted without username or password.")
    
    elif choice == "Login":
        st.subheader("🔑 Login to Your Account (Click the login button twice)")
        with st.form(key='login_form'):
            username = st.text_input("Username", key='login_username')
            password = st.text_input("Password", type='password', key='login_password')
            submit_button = st.form_submit_button(label='Login')
            
            if submit_button:
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"🎉 Welcome {username}!")
                    logger.info(f"User logged in: {username}")
                    # No need to call st.experimental_rerun()
                else:
                    st.error("❌ Invalid username or password.")
                    logger.warning(f"Failed login attempt for user: {username}")

# ---------------------------
# 5. Purchase and Sell Interface
# ---------------------------

def place_trade_ui():
    """UI for users to purchase and sell stocks."""
    st.header("🛒 Purchase & Sell Stocks")
    
    # Check market status
    market_status = check_market_status()
    
    # Fetch S&P 500 tickers
    tickers = get_sp500_tickers()
    
    # Select stock
    selected_stock = st.selectbox("Choose a Stock", options=tickers)
    
    if selected_stock:
        # Fetch current stock price
        with st.spinner(f"Fetching current price for `{selected_stock}`..."):
            hist = fetch_stock_data(selected_stock, period='1d')
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                st.write(f"**Current Price:** {current_price:.2f} Beans")
            else:
                st.error("Unable to fetch current price.")
                return
    
    # Fetch user balance
    user_balance = get_user_balance(st.session_state.username)
    
    if selected_stock and not hist.empty:
        st.write(f"**Your Balance:** {user_balance:.2f} Beans")
        st.write("**Price of One Share:** {:.2f} Beans".format(current_price))
        
        # Buying Section
        st.subheader("📈 Buy Stocks")
        buy_quantity = st.number_input("Quantity to Buy", min_value=1, step=1, value=1)
        total_buy_cost = buy_quantity * current_price
        st.write(f"**Total Cost:** {total_buy_cost:.2f} Beans")
        
        if st.button("🛒 Purchase"):
            if user_balance >= total_buy_cost:
                # Deduct total cost from balance
                update_user_balance(st.session_state.username, -total_buy_cost)
                
                # Record the purchase
                purchase_time = datetime.now(pytz.timezone('US/Eastern'))
                record_purchase(st.session_state.username, selected_stock, current_price, buy_quantity, purchase_time)
                
                st.success(f"✅ Successfully purchased {buy_quantity} shares of `{selected_stock}` for {total_buy_cost:.2f} Beans.")
                logger.info(f"User {st.session_state.username} purchased {buy_quantity} shares of {selected_stock} at {current_price} Beans each.")
            else:
                st.error("⚠️ Insufficient beans to complete this purchase.")
                logger.warning(f"User {st.session_state.username} attempted to purchase {selected_stock} with insufficient balance.")
        
        st.markdown("---")
        
        # Selling Section
        st.subheader("📉 Sell Stocks")
        user_holdings = get_user_holdings(st.session_state.username)
        user_stocks = user_holdings['Stock'].unique().tolist()
        
        if user_stocks:
            sell_stock = st.selectbox("Select Stock to Sell", options=user_stocks)
            sell_holdings = user_holdings[user_holdings['Stock'] == sell_stock]
            total_owned = sell_holdings['Quantity'].sum()
            st.write(f"**Total Shares Owned:** {total_owned}")
            
            sell_quantity = st.number_input("Quantity to Sell", min_value=1, step=1, value=1, max_value=int(total_owned))
            total_sell_value = sell_quantity * current_price
            st.write(f"**Total Sell Value:** {total_sell_value:.2f} Beans")
            
            if st.button("💰 Sell"):
                if sell_quantity <= total_owned:
                    # Calculate average purchase price
                    average_price = sell_holdings['Purchase Price (Beans)'].mean()
                    
                    # Calculate profit
                    profit = (current_price - average_price) * sell_quantity
                    
                    # Update user balance
                    update_user_balance(st.session_state.username, total_sell_value)
                    
                    # Remove sold shares from holdings
                    # Fetch holdings ordered by purchase_time (FIFO)
                    conn = sqlite3.connect('stock_machine.db')
                    c = conn.cursor()
                    c.execute("""
                        SELECT holding_id, quantity
                        FROM holdings
                        WHERE username=? AND stock_symbol=?
                        ORDER BY purchase_time ASC
                    """, (st.session_state.username, sell_stock))
                    holdings = c.fetchall()
                    
                    remaining = sell_quantity
                    for holding in holdings:
                        holding_id, qty = holding
                        if qty <= remaining:
                            # Delete the holding
                            c.execute("DELETE FROM holdings WHERE holding_id=?", (holding_id,))
                            remaining -= qty
                        else:
                            # Update the holding with reduced quantity
                            c.execute("UPDATE holdings SET quantity = ? WHERE holding_id=?", (qty - remaining, holding_id))
                            remaining = 0
                        if remaining == 0:
                            break
                    
                    conn.commit()
                    conn.close()
                    
                    st.success(f"✅ Successfully sold {sell_quantity} shares of `{sell_stock}` for {total_sell_value:.2f} Beans.")
                    if profit > 0:
                        st.write(f"**Profit:** {profit:.2f} Beans")
                    else:
                        st.write(f"**Loss:** {abs(profit):.2f} Beans")
                    logger.info(f"User {st.session_state.username} sold {sell_quantity} shares of {sell_stock} at {current_price} Beans each. Profit: {profit} Beans.")
                else:
                    st.error("⚠️ You do not have enough shares to sell this quantity.")
                    logger.warning(f"User {st.session_state.username} attempted to sell {sell_quantity} shares of {sell_stock} exceeding holdings.")
        else:
            st.write("You have no holdings to sell.")
            logger.debug(f"User {st.session_state.username} has no holdings to sell.")

# ---------------------------
# 6. Automatic Holdings Update
# ---------------------------

def update_holdings_on_run():
    """Updates holdings based on current stock prices."""
    try:
        conn = sqlite3.connect('stock_machine.db')
        c = conn.cursor()
        
        # Fetch all holdings
        c.execute("""
            SELECT holding_id, username, stock_symbol, purchase_price, quantity
            FROM holdings
        """)
        holdings = c.fetchall()
        
        conn.close()
        logger.debug(f"Fetched {len(holdings)} holdings for update.")
        
        for holding in holdings:
            holding_id, username, stock_symbol, purchase_price, quantity = holding
            hist = fetch_stock_data(stock_symbol, period='1d')
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                profit = (current_price - purchase_price) * quantity
                # Update user's balance based on profit
                update_user_balance(username, profit)
                logger.info(f"Updated holding {holding_id} for user {username}: Profit {profit} Beans.")
    except Exception as e:
        st.error("An error occurred while updating holdings.")
        logger.error(f"Error updating holdings: {e}")

# ---------------------------
# 7. Additional Features
# ---------------------------

def display_purchase_history():
    """Displays the user's purchase history."""
    try:
        st.header("📜 Your Purchase History")
        conn = sqlite3.connect('stock_machine.db')
        c = conn.cursor()
        c.execute("""
            SELECT stock_symbol, purchase_price, quantity, purchase_time
            FROM holdings
            WHERE username=?
            ORDER BY purchase_time DESC
            LIMIT 10
        """, (st.session_state.username,))
        purchases = c.fetchall()
        conn.close()
        
        if purchases:
            df_purchases = pd.DataFrame(purchases, columns=["Stock", "Purchase Price (Beans)", "Quantity", "Purchase Time"])
            # Formatting the datetime columns for better readability
            df_purchases["Purchase Time"] = pd.to_datetime(df_purchases["Purchase Time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(df_purchases)
            logger.debug(f"Displayed purchase history for user {st.session_state.username}.")
        else:
            st.write("No purchases made yet.")
            logger.debug(f"No purchase history found for user {st.session_state.username}.")
    except Exception as e:
        st.error("An error occurred while displaying your purchase history.")
        logger.error(f"Error displaying purchase history for user {st.session_state.username}: {e}")

def display_stock_performance():
    """Displays existing financial graphs and indicators."""
    try:
        st.header("📊 Your Stock Performance")
        tickers = get_sp500_tickers()
        if not tickers:
            st.warning("No tickers available to display stock performance.")
            return
        selected_stock = st.selectbox("Select a Stock to View Performance", options=tickers)
        selected_range = st.selectbox("Select Time Range (Chosen future model backtesting (historical accuracy) shows only on longer ranges)", options=["5d", "1mo", "6mo", "1y", "5y", "10y", "max"])
        
        if selected_stock:
            with st.spinner(f"Fetching data for `{selected_stock}`..."):
                hist = fetch_stock_data(selected_stock, period=selected_range)
            
            if not hist.empty:
                # Display Performance Metrics
                display_performance_metrics(hist)

                # Display Company Overview
                display_company_overview(selected_stock)

                # Plot Candlestick Chart
                plot_candlestick(hist, selected_stock)

                # Plot Bar Graph
                plot_bar_graph(hist, selected_stock)

                # Advanced Financial Indicators
                st.subheader("📊 Advanced Financial Indicators")
                
                # Bollinger Bands
                bb_window = st.slider("Bollinger Bands Window", min_value=5, max_value=50, value=20)
                bb_std = st.slider("Bollinger Bands Standard Deviation", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
                hist_bb = calculate_bollinger_bands(hist.copy(), window=bb_window, num_std=bb_std)
                
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(x=hist_bb.index, y=hist_bb['Close'], name="Close"))
                fig_bb.add_trace(go.Scatter(x=hist_bb.index, y=hist_bb['Upper_BB'], name="Upper BB"))
                fig_bb.add_trace(go.Scatter(x=hist_bb.index, y=hist_bb['Lower_BB'], name="Lower BB"))
                fig_bb.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_bb, use_container_width=True)

                # Stochastic Oscillator
                k_window = st.slider("Stochastic %K Window", min_value=5, max_value=30, value=14)
                d_window = st.slider("Stochastic %D Window", min_value=1, max_value=10, value=3)
                hist_stoch = calculate_stochastic_oscillator(hist.copy(), k_window=k_window, d_window=d_window)
                
                fig_stoch = go.Figure()
                fig_stoch.add_trace(go.Scatter(x=hist_stoch.index, y=hist_stoch['%K'], name="%K"))
                fig_stoch.add_trace(go.Scatter(x=hist_stoch.index, y=hist_stoch['%D'], name="%D"))
                fig_stoch.update_layout(title="Stochastic Oscillator", xaxis_title="Date", yaxis_title="Value")
                st.plotly_chart(fig_stoch, use_container_width=True)

                # Prediction and Backtesting
                st.subheader("📈 Future Price Estimates")
                prediction_models = {
                    "Momentum": predict_momentum,
                    "Mean Reversion": predict_mean_reversion,
                    "MACD": predict_macd,
                    "RSI": predict_rsi
                }
                
                selected_model = st.selectbox("Select Prediction Model", options=list(prediction_models.keys()))
                
                # Model-specific parameter customization
                model_params = {}
                if selected_model == "MACD":
                    short_window = st.slider("MACD Short Window", min_value=5, max_value=20, value=12)
                    long_window = st.slider("MACD Long Window", min_value=20, max_value=50, value=26)
                    signal_window = st.slider("MACD Signal Window", min_value=5, max_value=20, value=9)
                    model_params = {"short_window": short_window, "long_window": long_window, "signal_window": signal_window}
                elif selected_model == "RSI":
                    rsi_window = st.slider("RSI Window", min_value=5, max_value=30, value=14)
                    model_params = {"window": rsi_window}
                
                days_ahead = st.slider("Days Ahead for Prediction", min_value=1, max_value=60, value=30)
                
                future_predictions = prediction_models[selected_model](hist.copy(), days_ahead=days_ahead, **model_params)
                
                if not future_predictions.empty:
                    st.line_chart(future_predictions, use_container_width=True)
                    logger.debug(f"Displayed future predictions using {selected_model} model.")
                
                # Backtesting
                backtest_window = st.slider("Backtesting Window (days)", min_value=30, max_value=365, value=90)
                rmse = backtest_model(hist.copy(), prediction_models[selected_model], window=backtest_window, **model_params)
                if not np.isnan(rmse):
                    st.write(f"Backtesting RMSE (Root Mean Square Error): {rmse:.2f}")
                    st.write(f"This indicates the average prediction error over the last {backtest_window} days.")
                else:
                    st.write("No predictions were generated for backtesting.")
                    logger.warning("Backtesting returned NaN.")
                
                # Explanations for Financial Indicators
                st.write("### Explanations of Financial Indicators")
                st.write("**Bollinger Bands**: Bollinger Bands consist of a middle band (SMA) and two outer bands. The outer bands are standard deviations above and below the SMA. They indicate volatility; when the price touches the upper band, it might be overbought, while touching the lower band might indicate oversold conditions.")
                st.write("**Stochastic Oscillator**: This measures the level of the closing price relative to the high-low range over a given period. Values over 80 indicate overbought conditions, while values under 20 indicate oversold conditions.")
                st.write("**MACD**: Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price. It consists of the MACD line and the Signal line. A bullish signal occurs when the MACD line crosses above the Signal line, indicating potential upward momentum, while a bearish signal occurs when the MACD line crosses below the Signal line.")
                st.write("**RSI**: The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. The RSI moves between 0 and 100 and is typically used to identify overbought or oversold conditions in a market. An RSI above 70 is typically considered overbought, while an RSI below 30 is considered oversold.")
                
                # Display Raw Data (optional)
                st.subheader("📊 Historical Data")
                st.dataframe(hist.tail(100))  # Show last 100 entries
                logger.debug(f"Displayed historical data for {selected_stock}.")
            else:
                st.write("No data available for the selected stock.")
                logger.warning(f"No data available for the selected stock: {selected_stock}.")
    except Exception as e:
        st.error("An error occurred while displaying stock performance.")
        logger.error(f"Error displaying stock performance: {e}")

# ---------------------------
# 8. Main Application
# ---------------------------

def main_app():
    """Main application interface after user is logged in."""
    # Display Header
    st.title("📈 **Stock Machine** 📈")
    st.markdown("## **Enhance Your Investment Portfolio with Smart Purchases!**")
    
    # Sidebar - User Info and Logout
    st.sidebar.header(f"👤 {st.session_state.username}")
    
    # Display Market Status in Sidebar
    check_market_status()
    
    # Fetch user balance
    user_balance = get_user_balance(st.session_state.username)
    st.sidebar.write(f"**Beans Balance:** {user_balance:.2f} Beans")
    
    if st.sidebar.button("🔒 Logout"):
        logger.info(f"User {st.session_state.username} logged out.")
        st.session_state.logged_in = False
        st.session_state.username = ''
        # No need to call st.experimental_rerun()
    
    st.markdown("---")
    
    # Trade Section (Buy & Sell)
    place_trade_ui()
    
    st.markdown("---")
    
    # Display Purchase History
    display_purchase_history()
    
    st.markdown("---")
    
    # Display Leaderboard
    display_leaderboard()
    
    st.markdown("---")
    
    # Display Existing Graphs
    display_stock_performance()
    
    st.markdown("---")
    
    # Footer
    st.markdown("### 📌 Note:")
    st.write("""
        - **Virtual Currency ("Beans")**: All transactions are virtual. No real money is involved.
        - **Holdings Management**: Your purchased stocks are tracked in your holdings. Profits are updated based on current stock prices.
        - **Leaderboard**: Compete with other users to top the leaderboard!
    """)
    st.markdown("### 🔗 Data Source:")
    st.write("[Yahoo Finance](https://finance.yahoo.com/)")
    st.markdown("### 🛠️ Done with ChatGPT.")

def main():
    """Main function to run the Streamlit app."""
    logger.debug("Application started.")
    
    # Initialize the database
    init_db()
    
    # Update holdings based on current stock prices
    update_holdings_on_run()
    
    if not st.session_state.logged_in:
        authenticate()
    else:
        # Optional: Set custom slot machine frame
        # Uncomment and provide the path to your PNG file if desired
        # set_custom_frame("assets/slot_frame.png")
        
        main_app()
        
        # Optionally, show admin panel if user is admin
        if st.session_state.username.lower() == "admin":
            admin_panel()
    
    # Footer (already included in main_app(), but keeping if additional info is needed)
    # Additional footer can be added here if desired

if __name__ == "__main__":
    main()
