# 📈 Stock Machine! 📈

**Stock Machine!** is a powerful Streamlit-based web application that allows users to simulate buying and selling stocks using a virtual currency system called "Beans." Enhance your investment portfolio with smart purchases, track your holdings, analyze stock performance, and compete on the leaderboard—all within an interactive and user-friendly interface.

## 🚀 Features

- **User Authentication**
  - Secure registration and login system.
  - User-specific balance management in Beans.

- **Stock Transactions**
  - **Buy Stocks**: Purchase multiple shares of S&P 500 companies.
  - **Sell Stocks**: Sell your holdings to realize profits or cut losses.
  - **Quantity Selection**: Specify the number of shares to buy or sell.

- **Market Information**
  - Display current time and market status (Open, Pre-Market, After-Market, Closed).
  - Show operating hours based on Eastern Time (ET).

- **Stock Performance Analysis**
  - Interactive candlestick charts.
  - Daily percentage change bar graphs.
  - Advanced financial indicators:
    - Bollinger Bands
    - Stochastic Oscillator
    - MACD
    - RSI

- **Prediction Models**
  - **Momentum**
  - **Mean Reversion**
  - **MACD-Based**
  - **RSI-Based**
  - Visualize future price estimates with backtesting metrics.

- **User Dashboard**
  - View purchase history and current holdings.
  - Leaderboard showcasing top users based on their Bean balances.

- **Admin Panel**
  - Manage users and holdings.
  - Secure access with admin authentication.

- **Customizable UI**
  - Option to overlay a custom PNG frame for personalized branding.

## 📦 Installation

Follow these steps to set up and run **Stock Machine!** on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/anttiluode/stockmachine.git
cd stock-machine
2. Create a Virtual Environment (Optional but Recommended)
Create a virtual environment to manage dependencies.

bash
Copy code
python -m venv venv
Activate the virtual environment:

Windows:

bash
Copy code
venv\Scripts\activate
macOS/Linux:

bash
Copy code
source venv/bin/activate
3. Install Dependencies
Ensure you have pip installed, then install the required packages:

bash
Copy code
pip install -r requirements.txt
4. Initialize the Database
The application uses SQLite for data storage. Initialize the database by running the application once, which will automatically create the necessary tables.

bash
Copy code
streamlit run app.py
Close the application after the initial run to proceed to the next steps.

5. Configure Admin Credentials
By default, the admin password is set to admin123. For security, it's recommended to change this password.

Open the app.py file.
Locate the admin_panel function.
Replace "admin123" with your desired secure password.
python
Copy code
hashed_admin_password = hash_password("your_secure_password")  # Replace with your secure admin password
6. (Optional) Add a Custom Frame
If you wish to overlay a custom PNG frame over the app:

Place your PNG file (e.g., slot_frame.png) in the assets/ directory.
Uncomment and update the set_custom_frame line in the main function.
python
Copy code
# set_custom_frame("assets/slot_frame.png")
As of now this function has been commented (#) out. 
7. Run the Application
Start the Streamlit application:

bash
Copy code
streamlit run app.py
Access the app by navigating to the URL provided in the terminal, typically http://localhost:8501.

🛠️ Usage
Register an Account

Navigate to the registration page via the sidebar.
Create a new account by providing a unique username and password.
Login

Use your credentials to log into the application.
Purchase Stocks

Select a stock from the S&P 500 list.
Specify the number of shares to purchase.
Confirm the transaction to buy stocks using your Beans balance.
Sell Stocks

Navigate to your holdings.
Choose the stock and specify the number of shares to sell.
Confirm the sale to receive Beans based on the current stock price.
Analyze Stocks

View detailed performance metrics and interactive charts.
Utilize prediction models to estimate future stock prices.
Backtest models to evaluate their accuracy.
Leaderboard

Compete with other users to top the leaderboard based on your Bean balance.
Admin Panel

Accessible to admin users.
Manage user accounts and view all holdings.
📊 Screenshots
Add screenshots of your application here to showcase different features and interfaces.

📚 Technologies Used
Frontend: Streamlit
Backend: Python, SQLite
Data Fetching: yfinance
Data Visualization: Plotly
Authentication: bcrypt
Logging: Python's logging module
🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch for your feature or bugfix.
Commit your changes with clear messages.
Push to your forked repository.
Open a pull request detailing your changes.
📝 License
This project is licensed under the MIT License.
