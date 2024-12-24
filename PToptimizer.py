import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import minimize

# Functions needed

# Function to calculate portfolio statistics


def portfolio_statistics(weights, mean_returns, cov_matrix):
    portfolio_return = (1 + np.sum(mean_returns * weights))**252-1
    portfolio_volatility = np.sqrt(
        252*np.dot(weights, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Function to minimize negative Sharpe ratio


def negative_sharpe_ratio(weights, mean_returns, cov_matrix):
    return -portfolio_statistics(weights, mean_returns, cov_matrix)[2]

# Function defining weight constraint


def constraint_sum(weights):
    return np.sum(weights) - 1


# Update rcParams for font sizes
plt.rcParams.update({
    'axes.labelsize': 16,  # Font size for x and y labels
    'xtick.labelsize': 12,  # Font size for x-axis tick labels
    'ytick.labelsize': 12   # Font size for y-axis tick labels
})

# Set page configuration for wider layout
st.set_page_config(layout="wide")

# Streamlit app title
st.title("Portfolio Optimization App")

# Sidebar for user input
st.sidebar.header("Input Options")
tickers_input = st.sidebar.text_input(
    "Enter Stock Tickers (comma-separated):", "AAPL, MSFT, TSLA")
start_date = st.sidebar.date_input("Start Date:", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date:", pd.to_datetime("today"))

# Sidebar checkboxes
st.sidebar.title("Display:")
option_pie = st.sidebar.checkbox("Weights Pie Chart", value=True)
option_ret_vol = st.sidebar.checkbox("Annual Returns/Volatility", value=True)
option_price = st.sidebar.checkbox("Price Over Time", value=True)
option_return = st.sidebar.checkbox("Daily Returns", value=True)

# Convert input string to a list of tickers
tickers = [ticker.strip().upper()
           for ticker in tickers_input.split(",") if ticker.strip()]

st.write('This app calculates the optimized distribution of stocks in a portfolio by maximizing the Sharpe Ratio inside the given interval.')
# Fetch stock data
if tickers:
    try:
        # Download data using yfinance
        stock_data = yf.download(tickers, start=start_date, end=end_date)

        # if not stock_data.empty:
        if stock_data['Adj Close'].notna().all().all():

            # Calculate daily returns, mean returns, covariance and std
            daily_returns = stock_data['Adj Close'].pct_change().dropna()
            mean_returns = daily_returns.mean()
            cov_matrix = daily_returns.cov()
            std = daily_returns.std()
            num_assets = len(tickers)

            # Initial guess and constraints
            initial_weights = num_assets * [1. / num_assets]
            bounds = tuple((0, 1) for _ in range(num_assets))
            constraints = {'type': 'eq', 'fun': constraint_sum}

            # Optimize portfolio weights
            optimized_result = minimize(
                negative_sharpe_ratio,
                initial_weights,
                args=(mean_returns, cov_matrix),
                bounds=bounds,
                constraints=constraints
            )

            optimal_weights = optimized_result.x
            optimal_weights_string = [str((x*100).round(2)) + '%'
                                      for x in optimal_weights]
            df_optimal_weights = pd.DataFrame(
                {'Tickers': tickers, 'Weights (%)': (optimal_weights*100).round(1)})
            df_optimal_weights = df_optimal_weights.set_index('Tickers')
            st.subheader("Optimal Weights for Stocks " +
                         ", ".join(tickers) + ": ")
            st.dataframe(df_optimal_weights, width=500)

            # Portfolio performance visualization
            portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_statistics(
                optimal_weights, mean_returns, cov_matrix)

            if option_pie:
                st.subheader("Pie Chart of Optimal Portfolio Weights:")
                # Plot portfolio composition
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(optimal_weights, labels=tickers,
                       autopct='%1.1f%%', startangle=140)
                st.pyplot(fig)

            if option_ret_vol:
                # Create new dataframe with summary of stocks
                df_summary = pd.DataFrame(
                    {'Return': (1+mean_returns)**252-1, 'Volatility': std*np.sqrt(252), 'Sharpe Ratio': ((1+mean_returns)**252-1)/(std*np.sqrt(252))})
                df_summary.loc['Optimized PF'] = [
                    portfolio_return, portfolio_volatility, sharpe_ratio]

                # Display the summary of stocks vs optimized portfolio
                st.subheader(
                    "Annual Return and Volatility of the stocks compared to the optimized portfolio:")
                st.dataframe(df_summary, width=500)
                # Plot showing this data
                st.subheader("Annualized Returns and Volatilities:")
                fig, ax = plt.subplots(figsize=(10, 5))
                for x in range(len(df_summary.index)):
                    plt.scatter(
                        df_summary.iloc[x, 1], df_summary.iloc[x, 0], label=df_summary.index[x])
                ax.set_xlabel("Volatility")
                ax.set_ylabel("Return")
                ax.legend()
                st.pyplot(fig)

            if option_price:
                # Line chart for stock closing price with optimal portfolio
                st.subheader("Closing Price Over Time:")
                fig, ax = plt.subplots(figsize=(10, 5))
                for ticker in tickers:
                    ax.plot(stock_data['Adj Close'][ticker], label=ticker)
                stock_data_optimal = pd.Series(0,
                                               index=stock_data['Adj Close'].iloc[:, 0].index)
                for i in range(len(optimal_weights)):
                    stock_data_optimal += optimal_weights[i] * \
                        stock_data['Adj Close'].iloc[:, i]
                ax.plot(stock_data_optimal, label='Optimized Portfolio')
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (USD)")
                ax.legend()
                st.pyplot(fig)

                # Display stock data summary
                st.subheader("Stock Data Closing Prices:")
                points_shown = st.number_input(
                    "Number of data points shown:", value=5, step=5)
                st.dataframe(stock_data['Adj Close'].head(
                    points_shown))

            if option_return:
                # Daily returns with optimal portfolio returns
                st.subheader("Daily Returns:")
                tickers_shown = tickers
                tickers_shown.append('Optimized PT')
                selected_tickers = st.multiselect(
                    "Choose stocks shown:", tickers_shown, default=tickers_shown)
                returns_optimal = pd.Series(0,
                                            index=daily_returns.iloc[:, 0].index)
                for i in range(len(optimal_weights)):
                    returns_optimal += optimal_weights[i] * \
                        daily_returns.iloc[:, i]
                fig, ax = plt.subplots(figsize=(10, 5))
                daily_returns_complete = daily_returns.copy()
                daily_returns_complete['Optimized PT'] = returns_optimal
                for ticker in selected_tickers:
                    ax.plot(daily_returns_complete[ticker], label=ticker)
                # ax.plot(returns_optimal, label='Optimized Portfolio')
                ax.set_xlabel("Date")
                ax.set_ylabel("Daily Return (%)")
                ax.legend()
                st.pyplot(fig)

                # Histogram of daily returns
                st.subheader("Histograms of Daily Returns:")
                daily_returns_complete = daily_returns_complete[selected_tickers]
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(daily_returns_complete, bins=50, ax=ax)
                # sns.histplot(pd.DataFrame({'Optimized PT': returns_optimal}), bins=50,
                #              ax=ax)
                plt.xlabel('Daily Return')
                st.pyplot(fig)

        else:
            st.error(
                "No data found for some of the given stock symbols. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add forecast or other analysis sections as desired
st.sidebar.markdown("### Created by Alberto Nava")
