import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt 

class ValuationAnalyzer:
    def __init__(self):
        # Base industry metrics
        self.industry_pe = {
            'Technology': 25.5,
            'Energy': 15.2,
        }
        self.industry_pb = {
            'Technology': 6.8,
            'Energy': 1.8,
        }
        
        # Base weights (60% of total score)
        self.base_weights = {
            'P/E Score': 0.15,      
            'P/B Score': 0.10,      
            'PEG Score': 0.15,      
            'DCF Score': 0.20       
        }
        
        # Technical weights (20% of total score)
        self.technical_weights = {
            'RSI_Score': 0.05,     
            'MA_Score': 0.05,      
            'Growth_Score': 0.10    
        }
        
        # Sector-specific metrics and thresholds
        self.sector_metrics = {
            'Technology': {
                'min_gross_margin': 0.30,
                'min_r_and_d': 0.10,
                'max_debt_equity': 1.5,
                'min_revenue_growth': 0.15,
                'important_metrics': ['R&D_Spend', 'Patent_Count', 'Cloud_Revenue']
            },
            'Energy': {
                'min_current_ratio': 1.5,
                'max_debt_equity': 2.0,
                'min_operating_margin': 0.15,
                'important_metrics': ['Reserve_Life', 'Production_Cost', 'ESG_Score']
            }
        }

        # Sector-specific weights (20% of total score)
        self.sector_weights = {
            'Technology': {
                'R&D_Score': 0.07,
                'Market_Share_Score': 0.07,
                'Innovation_Score': 0.06
            },
            'Energy': {
                'Reserve_Score': 0.07,
                'Production_Cost_Score': 0.07,
                'ESG_Score': 0.06
            }
        }
        # Add industry averages
        self.industry_averages = {
            'Technology': {
                'Revenue_Growth': 0.15,
                'R&D_Ratio': 0.10,
                'Gross_Margin': 0.50,
                'Operating_Margin': 0.20
            },
            'Energy': {
                'Production_Cost': 35.0,  # Dollar per barrel/unit
                'Operating_Margin': 0.15,
                'Reserve_Life': 10,
                'Current_Ratio': 1.5
            }
        }

    def analyze_valuation(self, metrics, industry):
        """
        Analyze stock valuation using multiple metrics and return a comprehensive score
        Score range: 0 (extremely overvalued) to 100 (extremely undervalued)
        """
        scores = {}
        
        # 1. P/E Ratio Analysis (Weight: 25%)
        if metrics['P/E Ratio'] and metrics['P/E Ratio'] > 0:
            pe_score = self._score_pe_ratio(metrics['P/E Ratio'], self.industry_pe[industry])
            scores['P/E Score'] = pe_score
        
        # 2. P/B Ratio Analysis (Weight: 15%)
        if metrics['P/B Ratio'] and metrics['P/B Ratio'] > 0:
            pb_score = self._score_pb_ratio(metrics['P/B Ratio'], self.industry_pb[industry])
            scores['P/B Score'] = pb_score
        
        # 3. PEG Ratio Analysis (Weight: 20%)
        if metrics['PEG Ratio'] and metrics['PEG Ratio'] > 0:
            peg_score = self._score_peg_ratio(metrics['PEG Ratio'])
            scores['PEG Score'] = peg_score
        
        # 4. DCF Valuation Analysis (Weight: 30%)
        if metrics['Current Price'] and metrics['DCF Value']:
            dcf_score = self._score_dcf_value(metrics['Current Price'], metrics['DCF Value'])
            scores['DCF Score'] = dcf_score
        
        # 5. Financial Health Score (Weight: 10%)
        health_score = self._score_financial_health(metrics)
        scores['Financial Health Score'] = health_score
        
        # Calculate weighted average score
        weights = {
            'P/E Score': 0.25,
            'P/B Score': 0.15,
            'PEG Score': 0.20,
            'DCF Score': 0.30,
            'Financial Health Score': 0.10
        }
        
        final_score = 0
        valid_weights_sum = 0
        
        for metric, score in scores.items():
            if score is not None:
                final_score += score * weights[metric]
                valid_weights_sum += weights[metric]
        
        if valid_weights_sum > 0:
            final_score = final_score / valid_weights_sum
        
        # Generate recommendation
        recommendation = self.get_recommendation(metrics, final_score)  # Changed from _get_recommendation
            
        return {
            'detailed_scores': scores,
            'final_score': final_score,
            'valuation_status': self._get_valuation_status(final_score),
            'recommendation': recommendation
        }

    def _calculate_technical_score(self, metrics):
        """Calculate technical analysis score"""
        score = 50  # Start at neutral
    
        # RSI Analysis
        rsi = metrics.get('RSI')
        if rsi is not None:
            if rsi < 30:  # Oversold
                score += 20
            elif rsi < 40:
                score += 10
            elif rsi > 70:  # Overbought
                score -= 20
            elif rsi > 60:
                score -= 10
    
        # Moving Average Analysis
        ma50 = metrics.get('MA50')
        ma200 = metrics.get('MA200')
        if ma50 is not None and ma200 is not None:
            if ma50 > ma200:  # Golden Cross
                score += 15
            else:  # Death Cross
                score -= 15
    
        # Volume Analysis
        vol_avg = metrics.get('Volume_Average')
        vol_current = metrics.get('Volume_Current')
        if vol_avg is not None and vol_current is not None:
            if vol_current > vol_avg * 1.5:  # High volume
                score += 10
    
        return max(0, min(100, score))

    def _calculate_sector_score(self, metrics, sector):
        """Calculate sector-specific score"""
        score = 50  # Start at neutral
        thresholds = self.sector_metrics[sector]
    
        if sector == 'Technology':
            # Growth and Innovation Metrics
            if metrics.get('Revenue_Growth', 0) > thresholds['min_revenue_growth']:
                score += 15
        
            if metrics.get('R&D_Ratio', 0) > thresholds['min_r_and_d']:
                score += 10
        
            if metrics.get('Patent_Growth', 0) > 0:
                score += 10
        
            if metrics.get('Market_Share', 0) > 0.20:
                score += 15
            
        elif sector == 'Energy':
            # Resource and Efficiency Metrics
            if metrics.get('Reserve_Life', 0) > 10:
                score += 15
        
            if metrics.get('Production_Cost', float('inf')) < self.industry_averages[sector]['Production_Cost']:
                score += 15
        
            if metrics.get('ESG_Score', 0) > 70:
                score += 10
        
            if metrics.get('Portfolio_Diversity_Score', 0) > 0.7:
                score += 10
    
        return max(0, min(100, score))

    def _score_pe_ratio(self, pe_ratio, industry_avg):
        """Score P/E ratio relative to industry average"""
        if pe_ratio <= 0:
            return None
            
        if pe_ratio < industry_avg:
            return min(100, (1 - (pe_ratio / industry_avg)) * 100 + 50)
        else:
            return max(0, (2 - (pe_ratio / industry_avg)) * 50)
    
    def _score_pb_ratio(self, pb_ratio, industry_avg):
        """Score P/B ratio relative to industry average"""
        if pb_ratio <= 0:
            return None
            
        if pb_ratio < industry_avg:
            return min(100, (1 - (pb_ratio / industry_avg)) * 100 + 50)
        else:
            return max(0, (2 - (pb_ratio / industry_avg)) * 50)
    
    def _score_peg_ratio(self, peg_ratio):
        """Score PEG ratio (1 is considered fair value)"""
        if peg_ratio <= 0:
            return None
            
        if peg_ratio < 1:
            return min(100, (1 - peg_ratio) * 50 + 50)
        else:
            return max(0, (2 - peg_ratio) * 50)
    
    def _score_dcf_value(self, current_price, dcf_value):
        """Score based on DCF valuation"""
        if current_price <= 0 or dcf_value <= 0:
            return None
            
        ratio = current_price / dcf_value
        if ratio < 1:
            return min(100, (1 - ratio) * 100 + 50)
        else:
            return max(0, (2 - ratio) * 50)
    
    def _score_financial_health(self, metrics):
        """Score overall financial health"""
        score = 50  # Start at neutral
        
        # Check Free Cash Flow
        if metrics.get('FCF'):
            if metrics['FCF'] > 0:
                score += 10
            else:
                score -= 10
        
        # Check Debt/Equity
        if metrics.get('Debt/Equity'):
            if metrics['Debt/Equity'] < 1:
                score += 10
            elif metrics['Debt/Equity'] > 2:
                score -= 10
        
        return max(0, min(100, score))
    
    def _get_valuation_status(self, score):
        """Convert numerical score to valuation status"""
        if score >= 80:
            return "Significantly Undervalued"
        elif score >= 60:
            return "Moderately Undervalued"
        elif score >= 40:
            return "Fairly Valued"
        elif score >= 20:
            return "Moderately Overvalued"
        else:
            return "Significantly Overvalued"
    
    def get_recommendation(self, metrics, score):  # Changed from _get_recommendation to get_recommendation
        """Generate investment recommendation based on multiple factors"""
        # Base recommendation on valuation score
        if score >= 80:
            base_rec = "Strong Buy"
        elif score >= 60:
            base_rec = "Buy"
        elif score >= 40:
            base_rec = "Hold"
        elif score >= 20:
            base_rec = "Sell"
        else:
            base_rec = "Strong Sell"
        
        # Adjust recommendation based on additional factors
        adjustment_points = 0
        
        # Financial Health Adjustments
        fcf = metrics.get('FCF')
        if fcf is not None and fcf > 0:
            adjustment_points += 1

        debt_equity = metrics.get('Debt/Equity')
        if debt_equity is not None and debt_equity < 1:
            adjustment_points += 1
        
        # Risk Adjustments
        beta = metrics.get('Beta')
        if beta is not None:
            if beta < 0.8:  # Low volatility
                adjustment_points += 1
            elif beta > 1.5:  # High volatility
                adjustment_points -= 1
        
        # Dividend Consideration
        div_yield = metrics.get('Dividend Yield')
        if div_yield is not None and div_yield > 0.02:  # 2% yield threshold
            adjustment_points += 1
        
        # DCF Value vs Current Price
        current_price = metrics.get('Current Price')
        dcf_value = metrics.get('DCF Value')
        if current_price is not None and dcf_value is not None and current_price > 0:
            dcf_premium = (dcf_value - current_price) / current_price
            if dcf_premium > 0.3:  # 30% upside
                adjustment_points += 1
            elif dcf_premium < -0.3:  # 30% downside
                adjustment_points -= 1
        
        # Adjust final recommendation based on points
        rec_scale = {
            "Strong Buy": 2,
            "Buy": 1,
            "Hold": 0,
            "Sell": -1,
            "Strong Sell": -2
        }
        
        base_value = rec_scale[base_rec]
        adjusted_value = base_value + (adjustment_points * 0.5)  # Scale adjustment impact
        
        # Convert back to recommendation
        for rec, value in rec_scale.items():
            if adjusted_value >= value - 0.25:
                return rec
        
        return base_rec

class StockScreener:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.valuation_analyzer = ValuationAnalyzer()
        
    def get_financial_ratios(self, ticker):
        """Fetch financial ratios from FMP API"""
        endpoint = f"{self.base_url}/ratios-ttm/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return response.json()[0] if response.json() else None
    
    def get_company_profile(self, ticker):
        """Fetch company profile from FMP API"""
        endpoint = f"{self.base_url}/profile/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return response.json()[0] if response.json() else None
    
    def get_dcf_value(self, ticker):
        """Fetch DCF value from FMP API"""
        endpoint = f"{self.base_url}/discounted-cash-flow/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return response.json()[0] if response.json() else None
   
    def get_technical_indicators(self, ticker):
        """Fetch technical indicators from FMP API"""
        # RSI
        endpoint = f"{self.base_url}/technical_indicator/daily/{ticker}?period=14&type=rsi&apikey={self.api_key}"
        response = requests.get(endpoint)
        rsi_data = response.json() if response.json() else None
    
        # Moving Averages
        ma_endpoint = f"{self.base_url}/technical_indicator/daily/{ticker}?period=200&type=sma&apikey={self.api_key}"
        ma_response = requests.get(ma_endpoint)
        ma_data = ma_response.json() if ma_response.json() else None
    
        return {
            'RSI': rsi_data[0]['rsi'] if rsi_data else None,
            'MA50': self._calculate_ma(ticker, 50),
            'MA200': self._calculate_ma(ticker, 200),
            'Volume_Average': self._calculate_volume_average(ticker),
            'Volume_Current': self._get_current_volume(ticker)
        }

    def _calculate_ma(self, ticker, period):
        """Calculate moving average for specified period"""
        endpoint = f"{self.base_url}/historical-price-full/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        
        if not response.json():
            return None
            
        prices = pd.DataFrame(response.json()['historical'])
        if len(prices) >= period:
            return prices['close'].rolling(window=period).mean().iloc[0]
        return None

    def _calculate_volume_average(self, ticker, period=30):
        """Calculate average volume over specified period"""
        endpoint = f"{self.base_url}/historical-price-full/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        
        if not response.json():
            return None
            
        volumes = pd.DataFrame(response.json()['historical'])
        if len(volumes) >= period:
            return volumes['volume'].rolling(window=period).mean().iloc[0]
        return None

    def _get_current_volume(self, ticker):
        """Get most recent trading volume"""
        endpoint = f"{self.base_url}/historical-price-full/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        
        if not response.json():
            return None
            
        volumes = pd.DataFrame(response.json()['historical'])
        if not volumes.empty:
            return volumes['volume'].iloc[0]
        return None

    def get_sector_metrics(self, ticker, sector):
        """Fetch sector-specific metrics"""
        metrics = {}
    
        if sector == 'Technology':
            # Get growth metrics
            growth_data = self._get_growth_metrics(ticker)
            metrics.update({
                'Revenue_Growth': growth_data.get('revenue_growth'),
                'R&D_Ratio': self._calculate_rd_ratio(ticker),
                'Patent_Count': self._get_patent_data(ticker),
                'Market_Share': self._get_market_share(ticker)
            })
    
        elif sector == 'Energy':
            # Get energy-specific metrics
            energy_data = self._get_energy_metrics(ticker)
            metrics.update({
                'Reserve_Life': energy_data.get('reserve_life'),
                'Production_Cost': energy_data.get('production_cost'),
                'ESG_Score': self._get_esg_score(ticker),
                'Portfolio_Diversity_Score': self._calculate_portfolio_diversity(ticker)
            })
    
        return metrics

    def _get_growth_metrics(self, ticker):
        """Get growth related metrics"""
        endpoint = f"{self.base_url}/income-statement/{ticker}?limit=4&apikey={self.api_key}"
        response = requests.get(endpoint)
        
        if not response.json():
            return {'revenue_growth': None}
            
        statements = pd.DataFrame(response.json())
        if len(statements) >= 2:
            revenue_growth = (statements['revenue'].iloc[0] - statements['revenue'].iloc[1]) / statements['revenue'].iloc[1]
            return {'revenue_growth': revenue_growth}
        return {'revenue_growth': None}

    def _calculate_rd_ratio(self, ticker):
        """Calculate R&D spending ratio"""
        endpoint = f"{self.base_url}/income-statement/{ticker}?limit=1&apikey={self.api_key}"
        response = requests.get(endpoint)
        
        if not response.json():
            return None
            
        statement = response.json()[0]
        revenue = statement.get('revenue', 0)
        rd_expense = statement.get('researchAndDevelopmentExpenses', 0)
        
        if revenue > 0:
            return rd_expense / revenue
        return None

    def _get_patent_data(self, ticker):
        """Placeholder for patent data - would need separate API"""
        return None

    def _get_market_share(self, ticker):
        """Placeholder for market share calculation"""
        return None

    def _get_energy_metrics(self, ticker):
        """Placeholder for energy specific metrics"""
        return {
            'reserve_life': None,
            'production_cost': None
        }

    def _get_esg_score(self, ticker):
        """Placeholder for ESG score - would need separate API"""
        return None

    def _calculate_portfolio_diversity(self, ticker):
        """Placeholder for portfolio diversity calculation"""
        return None
    
    def get_historical_prices(self, ticker, period='1y'):
        """Fetch historical prices for beta and volatility calculation"""
        endpoint = f"{self.base_url}/historical-price-full/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        if response.json():
            df = pd.DataFrame(response.json()['historical'])
            df['date'] = pd.to_datetime(df['date'])
            return df
        return None
    
    def calculate_beta(self, ticker, market_index='^GSPC'):
        """Calculate beta relative to S&P 500"""
        stock_prices = self.get_historical_prices(ticker)
        market_prices = self.get_historical_prices(market_index)
        
        if stock_prices is None or market_prices is None:
            return None
        
        stock_returns = stock_prices['close'].pct_change().dropna()
        market_returns = market_prices['close'].pct_change().dropna()
        
        common_dates = stock_returns.index.intersection(market_returns.index)
        stock_returns = stock_returns[common_dates]
        market_returns = market_returns[common_dates]
        
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance != 0 else None
    
    def analyze_stock(self, ticker, sector):
        """Perform comprehensive stock analysis with enhanced metrics"""
        # Get base metrics
        ratios = self.get_financial_ratios(ticker)
        profile = self.get_company_profile(ticker)
        dcf = self.get_dcf_value(ticker)
        
        # Get new metrics
        technical_indicators = self.get_technical_indicators(ticker)
        sector_metrics = self.get_sector_metrics(ticker, sector)
        
        if not all([ratios, profile, dcf]):
            return None
        
        # Combine all metrics
        metrics = {
            'Ticker': ticker,
            'Company Name': profile['companyName'],
            'Current Price': profile['price'],
            'Market Cap': profile['mktCap'],
            'P/E Ratio': ratios['peRatioTTM'],
            'P/B Ratio': ratios['priceToBookRatioTTM'],
            'PEG Ratio': ratios.get('pegRatioTTM', None),
            'FCF': ratios.get('freeCashFlowTTM', None),
            'Dividend Yield': ratios.get('dividendYieldTTM', None),
            'Debt/Equity': ratios.get('debtEquityRatioTTM', None),
            'DCF Value': dcf['dcf'],
            'Beta': self.calculate_beta(ticker),
        }
        
        # Add technical and sector-specific metrics
        metrics.update(technical_indicators)
        metrics.update(sector_metrics)
        
        # Get valuation analysis with enhanced scoring
        valuation = self.valuation_analyzer.analyze_valuation(metrics, sector)
        
        # Add scores to metrics
        metrics.update({
            'Valuation Score': valuation['final_score'],
            'Technical Score': self._calculate_technical_score(metrics),
            'Sector Score': self._calculate_sector_score(metrics, sector),
            'Valuation Status': valuation['valuation_status'],
            'Recommendation': valuation['recommendation']
        })
        
        return metrics

    def _calculate_technical_score(self, metrics):
        """Calculate technical analysis score"""
        score = 50  # Start at neutral
        
        # RSI Analysis
        rsi = metrics.get('RSI')
        if rsi is not None:
            if rsi < 30:  # Oversold
                score += 20
            elif rsi < 40:
                score += 10
            elif rsi > 70:  # Overbought
                score -= 20
            elif rsi > 60:
                score -= 10
        
        # Moving Average Analysis
        ma50 = metrics.get('MA50')
        ma200 = metrics.get('MA200')
        if ma50 is not None and ma200 is not None:
            if ma50 > ma200:  # Golden Cross
                score += 15
            else:  # Death Cross
                score -= 15
        
        # Volume Analysis
        vol_avg = metrics.get('Volume_Average')
        vol_current = metrics.get('Volume_Current')
        if vol_avg is not None and vol_current is not None:
            if vol_current > vol_avg * 1.5:  # High volume
                score += 10
        
        return max(0, min(100, score))

    def _calculate_sector_score(self, metrics, sector):
        """Calculate sector-specific score"""
        score = 50  # Start at neutral
        
        if sector == "Technology":
            # Tech metrics scoring
            if metrics.get('Revenue_Growth'):
                if metrics['Revenue_Growth'] > 0.15:  # 15% growth
                    score += 15
                elif metrics['Revenue_Growth'] > 0.10:  # 10% growth
                    score += 10
                    
            if metrics.get('R&D_Ratio'):
                if metrics['R&D_Ratio'] > 0.15:  # 15% of revenue
                    score += 15
                elif metrics['R&D_Ratio'] > 0.10:  # 10% of revenue
                    score += 10
                    
            if metrics.get('Market_Share'):
                if metrics['Market_Share'] > 0.20:  # 20% market share
                    score += 20
                elif metrics['Market_Share'] > 0.10:  # 10% market share
                    score += 10
                    
        elif sector == "Energy":
            # Energy metrics scoring
            if metrics.get('Reserve_Life'):
                if metrics['Reserve_Life'] > 15:  # 15+ years
                    score += 15
                elif metrics['Reserve_Life'] > 10:  # 10+ years
                    score += 10
                    
            if metrics.get('Production_Cost'):
                industry_avg = 35  # Example industry average
                if metrics['Production_Cost'] < industry_avg * 0.8:  # 20% below average
                    score += 15
                elif metrics['Production_Cost'] < industry_avg:
                    score += 10
                    
            if metrics.get('ESG_Score'):
                if metrics['ESG_Score'] > 80:
                    score += 20
                elif metrics['ESG_Score'] > 70:
                    score += 10
        
        return max(0, min(100, score))
        
def main():
    st.title("Advanced Stock Screener with Enhanced Analysis")
    
    # Sidebar Configuration
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Enter your FMP API Key:", type="password")
    
    if api_key:
        screener = StockScreener(api_key)
        
        tech_stocks = ['META', 'TSLA', 'MSFT', 'NVDA', 'GOOG', 'AAPL', 'AMZN']
        energy_stocks = ['CCO.TO', 'ENB.TO', 'SU.TO', 'NXE.TO', 'XOM']
        
        sector = st.sidebar.selectbox("Select Sector:", ["Technology", "Energy"])
        stocks = tech_stocks if sector == "Technology" else energy_stocks
        
        if st.sidebar.button("Analyze Stocks"):
            with st.spinner('Performing comprehensive analysis...'):
                results = []
                progress_bar = st.progress(0)
                
                for i, ticker in enumerate(stocks):
                    analysis = screener.analyze_stock(ticker, sector)
                    if analysis:
                        results.append(analysis)
                    progress_bar.progress((i + 1) / len(stocks))
                
                if results:
                    df = pd.DataFrame(results)
                    
                    # Create tabs for different analysis views
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Summary Dashboard", 
                        "Technical Analysis", 
                        f"{sector} Metrics",
                        "Detailed Analysis"
                    ])
                    
                    with tab1:
                        # Summary Dashboard
                        st.subheader("Market Overview")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Average Valuation Score", 
                                f"{df['Valuation Score'].mean():.1f}",
                                delta=f"{df['Valuation Score'].mean() - 50:.1f} vs Neutral"
                            )
                        
                        with col2:
                            best_stock = df.loc[df['Valuation Score'].idxmax()]
                            st.metric(
                                "Top Pick", 
                                best_stock['Ticker'],
                                f"Score: {best_stock['Valuation Score']:.1f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Average Technical Score",
                                f"{df['Technical Score'].mean():.1f}",
                                delta=f"{df['Technical Score'].mean() - 50:.1f} vs Neutral"
                            )
                        
                        # Recommendations Distribution
                        st.subheader("Investment Recommendations")
                        rec_df = df.groupby('Recommendation').size().reset_index(name='Count')
                        fig_rec = px.pie(
                            rec_df, 
                            values='Count', 
                            names='Recommendation',
                            color='Recommendation',
                            color_discrete_map={
                                'Strong Buy': '#2E7D32',
                                'Buy': '#4CAF50',
                                'Hold': '#FFC107',
                                'Sell': '#F44336',
                                'Strong Sell': '#B71C1C'
                            }
                        )
                        st.plotly_chart(fig_rec)
                    
                    with tab2:
                        # Technical Analysis View
                        st.subheader("Technical Indicators")
                        
                        # RSI Analysis
                        fig_rsi = px.scatter(
                            df,
                            x='Ticker',
                            y='RSI',
                            color='Recommendation',
                            title='Relative Strength Index (RSI)',
                            color_discrete_map={
                                'Strong Buy': '#2E7D32',
                                'Buy': '#4CAF50',
                                'Hold': '#FFC107',
                                'Sell': '#F44336',
                                'Strong Sell': '#B71C1C'
                            }
                        )
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                        st.plotly_chart(fig_rsi)
                        
                        # Moving Averages
                        st.subheader("Moving Average Analysis")
                        ma_data = df[['Ticker', 'MA50', 'MA200']].melt(
                            id_vars=['Ticker'],
                            var_name='MA Type',
                            value_name='Value'
                        )
                        fig_ma = px.line(
                            ma_data,
                            x='Ticker',
                            y='Value',
                            color='MA Type',
                            title='Moving Averages Comparison'
                        )
                        st.plotly_chart(fig_ma)
                    
                    with tab3:
                        # Sector-Specific Metrics
                        st.subheader(f"{sector} Specific Analysis")
                        
                        if sector == "Technology":
                            # Tech metrics visualization
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_growth = px.bar(
                                    df,
                                    x='Ticker',
                                    y='Revenue_Growth',
                                    title='Revenue Growth Rate',
                                    color='Recommendation'
                                )
                                fig_growth.add_hline(
                                    y=screener.valuation_analyzer.industry_averages['Technology']['Revenue_Growth'],
                                    line_dash="dash",
                                    annotation_text="Industry Average"
                                )
                                st.plotly_chart(fig_growth)
                            
                            with col2:
                                fig_rd = px.bar(
                                    df,
                                    x='Ticker',
                                    y='R&D_Ratio',
                                    title='R&D Investment Ratio',
                                    color='Recommendation'
                                )
                                st.plotly_chart(fig_rd)
                            
                        else:  # Energy sector
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_reserves = px.bar(
                                    df,
                                    x='Ticker',
                                    y='Reserve_Life',
                                    title='Reserve Life (Years)',
                                    color='Recommendation'
                                )
                                st.plotly_chart(fig_reserves)
                            
                            with col2:
                                fig_esg = px.bar(
                                    df,
                                    x='Ticker',
                                    y='ESG_Score',
                                    title='ESG Scores',
                                    color='Recommendation'
                                )
                                st.plotly_chart(fig_esg)
                    
                    with tab4:
                        # Detailed Analysis Table
                        st.subheader("Comprehensive Analysis")
                        
                        # Define columns based on sector
                        base_cols = ['Ticker', 'Company Name', 'Current Price', 
                                   'Valuation Score', 'Technical Score', 'Sector Score',
                                   'Recommendation']
                        
                        tech_cols = ['Revenue_Growth', 'R&D_Ratio', 'Market_Share']
                        energy_cols = ['Reserve_Life', 'Production_Cost', 'ESG_Score']
                        
                        display_cols = base_cols + (tech_cols if sector == 'Technology' else energy_cols)
                        
                        # Style the dataframe
                        def color_recommendation(val):
                            colors = {
                                'Strong Buy': 'background-color: #2E7D32; color: white',
                                'Buy': 'background-color: #4CAF50; color: white',
                                'Hold': 'background-color: #FFC107',
                                'Sell': 'background-color: #F44336; color: white',
                                'Strong Sell': 'background-color: #B71C1C; color: white'
                            }
                            return colors.get(val, '')
                        
                        styled_df = df[display_cols].style\
                            .background_gradient(subset=['Valuation Score', 'Technical Score', 'Sector Score'], cmap='RdYlGn')\
                            .applymap(color_recommendation, subset=['Recommendation'])
                        
                        st.dataframe(styled_df)
                        
                        # Export functionality
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Complete Analysis",
                            data=csv,
                            file_name=f"stock_analysis_{sector}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )

if __name__ == "__main__":
    main()