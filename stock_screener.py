import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt 
import os 
from dotenv import load_dotenv
from sector_analysis import SectorType, SectorMetrics
from utils import get_api_key



class ValuationAnalyzer:
    def __init__(self):
        self.sector_metrics = SectorMetrics()
        
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

    def analyze_valuation(self, metrics, sector):
        """
        Analyze stock valuation using multiple metrics and return a comprehensive score
        Score range: 0 (extremely overvalued) to 100 (extremely undervalued)
        """
        try:
            sector_type = SectorType(sector)
        except ValueError:
            sector_type = SectorType.TECHNOLOGY  # Default fallback
            
        scores = {}
        
        # Get industry averages for the sector
        industry_pe = self.sector_metrics.INDUSTRY_PE[sector_type]
        industry_pb = self.sector_metrics.INDUSTRY_PB[sector_type]
        
        # Calculate scores using industry-specific benchmarks
        if metrics['P/E Ratio'] and metrics['P/E Ratio'] > 0:
            scores['P/E Score'] = self._score_pe_ratio(metrics['P/E Ratio'], industry_pe)
            
        if metrics['P/B Ratio'] and metrics['P/B Ratio'] > 0:
            scores['P/B Score'] = self._score_pb_ratio(metrics['P/B Ratio'], industry_pb)
        
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
        try:
            sector_type = SectorType(sector)
        except ValueError:
            sector_type = SectorType.TECHNOLOGY  # Default fallback
            
        score = 50  # Start at neutral
        sector_metrics = self.sector_metrics.BASE_METRICS.copy()
        
        # Add sector-specific metrics if available
        if sector_type in self.sector_metrics.SECTOR_CONFIGS:
            sector_metrics.update(
                self.sector_metrics.SECTOR_CONFIGS[sector_type]['specific_metrics']
            )
        
        total_weight = 0
        weighted_score = 0
        
        for metric_name, config in sector_metrics.items():
            value = metrics.get(metric_name)
            if value is not None and config['threshold'] is not None:
                # Calculate metric score
                metric_score = self._score_metric(value, config['threshold'])
                weighted_score += metric_score * config['weight']
                total_weight += config['weight']
        
        # Adjust for risk factors
        risk_adjustment = self._calculate_risk_adjustment(metrics, sector_type)
        
        # Calculate final score
        if total_weight > 0:
            final_score = (weighted_score / total_weight) * 0.8 + risk_adjustment * 0.2
            return max(0, min(100, final_score))
        
        return score

    def _score_metric(self, value, threshold):
        """Convert metric value to score between 0 and 100"""
        if threshold == 0:
            return 50
        
        relative_performance = value / threshold
        if relative_performance >= 2:
            return 100
        elif relative_performance >= 1:
            return 75 + (relative_performance - 1) * 25
        else:
            return max(0, relative_performance * 75)

    def _calculate_risk_adjustment(self, metrics, sector_type):
        """Calculate risk adjustment based on sector-specific risk factors"""
        risk_score = 50  # Neutral starting point
        
        # Get sector-specific risk factors
        sector_config = self.sector_metrics.SECTOR_CONFIGS.get(sector_type)
        if not sector_config or 'risk_factors' not in sector_config:
            return risk_score
        
        risk_factors = sector_config['risk_factors']
        
        # Technology sector risk adjustments
        if sector_type == SectorType.TECHNOLOGY:
            # Tech obsolescence risk
            if metrics.get('R&D_Ratio', 0) < 0.05:
                risk_score -= 10
            elif metrics.get('R&D_Ratio', 0) > 0.15:
                risk_score += 10
                
            # Cybersecurity risk
            if metrics.get('Security_Investment_Ratio', 0) > 0.05:
                risk_score += 5
                
            # Competition risk
            if metrics.get('Market_Share', 0) > 0.20:
                risk_score += 10
                
        # Financial sector risk adjustments
        elif sector_type == SectorType.FINANCIAL:
            # Interest rate risk
            if metrics.get('Interest_Rate_Sensitivity', 0) > 0.5:
                risk_score -= 10
                
            # Credit risk
            if metrics.get('NPL_Ratio', 0) < 0.02:
                risk_score += 10
            elif metrics.get('NPL_Ratio', 0) > 0.05:
                risk_score -= 10
                
            # Capital adequacy
            if metrics.get('Capital_Adequacy_Ratio', 0) > 0.12:
                risk_score += 10
                
        # Healthcare sector risk adjustments
        elif sector_type == SectorType.HEALTHCARE:
            # Regulatory risk
            if metrics.get('Regulatory_Compliance_Score', 0) > 0.8:
                risk_score += 10
                
            # Patent expiry risk
            if metrics.get('Patent_Protection_Years', 0) > 5:
                risk_score += 5
                
            # Clinical trial risk
            if metrics.get('Clinical_Trial_Success_Rate', 0) > 0.6:
                risk_score += 10
                
        # Energy sector risk adjustments
        elif sector_type == SectorType.ENERGY:
            # Environmental risk
            if metrics.get('ESG_Score', 0) > 70:
                risk_score += 10
                
            # Resource depletion risk
            if metrics.get('Reserve_Life', 0) > 15:
                risk_score += 10
                
            # Production efficiency
            if metrics.get('Production_Cost', 0) < self.sector_metrics.BASE_METRICS['Operating_Margin']['threshold']:
                risk_score += 5
        
        # Add similar conditions for other sectors...
        
        return max(0, min(100, risk_score))

    def get_sector_metrics_breakdown(self, metrics, sector_type):
        """Get detailed breakdown of sector-specific metrics"""
        breakdown = {
            'base_metrics': {},
            'sector_specific': {},
            'risk_factors': {}
        }
        
        # Add base metrics
        for metric, config in self.sector_metrics.BASE_METRICS.items():
            value = metrics.get(metric)
            if value is not None:
                breakdown['base_metrics'][metric] = {
                    'value': value,
                    'threshold': config['threshold'],
                    'score': self._score_metric(value, config['threshold'])
                }
        
        # Add sector-specific metrics
        sector_config = self.sector_metrics.SECTOR_CONFIGS.get(sector_type)
        if sector_config and 'specific_metrics' in sector_config:
            for metric, config in sector_config['specific_metrics'].items():
                value = metrics.get(metric)
                if value is not None:
                    breakdown['sector_specific'][metric] = {
                        'value': value,
                        'threshold': config['threshold'],
                        'score': self._score_metric(value, config['threshold'])
                    }
        
        # Add risk factors
        if sector_config and 'risk_factors' in sector_config:
            for risk_factor in sector_config['risk_factors']:
                value = metrics.get(f'{risk_factor}_Risk')
                if value is not None:
                    breakdown['risk_factors'][risk_factor] = value
        
        return breakdown

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
        self.sector_metrics = SectorMetrics()
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
        """Simplified patent data (since FMP doesn't provide direct patent info)"""
        try:
            # Use intangible assets as a proxy for patent value
            balance_sheet = self._get_balance_sheet(ticker)
            if not balance_sheet:
                return None
            
            intangible_assets = balance_sheet.get('intangibleAssets', 0)
            total_assets = balance_sheet.get('totalAssets', 1)
            
            return {
                'intangible_assets': intangible_assets,
                'intangible_ratio': intangible_assets / total_assets if total_assets > 0 else 0
            }
        except Exception as e:
            print(f"Error calculating patent data: {e}")
            return None

    def _get_market_share(self, ticker):
        """Placeholder for market share calculation"""
        return None

    def _get_energy_metrics(self, ticker):
        """Get energy sector specific metrics"""
        ratios = self._get_financial_ratios(ticker)
        esg_data = self._get_esg_scores(ticker)
        production = self._get_production_metrics(ticker)
        reserves = self._get_reserve_estimates(ticker)
        
        if not all([ratios, esg_data, production]):
            return {}
        
        return {
            'Production_Cost': production.get('costPerUnit'),
            'Reserve_Life': reserves.get('reserveLife'),
            'ESG_Score': esg_data.get('totalScore'),
            'Environmental_Score': esg_data.get('environmentalScore'),
            'Production_Efficiency': production.get('productionEfficiency'),
            'Reserve_Replacement_Ratio': reserves.get('replacementRatio'),
            'Carbon_Intensity': esg_data.get('carbonIntensity')
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
        try:
            # Get base metrics
            ratios = self._get_financial_ratios(ticker)
            profile = self._get_company_profile(ticker)
            dcf = self._get_dcf_value(ticker)
            
            if not all([ratios, profile, dcf]):
                st.error(f"Missing required base data for {ticker}")
                return None
                
            # Get enhanced metrics
            composite_metrics = self._calculate_composite_metrics(ticker, sector)
            technical_indicators = self.get_technical_indicators(ticker)
            sector_metrics = self.get_sector_metrics(ticker, sector)
            
            # Get raw sector from profile for display
            raw_sector = profile.get('sector', 'Unknown')
            
            # Calculate scores
            valuation_score = self._calculate_valuation_score(ratios, dcf, sector)
            technical_score = self._calculate_technical_score(technical_indicators) if technical_indicators else 50
            sector_score = self._calculate_sector_score(sector_metrics, sector) if sector_metrics else 50
            
            # Combine all metrics
            metrics = {
                'Ticker': ticker,
                'Company Name': profile.get('companyName', 'Unknown'),
                'Sector': raw_sector,  # Use raw sector name from API
                'Current Price': profile.get('price', 0),
                'Market Cap': profile.get('mktCap', 0),
                'P/E Ratio': ratios.get('peRatioTTM', 0),
                'P/B Ratio': ratios.get('priceToBookRatioTTM', 0),
                'Valuation Score': valuation_score,
                'Technical Score': technical_score,
                'Sector Score': sector_score,
                'RSI': technical_indicators.get('RSI', 50) if technical_indicators else 50,
                'MA50': technical_indicators.get('MA50', 0) if technical_indicators else 0,
                'MA200': technical_indicators.get('MA200', 0) if technical_indicators else 0,
                'Volume': technical_indicators.get('Volume', 0) if technical_indicators else 0,
                'Recommendation': self._generate_recommendation(
                    valuation_score=valuation_score,
                    technical_score=technical_score,
                    sector_score=sector_score
                )
            }
            
            # Add any sector-specific metrics if available
            if sector_metrics:
                metrics.update({
                    'Revenue': sector_metrics.get('Revenue', 0),
                    'Operating Margin': sector_metrics.get('Operating Margin', 0),
                    'Asset Turnover': sector_metrics.get('Asset Turnover', 0),
                    'Debt to Equity': sector_metrics.get('Debt to Equity', 0)
                })
            
            return metrics
            
        except Exception as e:
            st.error(f"Failed to analyze {ticker}")
            return None

    def get_technical_indicators(self, ticker):
        """Get technical indicators for the stock"""
        try:
            # Implement basic technical analysis
            return {
                'RSI': 50,  # Placeholder
                'MA50': 0,
                'MA200': 0,
                'Volume': 0
            }
        except Exception as e:
            st.write(f"Error getting technical indicators: {e}")
            return None

    def get_sector_metrics(self, ticker, sector):
        """Get sector-specific metrics"""
        try:
            # Get financial statements
            income_stmt = self._get_income_statement(ticker)
            balance_sheet = self._get_balance_sheet(ticker)
            
            if not income_stmt or not balance_sheet:
                return None
                
            # Calculate metrics
            revenue = float(income_stmt.get('revenue', 0))
            operating_income = float(income_stmt.get('operatingIncome', 0))
            total_assets = float(balance_sheet.get('totalAssets', 1))
            total_equity = float(balance_sheet.get('totalStockholdersEquity', 1))
            total_debt = float(balance_sheet.get('totalDebt', 0))
            
            # Calculate ratios
            operating_margin = operating_income / revenue if revenue > 0 else 0
            asset_turnover = revenue / total_assets if total_assets > 0 else 0
            debt_to_equity = total_debt / total_equity if total_equity > 0 else 0
            
            return {
                'Revenue': revenue,
                'Operating Margin': operating_margin,
                'Asset Turnover': asset_turnover,
                'Debt to Equity': debt_to_equity
            }
        
        except Exception as e:
            st.write(f"Error getting sector metrics: {e}")
            return None

    def _calculate_valuation_score(self, ratios, dcf, sector):
        """Calculate valuation score"""
        try:
            # Simple valuation score based on P/E ratio
            pe_ratio = ratios.get('peRatioTTM', 0)
            if pe_ratio <= 0:
                return 50
            
            # Compare to industry average
            industry_pe = self.sector_metrics.INDUSTRY_PE.get(sector, 20)
            if pe_ratio < industry_pe * 0.7:
                return 80
            elif pe_ratio < industry_pe:
                return 65
            elif pe_ratio < industry_pe * 1.3:
                return 45
            else:
                return 30
        except Exception as e:
            st.write(f"Error calculating valuation score: {e}")
            return 50

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
        try:
            sector_type = SectorType(sector)
        except ValueError:
            sector_type = SectorType.TECHNOLOGY  # Default fallback
            
        score = 50  # Start at neutral
        sector_metrics = self.sector_metrics.BASE_METRICS.copy()
        
        # Add sector-specific metrics if available
        if sector_type in self.sector_metrics.SECTOR_CONFIGS:
            sector_metrics.update(
                self.sector_metrics.SECTOR_CONFIGS[sector_type]['specific_metrics']
            )
        
        total_weight = 0
        weighted_score = 0
        
        for metric_name, config in sector_metrics.items():
            value = metrics.get(metric_name)
            if value is not None and config['threshold'] is not None:
                # Calculate metric score
                metric_score = self._score_metric(value, config['threshold'])
                weighted_score += metric_score * config['weight']
                total_weight += config['weight']
        
        # Adjust for risk factors
        risk_adjustment = self._calculate_risk_adjustment(metrics, sector_type)
        
        # Calculate final score
        if total_weight > 0:
            final_score = (weighted_score / total_weight) * 0.8 + risk_adjustment * 0.2
            return max(0, min(100, final_score))
        
        return score

    def _score_metric(self, value, threshold):
        """Convert metric value to score between 0 and 100"""
        if threshold == 0:
            return 50
        
        relative_performance = value / threshold
        if relative_performance >= 2:
            return 100
        elif relative_performance >= 1:
            return 75 + (relative_performance - 1) * 25
        else:
            return max(0, relative_performance * 75)

    def _calculate_risk_adjustment(self, metrics, sector_type):
        """Calculate risk adjustment based on sector-specific factors"""
        risk_score = 50  # Neutral starting point
        
        if sector_type == SectorType.TECHNOLOGY:
            # Tech Obsolescence Risk
            if metrics.get('Product_Lifecycle', 0) < 2:  # Years
                risk_score -= 10
            
            # Cybersecurity Risk
            security_investment = metrics.get('Security_Investment_Ratio', 0)
            if security_investment < 0.05:  # Less than 5% of revenue
                risk_score -= 10
            
            # Competition Risk
            market_share = metrics.get('Market_Share', 0)
            if market_share < 0.10:  # Less than 10% market share
                risk_score -= 10
                
        elif sector_type == SectorType.HEALTHCARE:
            # Regulatory Risk
            if metrics.get('Regulatory_Compliance_Score', 0) < 0.8:
                risk_score -= 15
            
            # Patent Expiry Risk
            patent_years = metrics.get('Patent_Protection_Years', 0)
            if patent_years < 5:
                risk_score -= 10
            
            # Clinical Trial Risk
            trial_success = metrics.get('Clinical_Trial_Success_Rate', 0)
            if trial_success < 0.5:
                risk_score -= 10
            
        elif sector_type == SectorType.FINANCIAL:
            # Interest Rate Risk
            rate_sensitivity = metrics.get('Interest_Rate_Sensitivity', 0)
            if abs(rate_sensitivity) > 0.2:
                risk_score -= 10
            
            # Credit Risk
            npl_ratio = metrics.get('NPL_Ratio', 0)
            if npl_ratio > 0.05:
                risk_score -= 15
            
            # Market Risk
            var_ratio = metrics.get('Value_At_Risk_Ratio', 0)
            if var_ratio > 0.03:
                risk_score -= 10
            
        elif sector_type == SectorType.CONSUMER_CYCLICAL:
            # Consumer Confidence Risk
            if metrics.get('Consumer_Confidence_Impact', 0) > 0.7:
                risk_score -= 10
            
            # Economic Cycle Risk
            beta = metrics.get('Beta', 0)
            if beta > 1.5:
                risk_score -= 10
            
            # Fashion/Trend Risk
            inventory_turnover = metrics.get('Inventory_Turnover', 0)
            if inventory_turnover < 4:
                risk_score -= 10
            
        elif sector_type == SectorType.ENERGY:
            # Environmental Risk
            esg_score = metrics.get('ESG_Score', 0)
            if esg_score < 70:
                risk_score -= 15
            
            # Resource Depletion Risk
            reserve_life = metrics.get('Reserve_Life', 0)
            if reserve_life < 10:
                risk_score -= 10
            
            # Production Efficiency
            production_cost = metrics.get('Production_Cost', 0)
            if production_cost > 35:
                risk_score -= 10
            
        elif sector_type == SectorType.UTILITIES:
            # Regulatory Risk
            regulatory_roe = metrics.get('Regulatory_ROE', 0)
            if regulatory_roe < 0.09:
                risk_score -= 15
            
            # Environmental Risk
            renewable_mix = metrics.get('Renewable_Mix', 0)
            if renewable_mix < 0.2:
                risk_score -= 10
            
            # Infrastructure Risk
            infrastructure_age = metrics.get('Infrastructure_Age', 0)
            if infrastructure_age > 25:
                risk_score -= 10
            
        elif sector_type == SectorType.REAL_ESTATE:
            # Interest Rate Risk
            rate_sensitivity = metrics.get('Interest_Rate_Sensitivity', 0)
            if abs(rate_sensitivity) > 0.15:
                risk_score -= 10
            
            # Market Cycle Risk
            occupancy_rate = metrics.get('Occupancy_Rate', 0)
            if occupancy_rate < 0.85:
                risk_score -= 15
            
            # Location Risk
            location_score = metrics.get('Location_Score', 0)
            if location_score < 0.7:
                risk_score -= 10
            
        elif sector_type == SectorType.COMMUNICATION:
            # Technology Change Risk
            network_upgrade_cost = metrics.get('Network_Upgrade_Cost_Ratio', 0)
            if network_upgrade_cost > 0.2:
                risk_score -= 10
            
            # Regulatory Risk
            regulatory_compliance = metrics.get('Regulatory_Compliance_Score', 0)
            if regulatory_compliance < 0.8:
                risk_score -= 15
            
            # Competition Risk
            market_share = metrics.get('Market_Share', 0)
            if market_share < 0.15:
                risk_score -= 10
            
        elif sector_type == SectorType.INDUSTRIALS:
            # Economic Cycle Risk
            if metrics.get('Economic_Sensitivity', 0) > 0.7:
                risk_score -= 10
            
            # Raw Materials Risk
            materials_cost_ratio = metrics.get('Raw_Materials_Cost_Ratio', 0)
            if materials_cost_ratio > 0.4:
                risk_score -= 10
            
            # Labor Relations Risk
            labor_risk_score = metrics.get('Labor_Relations_Score', 0)
            if labor_risk_score < 0.7:
                risk_score -= 10
            
        elif sector_type == SectorType.BASIC_MATERIALS:
            # Commodity Price Risk
            price_volatility = metrics.get('Price_Volatility', 0)
            if price_volatility > 0.3:
                risk_score -= 15
            
            # Environmental Impact Score
            environmental_impact = metrics.get('Environmental_Impact_Score', 0)
            if environmental_impact < 60:
                risk_score -= 10
            
            # Geopolitical Risk
            geopolitical_exposure = metrics.get('Geopolitical_Risk_Score', 0)
            if geopolitical_exposure > 0.6:
                risk_score -= 10
        
        # Ensure risk score stays within bounds
        return max(0, min(100, risk_score))

    def validate_ticker(self, ticker):
        """Verify if ticker exists and get basic info"""
        endpoint = f"{self.base_url}/quote/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return bool(response.json())

    def search_suggestions(self, partial_query):
        """Get stock suggestions as user types"""
        endpoint = f"{self.base_url}/search?query={partial_query}&apikey={self.api_key}"
        response = requests.get(endpoint)
        return [item['symbol'] for item in response.json()]

    def create_sector_visualizations(self, metrics, sector_type):
        """Create comprehensive sector-specific visualizations"""
        figures = {}
        
        # Get sector metrics breakdown
        breakdown = self.get_sector_metrics_breakdown(metrics, sector_type)
        
        # 1. Radar Chart for Base Metrics
        base_metrics_fig = go.Figure()
        
        metrics_values = []
        metrics_thresholds = []
        metric_names = []
        
        for metric, data in breakdown['base_metrics'].items():
            if data['value'] is not None and data['threshold'] is not None:
                metrics_values.append(data['value'])
                metrics_thresholds.append(data['threshold'])
                metric_names.append(metric)
        
        base_metrics_fig.add_trace(go.Scatterpolar(
            r=metrics_values,
            theta=metric_names,
            fill='toself',
            name='Current Values'
        ))
        
        base_metrics_fig.add_trace(go.Scatterpolar(
            r=metrics_thresholds,
            theta=metric_names,
            fill='toself',
            name='Industry Thresholds'
        ))
        
        base_metrics_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(metrics_values + metrics_thresholds) * 1.2])),
            showlegend=True,
            title='Base Metrics Analysis'
        )
        
        figures['base_metrics'] = base_metrics_fig
        
        # 2. Sector-Specific Metrics Bar Chart
        if breakdown['sector_specific']:
            sector_metrics_values = []
            sector_metrics_thresholds = []
            sector_metric_names = []
            
            for metric, data in breakdown['sector_specific'].items():
                if data['value'] is not None and data['threshold'] is not None:
                    sector_metrics_values.append(data['value'])
                    sector_metrics_thresholds.append(data['threshold'])
                    sector_metric_names.append(metric)
            
            sector_fig = go.Figure(data=[
                go.Bar(name='Current Values', x=sector_metric_names, y=sector_metrics_values),
                go.Bar(name='Industry Thresholds', x=sector_metric_names, y=sector_metrics_thresholds)
            ])
            
            sector_fig.update_layout(
                barmode='group',
                title=f'{sector_type.value} Specific Metrics',
                xaxis_title='Metrics',
                yaxis_title='Values'
            )
            
            figures['sector_specific'] = sector_fig
        
        # 3. Risk Factors Gauge Chart
        if breakdown['risk_factors']:
            risk_values = list(breakdown['risk_factors'].values())
            risk_names = list(breakdown['risk_factors'].keys())
            
            risk_fig = go.Figure()
            
            for i, (name, value) in enumerate(zip(risk_names, risk_values)):
                risk_fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    domain={'row': i, 'column': 0},
                    title={'text': name},
                    gauge={'axis': {'range': [0, 100]},
                           'steps': [
                               {'range': [0, 33], 'color': "lightgreen"},
                               {'range': [33, 66], 'color': "yellow"},
                               {'range': [66, 100], 'color': "red"}
                           ]}
                ))
            
            risk_fig.update_layout(
                grid={'rows': len(risk_names), 'columns': 1, 'pattern': "independent"},
                height=200 * len(risk_names),
                title='Risk Factor Analysis'
            )
            
            figures['risk_factors'] = risk_fig
        
        # 4. Composite Score Gauge
        composite_score = metrics.get('Valuation Score', 50)
        
        score_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=composite_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Score"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 20], 'color': "red"},
                    {'range': [20, 40], 'color': "orange"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "lightgreen"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': composite_score
                }
            }
        ))
        
        score_fig.update_layout(height=400)
        figures['composite_score'] = score_fig
        
        # 5. Historical Performance (if available)
        if 'historical_data' in metrics:
            hist_fig = go.Figure()
            hist_data = metrics['historical_data']
            
            hist_fig.add_trace(go.Scatter(
                x=hist_data['dates'],
                y=hist_data['values'],
                mode='lines',
                name='Historical Performance'
            ))
            
            hist_fig.update_layout(
                title='Historical Performance Analysis',
                xaxis_title='Date',
                yaxis_title='Value'
            )
            
            figures['historical'] = hist_fig
        
        return figures

    def display_sector_analysis(self, stock_data, sector_type):
        """Display sector-specific analysis"""
        # Get unique sectors from the data
        if isinstance(stock_data, pd.DataFrame):
            unique_sectors = stock_data['Sector'].unique()
        else:
            unique_sectors = [stock_data.get('Sector', 'Unknown')]
            
        # Create sector selector in sidebar
        selected_sector = st.sidebar.radio(
            "Select Sector for Analysis",
            unique_sectors
        )
            
        st.subheader(f"Sector Analysis: {selected_sector}")
        
        # Filter data for selected sector
        if isinstance(stock_data, pd.DataFrame):
            sector_stocks = stock_data[stock_data['Sector'] == selected_sector]
            sector_data = sector_stocks.iloc[0].to_dict()
        else:
            sector_data = stock_data
        
        # Display sector-specific metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Key Sector Metrics")
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Sector Score',
                    'Operating Margin',
                    'Asset Turnover',
                    'Debt to Equity'
                ],
                'Value': [
                    f"{sector_data.get('Sector Score', 0):.1f}",
                    f"{sector_data.get('Operating Margin', 0):.1%}",
                    f"{sector_data.get('Asset Turnover', 0):.2f}",
                    f"{sector_data.get('Debt to Equity', 0):.2f}"
                ]
            })
            st.dataframe(metrics_df)
        
        with col2:
            st.write("Sector Performance")
            if selected_sector == 'Technology':
                metrics = {
                    'R&D Intensity': f"{sector_data.get('R&D Ratio', 0):.1%}",
                    'Gross Margin': f"{sector_data.get('Gross Margin', 0):.1%}",
                    'Revenue Growth': f"{sector_data.get('Revenue Growth', 0):.1%}"
                }
            elif selected_sector == 'Financial Services':
                metrics = {
                    'Net Interest Margin': f"{sector_data.get('Net Interest Margin', 0):.1%}",
                    'Return on Assets': f"{sector_data.get('ROA', 0):.1%}",
                    'Capital Adequacy': f"{sector_data.get('Capital Ratio', 0):.1%}"
                }
            elif selected_sector == 'Healthcare':
                metrics = {
                    'R&D Intensity': f"{sector_data.get('R&D Ratio', 0):.1%}",
                    'Operating Margin': f"{sector_data.get('Operating Margin', 0):.1%}",
                    'Revenue per Employee': f"${sector_data.get('Revenue per Employee', 0):,.0f}"
                }
            else:
                metrics = {
                    'Operating Margin': f"{sector_data.get('Operating Margin', 0):.1%}",
                    'Asset Turnover': f"{sector_data.get('Asset Turnover', 0):.2f}",
                    'Revenue Growth': f"{sector_data.get('Revenue Growth', 0):.1%}"
                }
            
            for metric, value in metrics.items():
                st.metric(metric, value)
        
        # Add sector comparison chart
        st.write("Sector Comparison")
        
        # Get industry averages from SectorMetrics
        industry_pe = self.sector_metrics.INDUSTRY_PE.get(SectorType(selected_sector), 20)
        industry_pb = self.sector_metrics.INDUSTRY_PB.get(SectorType(selected_sector), 2)
        
        comparison_data = pd.DataFrame({
            'Metric': ['P/E Ratio', 'P/B Ratio', 'Operating Margin'],
            'Company': [
                sector_data.get('P/E Ratio', 0),
                sector_data.get('P/B Ratio', 0),
                sector_data.get('Operating Margin', 0)
            ],
            'Industry Average': [industry_pe, industry_pb, 0.15]  # Example industry average
        })
        
        # Create comparison chart
        fig = px.bar(
            comparison_data,
            x='Metric',
            y=['Company', 'Industry Average'],
            barmode='group',
            title='Company vs Industry Average'
        )
        st.plotly_chart(fig)

    def _calculate_composite_metrics(self, ticker, sector_type):
        """Calculate composite metrics from multiple data sources"""
        metrics = {}
        
        # Get base financial data
        income_stmt = self._get_income_statement(ticker)
        balance_sheet = self._get_balance_sheet(ticker)
        cash_flow = self._get_cash_flow_statement(ticker)
        key_metrics = self._get_key_metrics(ticker)
        
        if all([income_stmt, balance_sheet, cash_flow, key_metrics]):
            # Financial Health Score
            metrics['Financial_Health'] = self._calculate_financial_health(
                income_stmt, balance_sheet, cash_flow
            )
            
            # Growth Metrics
            metrics['Revenue_Growth_3Y'] = self._calculate_cagr(ticker, 'revenue', years=3)
            metrics['Earnings_Growth_3Y'] = self._calculate_cagr(ticker, 'netIncome', years=3)
            metrics['FCF_Growth_3Y'] = self._calculate_cagr(ticker, 'freeCashFlow', years=3)
            
            # Quality Metrics
            metrics['Earnings_Quality'] = self._calculate_earnings_quality(
                income_stmt, cash_flow
            )
            
            # Efficiency Metrics
            metrics['Capital_Efficiency'] = self._calculate_capital_efficiency(
                income_stmt, balance_sheet
            )
            
            # Sector-specific composite metrics
            if sector_type == SectorType.TECHNOLOGY:
                metrics.update(self._calculate_tech_composites(ticker))
            elif sector_type == SectorType.HEALTHCARE:
                metrics.update(self._calculate_healthcare_composites(ticker))
            # ... add other sectors as needed
        
        return metrics

    def _calculate_financial_health(self, income_stmt, balance_sheet, cash_flow):
        """Calculate comprehensive financial health score with better error handling"""
        try:
            score = 50  # Base score
            
            # Profitability checks with safety
            gross_profit_ratio = income_stmt.get('grossProfitRatio', 0)
            operating_income_ratio = income_stmt.get('operatingIncomeRatio', 0)
            
            if gross_profit_ratio > 0.3:
                score += 10
            if operating_income_ratio > 0.15:
                score += 10
                
            # Liquidity checks with safety
            try:
                current_assets = float(balance_sheet.get('totalCurrentAssets', 0))
                current_liabilities = float(balance_sheet.get('totalCurrentLiabilities', 0))
                
                if current_assets == 0 or current_liabilities == 0:
                    total_assets = float(balance_sheet.get('totalAssets', 1))
                    total_liabilities = float(balance_sheet.get('totalLiabilities', 1))
                    
                    if total_assets > 0 and total_liabilities > 0:
                        assets_to_liabilities = total_assets / total_liabilities
                        if assets_to_liabilities > 1.1:
                            score += 10
                else:
                    current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
                    if current_ratio > 1.5:
                        score += 10
                
            except (ValueError, TypeError, ZeroDivisionError):
                pass
            
            # Cash Flow Quality check with safety
            try:
                ocf = float(cash_flow.get('operatingCashFlow', 0))
                net_income = float(income_stmt.get('netIncome', 0))
                
                if net_income > 0 and ocf > net_income:
                    score += 10
                    
            except (ValueError, TypeError):
                pass
            
            # Additional financial institution specific metrics
            if 'totalDeposits' in balance_sheet:
                try:
                    tier1_capital = float(balance_sheet.get('tier1Capital', 0))
                    risk_weighted_assets = float(balance_sheet.get('riskWeightedAssets', 1))
                    capital_ratio = tier1_capital / risk_weighted_assets if risk_weighted_assets > 0 else 0
                    
                    if capital_ratio > 0.1:
                        score += 10
                        
                    net_interest_income = float(income_stmt.get('netInterestIncome', 0))
                    average_earning_assets = float(balance_sheet.get('averageEarningAssets', 1))
                    nim = net_interest_income / average_earning_assets if average_earning_assets > 0 else 0
                    
                    if nim > 0.02:
                        score += 10
                        
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            
            return min(100, score)
            
        except Exception:
            return 50  # Return neutral score on error

    def _calculate_earnings_quality(self, income_stmt, cash_flow):
        """Assess earnings quality through various metrics"""
        net_income = income_stmt.get('netIncome', 0)
        operating_cash_flow = cash_flow.get('operatingCashFlow', 0)
        
        if net_income <= 0:
            return 0
            
        # Calculate accruals ratio
        accruals = (operating_cash_flow - net_income) / abs(net_income)
        
        # Score based on accruals ratio
        if accruals > 0:
            return min(100, 50 + accruals * 50)  # Higher score for positive cash flow vs earnings
        else:
            return max(0, 50 + accruals * 50)

    def _calculate_capital_efficiency(self, income_stmt, balance_sheet):
        """Calculate capital efficiency metrics"""
        total_assets = balance_sheet.get('totalAssets', 0)
        net_income = income_stmt.get('netIncome', 0)
        
        if total_assets <= 0:
            return 0
            
        roa = net_income / total_assets
        return min(100, roa * 500)  # Scale ROA to 0-100 score

    def _calculate_cagr(self, ticker, metric, years=3):
        """Calculate Compound Annual Growth Rate for a given metric"""
        endpoint = f"{self.base_url}/income-statement/{ticker}?limit={years+1}&apikey={self.api_key}"
        response = requests.get(endpoint)
        
        if not response.json() or len(response.json()) < years+1:
            return None
            
        data = response.json()
        start_value = data[years].get(metric, 0)
        end_value = data[0].get(metric, 0)
        
        if start_value <= 0:
            return None
            
        return (pow(end_value / start_value, 1/years) - 1) * 100

    def _calculate_tech_composites(self, ticker):
        """Calculate technology sector composite metrics"""
        metrics = {}
        
        # Innovation Metrics
        rd_data = self._get_rd_metrics(ticker)
        patent_data = self._get_patent_data(ticker)
        
        if rd_data and patent_data:
            metrics['Innovation_Score'] = self._calculate_innovation_score(rd_data, patent_data)
        
        # Digital Transformation
        digital_metrics = self._get_digital_metrics(ticker)
        if digital_metrics:
            metrics['Digital_Transformation_Score'] = self._calculate_digital_score(digital_metrics)
        
        return metrics

    def _calculate_healthcare_composites(self, ticker):
        """Calculate healthcare sector composite metrics"""
        metrics = {}
        
        # Pipeline Success
        pipeline_data = self._get_drug_pipeline(ticker)
        if pipeline_data:
            metrics['Pipeline_Score'] = self._calculate_pipeline_score(pipeline_data)
        
        # Research Impact
        research_data = self._get_research_impact(ticker)
        if research_data:
            metrics['Research_Impact_Score'] = self._calculate_research_score(research_data)
        
        return metrics

    def _get_income_statement(self, ticker):
        """Fetch income statement data from FMP API"""
        endpoint = f"{self.base_url}/income-statement/{ticker}?limit=1&apikey={self.api_key}"
        response = requests.get(endpoint)
        return response.json()[0] if response.json() else None

    def _get_balance_sheet(self, ticker):
        """Fetch balance sheet data from FMP API"""
        endpoint = f"{self.base_url}/balance-sheet-statement/{ticker}?limit=1&apikey={self.api_key}"
        response = requests.get(endpoint)
        return response.json()[0] if response.json() else None

    def _get_cash_flow_statement(self, ticker):
        """Fetch cash flow statement data from FMP API"""
        endpoint = f"{self.base_url}/cash-flow-statement/{ticker}?limit=1&apikey={self.api_key}"
        response = requests.get(endpoint)
        return response.json()[0] if response.json() else None

    def _get_key_metrics(self, ticker):
        """Fetch key metrics from FMP API"""
        endpoint = f"{self.base_url}/key-metrics-ttm/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return response.json()[0] if response.json() else None

    def _get_financial_ratios(self, ticker):
        """Fetch financial ratios from FMP API"""
        endpoint = f"{self.base_url}/ratios-ttm/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return response.json()[0] if response.json() else None

    def _get_company_profile(self, ticker):
        """Fetch company profile from FMP API"""
        endpoint = f"{self.base_url}/profile/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return response.json()[0] if response.json() else None

    def _get_dcf_value(self, ticker):
        """Fetch DCF valuation from FMP API"""
        endpoint = f"{self.base_url}/discounted-cash-flow/{ticker}?apikey={self.api_key}"
        response = requests.get(endpoint)
        return response.json()[0] if response.json() else None

    def _get_rd_metrics(self, ticker):
        """Calculate R&D metrics from income statement"""
        try:
            income_stmt = self._get_income_statement(ticker)
            if not income_stmt:
                return None
            
            revenue = income_stmt.get('revenue', 0)
            rd_expense = income_stmt.get('researchAndDevelopmentExpenses', 0)
            
            return {
                'rd_expense': rd_expense,
                'rd_ratio': rd_expense / revenue if revenue > 0 else 0,
                'rd_growth': self._calculate_metric_growth(ticker, 'researchAndDevelopmentExpenses')
            }
        except Exception as e:
            print(f"Error calculating R&D metrics: {e}")
            return None

    def _get_patent_data(self, ticker):
        """Simplified patent data (since FMP doesn't provide direct patent info)"""
        try:
            # Use intangible assets as a proxy for patent value
            balance_sheet = self._get_balance_sheet(ticker)
            if not balance_sheet:
                return None
            
            intangible_assets = balance_sheet.get('intangibleAssets', 0)
            total_assets = balance_sheet.get('totalAssets', 1)
            
            return {
                'intangible_assets': intangible_assets,
                'intangible_ratio': intangible_assets / total_assets if total_assets > 0 else 0
            }
        except Exception as e:
            print(f"Error calculating patent data: {e}")
            return None

    def _get_digital_metrics(self, ticker):
        """Simplified digital transformation metrics"""
        try:
            income_stmt = self._get_income_statement(ticker)
            if not income_stmt:
                return None
            
            # Use operating expenses as a proxy for digital investment
            operating_expenses = income_stmt.get('operatingExpenses', 0)
            revenue = income_stmt.get('revenue', 1)
            
            return {
                'digital_investment_ratio': operating_expenses / revenue if revenue > 0 else 0,
                'efficiency_ratio': income_stmt.get('operatingIncomeRatio', 0)
            }
        except Exception as e:
            print(f"Error calculating digital metrics: {e}")
            return None

    def _calculate_metric_growth(self, ticker, metric_name, years=3):
        """Calculate growth rate for a specific metric"""
        try:
            endpoint = f"{self.base_url}/income-statement/{ticker}?limit={years+1}&apikey={self.api_key}"
            response = requests.get(endpoint)
            
            if not response.json() or len(response.json()) < years+1:
                return 0
            
            data = response.json()
            start_value = data[years].get(metric_name, 0)
            end_value = data[0].get(metric_name, 0)
            
            if start_value <= 0:
                return 0
            
            return (pow(end_value / start_value, 1/years) - 1) * 100
        except Exception as e:
            print(f"Error calculating metric growth: {e}")
            return 0

    def _calculate_innovation_score(self, rd_data, patent_data):
        """Calculate innovation score from R&D and patent data"""
        if not rd_data or not patent_data:
            return 50  # Default neutral score
        
        score = 50
        
        # R&D intensity
        if rd_data['rd_ratio'] > 0.10:  # More than 10% of revenue
            score += 15
        elif rd_data['rd_ratio'] > 0.05:  # More than 5% of revenue
            score += 10
        
        # R&D growth
        if rd_data['rd_growth'] > 10:  # More than 10% growth
            score += 15
        elif rd_data['rd_growth'] > 5:  # More than 5% growth
            score += 10
        
        # Patent/Intangible assets intensity
        if patent_data['intangible_ratio'] > 0.20:  # More than 20% of assets
            score += 20
        elif patent_data['intangible_ratio'] > 0.10:  # More than 10% of assets
            score += 10
        
        return min(100, score)

    def _calculate_digital_score(self, digital_metrics):
        """Calculate digital transformation score"""
        if not digital_metrics:
            return 50  # Default neutral score
        
        score = 50
        
        # Digital investment intensity
        if digital_metrics['digital_investment_ratio'] > 0.20:
            score += 15
        elif digital_metrics['digital_investment_ratio'] > 0.10:
            score += 10
        
        # Operational efficiency
        if digital_metrics['efficiency_ratio'] > 0.20:
            score += 15
        elif digital_metrics['efficiency_ratio'] > 0.10:
            score += 10
        
        return min(100, score)

    def _generate_recommendation(self, valuation_score, technical_score, sector_score):
        """Generate investment recommendation based on multiple factors"""
        # Calculate weighted average score
        composite_score = (
            valuation_score * 0.4 +    # 40% weight on valuation
            technical_score * 0.3 +     # 30% weight on technical
            sector_score * 0.3          # 30% weight on sector-specific
        )
        
        # Generate base recommendation based on composite score
        if composite_score >= 80:
            base_rec = "Strong Buy"
        elif composite_score >= 60:
            base_rec = "Buy"
        elif composite_score >= 40:
            base_rec = "Hold"
        elif composite_score >= 20:
            base_rec = "Sell"
        else:
            base_rec = "Strong Sell"
        
        # Add recommendation confidence level
        score_variance = abs(valuation_score - technical_score) + abs(technical_score - sector_score)
        if score_variance > 40:
            confidence = "Low"
        elif score_variance > 20:
            confidence = "Medium"
        else:
            confidence = "High"
        
        return f"{base_rec} (Confidence: {confidence})"

def main():
    st.title("Advanced Stock Screener with Enhanced Analysis")

    # Add description
    st.markdown("""
    This stock screener helps you analyze stocks across different sectors using:
    - 📊 Fundamental Analysis (P/E ratios, margins, growth)
    - 📈 Technical Analysis (RSI, Moving Averages)
    - 🏢 Sector-Specific Metrics
    - ⚠️ Risk Assessment
    """)
    
    # Create tabs for main content and educational content
    main_tab, education_tab = st.tabs(["Stock Analysis", "Learn More"])
    
    # Educational Content
    with education_tab:
        st.header("📚 Understanding Stock Analysis")
        
        with st.expander("Valuation Metrics Explained"):
            st.markdown("""
            ### Valuation Score (0-100)
            - **80+ : Significantly Undervalued** - Stock may be trading well below its fair value
            - **60-79: Moderately Undervalued** - Stock appears somewhat undervalued
            - **40-59: Fairly Valued** - Stock is trading close to its fair value
            - **20-39: Moderately Overvalued** - Stock appears somewhat overvalued
            - **<20: Significantly Overvalued** - Stock may be trading well above its fair value
            
            The valuation score combines multiple factors including:
            - P/E Ratio (Price to Earnings)
            - P/B Ratio (Price to Book)
            - PEG Ratio (Price/Earnings to Growth)
            - DCF Valuation (Discounted Cash Flow)
            """)
            
        with st.expander("Technical Analysis Explained"):
            st.markdown("""
            ### Technical Indicators
            - **RSI (Relative Strength Index)**
                - Above 70: Potentially overbought
                - Below 30: Potentially oversold
                - Between 30-70: Neutral territory
            
            - **Moving Averages**
                - Golden Cross (50-day MA crosses above 200-day MA): Bullish signal
                - Death Cross (50-day MA crosses below 200-day MA): Bearish signal
            """)
            
        with st.expander("Understanding Sector Analysis"):
            st.markdown("""
            ### Sector-Specific Metrics
            Different sectors require different analytical approaches:
            
            - **Technology**: R&D spending, margin trends, innovation metrics
            - **Financial**: Interest margins, loan quality, capital ratios
            - **Healthcare**: Drug pipeline, R&D efficiency, regulatory risks
            - **Consumer**: Brand value, market share, inventory turnover
            """)
    
    # Main Analysis Tab
    with main_tab:
        try:
            api_key = get_api_key()
            screener = StockScreener(api_key)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        
        # Sidebar with enhanced help
        with st.sidebar:
            st.header("Stock Selection")
            
            st.info("💡 How to use this screener:")
            st.markdown("""
            1. Enter a stock symbol (e.g., AAPL for Apple)
            2. Click 'Add to Analysis List'
            3. Add multiple stocks to compare
            4. Click 'Analyze Stocks' to see results
            """)
            
            # Stock search and add functionality
            stock_search = st.text_input(
                "Search for a stock (e.g., AAPL, TD.TO, SHOP.TO)",
                help="Enter ticker symbol. Add .TO for Canadian stocks"
            )
            
            # Initialize session state for watchlist
            if 'watchlist' not in st.session_state:
                st.session_state.watchlist = []
            
            # Add stock button with validation
            if st.button("Add to Analysis List") and stock_search:
                if stock_search not in st.session_state.watchlist:
                    with st.spinner(f"Validating {stock_search}..."):
                        if screener.validate_ticker(stock_search):
                            st.session_state.watchlist.append(stock_search)
                            display_stock_info(stock_search, screener)
                        else:
                            st.error("Invalid ticker symbol")
            
            # Display and manage watchlist
            st.subheader("Analysis List")
            for stock in st.session_state.watchlist:
                col1, col2 = st.columns([3, 1])
                col1.write(stock)
                if col2.button("🗑️", key=f"remove_{stock}"):
                    st.session_state.watchlist.remove(stock)
            
            if st.button("Clear List"):
                st.session_state.watchlist = []

            # Analysis button (moved back to sidebar)
            if st.session_state.watchlist:
                if st.button("Analyze Stocks"):
                    st.session_state.analyze_clicked = True
            else:
                st.info("Add stocks to analyze")

        # Main content area (analysis results only)
        if st.session_state.watchlist:
            if getattr(st.session_state, 'analyze_clicked', False):
                with st.spinner('Performing comprehensive analysis...'):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, ticker in enumerate(st.session_state.watchlist):
                        try:
                            profile = screener._get_company_profile(ticker)
                            if profile:
                                sector = profile.get('sector')
                                sector_type = map_sector_to_type(sector)
                                analysis = screener.analyze_stock(ticker, sector_type)
                                if analysis:
                                    results.append(analysis)
                            progress_bar.progress((i + 1) / len(st.session_state.watchlist))
                        except Exception as e:
                            st.error(f"Error analyzing {ticker}: {str(e)}")
                    
                    if results:
                        df = pd.DataFrame(results)
                        
                        # Create tabs for different views
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "Summary Dashboard",
                            "Technical Analysis",
                            "Sector Analysis",
                            "Stock Comparison",
                            "Detailed Analysis"
                        ])
                        
                        with tab1:
                            display_summary_dashboard(df)
                        
                        with tab2:
                            display_technical_analysis(df)
                        
                        with tab3:
                            display_sector_analysis(df, screener)
                        
                        with tab4:
                            display_stock_comparison(df)
                        
                        with tab5:
                            display_detailed_analysis(df)
                    else:
                        st.warning("No analysis results were generated")
        else:
            st.info("👈 Add stocks to your analysis list using the sidebar")

def display_summary_dashboard(df):
    """Display enhanced summary dashboard"""
    st.subheader("📊 Market Overview")
    
    # Enhanced metrics display
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_score = df['Valuation Score'].mean()
        delta = avg_score - 50
        st.metric(
            "Average Valuation Score", 
            f"{avg_score:.1f}",
            delta=f"{delta:.1f} vs Neutral",
            help="Scores above 50 indicate undervaluation"
        )
    
    with col2:
        best_stock = df.loc[df['Valuation Score'].idxmax()]
        st.metric(
            "Top Pick", 
            best_stock['Ticker'],
            f"Score: {best_stock['Valuation Score']:.1f}",
            help="Stock with the highest overall score"
        )
    
    with col3:
        avg_tech_score = df['Technical Score'].mean()
        tech_delta = avg_tech_score - 50
        st.metric(
            "Average Technical Score",
            f"{avg_tech_score:.1f}",
            delta=f"{tech_delta:.1f} vs Neutral",
            help="Technical strength indicator"
        )
    
    # Market interpretation
    if avg_score > 60:
        st.success("🔥 Overall market segment appears undervalued")
    elif avg_score < 40:
        st.warning("⚠️ Overall market segment appears overvalued")
    else:
        st.info("📊 Overall market segment appears fairly valued")
    
    # Enhanced recommendation distribution
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
        },
        hole=0.4
    )
    st.plotly_chart(fig_rec)
    
    # Key insights
    display_key_insights(df)

def display_technical_analysis(df):
    """Display enhanced technical analysis"""
    st.subheader("Technical Indicators")
    
    # Add technical analysis explanation
    with st.expander("Understanding Technical Indicators"):
        st.markdown("""
        - **RSI (Relative Strength Index)**: Momentum indicator showing overbought/oversold conditions
        - **Moving Averages**: Trend indicators showing short-term vs long-term price movements
        - **Volume**: Trading activity indicator
        """)
    
    # RSI Analysis with interpretation
    st.subheader("RSI Analysis")
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
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    st.plotly_chart(fig_rsi)
    
    # Moving Averages with crossover analysis
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
    
    # Add MA crossover signals
    for _, row in df.iterrows():
        if row['MA50'] > row['MA200']:
            st.info(f"🔵 {row['Ticker']}: Golden Cross (Bullish Signal)")
        elif row['MA50'] < row['MA200']:
            st.warning(f"🔴 {row['Ticker']}: Death Cross (Bearish Signal)")

def display_stock_comparison(df):
    """Display stock comparison analysis"""
    st.subheader("📊 Stock Comparison")
    
    # Select stocks to compare
    stocks_to_compare = st.multiselect(
        "Select stocks to compare (max 3)",
        df['Ticker'].tolist(),
        max_selections=3
    )
    
    if stocks_to_compare:
        comparison_df = df[df['Ticker'].isin(stocks_to_compare)]
        
        # Radar chart comparison
        metrics = ['Valuation Score', 'Technical Score', 'Sector Score']
        fig = go.Figure()
        
        for ticker in stocks_to_compare:
            stock_data = comparison_df[comparison_df['Ticker'] == ticker]
            fig.add_trace(go.Scatterpolar(
                r=[stock_data[metric].iloc[0] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=ticker
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Comparative Analysis"
        )
        st.plotly_chart(fig)
        
        # Detailed comparison table
        st.subheader("Detailed Comparison")
        comparison_metrics = ['Ticker', 'Current Price', 'Valuation Score', 
                            'Technical Score', 'Sector Score', 'Recommendation']
        st.dataframe(comparison_df[comparison_metrics])

def display_key_insights(df):
    """Display key insights from analysis"""
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Best Value Opportunities")
        best_value = df.nlargest(3, 'Valuation Score')
        for _, stock in best_value.iterrows():
            st.write(f"**{stock['Ticker']}**: Score {stock['Valuation Score']:.1f}")
            st.write(f"Recommendation: {stock['Recommendation']}")
            st.write("---")
    
    with col2:
        st.markdown("#### Technical Standouts")
        best_tech = df.nlargest(3, 'Technical Score')
        for _, stock in best_tech.iterrows():
            st.write(f"**{stock['Ticker']}**: Score {stock['Technical Score']:.1f}")
            st.write(f"RSI: {stock.get('RSI', 'N/A')}")
            st.write("---")

def display_detailed_analysis(df):
    """Display detailed analysis with filtering"""
    st.subheader("Comprehensive Analysis")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Minimum Valuation Score", 0, 100, 0)
    with col2:
        selected_recommendations = st.multiselect(
            "Filter by Recommendation",
            df['Recommendation'].unique().tolist(),
            default=df['Recommendation'].unique().tolist()
        )
    
    # Filter DataFrame
    filtered_df = df[
        (df['Valuation Score'] >= min_score) &
        (df['Recommendation'].isin(selected_recommendations))
    ]
    
    # Display filtered results
    display_cols = ['Ticker', 'Company Name', 'Sector', 'Current Price', 
                   'Valuation Score', 'Technical Score', 'Sector Score',
                   'Recommendation']
    
    styled_df = filtered_df[display_cols].style\
        .background_gradient(subset=['Valuation Score', 'Technical Score', 'Sector Score'], cmap='RdYlGn')\
        .applymap(color_recommendation, subset=['Recommendation'])
    
    st.dataframe(styled_df)
    
    # Export functionality
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Analysis Results",
        data=csv,
        file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def map_sector_to_type(sector):
    """Map FMP sectors to SectorType"""
    sector_map = {
        'Technology': SectorType.TECHNOLOGY,
        'Healthcare': SectorType.HEALTHCARE,
        'Financials': SectorType.FINANCIAL,
        'Financial': SectorType.FINANCIAL,
        'Consumer Discretionary': SectorType.CONSUMER_CYCLICAL,
        'Consumer Staples': SectorType.CONSUMER_DEFENSIVE,
        'Industrials': SectorType.INDUSTRIALS,
        'Materials': SectorType.BASIC_MATERIALS,
        'Energy': SectorType.ENERGY,
        'Utilities': SectorType.UTILITIES,
        'Real Estate': SectorType.REAL_ESTATE,
        'Communication Services': SectorType.COMMUNICATION,
        'Telecommunications': SectorType.COMMUNICATION
    }
    return sector_map.get(sector, SectorType.TECHNOLOGY)

def color_recommendation(val):
    """Style function for recommendations"""
    colors = {
        'Strong Buy': 'background-color: #2E7D32; color: white',
        'Buy': 'background-color: #4CAF50; color: white',
        'Hold': 'background-color: #FFC107',
        'Sell': 'background-color: #F44336; color: white',
        'Strong Sell': 'background-color: #B71C1C; color: white'
    }
    return colors.get(val, '')

def display_stock_info(ticker, screener):
    """Show basic info when stock is added"""
    info = screener.get_company_profile(ticker)
    if info:
        sector = info.get('sector', 'Unknown')
        exchange = info.get('exchange', 'Unknown')
        price = info.get('price', 0)
        
        st.sidebar.markdown(f"""
        **Added: {info['companyName']}**
        - Exchange: {exchange}
        - Sector: {sector}
        - Current Price: ${price:.2f}
        """)

def display_sector_analysis(df, screener):
    """Display enhanced sector analysis"""
    st.subheader("Sector Analysis")
    
    # Get unique sectors
    sectors = df['Sector'].unique().tolist()
    
    # Sector selector
    selected_sector = st.selectbox("Select Sector for Analysis", sectors)
    
    # Filter for selected sector
    sector_df = df[df['Sector'] == selected_sector]
    
    if not sector_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sector metrics
            st.markdown(f"### {selected_sector} Metrics")
            avg_sector_score = sector_df['Sector Score'].mean()
            st.metric(
                "Average Sector Score",
                f"{avg_sector_score:.1f}",
                delta=f"{avg_sector_score - 50:.1f} vs Market",
                help="Sector-specific performance score"
            )
            
            # Get sector-specific metrics
            sector_type = map_sector_to_type(selected_sector)
            if sector_type in screener.sector_metrics.SECTOR_CONFIGS:
                specific_metrics = screener.sector_metrics.SECTOR_CONFIGS[sector_type]['specific_metrics']
                st.write("Key Sector Metrics:")
                
                # Display available metrics for the sector
                metrics_found = False
                for metric, config in specific_metrics.items():
                    # Convert metric name to the format used in your DataFrame
                    df_metric_name = metric.lower()
                    if df_metric_name in sector_df.columns:
                        metrics_found = True
                        avg_value = sector_df[df_metric_name].mean()
                        threshold = config['threshold']
                        if threshold is not None:
                            st.metric(
                                metric.replace('_', ' ').title(),
                                f"{avg_value:.2f}",
                                delta=f"{avg_value - threshold:.2f} vs Threshold"
                            )
                
                if not metrics_found:
                    st.info(f"No specific metrics available for {selected_sector}")
        
        with col2:
            # Sector risk analysis
            st.markdown("### Risk Analysis")
            if sector_type in screener.sector_metrics.SECTOR_CONFIGS:
                risk_factors = screener.sector_metrics.SECTOR_CONFIGS[sector_type]['risk_factors']
                st.write("Key Risk Factors:")
                
                # Display available risk factors
                risks_found = False
                for risk in risk_factors:
                    risk_col = f"{risk.lower()}_risk"
                    if risk_col in sector_df.columns:
                        risks_found = True
                        risk_value = sector_df[risk_col].mean()
                        st.progress(risk_value/100)
                        st.write(f"{risk.replace('_', ' ')}: {risk_value:.1f}%")
                
                if not risks_found:
                    st.info(f"No risk factors available for {selected_sector}")
        
        # Sector performance visualization
        st.subheader("Sector Performance Distribution")
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=sector_df['Valuation Score'],
            name='Valuation Score',
            boxpoints='all'
        ))
        fig.add_trace(go.Box(
            y=sector_df['Technical Score'],
            name='Technical Score',
            boxpoints='all'
        ))
        fig.add_trace(go.Box(
            y=sector_df['Sector Score'],
            name='Sector Score',
            boxpoints='all'
        ))
        fig.update_layout(
            title=f"{selected_sector} Score Distribution",
            yaxis_title="Score",
            showlegend=True
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for selected sector")

if __name__ == "__main__":
    main()