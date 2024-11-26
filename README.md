📈 Advanced Stock Screener

A comprehensive stock screening tool that combines fundamental analysis, technical indicators, and sector-specific metrics to provide detailed investment insights.

📋 Table of Contents

Features
Prerequisites
Installation
API Setup
Usage Guide
Dashboard Overview
Contributing
License

✨ Features

Core Capabilities:

📊 Multi-factor valuation analysis
📉 Technical indicator integration
🎯 Sector-specific metrics (Technology and Energy sectors)
🖥️ Interactive dashboard with Streamlit
⚙️ Customizable screening parameters
📁 Export functionality for detailed analysis

Analysis Components:

Fundamental Analysis

- P/E and P/B ratios
- DCF valuation
- Financial health metrics


Technical Analysis

- RSI indicators
- Moving averages
- Volume analysis


Sector-Specific Metrics

Technology sector metrics
Energy sector metrics



🔧 Prerequisites

Required Software:

-Python 3.8 or higher
- pip (Python package installer)

API Requirements:

- Financial Modeling Prep API key (Free tier available)

🚀 Installation:

1. Clone the Repository
- git clone https://github.com/m-turnergane/stock-screener.git
- cd stock-screener

2. Set Up Virtual Environment
- Create virtual environment
- python -m venv venv

# Activate virtual environment

# For Windows:
venv\Scripts\activate

# For macOS/Linux:
source venv/bin/activate

3. Install Dependencies
- pip install -r requirements.txt

🔑 API Setup:

Getting Your Free API Key:
- Visit Financial Modeling Prep: https://site.financialmodelingprep.com/developer/docs/pricing register for a free account and get your API key.
- Click "Sign Up" for a free account


Access Your API Key:
- Log into your dashboard
- Navigate to "API Keys" section
- Copy your API key



Free Tier Limitations:

⏰ 250 API calls per day
📊 Limited historical data access
⚡ Real-time data with 15-minute delay
📑 Basic financial statements

💻 Usage Guide:

Starting the Application:

Launch the Streamlit app:
streamlit run stock_screener.py


Configuration Steps:

Enter your FMP API key in the sidebar

Select desired sector:

Technology
Energy


Click "Analyze Stocks" button

Best Practices

- Run analysis during market hours for most accurate data
- Monitor API call usage to stay within limits
- Export important analyses for offline use
- Hop into the code base and customize the analysis to your liking, also feel free to add more sectors, symbols, funds, etc.!

📊 Dashboard Overview:

1. Summary Dashboard

Market Overview:

- Sector performance metrics
- Average valuation scores
- Market trend indicators


Investment Insights:

- Top stock recommendations
- Risk analysis
- Sector comparison



Technical Analysis:

Price Indicators:

- RSI analysis
- Moving average trends
- Support and resistance levels


Volume Analysis:

- Trading volume patterns
- Volume-price relationships
- Unusual volume alerts



3. Sector Metrics

Technology Sector:

- R&D spending metrics
- Growth rate analysis
- Market share evaluation
- Innovation indicators


Energy Sector:

- Reserve life calculations
- Production cost analysis
- ESG score tracking
- Resource efficiency metrics



4. Detailed Analysis:

Comprehensive Metrics:

- Fundamental ratios
- Technical indicators
- Risk metrics


Export Options:

- CSV export functionality
- Custom report generation
- Data visualization export



🤝 Contributing:

Contribution Process:

Fork the repository

Create your feature branch:
git checkout -b feature/AmazingFeature

Commit your changes:
git commit -m 'Add some AmazingFeature'

Push to the branch:
git push origin feature/AmazingFeature

Open a Pull Request:

Development Guidelines:

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation as needed
- Maintain compatibility with free API tier

📄 License
This project is distributed under the MIT License. See LICENSE for more information.

📞 Support
For issues and feature requests, please use the GitHub Issue Tracker.

🙏 Acknowledgements:

Financial Modeling Prep API
Streamlit Framework
Contributing Developers