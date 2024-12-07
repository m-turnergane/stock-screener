📈 Advanced Stock Screener

A comprehensive stock screening and analysis platform that delivers in-depth market insights through multi-sector analysis, technical indicators, and automated valuation metrics. The platform combines fundamental analysis, technical indicators, and sector-specific metrics to provide detailed investment recommendations across all major market sectors.

Live Application for public usage via Streamlit Community Deployment site: https://better-stonk-screener.streamlit.app/

✨ Enhanced Features

Core Analysis Capabilities:
- 📊 Multi-factor valuation analysis across all market sectors
- 📈 Comprehensive technical indicator integration
- 🎯 Sector-specific metrics and risk analysis
- 📱 Modern, tab-based dashboard interface
- 🔍 Advanced filtering and comparison tools
- 📊 Interactive visualizations with Plotly
- 💾 Enhanced export functionality

Analysis Components:

Fundamental Analysis:
- Industry-adjusted P/E and P/B ratios
- DCF valuation modeling
- Financial health scoring
- Sector-specific performance metrics
- Risk-adjusted valuation scores

Technical Analysis:
- RSI indicators with overbought/oversold signals
- Moving average crossover detection
- Volume analysis with trend confirmation
- Technical score aggregation
- Interactive technical charts

Sector Analysis:
- Sector-specific performance metrics
- Risk factor analysis by sector
- Industry benchmark comparisons
- Sector trend visualization
- Peer comparison tools

Dashboard Features:
- Summary Dashboard with key metrics
- Technical Analysis visualization
- Sector-specific analysis
- Stock Comparison tools
- Detailed Analysis with filtering

Educational Components:
- Built-in metric explanations
- Technical indicator guides
- Sector-specific insights
- Risk factor education

🔄 Current Limitations & Development Opportunities:

Sector Analysis:
- Limited sector-specific metric collection for non-Tech/Energy sectors
- Opportunity to implement detailed metrics for:
  - Financial Services (banking ratios, credit metrics)
  - Healthcare (pipeline analysis, regulatory metrics)
  - Consumer sectors (brand value, market share)
  - Real Estate (occupancy rates, NOI analysis)

Risk Analysis:
- Basic risk factor implementation
- Opportunity to enhance:
  - Sector-specific risk calculations
  - Market risk integration
  - Volatility analysis
  - Correlation studies

Data Integration:
- Currently limited to Financial Modeling Prep API
- Potential to add:
  - Alternative data sources
  - Real-time news integration
  - Social sentiment analysis
  - Economic indicator correlation

Future Enhancement Opportunities:

Technical Analysis:
- Additional technical indicators (MACD, Bollinger Bands)
- Pattern recognition algorithms
- Custom indicator creation
- Backtesting capabilities

Portfolio Analytics:
- Portfolio optimization tools
- Risk-adjusted return calculations
- Correlation analysis
- Diversification metrics

Machine Learning Integration:
- Predictive analytics
- Pattern recognition
- Anomaly detection
- Sentiment analysis

API & Data:
- Multiple data source integration
- Real-time websocket support
- Custom API endpoint creation
- Enhanced data validation

🚀 Installation & Setup

1. Clone the Repository
```bash
git clone https://github.com/m-turnergane/stock-screener.git
cd stock-screener
```

2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Create .env File
```bash
# Create a .env file in the root directory
# Add your FMP API key:
FMP_API_KEY=your_api_key_here
```

🖥️ Usage Guide

Starting the Application:
```bash
streamlit run stock_screener.py
```

Application Workflow:

1. Stock Selection
   - Enter stock symbols in the sidebar
   - Add multiple stocks for comparison
   - View basic company info upon addition

2. Analysis Options
   - Summary Dashboard: Overall market view
   - Technical Analysis: Detailed technical indicators
   - Sector Analysis: Sector-specific metrics
   - Stock Comparison: Side-by-side analysis
   - Detailed Analysis: Comprehensive metrics with filtering

3. Interactive Features
   - Filter stocks by various metrics
   - Compare multiple stocks
   - Export analysis results
   - Access educational content

📊 Dashboard Components

1. Summary Dashboard
   - Market overview metrics
   - Valuation score distribution
   - Top stock recommendations
   - Key insights summary

2. Technical Analysis
   - RSI visualization with signals
   - Moving average crossover detection
   - Volume analysis
   - Technical signal alerts

3. Sector Analysis
   - Sector-specific metrics
   - Risk factor analysis
   - Industry comparisons
   - Performance distribution

4. Stock Comparison
   - Side-by-side metric comparison
   - Radar chart visualization
   - Relative performance analysis
   - Multiple stock selection

5. Detailed Analysis
   - Comprehensive metrics table
   - Custom filtering options
   - Export functionality
   - Sorting capabilities

🔧 Customization Options

1. Metric Weights
   - Adjust valuation weights
   - Modify technical score components
   - Customize sector-specific weightings

2. Analysis Parameters
   - RSI thresholds
   - Moving average periods
   - Volume significance levels
   - Risk tolerance settings

3. Visualization Options
   - Chart types
   - Color schemes
   - Data display preferences
   - Export formats

👥 Contributing

Development Focus Areas:

1. Sector Analysis Enhancement
   - Implement additional sector metrics
   - Develop sector-specific risk models
   - Create custom sector visualizations

2. Technical Analysis Expansion
   - Add new technical indicators
   - Implement pattern recognition
   - Enhance signal detection

3. Data Integration
   - Additional API integrations
   - Alternative data sources
   - Real-time data handling

4. UI/UX Improvements
   - Mobile responsiveness
   - Custom themes
   - Advanced filtering
   - Interactive tutorials

Contribution Process:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

Code Standards:
- Follow PEP 8 guidelines
- Include docstrings and comments
- Add unit tests for new features
- Update documentation

🔍 Testing

Running Tests:
```bash
python -m pytest tests/
```

Test Coverage:
- Unit tests for core functions
- Integration tests for API
- UI component testing
- Performance benchmarks

📚 Documentation

Code Documentation:
- Function and class documentation
- API endpoint descriptions
- Configuration options
- Custom metric calculations

User Documentation:
- Installation guide
- Usage tutorials
- Metric explanations
- Troubleshooting guide

🔒 Security & Performance

Security Considerations:
- API key protection
- Rate limiting
- Data validation
- Error handling

Performance Optimization:
- API call caching
- Data preprocessing
- Batch processing
- Memory management

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgements
- Financial Modeling Prep API
- Streamlit Framework
- Plotly Visualization Library
- Contributing Developers

💡 Support
For support, feature requests, or bug reports:
- Open an issue on GitHub
- Review existing issues
- Join discussions
- Contribute solutions
