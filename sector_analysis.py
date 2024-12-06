from enum import Enum
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

class SectorType(Enum):
    """Enum for different market sectors"""
    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare"
    FINANCIAL = "Financial Services"
    CONSUMER_CYCLICAL = "Consumer Cyclical"
    CONSUMER_DEFENSIVE = "Consumer Defensive"
    INDUSTRIALS = "Industrials"
    BASIC_MATERIALS = "Basic Materials"
    ENERGY = "Energy"
    UTILITIES = "Utilities"
    REAL_ESTATE = "Real Estate"
    COMMUNICATION = "Communication Services"

class SectorMetrics:
    """Base configuration for sector-specific metrics and thresholds"""
    
    # Common metrics across all sectors
    BASE_METRICS = {
        'Revenue_Growth': {'weight': 0.10, 'threshold': 0.05},
        'Operating_Margin': {'weight': 0.10, 'threshold': 0.10},
        'Current_Ratio': {'weight': 0.05, 'threshold': 1.5},
        'Debt_To_Equity': {'weight': 0.05, 'threshold': 2.0},
        'Return_On_Equity': {'weight': 0.10, 'threshold': 0.12},
        'Asset_Turnover': {'weight': 0.05, 'threshold': 0.5}
    }
    
    # Sector-specific metrics configurations
    SECTOR_CONFIGS = {
        SectorType.TECHNOLOGY: {
            'specific_metrics': {
                'R&D_Ratio': {'weight': 0.15, 'threshold': 0.10},
                'Gross_Margin': {'weight': 0.10, 'threshold': 0.30},
                'Cloud_Revenue_Growth': {'weight': 0.10, 'threshold': 0.20},
                'Patent_Count_Growth': {'weight': 0.05, 'threshold': 0.10}
            },
            'risk_factors': ['Tech_Obsolescence', 'Cybersecurity', 'Competition']
        },
        
        SectorType.HEALTHCARE: {
            'specific_metrics': {
                'R&D_Ratio': {'weight': 0.15, 'threshold': 0.15},
                'FDA_Approval_Rate': {'weight': 0.10, 'threshold': 0.60},
                'Pipeline_Value': {'weight': 0.10, 'threshold': None},
                'Patent_Protection': {'weight': 0.05, 'threshold': None}
            },
            'risk_factors': ['Regulatory', 'Patent_Expiry', 'Clinical_Trial_Risk']
        },
        
        SectorType.FINANCIAL: {
            'specific_metrics': {
                'Net_Interest_Margin': {'weight': 0.15, 'threshold': 0.03},
                'NPL_Ratio': {'weight': 0.10, 'threshold': 0.05},
                'Capital_Adequacy': {'weight': 0.15, 'threshold': 0.08},
                'Cost_To_Income': {'weight': 0.10, 'threshold': 0.60}
            },
            'risk_factors': ['Interest_Rate', 'Credit_Risk', 'Market_Risk']
        },
        
        SectorType.CONSUMER_CYCLICAL: {
            'specific_metrics': {
                'Same_Store_Sales': {'weight': 0.15, 'threshold': 0.02},
                'Inventory_Turnover': {'weight': 0.10, 'threshold': 4.0},
                'Online_Sales_Ratio': {'weight': 0.10, 'threshold': 0.15},
                'Brand_Value': {'weight': 0.05, 'threshold': None}
            },
            'risk_factors': ['Consumer_Confidence', 'Economic_Cycle', 'Fashion_Risk']
        },
        
        SectorType.CONSUMER_DEFENSIVE: {
            'specific_metrics': {
                'Market_Share': {'weight': 0.15, 'threshold': 0.10},
                'Brand_Strength': {'weight': 0.10, 'threshold': 0.70},
                'Distribution_Network': {'weight': 0.10, 'threshold': None},
                'Product_Portfolio': {'weight': 0.05, 'threshold': None}
            },
            'risk_factors': ['Supply_Chain', 'Commodity_Prices', 'Private_Label_Competition']
        },
        
        SectorType.ENERGY: {
            'specific_metrics': {
                'Reserve_Life': {'weight': 0.15, 'threshold': 10.0},
                'Production_Cost': {'weight': 0.10, 'threshold': 35.0},
                'ESG_Score': {'weight': 0.10, 'threshold': 70.0},
                'Reserve_Replacement': {'weight': 0.05, 'threshold': 1.0}
            },
            'risk_factors': ['Environmental', 'Resource_Depletion', 'Regulatory']
        },
        
        SectorType.UTILITIES: {
            'specific_metrics': {
                'Regulatory_ROE': {'weight': 0.15, 'threshold': 0.10},
                'Customer_Growth': {'weight': 0.10, 'threshold': 0.02},
                'Infrastructure_Age': {'weight': 0.10, 'threshold': 20.0},
                'Renewable_Mix': {'weight': 0.05, 'threshold': 0.30}
            },
            'risk_factors': ['Regulatory', 'Environmental', 'Infrastructure']
        },
        
        SectorType.REAL_ESTATE: {
            'specific_metrics': {
                'Occupancy_Rate': {'weight': 0.15, 'threshold': 0.90},
                'FFO_Growth': {'weight': 0.10, 'threshold': 0.05},
                'Property_Yield': {'weight': 0.10, 'threshold': 0.05},
                'Lease_Duration': {'weight': 0.05, 'threshold': 5.0}
            },
            'risk_factors': ['Interest_Rate', 'Market_Cycle', 'Location']
        },
        
        SectorType.COMMUNICATION: {
            'specific_metrics': {
                'ARPU': {'weight': 0.15, 'threshold': 50.0},
                'Churn_Rate': {'weight': 0.10, 'threshold': 0.15},
                'Network_Quality': {'weight': 0.10, 'threshold': 0.95},
                'Spectrum_Efficiency': {'weight': 0.05, 'threshold': None}
            },
            'risk_factors': ['Technology_Change', 'Regulation', 'Competition']
        },
        
        SectorType.INDUSTRIALS: {
            'specific_metrics': {
                'Order_Backlog': {'weight': 0.15, 'threshold': None},
                'Capacity_Utilization': {'weight': 0.10, 'threshold': 0.80},
                'Operating_Efficiency': {'weight': 0.10, 'threshold': 0.85},
                'R&D_Effectiveness': {'weight': 0.05, 'threshold': None}
            },
            'risk_factors': ['Economic_Cycle', 'Raw_Materials', 'Labor_Relations']
        },
        
        SectorType.BASIC_MATERIALS: {
            'specific_metrics': {
                'Resource_Grade': {'weight': 0.15, 'threshold': None},
                'Processing_Cost': {'weight': 0.10, 'threshold': None},
                'Capacity_Utilization': {'weight': 0.10, 'threshold': 0.75},
                'Environmental_Impact': {'weight': 0.05, 'threshold': None}
            },
            'risk_factors': ['Commodity_Prices', 'Environmental', 'Geopolitical']
        }
    }
    
    # Industry average P/E ratios
    INDUSTRY_PE = {
        SectorType.TECHNOLOGY: 25.5,
        SectorType.HEALTHCARE: 22.3,
        SectorType.FINANCIAL: 12.8,
        SectorType.CONSUMER_CYCLICAL: 20.1,
        SectorType.CONSUMER_DEFENSIVE: 19.2,
        SectorType.INDUSTRIALS: 18.4,
        SectorType.BASIC_MATERIALS: 14.6,
        SectorType.ENERGY: 15.2,
        SectorType.UTILITIES: 17.9,
        SectorType.REAL_ESTATE: 16.7,
        SectorType.COMMUNICATION: 21.3
    }
    
    # Industry average P/B ratios
    INDUSTRY_PB = {
        SectorType.TECHNOLOGY: 6.8,
        SectorType.HEALTHCARE: 4.2,
        SectorType.FINANCIAL: 1.5,
        SectorType.CONSUMER_CYCLICAL: 3.2,
        SectorType.CONSUMER_DEFENSIVE: 2.8,
        SectorType.INDUSTRIALS: 3.1,
        SectorType.BASIC_MATERIALS: 2.1,
        SectorType.ENERGY: 1.8,
        SectorType.UTILITIES: 1.7,
        SectorType.REAL_ESTATE: 1.9,
        SectorType.COMMUNICATION: 3.5
    } 