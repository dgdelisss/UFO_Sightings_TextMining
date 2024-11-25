import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FDICBankDataRetriever:
    def __init__(self):
        # Base FDIC API endpoint
        self.base_url = "https://banks.data.fdic.gov/api/institutions"
        self.branches_url = "https://banks.data.fdic.gov/api/locations"
    
    def fetch_comprehensive_bank_data(self, 
                                      state=None, 
                                      asset_size_min=None, 
                                      asset_size_max=None, 
                                      reporting_date=None):
        """
        Comprehensive bank data retrieval with deposit-specific insights
        """
        # If no reporting date specified, use most recent quarter end
        if not reporting_date:
            reporting_date = self._get_most_recent_quarter_end()
        
        # Expanded field set for deposit-specific analysis
        params = {
            "filters": {
                "active": "1",  # Only active banks
                "officeType": "0"  # Main office only
            },
            "fields": [
                # Identification
                "name", 
                "address", 
                "city", 
                "state", 
                "zipCode", 
                "institution_class",
                
                # Financial Metrics
                "totalAssets", 
                "netIncome", 
                "returnOnAssets",
                
                # Deposit-Specific Fields
                "deposits",
                "domesticDeposits",
                "foreignDeposits",
                "totalDepositsChange",
                "domesticOfficeDeposits",
                "interestedBearingDeposits",
                "nonInterestBearingDeposits",
                
                # Additional Context
                "commercialLoans",
                "consumerLoans",
                "totalLoans",
                "loanLeaseFinancing"
            ],
            "sortField": "deposits",
            "sortOrder": "DESC",
            "limit": 500  # Adjust as needed
        }
        
        # Add optional filters
        filters = params["filters"]
        if state:
            filters["state"] = state
        if asset_size_min:
            filters["totalAssets_gt"] = asset_size_min
        if asset_size_max:
            filters["totalAssets_lt"] = asset_size_max
        
        try:
            response = requests.post(self.base_url, json=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            return self._process_deposit_data(df)
        
        except requests.RequestException as e:
            print(f"API Request Error: {e}")
            return None
    
    def fetch_branch_analysis(self, bank_name=None, state=None):
        """
        Comprehensive branch location and distribution analysis
        """
        branch_params = {
            "filters": {
                "active": "1"
            },
            "fields": [
                "name",  # Bank name
                "city",
                "state",
                "county",
                "branchType",
                "serviceType",
                "latitude",
                "longitude",
                "deposit_amount"
            ],
            "sortField": "deposit_amount",
            "sortOrder": "DESC",
            "limit": 1000
        }
        
        # Add optional filters
        filters = branch_params["filters"]
        if bank_name:
            filters["name"] = bank_name
        if state:
            filters["state"] = state
        
        try:
            response = requests.post(self.branches_url, json=branch_params)
            response.raise_for_status()
            
            data = response.json()
            branch_df = pd.DataFrame(data['data'])
            
            return self._process_branch_data(branch_df)
        
        except requests.RequestException as e:
            print(f"Branch API Request Error: {e}")
            return None
    
    def _process_deposit_data(self, df):
        """
        Advanced processing of deposit-related data
        """
        # Numeric conversions with error handling
        deposit_columns = [
            'deposits', 'domesticDeposits', 'foreignDeposits', 
            'totalDepositsChange', 'domesticOfficeDeposits',
            'interestedBearingDeposits', 'nonInterestBearingDeposits'
        ]
        
        for col in deposit_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Derived deposit metrics
        df['deposit_composition_ratio'] = df['interestedBearingDeposits'] / df['deposits'] * 100
        df['non_interest_deposit_ratio'] = df['nonInterestBearingDeposits'] / df['deposits'] * 100
        df['foreign_deposit_percentage'] = df['foreignDeposits'] / df['deposits'] * 100
        
        return df
    
    def _process_branch_data(self, df):
        """
        Process and enrich branch location data
        """
        # Numeric conversions
        df['deposit_amount'] = pd.to_numeric(df['deposit_amount'], errors='coerce')
        
        # Derive branch distribution metrics
        branch_summary = df.groupby(['name', 'state']).agg({
            'city': 'count',  # Number of branches
            'deposit_amount': 'sum',  # Total deposits per bank/state
            'latitude': ['mean', 'min', 'max'],
            'longitude': ['mean', 'min', 'max']
        }).reset_index()
        
        # Rename columns for clarity
        branch_summary.columns = [
            'bank_name', 'state', 
            'total_branches', 'total_branch_deposits',
            'avg_latitude', 'min_latitude', 'max_latitude',
            'avg_longitude', 'min_longitude', 'max_longitude'
        ]
        
        return branch_summary
    
    def deposit_performance_analysis(self, reference_bank_asset_size, state=None):
        """
        Comprehensive deposit performance analysis
        """
        # Define asset size range (e.g., ±25% of reference bank)
        asset_size_min = reference_bank_asset_size * 0.75
        asset_size_max = reference_bank_asset_size * 1.25
        
        # Fetch comprehensive bank data
        bank_data = self.fetch_comprehensive_bank_data(
            state=state, 
            asset_size_min=asset_size_min, 
            asset_size_max=asset_size_max
        )
        
        # Fetch branch data for the same banks
        branch_data = self.fetch_branch_analysis(state=state)
        
        # Deposit performance summary
        deposit_summary = {
            'mean_total_deposits': bank_data['deposits'].mean(),
            'median_total_deposits': bank_data['deposits'].median(),
            'mean_deposit_growth': bank_data['totalDepositsChange'].mean(),
            'interest_bearing_deposit_ratio': bank_data['deposit_composition_ratio'].mean(),
            'non_interest_deposit_ratio': bank_data['non_interest_deposit_ratio'].mean(),
            'total_peer_banks': len(bank_data),
            'total_branches': branch_data['total_branches'].sum(),
            'avg_deposits_per_branch': branch_data['total_branch_deposits'].mean()
        }
        
        return {
            'deposit_performance': deposit_summary,
            'bank_data': bank_data,
            'branch_data': branch_data
        }

# Example Usage
def main():
    # Initialize FDIC Data Retriever
    fdic_retriever = FDICBankDataRetriever()
    
    # Example: Find peers for a bank with $500 million in assets
    reference_bank_asset_size = 500_000_000
    
    # Comprehensive deposit and branch analysis
    deposit_analysis = fdic_retriever.deposit_performance_analysis(
        reference_bank_asset_size, 
        state='CA'  # Optional: filter by state
    )
    
    # Print Deposit Performance Summary
    print("Deposit Performance Summary:")
    for metric, value in deposit_analysis['deposit_performance'].items():
        print(f"{metric}: {value}")
    
    # Optional: Save detailed data to CSV for further analysis
    deposit_analysis['bank_data'].to_csv('bank_deposit_data.csv', index=False)
    deposit_analysis['branch_data'].to_csv('bank_branch_data.csv', index=False)

if __name__ == "__main__":
    main()
