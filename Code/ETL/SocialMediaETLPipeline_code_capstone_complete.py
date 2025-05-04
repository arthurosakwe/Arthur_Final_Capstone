"""
Capstone Project
ETL Pipeline
Author: Arthur
Version: 2.0
"""

import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import datetime, timedelta
import logging
from pathlib import Path

#Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("etl_log.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

class SocialMediaETL:
    def __init__(self, data_dir, output_dir):

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        #Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / 'dimensions', exist_ok=True)
        os.makedirs(self.output_dir / 'facts', exist_ok=True)
        
        #client name standardization mapping
        raw_name_map = {
            #STEP Learning Collaborative
            'mindmap': 'STEP Learning Collaborative',
            'mindmapct': 'STEP Learning Collaborative',
            'steplearningcollaborative': 'STEP Learning Collaborative',
        
            #Calm Nola
            'calmnola': 'Calm Nola',
            'calm nola': 'Calm Nola',
        
            #APT Foundation
            'aptfoundationct': 'APT Foundation',
            'apt': 'APT Foundation',
            'apt foundation': 'APT Foundation',
        
            #Conscious Business Collaborative
            'consciouscapitalism': 'Conscious Business Collaborative',
            'consciousbusinesscollab': 'Conscious Business Collaborative',
            'conscious capitalism': 'Conscious Business Collaborative',
            'conscious business collaborative': 'Conscious Business Collaborative',
        
            #Parker Marketing
            'parkermarketingandmanagement': 'Parker Marketing',
            'parkermarketingandmanagement.com': 'Parker Marketing',
            'parker marketing': 'Parker Marketing',

            #Red Rock
            'redrockbranding': 'Red Rock',
            'red rock': 'Red Rock',
        
            #ManufactureCT
            'manufacture.ct': 'ManufactureCT',
            'manufacturect': 'ManufactureCT',
        
            #Michigan Minds
            'michiganminds': 'Michigan Minds',
            'michigan minds': 'Michigan Minds',
        
            #Fempire
            'thefempireco': 'The Fempire Co.',
            'the fempire co': 'The Fempire Co.',
        
            #Amy Hughes
            'amyhughesphotographyaz': 'Amy Hughes Photography',
            'amy hughes photography': 'Amy Hughes Photography',
        
            #Barbell Saves
            'thebarbellsaves': 'The Barbell Saves Project',
            'the barbell saves project': 'The Barbell Saves Project',
        
            #Sunny Dublick
            'sunnydublickmarketing': 'Sunny Dublick Marketing',
            'sunny dublick marketing': 'Sunny Dublick Marketing',
        
            #NYC TCTTAC
            'nyctcttac': 'NYC TCTTAC',
            'nyc tcttac': 'NYC TCTTAC',
            
            #Catalyst Personal Training
            'catalyst personal training oceanside': 'Catalyst Personal Training Oceanside',
            'catalystpersonaltrainingoceanside': 'Catalyst Personal Training Oceanside'
        }

        #normalize keys
        self.client_name_map = {k.strip().lower(): v for k, v in raw_name_map.items()}

        
        #platform name mapping
        self.platform_map = {
            'website_daily_analytics_all.csv': 'Website',
            'facebook_all_pages_daily_summary.xlsx': 'Facebook Pages',
            'fb_ad_metrics.csv': 'Facebook Ads', 
            'mailchimp_campaign_metrics.csv': 'Mailchimp',
            'linkedin_daily_all_pages.csv': 'LinkedIn',
            'instagram_all_accounts_daily_summary.xlsx': 'Instagram'}
        
        #date column mapping
        self.date_column_map = {
            'Website': 'Date',
            'Facebook Ads': 'Date',
            'Facebook Pages': 'date',
            'Mailchimp': ['Start Date', 'End Date'],
            'LinkedIn': 'Date',
            'Instagram': 'date' }
        
        #initialize dimension df
        self.dim_client = pd.DataFrame(columns=['client_id', 'Client_Name', 'original_client_name'])
        self.dim_date = pd.DataFrame(columns=['date_id', 'full_date', 'day', 'month', 'year', 'quarter', 'week', 'is_weekend', 'month_name', 'period_type'])
        self.dim_platform = pd.DataFrame(columns=['platform_id', 'platform_name', 'data_source'])
        self.dim_metric = pd.DataFrame(columns=['metric_id', 'metric_name', 'metric_category', 'is_calculated'])
        
        #initialize fact DataFrame
        self.fact_metrics = pd.DataFrame(columns=['fact_id', 'client_id', 'date_id', 'platform_id', 'metric_id', 'value'])
        
        #initialize dictionaries to store dimension ids
        self.client_ids = {}
        self.date_ids = {}
        self.platform_ids = {}
        self.metric_ids = {}
        
        #Initialize counter for fact IDs
        self.fact_id_counter = 1
    
    def standardize_client_name(self, name):
        if not name or pd.isna(name):
            return "Unknown Client"
        
        cleaned_name = str(name).strip().lower()
        
        if cleaned_name in self.client_name_map:
            return self.client_name_map[cleaned_name]
        
        logger.warning(f"No standardized name found for '{name} - please review")
        return name

    #file extractor #TODO: standardize extension types before loading into pipeline
    def extract_file(self, file_path):
        file_path = Path(file_path)
        file_name = file_path.name
        platform_name = self.platform_map.get(file_name, "Unknown")
        
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                logger.info(f"CSV file extracted: {file_path} - {len(df)} rows")
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                logger.info(f"Excel file extracted: {file_path} - {len(df)} rows")
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return None, platform_name
            if platform_name == "LinkedIn" and 'Organization' in df.columns and 'Client_Name' not in df.columns:
                df['Client_Name'] = df['Organization']
                
            return df, platform_name
        except Exception as e:
            logger.error(f"Error extracting {file_path}: {str(e)}")
            return None, platform_name
   
    #standardize dates
    def transform_dates(self, df, platform_name):
        date_columns = self.date_column_map.get(platform_name, 'Date')     
        #Handle different date formats
        if isinstance(date_columns, list):
            #For platforms like Mailchimp with start and end dates
            for col in date_columns:
                if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            #For platforms with a single date column
            if date_columns in df.columns:
                    df[date_columns] = pd.to_datetime(df[date_columns], errors='coerce')

        
        return df

    #generate the date dimensions - optimized 3/11/25 to reduce code length
        #revert to old cod eif issue persists
    
    def generate_date_dimension(self):
        #Determine min/max dates
        min_date = datetime(2023, 1, 1)
        max_date = datetime.now()
    
        #Generate full date range
        daily_df = pd.DataFrame({'full_date': pd.date_range(min_date, max_date)})
    
        daily_df['date_id'] = range(1, len(daily_df) + 1)
        daily_df['day'] = daily_df['full_date'].dt.day
        daily_df['month'] = daily_df['full_date'].dt.month
        daily_df['year'] = daily_df['full_date'].dt.year
        daily_df['quarter'] = daily_df['full_date'].dt.quarter
        daily_df['week'] = daily_df['full_date'].dt.isocalendar().week
        daily_df['is_weekend'] = daily_df['full_date'].dt.dayofweek >= 5
        daily_df['month_name'] = daily_df['full_date'].dt.strftime('%B')
        daily_df['period_type'] = 'Day'
    
        #Store date ID mappings
        self.date_ids = {
            row['full_date'].strftime('%Y-%m-%d'): row['date_id']
            for _, row in daily_df.iterrows() }
    
        date_id_counter = daily_df['date_id'].max() + 1
        #Monthly periods
        monthly = daily_df.groupby(['year', 'month']).first().reset_index()
        monthly['date_id'] = range(date_id_counter, date_id_counter + len(monthly))
        monthly['day'] = 1
        monthly['quarter'] = (monthly['month'] - 1) // 3 + 1
        monthly['week'] = None
        monthly['is_weekend'] = False
        monthly['month_name'] = monthly['full_date'].dt.strftime('%B')
        monthly['period_type'] = 'Month'
        
        for _, row in monthly.iterrows():
            key = f"M{row['year']}-{row['month']:02d}"
            self.date_ids[key] = row['date_id']
        date_id_counter += len(monthly)
        
        #Yearly periods
        years = daily_df['year'].unique()
        yearly = pd.DataFrame({
            'year': years,
            'month': 1,
            'day': 1,
            'full_date': pd.to_datetime([f"{y}-01-01" for y in years]),
            'quarter': 1,
            'week': None,
            'is_weekend': False,
            'month_name': 'January',
            'period_type': 'Year',
            'date_id': range(date_id_counter, date_id_counter + len(years))})
        
        for _, row in yearly.iterrows():
            self.date_ids[f"Y{row['year']}"] = row['date_id']
        self.dim_date = pd.concat([daily_df, monthly, yearly], ignore_index=True)
        logger.info(f"Generated date dimension with {len(self.dim_date)} records")
        return self.dim_date

    #create platform dims
    def generate_platform_dimension(self):
        platform_records = [
            {'platform_id': i+1, 'platform_name': platform, 'data_source': file_name}
            for i, (file_name, platform) in enumerate(self.platform_map.items())
        ]
        
        self.dim_platform = pd.DataFrame(platform_records)
        
        #Create mapping of platform names to IDs
        for _, row in self.dim_platform.iterrows():
            self.platform_ids[row['platform_name']] = row['platform_id']
            
        logger.info(f"Generated platform dimension with {len(self.dim_platform)} records")
        return self.dim_platform
    
    #extract key metrics
    def extract_metrics(self, df, platform_name):
        metrics = []
        
        #Skip common non-metric columns
        skip_columns = [
            'Client_Name', 'original_client_name', 'Date', 'date', 'Start Date', 'End Date',
            'Report Type', 'Page Title', 'Page Path', 'Country', 'State', 'Property ID',
            'Campaign Name', 'Campaign ID', 'Period', 'Month', 'Year', 'Organization', 
            'Organization URN', 'page_id', 'page_name', 'page_category', 'instagram_id', 'username'
        ]
        
        #Map metric categories based on platform and column names
        category_mapping = {
                'Website': {
                'Views': 'Engagement',
                'Active Users': 'Audience',
                'Engaged Sessions': 'Engagement',
                'New Users': 'Audience',
                'Event Count': 'Engagement',
                'Sessions': 'Engagement',
                'Engagement Rate (%)': 'Engagement',
                'User Engagement Duration': 'Engagement'
                        },
                'Facebook Ads': {
                'Reach': 'Reach',
                'Impressions': 'Reach',
                'Frequency': 'Reach',
                'CPM': 'Cost',
                'Clicks': 'Engagement',
                'Amount Spent': 'Cost',
                'Link Clicks': 'Engagement',
                'Cost_': 'Cost',
                'Action_': 'Action'
                        },
                'Facebook Pages': {
                'New Page Likes': 'Audience',
                'Page Impressions': 'Reach',
                'Interactions': 'Engagement',
                'Likes': 'Engagement',
                'Page Views': 'Engagement',
                'Post Likes': 'Engagement',
                'Comments': 'Engagement',
                'Shares': 'Engagement',
                'Total Posts': 'Content'
                        },
                'Mailchimp': {
                'Emails Sent': 'Reach',
                'Open Rate': 'Engagement',
                'Click Rate': 'Engagement',
                'Total Clicks': 'Engagement',
                'Unique Clicks': 'Engagement',
                'Unique Subscriber Clicks': 'Engagement'
                        },
                'LinkedIn': {
                'New Followers': 'Audience',
                'Reach': 'Reach',
                'Impressions': 'Reach',
                'Comments': 'Engagement',
                'Likes': 'Engagement',
                'Clicks': 'Engagement',
                'Shares': 'Engagement',
                'Total Interactions': 'Engagement',
                'Engagement Rate': 'Engagement',
                'Avg Profile Views per New Follower': 'Engagement',
                'Profile Views': 'Engagement'
                        },
                'Instagram': {
                'New Follows': 'Audience',
                'Unfollows': 'Audience',
                'Net Follower Change': 'Audience',
                'Followers': 'Audience',
                'Likes': 'Engagement',
                'Comments': 'Engagement',
                'Saved': 'Engagement',
                'Views': 'Engagement',
                'Media Reach': 'Reach',
                'Total Engagement': 'Engagement',
                'Posts Count': 'Content',
                'Engagement Rate': 'Engagement'} }
        
        for col in df.columns:
            #Skip non-metric columns
            if col in skip_columns:
                continue
                
            #Determine metric category
            category = 'Other'
            
            #Check if column matches any pattern in the category mapping
            platform_mapping = category_mapping.get(platform_name, {})
            for pattern, cat in platform_mapping.items():
                if pattern in col or col.startswith(pattern):
                    category = cat
                    break
            
            #Determine if metric is calculated
            is_calculated = False
            if 'rate' in col.lower() or 'ratio' in col.lower() or '%' in col or 'avg' in col.lower():
                is_calculated = True
                
            metrics.append((col, category, is_calculated))
        
        return metrics
    
    def add_client(self, client_name, original_client_name=None):
        if client_name in self.client_ids:
            return self.client_ids[client_name]
            
        #Create new client ID
        client_id = len(self.client_ids) + 1
        
        #Add to client dimension
        new_client = {
            'client_id': client_id,
            'Client_Name': client_name,  
            'original_client_name': original_client_name or client_name
        }
        
        self.dim_client = pd.concat([self.dim_client, pd.DataFrame([new_client])], ignore_index=True)
        self.client_ids[client_name] = client_id
        
        return client_id
    #add metric to dimension
    def add_metric(self, metric_name, metric_category, is_calculated):

        if metric_name in self.metric_ids:
            return self.metric_ids[metric_name]
            
        #Create new metric ID
        metric_id = len(self.metric_ids) + 1
        
        #Add to metric dimension
        new_metric = {
            'metric_id': metric_id,
            'metric_name': metric_name,
            'metric_category': metric_category,
            'is_calculated': is_calculated
        }
        
        self.dim_metric = pd.concat([self.dim_metric, pd.DataFrame([new_metric])], ignore_index=True)
        self.metric_ids[metric_name] = metric_id
        
        return metric_id
   
    
    def transform_and_load_file(self, file_path):
        file_path = Path(file_path)
        file_name = file_path.name
        
        #Extract data
        df, platform_name = self.extract_file(file_path)
        if df is None:
            logger.error(f"Failed to extract data from {file_path}")
            return False
        
        #Check if platform is valid
        platform_id = self.platform_ids.get(platform_name)
        if platform_id is None:
            logger.error(f"Platform {platform_name} not found in dimension table for file {file_path}")
            return False
                
        
        #Transform dates
        df = self.transform_dates(df, platform_name)
        
        #Extract metrics
        metrics_list = self.extract_metrics(df, platform_name)
        
        #Create fact records
        fact_records = []
        records_loaded = 0
        
        #Helper function for converting metric values
        def convert_metric_value(metric_name, value):
            if pd.isna(value):
                return None
            
            #If already numeric, return as is
            if isinstance(value, (int, float)):
                return float(value)
            
            #Handle string values
            if isinstance(value, str):
                #Handle duration format "XXm YYs"
                if "m" in value and "s" in value:
                    minutes = re.search(r'(\d+)m', value)
                    seconds = re.search(r'(\d+)s', value)
                    total_seconds = 0
                    if minutes:
                        total_seconds += int(minutes.group(1)) * 60
                    if seconds:
                        total_seconds += int(seconds.group(1))
                    return total_seconds
                
                #Handle percentage format "XX.XX%"
                if "%" in value:
                    try:
                        return float(value.strip('%'))
                    except:
                        pass
                        
                #Try direct conversion
                try:
                    return float(value)
                except:
                    pass
            
            #Couldn't convert
            return None
        
        for _, row in df.iterrows():
            #Get client ID
            client_name = self.standardize_client_name(row.get('Client_Name', 'Unknown'))
            client_id = self.add_client(client_name, row.get('original_client_name', client_name))
            
            #Determine date for this record
            date_columns = self.date_column_map.get(platform_name, 'Date')
            record_date = None
            period_type = 'Day'  #Default period type
            date_id = None
            
            try:
                if isinstance(date_columns, list):
                    #For platforms like Mailchimp with start and end dates
                    if 'Period' in row and row['Period'] == 'Monthly':
                        period_type = 'Month'
                    elif 'Period' in row and row['Period'] == 'Yearly':
                        period_type = 'Year'
                    
                    #Use mid-point of date range as reference
                    start_date = pd.to_datetime(row.get(date_columns[0]), errors='coerce')
                    end_date = pd.to_datetime(row.get(date_columns[1]), errors='coerce')
                    
                    if not pd.isna(start_date) and not pd.isna(end_date):
                        if period_type == 'Month':
                            #Use start date for monthly data
                            record_date = start_date
                            #Use month ID
                            date_id = self.date_ids.get(f"M{record_date.year}-{record_date.month:02d}")
                        elif period_type == 'Year':
                            #Use year ID
                            date_id = self.date_ids.get(f"Y{start_date.year}")
                        else:
                            #Use midpoint for other periods
                            record_date = start_date + (end_date - start_date) / 2
                            #Use day ID
                            date_id = self.date_ids.get(record_date.strftime('%Y-%m-%d'))
                else:
                    #For platforms with a single date column
                    record_date = pd.to_datetime(row.get(date_columns), errors='coerce')
                    if not pd.isna(record_date):
                        date_id = self.date_ids.get(record_date.strftime('%Y-%m-%d'))
                        
                        #If daily date not found, try to use monthly date
                        if date_id is None:
                            date_id = self.date_ids.get(f"M{record_date.year}-{record_date.month:02d}")
                            
                            #If monthly date not found, try to use yearly date
                            if date_id is None:
                                date_id = self.date_ids.get(f"Y{record_date.year}")
            except Exception as e:
                logger.error(f"Error processing date for record in {file_name}: {str(e)}")
            
            if record_date is None or pd.isna(record_date) or date_id is None:
                logger.warning(f"Could not determine date for record {records_loaded + 1} in {file_name}")
                continue
            
            #Load metrics for this record
            for metric_name, category, is_calculated in metrics_list:
                if metric_name not in row or pd.isna(row[metric_name]):
                    continue
                
                #Get metric value
                metric_value = convert_metric_value(metric_name, row[metric_name])
                if metric_value is None:
                    continue
                
                #Get metric id
                metric_id = self.add_metric(metric_name, category, is_calculated)
                
                #Create fact record
                fact_record = {
                    'fact_id': self.fact_id_counter,
                    'client_id': client_id,
                    'date_id': date_id,
                    'platform_id': platform_id,
                    'metric_id': metric_id,
                    'value': metric_value}
                
                fact_records.append(fact_record)
                self.fact_id_counter += 1
                records_loaded += 1
        
        #Add new fact records if any
        if fact_records:
            new_facts_df = pd.DataFrame(fact_records)
            self.fact_metrics = pd.concat([self.fact_metrics, new_facts_df], ignore_index=True)
        
        logger.info(f"Loaded {records_loaded} fact records from {file_name}")
        return True
    
    def save_dimension_tables(self):
        self.dim_client.to_csv(self.output_dir / 'dimensions' / 'dim_client.csv', index=False)
        self.dim_date.to_csv(self.output_dir / 'dimensions' / 'dim_date.csv', index=False)
        self.dim_platform.to_csv(self.output_dir / 'dimensions' / 'dim_platform.csv', index=False)
        self.dim_metric.to_csv(self.output_dir / 'dimensions' / 'dim_metric.csv', index=False)
        
        logger.info("Saved dimension tables to CSV files")
    
    def save_fact_table(self):
        self.fact_metrics.to_csv(self.output_dir / 'facts' / 'fact_metrics.csv', index=False)
        logger.info(f"Saved fact table with {len(self.fact_metrics)} records to CSV file")
    
    def create_analysis_views(self):
        #Daily metrics view
        daily_metrics = self.fact_metrics.merge(self.dim_client, on='client_id').merge(self.dim_date, on='date_id' ).merge(self.dim_platform, on='platform_id').merge(self.dim_metric, on='metric_id')

        #Filter to daily records only
        daily_metrics = daily_metrics[daily_metrics['period_type'] == 'Day']

        #Exclude unreliable audience metrics for FB and IG
        exclude_condition = (
            (daily_metrics['metric_category'] == 'Audience') &
            (daily_metrics['platform_name'].isin(['Facebook Pages', 'Instagram', 'Facebook Ads'])) )
        daily_metrics = daily_metrics[~exclude_condition]

        #Select core fields
        daily_metrics = daily_metrics[[
            'Client_Name', 'full_date', 'platform_name', 
            'metric_name', 'metric_category', 'value'
        ]]

        #Flag rows where we can infer when a post occurred - probably not super reliable
        #TODO - find a way to extract this pre loading data in api extractor
        daily_metrics['inferred_post'] = daily_metrics['metric_category'].isin(['Engagement', 'Impressions']).astype(int)

        #Group by client + date + platform to collapse to one flag per day
        inferred_post_flags = daily_metrics.groupby(
            ['Client_Name', 'full_date', 'platform_name']
        )['inferred_post'].max().reset_index()

        inferred_post_flags.rename(columns={'inferred_post': 'inferred_post_flag'}, inplace=True)

        #Merge flag back into full view
        daily_metrics = daily_metrics.merge(
            inferred_post_flags, 
            on=['Client_Name', 'full_date', 'platform_name'], 
            how='left' )

        #Save final result
        daily_metrics.to_csv(self.output_dir / 'vw_daily_metrics.csv', index=False)
        logger.info(f"Created daily metrics view with inferred posts: {len(daily_metrics)} records")

        
        #Monthly metrics view
        #First create a copy of the data
        monthly_data = daily_metrics.copy()
        
        #Convert date to datetime
        monthly_data['full_date'] = pd.to_datetime(monthly_data['full_date'])
        
        #Extract year and month
        monthly_data['year'] = monthly_data['full_date'].dt.year
        monthly_data['month'] = monthly_data['full_date'].dt.month
        monthly_data['month_name'] = monthly_data['full_date'].dt.strftime('%B')
        
        #Group by month - using 'Client_Name' instead of 'client_name'
        monthly_metrics = monthly_data.groupby([
            'Client_Name', 'year', 'month', 'month_name', 
            'platform_name', 'metric_name', 'metric_category']).agg({
            'value': ['mean', 'sum', 'count']}).reset_index()
        
        #Flatten MultiIndex
        monthly_metrics.columns = ['_'.join(col).strip('_') for col in monthly_metrics.columns.values]
        
        #Rename aggregation columns
        monthly_metrics = monthly_metrics.rename(columns={
            'value_mean': 'avg_value',
            'value_sum': 'sum_value',
            'value_count': 'count_value'})
 
        #Save to CSV
        monthly_metrics.to_csv(self.output_dir / 'vw_monthly_metrics.csv', index=False)
        logger.info(f"Created monthly metrics view with {len(monthly_metrics)} records")
        
        #Platform comparison view - already excluding audience metrics
        #Using 'Client_Name' instead of 'client_name'
        platform_comparison = monthly_data[monthly_data['metric_category'].isin(['Engagement', 'Reach'])]
        
        platform_comparison = platform_comparison.groupby([
            'Client_Name', 'year', 'month', 'platform_name', 'metric_category'
        ]).agg({
            'value': 'sum'
        }).reset_index()
        
        platform_comparison.rename(columns={'value': 'total_value'}, inplace=True)
        
        platform_comparison.to_csv(self.output_dir / 'vw_platform_comparison.csv', index=False)
        logger.info(f"Created platform comparison view with {len(platform_comparison)} records")
        
        #Mailchimp view
        mailchimp_data = self.fact_metrics.merge(self.dim_client, on='client_id').merge(self.dim_date, on='date_id').merge(self.dim_platform, on='platform_id').merge(self.dim_metric, on='metric_id')
        
        mailchimp_data = mailchimp_data[mailchimp_data['platform_name'] == 'Mailchimp']
        
        mailchimp_view = mailchimp_data[[
            'Client_Name', 'year', 'month', 'metric_name', 'value']]
        
        mailchimp_view.to_csv(self.output_dir / 'vw_mailchimp_campaigns.csv', index=False)
        logger.info(f"Created Mailchimp view with {len(mailchimp_view)} records")    
    
    def run_pipeline(self):
        try:
            #Generate date dimension
            self.generate_date_dimension()
            
            #Generate platform dimension
            self.generate_platform_dimension()
            
            #Get list of all files
            all_files = []
            for ext in ['*.csv', '*.xlsx', '*.xls']:
                all_files.extend(glob.glob(str(self.data_dir / ext)))
                
            #Process each file
            for file_path in all_files:
                self.transform_and_load_file(file_path)
                
            #Save all dimension and fact tables
            self.save_dimension_tables()
            self.save_fact_table()
            
            #Create analysis views
            self.create_analysis_views()
            
            #Export client dimension to CSV for reference
            self.dim_client.to_csv(self.output_dir / 'client_mapping.csv', index=False)
            
            logger.info("ETL pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in ETL pipeline: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    #Set paths for data and output
    data_dir = r"C:\Users\Arthu\Documents\Capstone\1_ETL\2. Original Data\Combined"
    output_dir = r"C:\Users\Arthu\Documents\Capstone\1_ETL\3. Processed Data"
    
    #Run ETL pipeline
    etl = SocialMediaETL(data_dir, output_dir)
    etl.run_pipeline()