"""
Capstone Project
FacebookAds data extraction using Meta Graph API
Author: Arthur Osakwe
Version: 2.0
"""


from datetime import datetime, timedelta
import os
import time
import csv
import sys
import concurrent.futures
from functools import partial

from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.adsinsights import AdsInsights
from facebook_business.adobjects.adreportrun import AdReportRun
from facebook_business.adobjects.campaign import Campaign
from facebook_business.exceptions import FacebookRequestError

#read api config from file
def read_config(config_path):
    config = {'app_id': '', 'app_secret': '', 'access_token': '', 'account_id': ''}
    
    try:
        with open(config_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip() in config:
                        config[key.strip()] = value.strip()
        
        #Check for missing values
        missing = [k for k, v in config.items() if not v]
        if missing:
            print(f"Warning: Missing values: {', '.join(missing)}")
        return config
    except Exception as e:
        print(f"Config error: {e}")
        return config

#inititlize api
def init_api(config):
    app_id = config['app_id'] or os.getenv('FB_APP_ID', '')
    app_secret = config['app_secret'] or os.getenv('FB_APP_SECRET', '')
    access_token = config['access_token'] or os.getenv('FB_ACCESS_TOKEN', '')
    
    if app_id and app_secret and access_token:
        FacebookAdsApi.init(app_id=app_id, app_secret=app_secret, 
                          access_token=access_token, api_version='v22.0')
        print("API initialized successfully")
        return True
    else:
        print("Error: Missing API configuration")
        return False


def get_active_campaigns(account_id):
    try:
        account = AdAccount(account_id)
        campaigns = account.get_campaigns(
            fields=['name', 'id', 'effective_status', 'objective', 'status'],
            params={'limit': 1000, 'effective_status': ['ACTIVE', 'PAUSED']}
        )
        active_campaigns = [c for c in campaigns if c.get('effective_status') == 'ACTIVE']
        
        if not active_campaigns:
            print("No active campaigns found")
            return []
        
        print(f"\nAuto-selected all {len(active_campaigns)} active campaigns.")
        return active_campaigns
    
    except Exception as e:
        print(f"Error fetching campaigns: {e}")
        return []


#extract client name
def extract_client_name(campaign_name):
    if '|' in campaign_name:
        client_name = campaign_name.split('|')[0].strip()
    else:
        #If no delimiter, use the whole campaign name as client name
        client_name = campaign_name
    
    return client_name

#default date range
def get_date_range():
    weeks = 13
    today = datetime.now()
    end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d') 
    start_date = (today - timedelta(weeks=weeks)).strftime('%Y-%m-%d')

    print(f"Auto-processing last {weeks} weeks:")
    print(f"  Period: {start_date} to {end_date}")
    return start_date, end_date



def wait_for_report(async_report, campaign_name, max_retries=5):
    retry_count = 0
    total_wait = 0
    print(f"Waiting for report '{campaign_name}'", end="")
    
    while retry_count < max_retries:
        try:
            start_time = time.time()
            while async_report['async_status'] != 'Job Completed':
                time.sleep(2)  #Shorter sleep interval
                total_wait += 2
                
                if total_wait % 10 == 0:
                    print(".", end="", flush=True)
                
                async_report.api_get()
                
                #Check for error status
                if async_report['async_status'] == 'Job Failed':
                    print(f" Error: Report failed after {total_wait} seconds!")
                    return None
                
                #Timeout after 300 seconds (5 minutes)
                if total_wait > 300:
                    print(f" Timeout after {total_wait} seconds!")
                    return None
            
            return async_report
        
        except FacebookRequestError as e:
            retry_count += 1
            wait_time = min(2 ** retry_count, 32)  #Exponential backoff
            print(f"\nAPI Error (retry {retry_count}/{max_retries}), waiting {wait_time}s: {e}")
            time.sleep(wait_time)
    
    print(f" Failed after {retry_count} retries.")
    return None


def get_campaign_metrics(campaign, start_date, end_date):
    try:
        campaign_id = campaign.get('id')
        campaign_name = campaign.get('name')
        client_name = extract_client_name(campaign_name)  #Extract client name
        campaign_obj = Campaign(campaign_id)
        
        print(f"Fetching data for '{campaign_name}'...")
        
        #Request all fields in one call
        params = {
            'time_range': {'since': start_date, 'until': end_date},
            'fields': [
                'date_start', 'date_stop', 'reach', 'impressions', 
                'frequency', 'cpm', 'clicks', 'outbound_clicks',
                'actions', 'spend', 'cost_per_action_type'
            ],
            'level': 'campaign',
            'time_increment': 1,  #Daily breakdown
            'limit': 500  #Increased the limit to reduce pagination
        }
        
        async_report = campaign_obj.get_insights(params=params, is_async=True)
        async_report.api_get()
        
        #Wait for report with better retry logic
        async_report = wait_for_report(async_report, campaign_name)
        
        if not async_report:
            return []
        
        insights = async_report.get_result()
        daily_metrics = []
        
        for insight in insights:
            daily_data = {
                "Client_Name": client_name,  #Add client name as first column
                "Campaign Name": campaign_name,
                "Campaign ID": campaign_id,
                "Date": insight.get('date_start'),
                "Reach": int(insight.get('reach', 0)),
                "Impressions": int(insight.get('impressions', 0)),
                "Frequency": float(insight.get('frequency', 0)),
                "CPM": float(insight.get('cpm', 0)),
                "Clicks": int(insight.get('clicks', 0)),
                "Amount Spent": float(insight.get('spend', 0)),
                "cost_per_result": {},
                "actions": {}
            }
            
            #Process link clicks - faster method
            outbound_clicks = insight.get('outbound_clicks', [])
            if isinstance(outbound_clicks, list):
                daily_data["Link Clicks"] = sum(int(item.get('value', 0)) for item in outbound_clicks)
            else:
                daily_data["Link Clicks"] = int(outbound_clicks)
            
            #Process actions - optimize by using dictionary comprehension
            daily_data['actions'] = {
                action.get('action_type'): int(action.get('value', 0))
                for action in insight.get('actions', [])
            }
            
            #Process cost per action - optimize using dictionary comprehension
            daily_data['cost_per_result'] = {
                entry.get('action_type'): float(entry.get('value', 0))
                for entry in insight.get('cost_per_action_type', [])
            }
            
            daily_metrics.append(daily_data)
        
        print(f"Retrieved {len(daily_metrics)} days of data for {campaign_name}.")
        return daily_metrics
    except Exception as e:
        print(f"Error for campaign {campaign.get('name')}: {e}")
        return []


def write_csv_report(filepath, rows):
    if not rows:
        print("No data to write")
        return
    
    #Get unique keys for cost_per_result and actions
    cost_keys = set()
    action_keys = set()
    for row in rows:
        cost_keys.update(row.get("cost_per_result", {}).keys())
        action_keys.update(row.get("actions", {}).keys())
    
    cost_keys = sorted(list(cost_keys))
    action_keys = sorted(list(action_keys))
    
    #Get base fields excluding nested dictionaries
    base_fields = [k for k in rows[0].keys() if k not in ("cost_per_result", "actions")]
    
    #Create flattened field names
    fieldnames = base_fields + [f"Cost_{ck}" for ck in cost_keys] + [f"Action_{ak}" for ak in action_keys]
    
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        #Use a list comprehension to create flattened rows
        flattened_rows = []
        for row in rows:
            flat_row = {f: row.get(f, "") for f in base_fields}
            
            #Flatten cost_per_result and actions in a single pass
            for ck in cost_keys:
                flat_row[f"Cost_{ck}"] = row.get("cost_per_result", {}).get(ck, 0)
            for ak in action_keys:
                flat_row[f"Action_{ak}"] = row.get("actions", {}).get(ak, 0)
            
            flattened_rows.append(flat_row)
        
        #Write all rows at once
        writer.writerows(flattened_rows)
    
    print(f"CSV report with {len(rows)} rows written to: {filepath}")

#process multiple campaigns in parallel using threads - should speed up code
def process_campaigns_parallel(campaigns, start_date, end_date, max_workers=5):
    all_data = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        #Create a partial function with fixed parameters
        get_metrics = partial(get_campaign_metrics, start_date=start_date, end_date=end_date)
        
        #Map the function to all campaigns and get results
        future_to_campaign = {executor.submit(get_metrics, campaign): campaign for campaign in campaigns}
        
        for future in concurrent.futures.as_completed(future_to_campaign):
            campaign = future_to_campaign[future]
            try:
                campaign_data = future.result()
                all_data.extend(campaign_data)
            except Exception as e:
                print(f"Campaign processing error for {campaign.get('name')}: {e}")
    
    return all_data


def main():
    #Initialize
    config = read_config("fb_ad_config.txt")
    if not init_api(config):
        return
    
    #Set output directory
    output_dir = "facebook_ad_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    #Get date range
    start_date, end_date = get_date_range()
    
    #Get campaigns
    campaigns = get_active_campaigns(config['account_id'])
    
    #Process campaigns in parallel with threads
    all_data = process_campaigns_parallel(campaigns, start_date, end_date)
    
    if not all_data:
        print("No data retrieved for the specified period.")
        return
    
    #Write to CSV
    output_file = os.path.join(output_dir, f"fb_ad_metrics.csv")
    write_csv_report(output_file, all_data)
    
    print(f"\nExtracted {len(all_data)} days of data from {len(campaigns)} campaigns")
    print(f"CSV report: {output_file}")


if __name__ == "__main__":
    main()