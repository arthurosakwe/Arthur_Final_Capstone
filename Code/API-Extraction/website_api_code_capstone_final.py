"""
Capstone Project
Website data extraction using Google Analytics API
Author: Arthur Osakwe
Version: 1.0
"""

import os
import csv
import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

#Configuration
SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
CLIENT_SECRETS_FILE = 'client_secrets_p.json'
CSV_SAVE_DIR = r'C:\Users\Arthu\Documents\Capstone\1_ETL\2. Original Data\Combined'
CSV_FILENAME = "website_daily_analytics_all.csv"
TOKEN_PATH = 'token.json'

#authenticate with google analytics
def authenticate():
    creds = None
    #Check if token file exists and is valid
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
            print("Using existing token.")
            return creds
        except Exception as e:
            print(f"Error with existing token: {e}")    

#List each GA4 account
def list_accounts_and_properties():
    try:
        creds = authenticate()
        admin_service = build('analyticsadmin', 'v1alpha', credentials=creds)
        
        accounts = admin_service.accounts().list().execute() \
                                .get('accounts', [])
        if not accounts:
            print("No accounts found.")
            return []
        
        account_list = []
        for i, acct in enumerate(accounts, 1):
            acct_id = acct['name'].split('/')[-1]
            acct_name = acct.get('displayName', f'Account {acct_id}')
            
            props = admin_service.properties().list(
                filter=f"parent:accounts/{acct_id}"
            ).execute().get('properties', [])
            
            prop_ids = [p['name'].split('/')[-1] for p in props]
            ids_str = ", ".join(prop_ids) if prop_ids else "(no properties)"
            
            #Print in the format: "1. Account Name – propID1, propID2"
            print(f"{i}. {acct_name} - {ids_str}")
            
            account_list.append({
                'name': acct_name,
                'properties': prop_ids
            })
        
        return account_list

    except Exception as e:
        print(f"Error listing accounts and properties: {e}")
        return []

#auto select all propoerties
def select_properties_to_process(account_list):
    if not account_list:
        print("No accounts available to process.")
        return []

    selected_properties = []

    for account in account_list:
        if not account['properties']:
            print(f"{account['name']} has no properties. Skipping.")
            continue

        #Automatically use the first property
        prop_id = account['properties'][0]
        selected_properties.append({
            'id': prop_id,
            'account_name': account['name']})

    print(f"\nAuto-selected {len(selected_properties)} properties:")
    for i, prop in enumerate(selected_properties, 1):
        print(f"{i}. {prop['account_name']} - {prop['id']}")

    return selected_properties


def get_property_info(property_id):
    #Get account and property names for a GA4 property
    try:
        creds = authenticate()
        admin_service = build('analyticsadmin', 'v1alpha', credentials=creds)
        
        property_details = admin_service.properties().get(
            name=f"properties/{property_id}"
        ).execute()
        
        property_name = property_details.get("displayName", "Unknown Property")
        account_name = "Unknown Account"
        
        account_id = property_details.get("account", "")
        if account_id:
            try:
                account_details = admin_service.accounts().get(name=account_id).execute()
                account_name = account_details.get("displayName", "Unknown Account")
            except Exception as e:
                print(f"Error getting account info: {e}")
        
        return account_name, property_name
    except Exception as e:
        print(f"Error getting property info: {e}")
        return "Unknown Account", "Unknown Property"

def get_manual_properties():
    #func to get properties manually from user input
    print("\nPlease enter the GA4 property IDs you want to process.")
    custom_ids = input("Enter property IDs separated by commas: ").strip()
    if not custom_ids:
        return []
        
    properties = []
    for pid in custom_ids.split(','):
        if pid.strip():
            pid = pid.strip()
            account_name, property_name = get_property_info(pid)
            properties.append({
                'id': pid,
                'account_name': account_name,
                'display_name': property_name
            })
    
    return properties

def run_ga4_report(property_id, start_date, end_date, report_type):
    #Run various GA4 reports based on type
    try:
        creds = authenticate()
        service = build('analyticsdata', 'v1beta', credentials=creds)
        
        request = {
            'dateRanges': [{'startDate': start_date, 'endDate': end_date}],
            'dimensions': [{'name': 'date'}],
            'orderBys': [{'dimension': {'dimensionName': 'date'}}]
        }
        
       
        if report_type == 'metrics':
            request['metrics'] = [
                {'name': 'activeUsers'},
                {'name': 'engagedSessions'},
                {'name': 'newUsers'},
                {'name': 'eventCount'},
                {'name': 'screenPageViews'},
                {'name': 'sessions'},
                {'name': 'engagementRate'},
                {'name': 'userEngagementDuration'}
            ]
        elif report_type == 'pages':
            request['metrics'] = [{'name': 'screenPageViews'}]
            request['dimensions'].extend([
                {'name': 'pageTitle'},
                {'name': 'pagePath'}
            ])
            request['orderBys'].append({
                'metric': {'metricName': 'screenPageViews'}, 
                'desc': True
            })
        elif report_type == 'countries':
            request['metrics'] = [{'name': 'screenPageViews'}]
            request['dimensions'].append({'name': 'country'})
            request['orderBys'].append({
                'metric': {'metricName': 'screenPageViews'}, 
                'desc': True
            })
        elif report_type == 'states':
            request['metrics'] = [{'name': 'screenPageViews'}]
            request['dimensions'].append({'name': 'region'})
            request['dimensionFilter'] = {
                'filter': {
                    'fieldName': 'country',
                    'stringFilter': {
                        'value': 'United States',
                        'matchType': 'EXACT'
                    }
                }
            }
            request['orderBys'].append({
                'metric': {'metricName': 'screenPageViews'}, 
                'desc': True
            })
        
        response = service.properties().runReport(
            property=f'properties/{property_id}',
            body=request
        ).execute()
        
        return response
    except Exception as e:
        print(f"Error running {report_type} report for {property_id}: {e}")
        return None

def format_date(date_str):
    #Format GA4 date (YYYYMMDD) to YYYY-MM-DD
    try:
        return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
    except Exception:
        return date_str

def format_duration(seconds):
    #Convert seconds to minutes:seconds format
    try:
        seconds = float(seconds)
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    except Exception:
        return seconds

def main():    
    try:
        #List accounts and properties
        print("Listing your Google Analytics accounts and properties:")
        account_list = list_accounts_and_properties()
        
        #Let user select properties
        properties = select_properties_to_process(account_list)
        

        manual_properties = get_manual_properties()
        if manual_properties:
            properties = manual_properties     
            
        #Ask for number of weeks to analyze
        weeks = 13  
        print(f"\nAuto-analyzing the last {weeks} weeks.")

        
        #Calculate start and end date based on weeks
        today = datetime.datetime.now()
        end_date = (today - datetime.timedelta(days=1)).strftime('%Y-%m-%d')  #Yesterday
        start_date = (today - datetime.timedelta(weeks=weeks)).strftime('%Y-%m-%d')
    
        print(f"  Date range: {start_date} to {end_date}")

        all_data = []
        
        #Process each property
        for prop in properties:
            property_id = prop['id']
            account_name, property_name = get_property_info(property_id)
            print(f"\nProcessing property: {property_id} - {property_name}")

            
            #Get all reports
            metrics_report = run_ga4_report(property_id, start_date, end_date, 'metrics')   
            page_report = run_ga4_report(property_id, start_date, end_date, 'pages')
            country_report = run_ga4_report(property_id, start_date, end_date, 'countries')
            state_report = run_ga4_report(property_id, start_date, end_date, 'states')
            
            #Process top pages by date
            top_pages = {}
            if page_report and 'rows' in page_report:
                for row in page_report['rows']:
                    date = format_date(row['dimensionValues'][0]['value'])
                    path = row['dimensionValues'][2]['value']
                    views = int(row['metricValues'][0]['value'])
                    
                    if date not in top_pages or views > top_pages[date]['views']:
                        top_pages[date] = {
                            'url': f"https://{property_id}.com{path}",
                            'views': views
                        }
            
            #Process top countries by date
            top_countries = {}
            if country_report and 'rows' in country_report:
                for row in country_report['rows']:
                    date = format_date(row['dimensionValues'][0]['value'])
                    country = row['dimensionValues'][1]['value']
                    views = int(row['metricValues'][0]['value'])
                    
                    if date not in top_countries or views > top_countries[date]['views']:
                        top_countries[date] = {
                            'country': country,
                            'views': views
                        }
            
            #Process top states by date
            top_states = {}
            if state_report and 'rows' in state_report:
                for row in state_report['rows']:
                    date = format_date(row['dimensionValues'][0]['value'])
                    state = row['dimensionValues'][1]['value']
                    views = int(row['metricValues'][0]['value'])
                    
                    if date not in top_states or views > top_states[date]['views']:
                        top_states[date] = {
                            'state': state,
                            'views': views
                        }
            
            #Create combined dataset
            if metrics_report and 'rows' in metrics_report:
                print(f"Processing {len(metrics_report['rows'])} rows of data...")
                for row in metrics_report['rows']:
                    date = format_date(row['dimensionValues'][0]['value'])
                    metrics = row['metricValues']
                    
                    data_row = {
                        "Client_Name": prop['account_name'],
                        "Property ID": property_id,
                        "Date": date,
                        "Views": metrics[4]['value'],
                        "Active Users": metrics[0]['value'],
                        "Engaged Sessions": metrics[1]['value'],
                        "New Users": metrics[2]['value'],
                        "Event Count": metrics[3]['value'],
                        "Sessions": metrics[5]['value'],
                        "Engagement Rate (%)": f"{float(metrics[6]['value']) * 100:.2f}%",
                        "User Engagement Duration": format_duration(metrics[7]['value']),
                        "Top Page URL": top_pages.get(date, {}).get('url', ''),
                        "Top Country": top_countries.get(date, {}).get('country', ''),
                        "Top State": top_states.get(date, {}).get('state', '')
                    }
                    
                    all_data.append(data_row)
            else:
                print(f"No metrics data found for property {property_id}")
        
        #Create dict if it doesn't exist
        os.makedirs(CSV_SAVE_DIR, exist_ok=True)
        
        #Write results to CSV
        output_path = os.path.join(CSV_SAVE_DIR, CSV_FILENAME)
        
        if all_data:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=all_data[0].keys())
                writer.writeheader()
                writer.writerows(all_data)
            
            print(f"\nData saved to: {output_path}")
            print(f"Total records: {len(all_data)}")
        else:
            print("No data collected. CSV not created.")
            
    except Exception as e:
        print(f"Error in main process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()