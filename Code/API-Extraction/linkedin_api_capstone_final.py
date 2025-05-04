"""
Capstone Project
Linkedin insights data extraction using LinkedInAPI
Author: Arthur Osakwe
Version: 3.0
"""
#import libraries
import os, csv, json, time, urllib.parse, urllib.request, urllib.error
from datetime import datetime, timedelta
from pathlib import Path
# this helps speed the code
import concurrent.futures 
import threading
#import pandas as pd  #TODO: Add pandas export option

#Configuration
CREDENTIALS_FILE = Path(r"C:\Users\Arthu\Documents\Capstone\1_ETL\1. Data Extraction Code via API\LinkedIn\linkedin_creds.txt")
CSV_SAVE_DIR = Path(r"C:\Users\Arthu\Documents\Capstone\1_ETL\2. Original Data\Combined")
API_VERSION = "202504"

#for getting around thread issues
thread_local = threading.local()

class LinkedInAPI:
    def __init__(self):
        self.token = None
        self.config = self._load_config()
        self._rate_limit_lock = threading.Lock()
        self._last_request_time = 0
        self._min_request_interval = 0.25  #250ms minimum between requests
    
    #load credentials    
    def _load_config(self):
        defaults = {
            'CLIENT_ID': '',
            'CLIENT_SECRET': '',
            'REDIRECT_URI': 'https://67a76e73aa02f.site123.me',
            'SCOPES': 'openid profile email r_organization_admin rw_organization_admin w_member_social r_organization_social'
        }
        
        if not CREDENTIALS_FILE.exists():
            print(f"Missing credentials file: {CREDENTIALS_FILE}")
            return defaults
            
        try:
            with open(CREDENTIALS_FILE, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key, value = key.strip(), value.strip()
                        if key in defaults:
                            defaults[key] = value
            print(f"Credentials loaded from {CREDENTIALS_FILE}")
        except Exception as e:
            print(f"Error reading credentials: {e}")
        
        return defaults
    
    def _request(self, method, url, headers=None, params=None, data=None, json_data=None):
        #rate limiting with thread safety
        with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < self._min_request_interval:
                sleep_time = self._min_request_interval - time_since_last
                time.sleep(sleep_time)
            self._last_request_time = time.time()
        
        #add query params to url if provided
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        
        #set up data and content type
        if json_data:
            data = json.dumps(json_data).encode('utf-8')
            headers = headers or {}
            headers['Content-Type'] = 'application/json'
        elif data and isinstance(data, dict):
            data = urllib.parse.urlencode(data).encode('utf-8')
            headers = headers or {}
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
        
        #create request
        req = urllib.request.Request(url, data=data, method=method)
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        
        #execute request and handle response
        try:
            with urllib.request.urlopen(req) as response:
                text = response.read().decode('utf-8')
                return {'status': response.status, 'text': text, 'json': json.loads(text) if text else {}}
        except urllib.error.HTTPError as e:
            text = e.read().decode('utf-8') if hasattr(e, 'read') else str(e)
            print(f"HTTP Error {e.code}: {e.reason}")
            
            #handle rate limiting
            if e.code == 429:
                print("Rate limited - waiting 5s...")
                time.sleep(5)
                
            return {
                'status': e.code, 
                'text': text, 
                'json': json.loads(text) if text and '{' in text else {},
                'error': True
            }
        except Exception as e:
            print(f"Request error: {e}")
            return {'status': 500, 'text': str(e), 'error': True}
    
    def authenticate(self):
        #generate authorization URL
        auth_url = (
            f"https://www.linkedin.com/oauth/v2/authorization?"
            f"response_type=code&"
            f"client_id={self.config['CLIENT_ID']}&"
            f"redirect_uri={urllib.parse.quote(self.config['REDIRECT_URI'])}&"
            f"scope={urllib.parse.quote(self.config['SCOPES'])}&"
            f"state=analytics"
        )
        print(f"Authorize here: {auth_url}")
        code = input("Enter authorization code: ").strip()

        #exchange code for token
        response = self._request('POST', "https://www.linkedin.com/oauth/v2/accessToken", 
                                data={
                                    "grant_type": "authorization_code",
                                    "code": code,
                                    "redirect_uri": self.config["REDIRECT_URI"],
                                    "client_id": self.config["CLIENT_ID"],
                                    "client_secret": self.config["CLIENT_SECRET"]
                                })
        
        if response.get('error'):
            raise Exception(f"Auth failed: {response.get('text')}")
            
        self.token = response['json'].get("access_token")
        return self.token
    
    #get api headers
    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "LinkedIn-Version": API_VERSION,
            "X-Restli-Protocol-Version": "2.0.0"
        }
    
    #get all orgs client has access to
    def get_organizations(self):
        url = "https://api.linkedin.com/rest/organizationAcls?q=roleAssignee&role=ADMINISTRATOR&state=APPROVED&count=100"
        response = self._request('GET', url, headers=self.get_headers())
        
        if response.get('error'):
            print(f"Error getting orgs: {response.get('status')}")
            return []
        
        orgs = []
        for elem in response['json'].get("elements", []):
            org_urn = elem.get("organization") or elem.get("organizationTarget")
            if not org_urn:
                continue
                
            #get organization name
            org_id = org_urn.split(":")[-1]
            name_response = self._request(
                'GET', 
                f"https://api.linkedin.com/v2/organizations/{org_id}", 
                headers=self.get_headers(),
                params={"projection": "(localizedName)"}
            )
            
            name = name_response['json'].get("localizedName", f"Unknown Page ({org_id})")
            orgs.append((org_urn, name))
            
        return orgs
    
    def get_analytics(self, org_urn, start_ms, end_ms):
        #format time interval string for api
        time_interval = f"(timeRange:(start:{start_ms},end:{end_ms}),timeGranularityType:DAY)"
        time_encoded = urllib.parse.quote(time_interval, safe="():,")
        org_encoded = urllib.parse.quote(org_urn)
        
        #define endpoints to query
        endpoints = {
            "follower_stats": f"https://api.linkedin.com/rest/organizationalEntityFollowerStatistics?q=organizationalEntity&organizationalEntity={org_encoded}&timeIntervals={time_encoded}",
            "share_stats": f"https://api.linkedin.com/rest/organizationalEntityShareStatistics?q=organizationalEntity&organizationalEntity={org_encoded}&timeIntervals={time_encoded}",
            "page_views": f"https://api.linkedin.com/rest/organizationPageStatistics?q=organization&organization={org_encoded}&timeIntervals={time_encoded}"
        }
        
        #get data for each endpoint
        results = {}
        for key, url in endpoints.items():
            response = self._request('GET', url, headers=self.get_headers())
            results[key] = response['json'] if not response.get('error') else {"elements": []}
            
        return results


def process_metrics(data, date_str):
    #initialize metrics
    metrics = {
        "Date": date_str,
        "New Followers": 0,
        "Reach": 0,
        "Impressions": 0,
        "Comments": 0,
        "Likes": 0, 
        "Clicks": 0,
        "Shares": 0,
        "Total Interactions": 0,
        "Profile Views (All Pages)": 0,
        "Profile Views (Overview Page)": 0,
        "Profile Views (About Page)": 0,
        "Engagement Rate": 0,
        "Avg Profile Views per New Follower": 0
    }
    
    #follower stats
    for elem in data.get("follower_stats", {}).get("elements", []):
        gains = elem.get("followerGains", {})
        metrics["New Followers"] += gains.get("organicFollowerGain", 0) + gains.get("paidFollowerGain", 0)
    
    #share stats
    for elem in data.get("share_stats", {}).get("elements", []):
        stats = elem.get("totalShareStatistics", {})
        metrics["Impressions"] += stats.get("impressionCount", 0)
        metrics["Reach"] += stats.get("uniqueImpressionsCount", 0)
        metrics["Comments"] += stats.get("commentCount", 0)
        metrics["Likes"] += stats.get("likeCount", 0)
        metrics["Clicks"] += stats.get("clickCount", 0)
        metrics["Shares"] += stats.get("shareCount", 0)
    
    #process page view stats
    for elem in data.get("page_views", {}).get("elements", []):
        views = elem.get("totalPageStatistics", {}).get("views", {})
        metrics["Profile Views (All Pages)"] += views.get("allPageViews", {}).get("pageViews", 0)
        metrics["Profile Views (Overview Page)"] += views.get("overviewPageViews", {}).get("pageViews", 0)
        metrics["Profile Views (About Page)"] += views.get("aboutPageViews", {}).get("pageViews", 0)
    
    #calculate derived metrics
    metrics["Total Interactions"] = metrics["Comments"] + metrics["Likes"] + metrics["Clicks"] + metrics["Shares"]
    
    if metrics["Impressions"] > 0:
        metrics["Engagement Rate"] = metrics["Total Interactions"] / metrics["Impressions"]
        
    if metrics["New Followers"] > 0:
        metrics["Avg Profile Views per New Follower"] = metrics["Profile Views (All Pages)"] / metrics["New Followers"]
    
    return metrics

#generate date ranges
def get_date_ranges(weeks=8):
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=weeks)
    ranges = []
    
    current = start_date
    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        day_start = datetime(current.year, current.month, current.day, 0, 0, 0)
        day_end = datetime(current.year, current.month, current.day, 23, 59, 59)
        
        #convert to milliseconds
        start_ms = int(day_start.timestamp() * 1000)
        end_ms = int(day_end.timestamp() * 1000)
        
        ranges.append((start_ms, end_ms, date_str))
        current += timedelta(days=1)
    
    return ranges

#allow user select organization - might consider removing for future automation work
def select_organizations(organizations):
    if not organizations:
        return []
        
    print("\nManaged LinkedIn Pages:")
    for idx, (urn, name) in enumerate(organizations, 1):
        print(f"{idx}. {name} ({urn})")
    
    selection = input("\nEnter Pages to analyze (e.g., 1, 2-4, or all): ").strip()
    
    if selection.lower() == "all":
        return [org[0] for org in organizations]
    
    selected = []
    try:
        for part in selection.replace(" ", "").split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                if 1 <= start <= end <= len(organizations):
                    selected.extend([organizations[i-1][0] for i in range(start, end+1)])
            else:
                idx = int(part)
                if 1 <= idx <= len(organizations):
                    selected.append(organizations[idx-1][0])
    except (ValueError, IndexError) as e:
        print(f"Selection error: {e}. Using first organization.")
        if organizations:
            selected = [organizations[0][0]]
    
    return selected

#helper function to get thread-local API instance
def get_api_instance():
    if not hasattr(thread_local, "api"):
        #Clone the main API instance for this thread
        thread_local.api = main_api
    return thread_local.api

#function to process a single organization
def process_organization(org_data, date_ranges, progress_lock):
    urn, name, org_idx, total_orgs = org_data
    api = get_api_instance()
    
    with progress_lock:
        print(f"\nProcessing organization {org_idx}/{total_orgs}: {name}")
    
    org_rows = []
    
    for day_idx, (start_ms, end_ms, date_str) in enumerate(date_ranges, 1):
        with progress_lock:
            if day_idx % 10 == 0 or day_idx == 1:
                print(f"[{name}] Day {day_idx}/{len(date_ranges)}: {date_str}")
        
        #Get and process data
        analytics = api.get_analytics(urn, start_ms, end_ms)
        metrics = process_metrics(analytics, date_str)
        
        #Add organization info
        org_rows.append({
            "Organization": name,
            "Organization URN": urn,
            **metrics
        })
    
    with progress_lock:
        print(f"Completed organization: {name}")
        
    return org_rows

def main():
    try:
        global main_api  #We'll use this as the template for thread-local instances
        
        #initialize API and authenticate
        main_api = LinkedInAPI()
        main_api.authenticate()
        
        #get organizations and let user select
        organizations = main_api.get_organizations()
        if not organizations:
            print("No organizations found.")
            return
            
        selected_urns = [org[0] for org in organizations]  #Auto-select all
        print(f"\nAuto-selected {len(selected_urns)} organizations.")
        
        if not selected_urns:
            print("No organizations selected.")
            return
        
        #create dictionary for names
        org_names = dict(organizations)
        
        #get number of weeks to analyze - defaulted to 13 weeks
        weeks = 13  #Or load from environment/config if needed
        print(f"\nAuto-analyzing past {weeks} weeks.")
        
        #generate date ranges
        date_ranges = get_date_ranges(weeks=weeks)
        print(f"\nAnalyzing {len(date_ranges)} days from {date_ranges[0][2]} to {date_ranges[-1][2]}")
        
        #Set the maximum number of concurrent tasks
        #Start with a modest number to avoid hitting rate limits
        max_workers = min(4, len(selected_urns))
        
        #Use the default maximum workers
        print(f"Using {max_workers} parallel workers")
        
        #Prepare organization data tuples (urn, name, index, total)
        org_data = [
            (urn, org_names.get(urn, "Unknown Organization"), i+1, len(selected_urns))
            for i, urn in enumerate(selected_urns)
        ]
        
        #Create a lock for thread-safe progress reporting
        progress_lock = threading.Lock()
        all_rows = []
        
        #Process organizations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            #Submit all tasks
            future_to_org = {
                executor.submit(process_organization, data, date_ranges, progress_lock): data 
                for data in org_data
            }
            
            #Process results as they complete
            for future in concurrent.futures.as_completed(future_to_org):
                org_data = future_to_org[future]
                try:
                    org_rows = future.result()
                    all_rows.extend(org_rows)
                except Exception as exc:
                    with progress_lock:
                        print(f"Error processing {org_data[1]}: {exc}")
        
        #save to CSV
        if all_rows:
            csv_path = CSV_SAVE_DIR / f"linkedin_daily_all_pages.csv"
            os.makedirs(csv_path.parent, exist_ok=True)
            
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=all_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_rows)
            
            print(f"\nReport saved to: {csv_path}")
        
        print("\nData extraction complete!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()