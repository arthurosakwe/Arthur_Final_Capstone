"""
Capstone Project
Instagram insights data extraction using Meta Graph API
Author: Arthur Osakwe
Version: 4.0
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import logging
from typing import Dict, Any, List, Optional
import re
import sys  # Used for debug printing occasionally
import random  # For jittering API delay
from pathlib import Path  # Not using but might switch to this later


#setup logging - TODO: Move this to a config file maybe?
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("instagram_data_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

#constants
API_DELAY = 0.5  # Delay between API calls to avoid rate limits
# DEBUG_MODE = False  # Keeping this around in case I need it later
OUTPUT_DIR = r"C:\Users\Arthu\Documents\Capstone\1_ETL\2. Original Data\Combined"
MAX_RETRIES = 3  # Add retry logic for flaky API calls

#read access token from txt file
def read_access_token_from_file(file_path="fb_access_token.txt"):
    try:
        with open(file_path, 'r') as file:
            access_token = file.readline().strip()
            logger.info("Retrieved access token")
            return access_token
    except Exception as e:
        logger.error(f"Couldn't read access token: {str(e)}")
        return None

#get ig account list
def get_user_instagram_accounts(access_token):
    url = "https://graph.facebook.com/v22.0/me/accounts"
    params = {
        "access_token": access_token,
        "fields": "name,id,access_token,instagram_business_account"
    }
    
    insta_accounts = []
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        pages = response.json().get("data", [])
        
        #filter pages with Instagram business accounts
        for page in pages:
            if "instagram_business_account" in page:
                ig_id = page["instagram_business_account"]["id"]
                #get Instagram account details
                ig_url = f"https://graph.facebook.com/v22.0/{ig_id}"
                ig_params = {
                    "access_token": page["access_token"],
                    "fields": "id,username,name,profile_picture_url,followers_count,media_count,biography"
                }
                
                #sometimes the API is flaky -retry a couple times
                success = False
                retries = 0
                
                while not success and retries < MAX_RETRIES:
                    try:
                        ig_response = requests.get(url=ig_url, params=ig_params)
                        ig_response.raise_for_status()
                        ig_data = ig_response.json()
                        
                        # Add page access token to Instagram data for later use
                        ig_data["page_access_token"] = page["access_token"]
                        ig_data["page_id"] = page["id"]
                        ig_data["page_name"] = page["name"]
                        
                        insta_accounts.append(ig_data)
                        logger.info(f"Found account: {ig_data.get('username', 'Unknown')} (ID: {ig_id})")
                        success = True
                    except Exception as e:
                        retries += 1
                        logger.warning(f"Attempt {retries}/{MAX_RETRIES} failed: {str(e)}")
                        time.sleep(API_DELAY * 2)  # Wait a bit longer between retries
                

                sleep_time = API_DELAY + random.uniform(-0.1, 0.2)
                time.sleep(sleep_time)
        
        logger.info(f"Found {len(insta_accounts)} accounts")
        return insta_accounts
    except Exception as e:
        logger.error(f"Error fetching pages: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.debug(f"Response details: {e.response.text}")
        return []

def get_follows_and_unfollows_weekly(ig_account_id, page_token, since_date, until_date):
    url = f"https://graph.facebook.com/v22.0/{ig_account_id}/insights"
    
    #nconvert dates to UNIX timestamps
    if 'T' in since_date:
        since_date = since_date.split('T')[0]
    if 'T' in until_date:
        until_date = until_date.split('T')[0]
        
    #timestamp conversion
    since_timestamp = int(datetime.strptime(since_date, "%Y-%m-%d").timestamp())
    until_timestamp = int(datetime.strptime(until_date, "%Y-%m-%d").timestamp()) + 86400  # Add a day in seconds
    
    params = {
        "access_token": page_token,
        "metric": "follows_and_unfollows",
        "period": "day",
        "since": since_timestamp,
        "until": until_timestamp,
        "breakdown": "follow_type",
        "metric_type": "total_value"
    }
    
    follows_data_by_day = {}
    
    try:
        logger.info(f"Getting follows/unfollows: {since_date} to {until_date}")
        response = requests.get(url=url, params=params)
        response.raise_for_status()
        data = response.json().get("data", [])
        
        #process the data - organize by day
        if data and len(data) > 0:
            for metric in data:
                if metric.get("name") == "follows_and_unfollows" and "total_value" in metric:
                    #look for breakdowns
                    if "breakdowns" in metric["total_value"]:
                        for breakdown_set in metric["total_value"]["breakdowns"]:
                            for result in breakdown_set.get("results", []):
                                #get the day this result applies to
                                if "end_time" in result:
                                    day = result["end_time"].split("T")[0]
                                    if day not in follows_data_by_day:
                                        follows_data_by_day[day] = {"follows": 0, "unfollows": 0}
                                    
                                    dim_vals = result.get("dimension_values", [])
                                    if dim_vals and len(dim_vals) > 0:
                                        # Check if it's a follow or unfollow
                                        if dim_vals[0] == "follow":
                                            follows_data_by_day[day]["follows"] = result.get("value", 0)
                                        elif dim_vals[0] == "unfollow":
                                            follows_data_by_day[day]["unfollows"] = result.get("value", 0)
        
        # print(f"DEBUG: Got {len(follows_data_by_day)} days of follow data") 
        time.sleep(API_DELAY)
        return follows_data_by_day
    except Exception as e:
        logger.error(f"follow data API failed: {str(e)}")
        # TODO: Maybe add retry logic here?
        return {}
#get post insights
def get_media_insights(media_id: str, page_token: str) -> Dict[str, Any]:
    url = f"https://graph.facebook.com/v22.0/{media_id}/insights"
    params = {
        "access_token": page_token,
        "metric": "engagement,reach,impressions,saved,video_views"
    }
    
    #ffix for rate limits - adding more delay for video posts
    extra_delay = 0
    
    try:
        response = requests.get(url=url, params=params)
        response.raise_for_status()
        data = response.json().get("data", [])
        
        insights = {}
        for metric in data:
            metric_name = metric.get("name")
            metric_value = metric.get("values", [{}])[0].get("value", 0)
            
            # Sometimes video posts need more time between requests
            if metric_name == "video_views" and metric_value > 1000:
                extra_delay = 0.2
                
            insights[metric_name] = metric_value
        
        if extra_delay > 0:
            time.sleep(extra_delay)
            
        return insights
    except Exception as e:
        #skip insights for problematic media rather than crashing
        logger.warning(f"Couldn't get insights for {media_id[:8]}... - skipping")
        return {}


def get_instagram_media_for_period(ig_account_id, page_token, since_date, until_date, limit=100):
    url = f"https://graph.facebook.com/v22.0/{ig_account_id}/media"
    
    #clean up dates
    if 'T' in since_date:
        since_date = since_date.split('T')[0]
    if 'T' in until_date:
        until_date = until_date.split('T')[0]
    
    #add one day to until_date to include the full day
    until_dt = datetime.strptime(until_date, "%Y-%m-%d") + timedelta(days=1)
    until_date = until_dt.strftime("%Y-%m-%d")
    
    #enhanced fields to get more post data - these are usually enough
    params = {
        "access_token": page_token,
        "fields": "id,caption,media_type,permalink,timestamp,like_count,comments_count",
        "limit": limit
    }
    
    all_posts = []  
    
    try:
        logger.info(f"Getting posts for account {ig_account_id} from {since_date} to {until_date}")
        response = requests.get(url=url, params=params)
        # should probably add error handling here if API is acting up
        response.raise_for_status()
        
        data = response.json()
        media_items = data.get("data", [])
        
        #filter by date range
        since_dt = datetime.strptime(since_date, "%Y-%m-%d")
        until_dt = datetime.strptime(until_date, "%Y-%m-%d")
        since_ts = since_dt.timestamp()
        until_ts = until_dt.timestamp()
        
        #process initial batch
        for item in media_items:
            if "timestamp" in item:
                try:
                    post_time = datetime.strptime(item["timestamp"], "%Y-%m-%dT%H:%M:%S%z").timestamp()
                    if since_ts <= post_time < until_ts:
                        #extract date for organizing
                        post_date = datetime.fromtimestamp(post_time).strftime("%Y-%m-%d")
                        item["post_date"] = post_date
                        all_posts.append(item)
                except ValueError:
                    try:
                        post_time = datetime.strptime(item["timestamp"], "%Y-%m-%dT%H:%M:%S+0000").timestamp()
                        if since_ts <= post_time < until_ts:
                            post_date = datetime.fromtimestamp(post_time).strftime("%Y-%m-%d")
                            item["post_date"] = post_date
                            all_posts.append(item)
                    except:
                        # If all parsing fails, skip this item
                        pass
        
        # Handle pagination to get all media within date range
        page_count = 1 #For logging
        while "paging" in data and "next" in data["paging"] and len(all_posts) < 200:
            page_count += 1
            logger.info(f"Getting page {page_count} of posts (have {len(all_posts)} so far)")
            next_url = data["paging"]["next"]
            
            try:
                response = requests.get(url=next_url)
                response.raise_for_status()
                
                data = response.json()
                media_items = data.get("data", [])
                
                #Filter media by date
                for item in media_items:
                    if "timestamp" in item:
                        try:
                            post_time = datetime.strptime(item["timestamp"], "%Y-%m-%dT%H:%M:%S%z").timestamp()
                            if since_ts <= post_time < until_ts:
                                post_date = datetime.fromtimestamp(post_time).strftime("%Y-%m-%d")
                                item["post_date"] = post_date
                                all_posts.append(item)
                        except ValueError:
                            try:
                                # Try alternative format
                                post_time = datetime.strptime(item["timestamp"], "%Y-%m-%dT%H:%M:%S+0000").timestamp()
                                if since_ts <= post_time < until_ts:
                                    post_date = datetime.fromtimestamp(post_time).strftime("%Y-%m-%d")
                                    item["post_date"] = post_date
                                    all_posts.append(item)
                            except:
                                # Skip items with weird timestamps
                                pass
            except Exception as e:
                logger.error(f"Pagination failed on page {page_count}: {str(e)}")
                break
            
            time.sleep(API_DELAY + random.uniform(0, 0.3))
            
            #chk if we have enough media or reached the end
            if len(all_posts) >= 200 or len(media_items) == 0:
                if len(all_posts) >= 200:
                    logger.info("Hit the 200 post limit, good enough for analysis")
                else:
                    logger.info("No more posts available")
                break
        
        #organize by day
        posts_by_day = {}
        for item in all_posts:
            day = item.get("post_date")
            if day:
                if day not in posts_by_day:
                    posts_by_day[day] = []
                posts_by_day[day].append(item)
        
        
        logger.info(f"Got {len(all_posts)} posts from account {ig_account_id}")
        return posts_by_day
    except Exception as e:
        logger.error(f"Error getting posts for {ig_account_id}: {str(e)}")
        return {}

#Quick function to extract hashtags - keeping this for backwards compatibility
def extract_hashtags(text):
    return re.findall(r'#(\w+)', text) if text else []

def process_account_weekly(account, weekly_ranges):
    account_id = account["id"]
    username = account.get("username", "Unknown")
    page_token = account["page_access_token"]
    followers = account.get("followers_count", 0)  # This name is clearer
    
    logger.info(f"Starting to process account: {username} (ID: {account_id}) - {followers} followers")
    
    #create a list to store data for all days
    account_daily_data = []
    
    #get current follower count  
    current_followers = followers
    
    #keep track of processing time - this helps me optimize later
    start_time = time.time()
    
    #process each week - faster
    for i, week_range in enumerate(weekly_ranges):
        since_date = week_range["start"]
        until_date = week_range["end"]
        week_name = week_range["name"]
        
        logger.info(f"Week {i+1}/{len(weekly_ranges)}: {week_name}")
        
        #get follows/unfollows for the week
        follows_data = get_follows_and_unfollows_weekly(account_id, page_token, since_date, until_date)
        
        #get media for the week - most important data
        media_by_day = get_instagram_media_for_period(account_id, page_token, since_date, until_date)
        
        #process each day's data within this week
        #sort for readability in logs
        for day_date in sorted(set(list(follows_data.keys()) + list(media_by_day.keys()))):
            #get follows/unfollows for this day
            day_follows = follows_data.get(day_date, {"follows": 0, "unfollows": 0})
            
            #get media for this day
            day_media = media_by_day.get(day_date, [])
            
            #create day data entry
            day_data = {
                "Client_Name": username,
                "instagram_id": account_id,
                "username": username,
                "date": day_date,
                "Followers": current_followers,  # Use current follower count
                "New Follows": day_follows["follows"],
                "Unfollows": day_follows["unfollows"],
                "Net Follower Change": day_follows["follows"] - day_follows["unfollows"],
                "Posts Count": len(day_media),  # Actual number of posts made this day
                "Has Post": 1 if len(day_media) > 0 else 0  # Flag for ETL integration
            }
            
            #get insights for ALL posts to have accurate data
            total_likes = 0
            total_comments = 0
            total_saved = 0
            total_reach = 0
            total_views = 0
            total_engagement = 0
            top_post = None
            top_engagement = 0
            
            # track hashtags - this is super useful for clients
            all_hashtags = []
            
            #process each post
            for post in day_media:
                #get post ID and type
                post_id = post.get("id")
                media_type = post.get("media_type", "IMAGE")
                
                #extract hashtags from caption
                caption = post.get("caption", "")
                
                #one of these ways is probably better but whatever works
                #hashtags = extract_hashtags(caption)
                hashtags = re.findall(r'#(\w+)', caption) if caption else []
                
                all_hashtags.extend(hashtags)
                post["hashtags"] = hashtags
                
                #get insights for this post
                if post_id:
                    insights = get_media_insights(post_id, page_token)
                    
                    #add insights to the post
                    for key, value in insights.items():
                        post[key] = value
                    
                    post_likes = insights.get("likes", post.get("like_count", 0))
                    post_comments = insights.get("comments", post.get("comments_count", 0))
                    post_saved = insights.get("saved", 0)
                    post_reach = insights.get("reach", 0)
                    post_views = insights.get("views", 0) if media_type in ["VIDEO", "REELS"] else 0
                    
                    # clculate engagement
                    post_engagement = post_likes + post_comments + post_saved
                    post["engagement"] = post_engagement
                    
                    #track totals
                    total_likes += post_likes
                    total_comments += post_comments
                    total_saved += post_saved
                    total_reach += post_reach
                    total_views += post_views
                    total_engagement += post_engagement
                    
                    #track top post
                    if post_engagement > top_engagement:
                        top_post = post
                        top_engagement = post_engagement
            
            #add engagement metrics to day data
            day_data["Likes"] = total_likes
            day_data["Comments"] = total_comments
            day_data["Saved"] = total_saved
            day_data["Views"] = total_views
            day_data["Media Reach"] = total_reach
            day_data["Total Engagement"] = total_engagement
            
            #calculate engagement rate - I should probably make this a helper function
            if day_data["Followers"] > 0 and len(day_media) > 0:
                eng_rate = (total_engagement / day_data["Followers"]) * 100
                day_data["Engagement Rate"] = round(eng_rate, 2)
            else:
                day_data["Engagement Rate"] = 0
            
            # ad top post information
            if top_post:
                day_data["Top Post URL"] = top_post.get("permalink", "")
                
                #truncate long captions
                caption = top_post.get("caption", "")
                if caption and len(caption) > 100:
                    day_data["Top Post Message"] = caption[:100] + "..."
                else:
                    day_data["Top Post Message"] = caption
                    
                day_data["Top Post Engagement"] = top_post.get("engagement", 0)
                day_data["Top Post Media Type"] = top_post.get("media_type", "")
                
                # Add hashtags used in top post
                day_data["Top Post Hashtags"] = top_post.get("hashtags", [])
            else:
                day_data["Top Post URL"] = ""
                day_data["Top Post Message"] = ""
                day_data["Top Post Engagement"] = 0
                day_data["Top Post Media Type"] = ""
                day_data["Top Post Hashtags"] = []
            
            #simplified post details - focus on what's needed for post verification
            # his bloats the JSON file but it's useful for debugging
            day_data["Posts"] = [
                {
                    "id": post.get("id", ""),
                    "media_type": post.get("media_type", ""),
                    "permalink": post.get("permalink", "")
                    # Removed timestamp and engagement to reduce data size
                }
                for post in day_media
            ]
            
            account_daily_data.append(day_data)
        
    total_time = time.time() - start_time
    logger.info(f"Processed {username} in {total_time:.1f} seconds. {len(account_daily_data)} days of data.")
        
    return username, account_daily_data

def main():
    # Get access token
    access_token = read_access_token_from_file()
    if not access_token:
        logger.info("No access token found or it expired. Get a new one!")
        return
    
    #get Instagram accounts linked to Facebook pages
    instagram_accounts = get_user_instagram_accounts(access_token)
    if not instagram_accounts:
        logger.error("No Instagram accounts found or can't access them. Check permissions?")
        return
    
    #display accounts
    print("\nAvailable Instagram Accounts:")
    for i, account in enumerate(instagram_accounts, 1):
        print(f"{i}. {account.get('username', 'Unknown')} (Followers: {account.get('followers_count', 'N/A')})")
    
    #select all accounts automatically
    # TODO:Add interactive selection here again for future testing
    selected_accounts = instagram_accounts
    logger.info(f"Auto-selected all {len(selected_accounts)} accounts for processing.")
    
    #get number of weeks to process
    weeks = int(os.getenv("IG_WEEKS_BACK", "13"))  # Default to 13 weeks / ~3 months
    
    #generate weekly date ranges
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=weeks)
    
    weekly_ranges = []
    current_date = start_date
    
    #this could be a separate function but it's simple enough inline
    while current_date < end_date:
        week_end = min(current_date + timedelta(days=6), end_date)
        weekly_ranges.append({
            "name": f"{current_date.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}",
            "start": current_date.strftime('%Y-%m-%d') + "T00:00:00",
            "end": week_end.strftime('%Y-%m-%d') + "T23:59:59"
        })
        current_date += timedelta(days=7)
    
    logger.info(f"Processing data for {weeks} weeks: {weekly_ranges[0]['start']} to {weekly_ranges[-1]['end']}")
    
    #create output directory 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    #process accounts
    all_summary_data = []
    
    #maybe add a progress bar here? Nah, logging is fine
    for i, account in enumerate(selected_accounts):
        try:
            logger.info(f"Processing account {i+1}/{len(selected_accounts)}")
            username, account_daily_data = process_account_weekly(account, weekly_ranges)
            
            # Save individual account report
            safe_username = "".join(c if c.isalnum() or c == '_' else '_' for c in username)
            filename = f"{safe_username}_daily_insights.json"
            
            # Make the JSON pretty for easier debugging
            with open(os.path.join(OUTPUT_DIR, filename), "w", encoding='utf-8') as f:
                json.dump(account_daily_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Saved daily insights for {username}")
            
            #create summary data for Excel (excluding detailed post list)
            summary_data = []
            for day_data in account_daily_data:
                #copy everything except the big "Posts" list
                summary_row = {k: v for k, v in day_data.items() if k != "Posts"}
                summary_data.append(summary_row)
            
            all_summary_data.extend(summary_data)
            logger.info(f"Finished processing account: {username}")
        except Exception as exc:
            logger.error(f"Account processing failed for {account.get('username', 'Unknown')}: {exc}")
    
    #create a summary Excel report with all accounts and days
    if all_summary_data:
        # Convert to DataFrame for Excel export
        df = pd.DataFrame(all_summary_data)
        
        #make sure client name is first - makes the Excel easier to read
        if 'Client_Name' in df.columns:
            column_order = ['Client_Name'] + [col for col in df.columns if col != 'Client_Name']
            df = df[column_order]
            
        excel_path = os.path.join(OUTPUT_DIR, "instagram_all_accounts_daily_summary.xlsx")
        df.to_excel(excel_path, index=False)
        logger.info(f"\nSummary Excel report saved to: {excel_path}")
    
    logger.info("\nInstagram data extraction complete!")
    
if __name__ == "__main__":
    # Uncomment to profile execution time
    #import cProfile
    #cProfile.run('main()')
    main()