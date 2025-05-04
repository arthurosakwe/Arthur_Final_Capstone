"""
Capstone Project
Facebook Pages insights data extraction using Meta Graph API
Author: Arthur Osakwe

Version: 3.0
Last Updated: April 25, 2025
"""

#import libraries
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import logging
from typing import List, Dict, Any, Optional, Union

#set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("facebook_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class FacebookAPIExtractor:    
    API_VERSION = "v22.0"
    API_DELAY = 0.5  # Delay between API calls to avoid rate limits
    PLATFORM_NAME = "Facebook"
    
    def __init__(self, access_token: str, output_dir: str = "output"):
        self.access_token = access_token
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def get_user_pages(self) -> List[Dict[str, Any]]:
        url = f"https://graph.facebook.com/{self.API_VERSION}/me/accounts"
        params = {
            "access_token": self.access_token,
            "fields": "name,id,access_token,category"
        }
        
        try:
            response = requests.get(url=url, params=params)
            response.raise_for_status()
            pages = response.json().get("data", [])
            logger.info(f"Successfully found {len(pages)} pages")
            return pages
        except Exception as e:
            logger.error(f"Error fetching pages: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.debug(f"Response details: {e.response.text}")
            return []
    
    def get_page_insights(self, page_id: str, page_token: str, metrics: List[str], 
                         since: str, until: str, period: str = "day") -> Dict[str, Any]:
        url = f"https://graph.facebook.com/{self.API_VERSION}/{page_id}/insights"
        params = {
            "access_token": page_token,
            "metric": ','.join(metrics),
            "period": period,
            "since": since,
            "until": until
        }
        
        try:
            response = requests.get(url=url, params=params)
            response.raise_for_status()
            data = response.json().get("data", [])
            
            #create a dictionary to store results by metric name
            results = {}
            for metric_data in data:
                metric_name = metric_data.get("name")
                if metric_name:
                    results[metric_name] = metric_data
            
            return results
        except Exception as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 400:
                # This is likely a metric not available for this page
                logger.warning(f"Some metrics not available for page {page_id}")
                return {}
            else:
                logger.error(f"Error fetching metrics for page {page_id}: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.debug(f"Response details: {e.response.text}")
                return {}
    
    def get_page_feed(self, page_id: str, page_token: str, since_date: str, 
                     until_date: str, limit: int = 100) -> List[Dict[str, Any]]:
        url = f"https://graph.facebook.com/{self.API_VERSION}/{page_id}/feed"
        params = {
            "access_token": page_token,
            "fields": "id,message,created_time,likes.summary(true),comments.summary(true),shares,attachments{type,media_type,title,url},permalink_url",
            "since": since_date,
            "until": until_date,
            "limit": limit
        }
        
        all_posts = []
        
        try:
            logger.info(f"Requesting feed for page {page_id} from {since_date} to {until_date}")
            response = requests.get(url=url, params=params)
            response.raise_for_status()
            
            data = response.json()
            posts = data.get("data", [])
            all_posts.extend(posts)
            
            #handle pagination to get all posts
            while "paging" in data and "next" in data["paging"]:
                logger.info(f"Getting next page of posts (currently have {len(all_posts)})")
                next_url = data["paging"]["next"]
                response = requests.get(url=next_url)
                response.raise_for_status()
                
                data = response.json()
                posts = data.get("data", [])
                all_posts.extend(posts)
                
                #add delay to avoid rate limits
                time.sleep(self.API_DELAY)
                
                #limit to 1000 posts to prevent excessive API calls
                if len(all_posts) >= 1000:
                    logger.info("Reached maximum post limit (1000), stopping pagination")
                    break
            
            logger.info(f"Retrieved {len(all_posts)} posts from feed for page {page_id}")
            
            #process posts to add calculated fields
            for post in all_posts:
                likes = post.get("likes", {}).get("summary", {}).get("total_count", 0)
                comments = post.get("comments", {}).get("summary", {}).get("total_count", 0)
                shares = 0
                if "shares" in post:
                    shares = post.get("shares", {}).get("count", 0)
                
                post["engagement"] = likes + comments + shares
                post["likes_count"] = likes
                post["comments_count"] = comments
                post["shares_count"] = shares
                
                #extract media_type from attachments
                post["media_type"] = "text"  # Default if no attachments
                if "attachments" in post and "data" in post["attachments"]:
                    for attachment in post["attachments"]["data"]:
                        if "type" in attachment:
                            post["media_type"] = attachment.get("type", "").lower()
                        if "media_type" in attachment:
                            post["media_type"] = attachment.get("media_type", "").lower()
                
                #parse the created_time to get just the date part
                if "created_time" in post:
                    try:
                        created_datetime = datetime.strptime(post["created_time"], "%Y-%m-%dT%H:%M:%S%z")
                        post["created_date"] = created_datetime.strftime("%Y-%m-%d")
                    except:
                        post["created_date"] = post["created_time"].split("T")[0]
                        
                #add URL if not present using permalink_url or construct from ID
                if "permalink_url" not in post:
                    post_id = post.get("id", "").split("_")[-1] if "_" in post.get("id", "") else ""
                    post["permalink_url"] = f"https://www.facebook.com/{page_id}/posts/{post_id}" if post_id else ""
                
            return all_posts
        except Exception as e:
            logger.error(f"Error fetching feed for page {page_id}: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.debug(f"Response details: {e.response.text}")
            return []
     #get comments for specific post       
    def get_post_comments(self, post_id: str, page_token: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get comments for a specific post, ordered by engagement"""
        url = f"https://graph.facebook.com/{self.API_VERSION}/{post_id}/comments"
        params = {
            "access_token": page_token,
            "fields": "id,message,created_time,like_count",
            "limit": limit,
            "order": "reverse_chronological"  # Get newest first as a proxy for relevance
        }
        
        try:
            response = requests.get(url=url, params=params)
            response.raise_for_status()
            comments = response.json().get("data", [])
            
            # Sort by like_count to get most engaging comment
            comments.sort(key=lambda x: x.get("like_count", 0), reverse=True)
            return comments
        except Exception as e:
            logger.error(f"Error fetching comments for post {post_id}: {str(e)}")
            return []
    
    #extrsact, process data for single page
    def fetch_facebook_data(self, page: Dict[str, Any], days_back: int) -> pd.DataFrame:
        page_id = page["id"]
        page_name = page["name"]
        page_token = page["access_token"]
        page_category = page.get("category", "Unknown Category")
        
        logger.info(f"Processing page: {page_name} (ID: {page_id}, Category: {page_category})")
        
        #calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        since_date = start_date.strftime("%Y-%m-%d") + "T00:00:00"
        until_date = end_date.strftime("%Y-%m-%d") + "T23:59:59"
        
        #metrics to extract based on correct API names
        metrics = {
            "page_impressions": "total_impressions",
            "page_impressions_unique": "total_reach",
            "page_post_engagements": "total_engagement",
            "page_consumptions": "total_clicks",
            "page_fan_adds_unique": "new_page_likes"
        }
        
        #get insights data
        insights = self.get_page_insights(
            page_id, 
            page_token, 
            list(metrics.keys()), 
            since_date, 
            until_date, 
            "day"
        )
        
        #get all posts for the page within the date range
        posts = self.get_page_feed(page_id, page_token, since_date, until_date)
        
        #create a dictionary to store data by date
        daily_data = {}
        
        #process insights data
        for api_metric, display_name in metrics.items():
            if api_metric in insights and "values" in insights[api_metric]:
                for daily_value in insights[api_metric]["values"]:
                    #extract the date from the 'end_time' field
                    end_time = daily_value.get("end_time")
                    if end_time:
                        day_date = end_time.split("T")[0]
                        
                        #create or update the entry for this day
                        if day_date not in daily_data:
                            daily_data[day_date] = {
                                "client_name": page_name,
                                "platform_name": self.PLATFORM_NAME,
                                "date": day_date,
                                "datetime": f"{day_date}T00:00:00",
                                "posts_metadata": []
                            }
                        
                        #add the metric value
                        daily_data[day_date][display_name] = daily_value.get("value", 0)
        
        #group posts by day and process
        for post in posts:
            day = post.get("created_date")
            if not day:
                continue
                
            #ensure day entry exists
            if day not in daily_data:
                daily_data[day] = {
                    "client_name": page_name,
                    "platform_name": self.PLATFORM_NAME,
                    "date": day,
                    "datetime": f"{day}T00:00:00",
                    "posts_metadata": [],
                    "total_reach": 0,
                    "total_engagement": 0,
                    "total_impressions": 0,
                    "total_clicks": 0
                }
            
            #get top comment if available
            top_comment_text = None
            comments = self.get_post_comments(post["id"], page_token, limit=3)
            if comments:
                top_comment_text = comments[0].get("message", "")
            
            #create post metadata
            post_metadata = {
                "post_id": post.get("id", ""),
                "post_text": post.get("message", ""),
                "post_time": post.get("created_time", ""),
                "url": post.get("permalink_url", ""),
                "media_type": post.get("media_type", "text"),
                "likes": post.get("likes_count", 0),
                "comments": post.get("comments_count", 0),
                "shares": post.get("shares_count", 0),
                "video_views": None,  # Not available in basic feed data
                "top_comment": top_comment_text
            }
            
            #add to posts metadata for the day
            daily_data[day]["posts_metadata"].append(post_metadata)
            
            #update datetime to earliest post time if needed
            try:
                post_datetime = datetime.strptime(post.get("created_time", ""), "%Y-%m-%dT%H:%M:%S%z")
                day_datetime = datetime.strptime(daily_data[day]["datetime"], "%Y-%m-%dT%H:%M:%S")
                if post_datetime.time() < day_datetime.time():
                    daily_data[day]["datetime"] = post.get("created_time", "")
            except:
                pass
            
            #add to day's engagement totals if not already accounted for in insights
            if "total_engagement" not in daily_data[day] or daily_data[day]["total_engagement"] == 0:
                daily_data[day]["total_engagement"] = daily_data[day].get("total_engagement", 0) + post.get("engagement", 0)
        
        #convert posts_metadata to JSON strings
        for day, data in daily_data.items():
            data["posts_metadata"] = json.dumps(data["posts_metadata"])
            
            # Ensure all required metrics are present with defaults
            for field in ["total_reach", "total_engagement", "total_impressions", "total_clicks"]:
                if field not in data:
                    data[field] = 0
        
        #convert to DataFrame
        df = pd.DataFrame(list(daily_data.values()))
        
        return df
    
    #Fetch and consolidate metrics for multiple Facebook pages
    def consolidate_metrics(self, days_back: int, clients: List[Dict[str, Any]]) -> pd.DataFrame:
        all_data = []
        
        for client in clients:
            try:
                client_data = self.fetch_facebook_data(client, days_back)
                all_data.append(client_data)
                logger.info(f"Successfully processed data for {client['name']}")
            except Exception as e:
                logger.error(f"Error processing data for {client['name']}: {e}")
        
        #combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        else:
            #return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                "client_name", "platform_name", "date", "datetime",
                "total_reach", "total_engagement", "total_impressions", 
                "total_clicks", "posts_metadata"
            ])
    #save the data to csv file
    def save_output(self, data: pd.DataFrame, filename: str) -> str:
        filepath = os.path.join(self.output_dir, filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return filepath

def read_access_token_from_file(file_path="fb_access_token.txt"):
    try:
        with open(file_path, 'r') as file:
            # Read the first line and strip any whitespace
            access_token = file.readline().strip()
            logger.info("Successfully read access token from file")
            return access_token
    except Exception as e:
        logger.error(f"Error reading access token from file: {str(e)}")
        return None


def main():
    #get access token from file
    access_token = read_access_token_from_file()
    if not access_token:
        logger.error("Access token not found in file. Exiting.")
        return

    #output directory
    OUTPUT_DIR = r"C:\Users\Arthu\Documents\Capstone\1_ETL\2. Original Data\Combined"
    extractor = FacebookAPIExtractor(access_token, output_dir=OUTPUT_DIR)

    #get all user-managed pages
    pages = extractor.get_user_pages()
    if not pages:
        logger.error("No pages found or error accessing pages.")
        return
    #this will be better for prod
    selected_pages = pages
    logger.info(f"Automatically selected all {len(selected_pages)} pages for processing.")

    # Use DAYS_BACK from env var, fallback to 90
    try:
        days_back = int(os.getenv("DAYS_BACK", "90"))
    except ValueError:
        days_back = 90
        logger.warning("Invalid DAYS_BACK value in environment. Defaulting to 90.")

    logger.info(f"Fetching data for the past {days_back} days.")

    # Run extraction and consolidation
    result_df = extractor.consolidate_metrics(days_back, selected_pages)

    # Save one combined CSV output
    output_file = extractor.save_output(result_df, "facebook_all_clients_daily_report.csv")

    print("\n Facebook data extraction complete!")
    print(f"File saved to: {output_file}")
    print(f"Processed {len(selected_pages)} pages over {days_back} days")
    print(f"Total records: {len(result_df)}")



if __name__ == "__main__":
    main()