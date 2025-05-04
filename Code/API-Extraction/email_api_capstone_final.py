"""
Capstone Project
Email insights data extraction using Mailchimp API
Author: Arthur Osakwe
Version: 4.0
Last Updated: April 12, 2025 - fixed the pagination issue
"""
#Import Libraries
import argparse
import calendar
import csv
import datetime
import logging
from pathlib import Path

import requests
from dateutil.relativedelta import relativedelta

#Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


#configuration

API_KEY_FILE = Path("mailchimp_key.txt")
DC = "us10"  # e.g., "us1", "us2", etc.
BASE_URL = f"https://{DC}.api.mailchimp.com/3.0/"
CSV_SAVE_DIR = Path.home() / "Documents" / "Capstone" / "1_ETL" / "2. Original Data" / "Combined"


def load_api_key(filepath: Path) -> str | None:
    ##Load API key from a file, return None if not found
    try:
        return filepath.read_text().strip()
    except FileNotFoundError:
        logger.error("API key file not found: %s", filepath)
        return None


MAILCHIMP_API_KEY = load_api_key(API_KEY_FILE)
AUTH = ("anystring", MAILCHIMP_API_KEY) if MAILCHIMP_API_KEY else None



#date range helpers
def get_month_date_ranges(start_date: datetime.date, num_months: int = 24) -> list[tuple[str, str, str, int]]:
    #Generate a list of month date ranges.
   # returns list of tuples:start_iso, end_iso, month_name, year
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()

    current = start_date.replace(day=1)
    ranges = []

    for _ in range(num_months):
        year, month = current.year, current.month
        _, last_day = calendar.monthrange(year, month)
        start_iso = current.isoformat()
        end_iso = current.replace(day=last_day).isoformat()
        ranges.append((start_iso, end_iso, calendar.month_name[month], year))
        current += relativedelta(months=1)

    return ranges



#mailchimp api functions
def list_campaigns(since: str, before: str, count: int = 1000) -> list[dict]:
    #retrieve campaigns sent between since YYYY-MM-DD and before YYYY-MM-DD
    if AUTH is None:
        logger.error("No Mailchimp auth configured.")
        return []

    params = {
        "since_send_time": f"{since}T00:00:00+00:00",
        "before_send_time": f"{before}T23:59:59+00:00",
        "count": count,
    }

    resp = requests.get(BASE_URL + "campaigns", auth=AUTH, params=params)
    if resp.ok:
        return resp.json().get("campaigns", [])
    logger.error("Error fetching campaigns: %s %s", resp.status_code, resp.text)
    return []


def get_campaign_report(campaign_id: str) -> dict | None:
    #fetch the report JSON for a given campaign ID."""
    if AUTH is None:
        return None

    resp = requests.get(BASE_URL + f"reports/{campaign_id}", auth=AUTH)
    if resp.ok:
        return resp.json()
    logger.error("Error fetching report for ID %s: %s %s", campaign_id, resp.status_code, resp.text)
    return None


def extract_campaign_metrics(report: dict | None) -> dict:
    #pull key metrics from the report JSON
    if not report:
        return {}

    opens = report.get("opens", {})
    clicks = report.get("clicks", {})

    return {
        "Emails Sent": report.get("emails_sent", "N/A"),
        "Open Rate": opens.get("open_rate", "N/A"),
        "Click Rate": clicks.get("click_rate", "N/A"),
        "Total Clicks": clicks.get("clicks_total", "N/A"),
        "Unique Clicks": clicks.get("unique_clicks", "N/A"),
        "Unique Subscriber Clicks": clicks.get("unique_subscriber_clicks", "N/A"),
    }


#write to csv
def write_csv_report(path: Path, rows: list[dict]) -> None:
    #write list of dicts to CSV at the given path
    if not rows:
        logger.warning("No data to write to CSV.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV report generated: %s", path)



#Execute main function
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Mailchimp campaign metrics for a specified time period."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date YYYY-MM-DD (defaults to X months ago)",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=24,
        help="Number of months to analyze (default: 24)",
    )
    args = parser.parse_args()

    #determine start date
    if args.start_date:
        try:
            start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD.")
            return
    else:
        start_date = datetime.date.today() - relativedelta(months=args.months)

    #build date ranges & collect metrics
    all_rows = []
    for start_iso, end_iso, month_name, year in get_month_date_ranges(start_date, args.months):
        logger.info("Processing %s %d...", month_name, year)
        campaigns = list_campaigns(start_iso, end_iso)
        if not campaigns:
            logger.info("  No campaigns found.")
            continue

        for camp in campaigns:
            cid = camp.get("id")
            name = camp.get("settings", {}).get("title", "N/A")
            report = get_campaign_report(cid)
            metrics = extract_campaign_metrics(report)

            all_rows.append({
                "Campaign Name": name,
                "Campaign ID": cid,
                "Period": "Monthly",
                "Month": month_name,
                "Year": year,
                "Start Date": start_iso,
                "End Date": end_iso,
                **metrics,
            })

    #write CSV
    end_date = start_date + relativedelta(months=args.months) - relativedelta(days=1)
    csv_path = CSV_SAVE_DIR / f"mailchimp_metrics.csv"
    write_csv_report(csv_path, all_rows)


if __name__ == "__main__":
    main()
