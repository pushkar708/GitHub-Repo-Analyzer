import streamlit as st
import requests
import json
import pandas as pd
import os
import re
from collections import defaultdict
import operator
import plotly.express as px
from datetime import datetime, date
from dotenv import load_dotenv
import time # Added for exponential backoff

# Note: In a real-world Streamlit deployment, you should set GITHUB_TOKEN
# and GEMINI_API_KEY in Streamlit Secrets (secrets.toml). We use os.environ here 
# for compatibility with the original script's structure.
load_dotenv()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
# IMPORTANT: In the canvas environment, the API key is automatically provided if left empty.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") 

# Streamlit Page Setup
st.set_page_config(
    page_title="GitHub Profile & Repo Analyzer",
    page_icon="⭐",
    layout="wide"
)

# --- NEW: AI Analysis Function (Python equivalent of ai_analysis_utility.js) ---

def get_ai_analysis_markdown(user_report_dict):
    """
    Calls the Gemini API to analyze the GitHub user report and returns a Markdown string.
    Implements basic exponential backoff for robustness.
    
    FIX: Converts DataFrames to JSON-serializable lists of records before serialization.
    """
    
    # Create a shallow copy of the report dictionary to avoid modifying the original
    ai_safe_report = user_report_dict.copy()
    
    # DataFrames are not JSON serializable. We convert them to a list of dictionaries ('records').
    
    # 1. Commit Activity Data
    commit_df = ai_safe_report.pop('repo_commit_activity_df', pd.DataFrame())
    if not commit_df.empty:
        # We rename the key to remove the '_df' suffix for cleaner AI interpretation
        ai_safe_report['repo_commit_activity'] = commit_df.to_dict('records')
    
    # 2. Code Frequency Data
    code_df = ai_safe_report.pop('repo_code_frequency_df', pd.DataFrame())
    if not code_df.empty:
        # We rename the key to remove the '_df' suffix for cleaner AI interpretation
        ai_safe_report['repo_code_frequency'] = code_df.to_dict('records')
        
    # Now, ai_safe_report contains only JSON-serializable types.
    reportDataJson = json.dumps(ai_safe_report, indent=4)
    
    # 1. Configuration
    apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
    
    # 2. System and User Prompts
    systemPrompt = """You are a world-class GitHub Profile Analyst. 
        Your task is to interpret the provided JSON data about a user's GitHub profile 
        and provide a concise, professional analysis. The analysis MUST contain 
        the following three sections:
        1. Executive Summary (3 sentences, covering their specialty and activity level).
        2. Key Strengths (3 bullet points, highlighting the best aspects of the profile).
        3. Actionable Recommendations (3 concrete, next-step suggestions for improvement).
        Present the response strictly in Markdown format, using headings. Use emojis for emphasis."""
    
    userQuery = f"""Analyze the following user data and provide the requested summary, strengths, and recommendations. 
        The data is provided below: \n\n{reportDataJson}"""
    
    payload = {
        "contents": [{"parts": [{"text": userQuery}]}],
        "systemInstruction": {"parts": [{"text": systemPrompt}]},
    }
    
    # 3. API Call with Exponential Backoff (simplified to 3 retries)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            headers = {'Content-Type': 'application/json'}
            # Note: We append the API key to the URL as required by the environment context.
            response = requests.post(f"{apiUrl}?key={GEMINI_API_KEY}", headers=headers, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0 and 'parts' in result['candidates'][0]['content']:
                generated_text = result['candidates'][0]['content']['parts'][0]['text']
                return generated_text
            
            return "AI Analysis: Could not parse response from the model."

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                # Calculate sleep time: 2^attempt seconds
                sleep_time = 2 ** attempt
                print(f"API Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                return f"AI Analysis Failed: Could not connect or retrieve data after {max_retries} attempts. Error: {e}"

    return "AI Analysis Failed: Unknown error."

# --- 2. CORE UTILITY FUNCTIONS (Adapted for Streamlit) ---

@st.cache_data(ttl=3600)
def _make_github_request(url, params=None):
    """
    Makes a rate-limited request to the GitHub API.
    Uses the GITHUB_TOKEN for authentication.
    """
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.star+json, application/json",
    }
    
    if not GITHUB_TOKEN:
        st.warning("⚠️ No `GITHUB_TOKEN` found. Requests are limited to 60 per hour. Please set the token in your environment/secrets.")
        headers.pop("Authorization")

    response = None
    try:
        response = requests.get(url, headers=headers, params=params)
        
        limit = response.headers.get('X-RateLimit-Limit')
        remaining = response.headers.get('X-RateLimit-Remaining')
        if remaining and int(remaining) < 10:
             st.warning(f"GitHub Rate Limit Low: {remaining}/{limit} remaining requests.")
             
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        if response is not None and response.status_code == 404:
             st.error(f"Error 404: The resource at `{url}` was not found. Please check the URL/Username.")
        elif response is not None and response.status_code == 403:
             st.error(f"Error 403 (Forbidden): You might be rate-limited or the token is invalid/lacks scope. Remaining: {remaining}")
        else:
             st.error(f"Error fetching data from GitHub: {e}")
        return None

def parse_github_url(url):
    """Parses a GitHub URL or username into owner and repository name."""
    pattern = r'(?:https?://github\.com/|git@github\.com:)?([\w\-\.]+)/?([\w\-\.]+)?'
    match = re.search(pattern, url, re.IGNORECASE)
    if match:
        owner = match.group(1)
        repo = match.group(2)
        if repo and repo.endswith('.git'):
            repo = repo[:-4]
        return owner, repo or None
    return None, None

@st.cache_data(ttl=3600)
def fetch_repo_details(owner, repo):
    """Fetches all stargazers for a repository."""
    star_data = []
    page = 1
    per_page = 100
    base_url = f"https://api.github.com/repos/{owner}/{repo}/stargazers"
    
    with st.spinner(f"Fetching star history for {owner}/{repo}..."):
        while True:
            params = {"page": page, "per_page": per_page, "media": "application/vnd.github.v3.star+json"}
            response = _make_github_request(base_url, params=params)
            
            if response is None:
                return None

            current_page_data = response.json()
            star_data.extend(current_page_data)

            link_header = response.headers.get('Link')
            if 'rel="next"' in str(link_header):
                page += 1
            else:
                break
    return star_data

@st.cache_data(ttl=3600)
def fetch_repo_languages(owner, repo):
    """Fetches detailed language breakdown (bytes) for a single repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/languages"
    response = _make_github_request(url)
    return response.json() if response else None

@st.cache_data(ttl=3600)
def fetch_repo_stats(owner, repo):
    """
    Fetches heavy repository statistics (commit activity, code frequency, participation).
    """
    base_url = f"https://api.github.com/repos/{owner}/{repo}/stats/"
    stats = {}
    
    with st.spinner(f"Fetching deep stats for {owner}/{repo}..."):
        # 1. Weekly Commit Activity (52 weeks)
        url_commit = f"{base_url}commit_activity"
        response = _make_github_request(url_commit)
        stats['commit_activity'] = response.json() if response else None

        # 2. Code Frequency (Additions/Deletions)
        url_code = f"{base_url}code_frequency"
        response = _make_github_request(url_code)
        stats['code_frequency'] = response.json() if response else None
        
        # 3. Contributor Participation (Owner vs. others)
        url_part = f"{base_url}participation"
        response = _make_github_request(url_part)
        stats['participation'] = response.json() if response else None
        
    return stats

@st.cache_data(ttl=3600)
def fetch_user_orgs(owner):
    """Fetches public organization memberships for a user."""
    url = f"https://api.github.com/users/{owner}/orgs"
    response = _make_github_request(url)
    return response.json() if response else None


@st.cache_data(ttl=3600)
def fetch_user_profile(owner):
    """Fetches public user profile data."""
    url = f"https://api.github.com/users/{owner}"
    response = _make_github_request(url)
    return response.json() if response else None

@st.cache_data(ttl=3600)
def fetch_user_repos(owner):
    """Fetches all public repositories owned by the user."""
    repos_data = []
    page = 1
    per_page = 100
    url = f"https://api.github.com/users/{owner}/repos"

    while True:
        params = {"page": page, "per_page": per_page, "type": "owner"}
        response = _make_github_request(url, params=params)
        if response is None:
            break

        current_page_data = response.json()
        if not current_page_data:
            break

        repos_data.extend(current_page_data)

        link_header = response.headers.get('Link')
        if 'rel="next"' in str(link_header):
            page += 1
        else:
            break
    return repos_data

@st.cache_data(ttl=3600)
def fetch_user_events(owner):
    """Fetches recent public activity events (up to 300)."""
    # Note: GitHub only provides the last 90 days of public events.
    url = f"https://api.github.com/users/{owner}/events/public"
    response = _make_github_request(url, params={"per_page": 100})
    return response.json() if response else None


def process_star_data(raw_star_data):
    """Converts star data into a cumulative DataFrame for plotting."""
    if not raw_star_data:
        return pd.DataFrame()

    stars = [{'starred_at': entry.get('starred_at')} for entry in raw_star_data if entry.get('starred_at')]
    df = pd.DataFrame(stars)
    if df.empty:
        return df

    df['starred_at'] = pd.to_datetime(df['starred_at'], utc=True)
    df = df.sort_values('starred_at').reset_index(drop=True)
    df['cumulative_stars'] = df.index + 1
    
    # Resample to daily max and forward fill for the line chart (used in Streamlit display)
    df_plot = df.set_index('starred_at')['cumulative_stars'].resample('D').max().ffill().reset_index()
    
    # Return the raw list of star events for the PDF report table and CSV
    df_raw = df[['starred_at', 'cumulative_stars']].copy()
    df_raw['starred_at'] = df_raw['starred_at'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    
    return df_plot, df_raw

# --- NEW HELPER FUNCTION FOR USER REPORT CSV EXPORT ---
def consolidate_user_metrics_to_csv(report):
    """
    Consolidates core profile metrics and top repository data into a single CSV string.
    """
    # 1. Profile Metrics (Vertical Table)
    metric_data = {
        'Metric': [
            'User ID', 'Name', 'Hirable Status', 'Location', 'Member Since', 'Followers',
            'Public Repositories Count', 'Public Gists Count', 'Total Stars Owned', 
            'Total Forks Owned', 'Total Open Issues', 'Total Watchers', 
            'Top Language', 'Activity Score', 'Active Repos Count'
        ],
        'Value': [
            report['user_id'], report['name'], 'Yes' if report['hireable'] else 'No/Unknown',
            report['location'], report['member_since'][:10], report['followers'],
            report['public_repos_count'], report['public_gists_count'], report['total_stars_owned'], 
            report['total_forks_owned'], report['total_open_issues'], report['total_watchers'], 
            report['top_language'], report['recent_activity_score'], report['repos_with_recent_activity']
        ]
    }
    metrics_df = pd.DataFrame(metric_data)

    # 2. Top Starred Repositories (Horizontal Table)
    star_df = pd.DataFrame(list(report['repo_star_counts'].items()), columns=['Repository', 'Stars'])
    top_repos_df = star_df.sort_values('Stars', ascending=False).head(10)
    
    # Combine into a single string with separators
    csv_output = "--- CORE PROFILE METRICS ---\n"
    csv_output += metrics_df.to_csv(index=False)
    csv_output += "\n--- TOP 10 STARRED REPOSITORIES ---\n"
    csv_output += top_repos_df.to_csv(index=False)
    
    return csv_output.encode('utf-8')


def analyze_user_data(owner, profile_data, repos_data, events_data):
    """
    Analyzes profile, repo, and event data to create a summary report.
    """
    
    total_open_issues = 0
    total_watchers = 0
    repo_languages_count = defaultdict(int)
    repo_star_counts = defaultdict(int) # This holds the stars per repo name
    most_popular_repo = {'name': 'N/A', 'stars': 0, 'forks': 0}
    active_repos = 0
    activity_threshold = (pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=1)).isoformat()
    total_stars_gained = 0
    total_forks_gained = 0


    for repo in repos_data:
        stars = repo.get('stargazers_count', 0)
        forks = repo.get('forks_count', 0)
        total_stars_gained += stars
        total_forks_gained += forks
        
        # Aggregate issues (Includes open PRs) and watchers
        total_open_issues += repo.get('open_issues_count', 0)
        total_watchers += repo.get('watchers_count', 0)

        if stars > 0:
            repo_star_counts[repo['name']] = stars

        if stars > most_popular_repo['stars']:
            most_popular_repo.update({'name': repo['name'], 'stars': stars, 'forks': forks})

        if repo.get('language'):
            repo_languages_count[repo['language']] += 1

        if repo.get('pushed_at', '1970-01-01T00:00:00Z') > activity_threshold:
            active_repos += 1

    # --- Top Repo Deep Dive Data Fetching ---
    detailed_languages = None
    repo_stats = None
    top_repo_data = {}
    commit_df = pd.DataFrame()
    code_df = pd.DataFrame()
    top_repo_name = most_popular_repo['name']
    
    if top_repo_name != 'N/A':
        # Find the full repo object for extra fields like topics, has_pages, has_wiki
        full_top_repo = next((r for r in repos_data if r['name'] == top_repo_name), None)
        if full_top_repo:
            top_repo_data['topics'] = full_top_repo.get('topics', [])
            top_repo_data['has_pages'] = full_top_repo.get('has_pages', False)
            top_repo_data['has_wiki'] = full_top_repo.get('has_wiki', False)
            # NEW: Issue Lifespan (Basic metric: days since creation of oldest open issue)
            oldest_open_issue_days = "N/A"
            if full_top_repo.get('open_issues_count', 0) > 0:
                # We use the repo's creation date as a proxy for the absolute maximum lifespan of *any* issue/PR.
                created_at = pd.to_datetime(full_top_repo.get('created_at'))
                days_since_creation = (pd.Timestamp.now(tz='UTC') - created_at).days
                oldest_open_issue_days = days_since_creation
            top_repo_data['oldest_open_issue_lifespan_days'] = oldest_open_issue_days


        # 1. Detailed Language Breakdown (bytes)
        detailed_languages = fetch_repo_languages(owner, top_repo_name)
        
        # 2. Heavy Repository Stats (Commit Activity, Code Frequency, Participation)
        repo_stats = fetch_repo_stats(owner, top_repo_name)
        
        # Process Commit Activity into DataFrame
        if repo_stats and repo_stats['commit_activity']:
            dates = [datetime.fromtimestamp(w['week'], tz=pd.Timestamp.now(tz='UTC').tz).strftime('%Y-%m-%d') for w in repo_stats['commit_activity']]
            commits = [w['total'] for w in repo_stats['commit_activity']]
            commit_df = pd.DataFrame({'Week Start Date': dates, 'Total Commits': commits})
            
        # Process Code Frequency into DataFrame
        if repo_stats and repo_stats['code_frequency']:
            # Format: [[timestamp, additions, deletions], ...]
            data = []
            for week in repo_stats['code_frequency']:
                week_start = datetime.fromtimestamp(week[0], tz=pd.Timestamp.now(tz='UTC').tz).strftime('%Y-%m-%d')
                data.append({
                    'Week Start Date': week_start, 
                    'Additions': week[1], 
                    'Deletions': abs(week[2]),
                    'Net Change': week[1] + week[2]
                })
            code_df = pd.DataFrame(data)

    # User Organizations
    user_orgs = fetch_user_orgs(owner)
    
    # --- Event Analysis (Recent Activity) ---
    recent_activity_score = 0
    event_counts = defaultdict(int)
    external_contributions = defaultdict(lambda: defaultdict(int))
    push_hours = defaultdict(int)

    for event in events_data if events_data else []:
        event_type = event.get('type')
        event_counts[event_type] += 1
        
        if event_type == 'PushEvent':
            recent_activity_score += 5
        elif event_type in ('PullRequestEvent', 'IssuesEvent', 'CreateEvent'):
            recent_activity_score += 3
        elif event_type == 'ForkEvent':
            recent_activity_score += 1

        if event_type in ('PushEvent', 'PullRequestEvent'):
            # External Contribution Check
            try:
                repo_owner = event['repo']['name'].split('/')[0]
                if repo_owner.lower() != owner.lower():
                    external_contributions[repo_owner][event_type] += 1
            except (ValueError, IndexError):
                pass 
            
            # Commit Time Analysis (UTC)
            try:
                event_time = pd.to_datetime(event['created_at']).tz_convert('UTC') 
                push_hours[event_time.hour] += 1
            except:
                pass

    report = {
        'user_id': profile_data.get('login'),
        'name': profile_data.get('name', 'N/A'),
        'hireable': profile_data.get('hireable', False),
        'location': profile_data.get('location', 'N/A'),
        'member_since': profile_data.get('created_at', 'N/A'),
        'followers': profile_data.get('followers', 0),
        'public_repos_count': profile_data.get('public_repos', 0),
        'public_gists_count': profile_data.get('public_gists', 0),
        'total_stars_owned': total_stars_gained,
        'total_forks_owned': total_forks_gained,
        'total_open_issues': total_open_issues,
        'total_watchers': total_watchers,
        'most_popular_repo': most_popular_repo,
        'repo_languages': repo_languages_count,
        'repo_star_counts': repo_star_counts,
        'detailed_languages_top_repo': detailed_languages,
        'top_repo_data': top_repo_data,
        # THESE ARE THE DATAFRAMES: Kept for Streamlit charting
        'repo_commit_activity_df': commit_df,
        'repo_code_frequency_df': code_df,
        'repo_participation_stats': repo_stats['participation'] if repo_stats and 'participation' in repo_stats else None,
        'user_orgs': user_orgs,
        'top_language': max(repo_languages_count.items(), key=operator.itemgetter(1))[0] if repo_languages_count else 'N/A',
        'repos_with_recent_activity': active_repos,
        'recent_activity_score': recent_activity_score,
        'recent_event_types': dict(event_counts),
        'top_external_projects': sorted(
            external_contributions.items(),
            key=lambda item: sum(item[1].values()),
            reverse=True
        )[:5],
        'most_active_time': 'N/A',
        'push_hours': push_hours
    }

    if push_hours:
        most_active_hour = max(push_hours.items(), key=operator.itemgetter(1))[0]
        start_time = f"{most_active_hour:02}:00"
        end_time = f"{(most_active_hour + 1) % 24:02}:00"
        report['most_active_time'] = f"{start_time} - {end_time} UTC Hour"

    return report


def generate_activity_level(score):
    """Translates activity score into a badge/level."""
    if score > 500:
        return "**Extremely Active** 🚀"
    elif score > 200:
        return "**Highly Active** 🔥"
    elif score > 50:
        return "**Moderately Active** ✅"
    else:
        return "Low/Inconsistent Activity 🕰️"

# --- NEW: HTML Report Generation for Print-to-PDF (TABLES ONLY) ---

def generate_html_report(report, star_df_raw=None):
    """Generates a comprehensive, printable HTML string with only tables from the report data."""
    
    # Define CSS for clean printing (no backgrounds, optimized layout)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>GitHub Analysis Report - {report.get('user_id', 'Repo')}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                color: #333;
            }}
            h1, h2, h3 {{ color: #2C3E50; border-bottom: 2px solid #ECF0F1; padding-bottom: 5px; margin-top: 25px; }}
            .metric-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            .metric-table td {{ padding: 10px; border: 1px solid #ddd; }}
            .metric-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric-table td:first-child {{ font-weight: bold; width: 30%; background-color: #ECF0F1; }}
            table.data-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 12px; }}
            table.data-table th, table.data-table td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
            table.data-table th {{ background-color: #3498DB; color: white; }}
            
            /* Print Specific Styles */
            @media print {{
                body {{ margin: 0; padding: 0; font-size: 10pt; }}
                h1 {{ font-size: 18pt; }}
                h2 {{ font-size: 16pt; }}
                h3 {{ font-size: 14pt; }}
                /* Force tables to use minimal padding */
                .metric-table td, table.data-table th, table.data-table td {{ padding: 5px; }}
            }}
        </style>
        <!-- JavaScript to automatically trigger the print dialog upon opening -->
        <script>
            window.onload = function() {{
                // Wait a moment for the page to fully render before printing
                setTimeout(function() {{
                    window.print();
                }}, 500);
            }};
        </script>
    </head>
    <body>
    """

    # --- Header ---
    if report.get('user_id'):
        title = f"GitHub User Analysis Report: {report['user_id']}"
    else:
        title = f"GitHub Repository Star Report: (Star History)"
        
    html_content += f"<h1>{title}</h1>"
    html_content += f"<p><em>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>"
    html_content += "<hr>"
    
    if star_df_raw is not None:
        # --- Repository Star History Report Mode ---
        html_content += "<h2>Star History Raw Data</h2>"
        if not star_df_raw.empty:
            html_content += f"<h3>Star Events ({len(star_df_raw)} Total Stars)</h3>"
            star_df_raw.columns = ['Starred At (UTC)', 'Cumulative Stars']
            html_content += star_df_raw.to_html(index=False, classes='data-table')
        else:
            html_content += "<p><em>No star data available for this repository.</em></p>"

    elif report.get('user_id'):
        # --- User Profile Report Mode (Tables Only) ---
        
        # --- 1. Profile and Core Metrics ---
        html_content += "<h2>1. Profile and Core Metrics</h2>"
        html_content += "<table class='metric-table'>"
        html_content += f"<tr><td>**User ID**</td><td>{report['user_id']}</td></tr>"
        html_content += f"<tr><td>**Name**</td><td>{report['name']}</td></tr>"
        html_content += f"<tr><td>**Hirable Status**</td><td>{'Yes' if report['hireable'] else 'No/Unknown'}</td></tr>"
        html_content += f"<tr><td>Location</td><td>{report['location']}</td></tr>"
        html_content += f"<tr><td>Member Since</td><td>{report['member_since'][:10]}</td></tr>"
        html_content += f"<tr><td>Followers</td><td>{report['followers']}</td></tr>"
        html_content += f"<tr><td>Public Repositories</td><td>{report['public_repos_count']}</td></tr>"
        html_content += f"<tr><td>Total Stars Owned</td><td>{report['total_stars_owned']}</td></tr>"
        html_content += f"<tr><td>Total Forks Owned</td><td>{report['total_forks_owned']}</td></tr>"
        html_content += f"<tr><td>Total Open Issues</td><td>{report['total_open_issues']}</td></tr>"
        html_content += f"<tr><td>Total Watchers/Subscribers</td><td>{report['total_watchers']}</td></tr>"
        html_content += f"<tr><td>Top Language (Repo Count)</td><td>{report['top_language']}</td></tr>"
        html_content += "</table>"
        
        # --- 2. Activity Summary ---
        html_content += "<h2>2. Activity and Engagement</h2>"
        html_content += f"<p><strong>Activity Level:</strong> {generate_activity_level(report['recent_activity_score'])}</p>"
        html_content += f"<p><strong>Most Active Commit Time:</strong> {report['most_active_time']}</p>"
        html_content += f"<p><strong>Repos with Recent Activity:</strong> {report['repos_with_recent_activity']}</p>"

        html_content += "<h3>Recent Public Event Types (Top 100 Events)</h3>"
        event_df = pd.DataFrame(list(report['recent_event_types'].items()), columns=['Event Type', 'Count'])
        html_content += event_df.to_html(index=False, classes='data-table')
        
        # --- Commit Hour Analysis (Table) ---
        html_content += "<h3>Commit Hour Analysis (UTC)</h3>"
        push_df = pd.DataFrame(list(report['push_hours'].items()), columns=['Hour (UTC)', 'Pushes'])
        push_df = push_df.sort_values('Hour (UTC)')
        if not push_df.empty:
            html_content += push_df.to_html(index=False, classes='data-table')
        else:
            html_content += "<p><em>No push event data available.</em></p>"
        
        # --- 3. Repository Analysis ---
        html_content += "<h2>3. Repository Analysis</h2>"
        
        html_content += "<h3>Top Starred Repositories</h3>"
        star_df = pd.DataFrame(list(report['repo_star_counts'].items()), columns=['Repository', 'Stars'])
        star_df = star_df.sort_values('Stars', ascending=False).head(10)
        html_content += star_df.to_html(index=False, classes='data-table')
        
        html_content += "<h3>Overall Language Distribution (by Repo Count)</h3>"
        lang_df = pd.DataFrame(list(report['repo_languages'].items()), columns=['Language', 'Repository Count'])
        html_content += lang_df.to_html(index=False, classes='data-table')
        
        # --- 4. Deep Dive on Most Popular Repository ---
        top_repo_name = report['most_popular_repo']['name']
        html_content += f"<h2>4. Deep Dive: {top_repo_name}</h2>"
        
        # Metadata
        html_content += "<h3>Repository Metadata</h3>"
        html_content += "<table class='metric-table'>"
        html_content += f"<tr><td>Stars</td><td>{report['most_popular_repo']['stars']}</td></tr>"
        html_content += f"<tr><td>Forks</td><td>{report['most_popular_repo']['forks']}</td></tr>"
        html_content += f"<tr><td>Has Pages</td><td>{'Yes' if report['top_repo_data'].get('has_pages') else 'No'}</td></tr>"
        html_content += f"<tr><td>Has Wiki</td><td>{'Yes' if report['top_repo_data'].get('has_wiki') else 'No'}</td></tr>"
        html_content += f"<tr><td>Estimated Issue Lifespan (Days)</td><td>{report['top_repo_data'].get('oldest_open_issue_lifespan_days', 'N/A')}</td></tr>"
        html_content += f"<tr><td>Repository Topics</td><td>{', '.join(report['top_repo_data'].get('topics', [])) or 'N/A'}</td></tr>"
        html_content += "</table>"

        # Detailed Language Breakdown
        html_content += "<h3>Detailed Language Breakdown (Code Volume in Bytes)</h3>"
        if report['detailed_languages_top_repo']:
            lang_df_detailed = pd.DataFrame(list(report['detailed_languages_top_repo'].items()), columns=['Language', 'Bytes'])
            html_content += lang_df_detailed.to_html(index=False, classes='data-table')
        else:
            html_content += "<p><em>No detailed language data available.</em></p>"

        # Contributor Participation
        html_content += "<h3>Contributor Participation (Last Year)</h3>"
        participation = report['repo_participation_stats']
        if participation:
            total_owner = sum(participation.get('owner', []))
            total_all = sum(participation.get('all', []))
            total_others = total_all - total_owner
            
            participation_df = pd.DataFrame({
                'Metric': ['Total Commits (All)', 'Owner Commits', 'External Commits'],
                'Value': [total_all, total_owner, total_others]
            })
            html_content += participation_df.to_html(index=False, classes='data-table')

        # Commit Activity
        html_content += "<h3>Weekly Commit Activity (Last 52 Weeks)</h3>"
        # Access the DataFrame stored in the report dictionary
        if not report['repo_commit_activity_df'].empty:
            html_content += report['repo_commit_activity_df'].tail(52).to_html(index=False, classes='data-table')
        else:
            html_content += "<p><em>Commit activity data not available.</em></p>"
        
        # Code Frequency
        html_content += "<h3>Code Frequency (Weekly Additions/Deletions, Last 52 Weeks)</h3>"
        # Access the DataFrame stored in the report dictionary
        if not report['repo_code_frequency_df'].empty:
            html_content += report['repo_code_frequency_df'].tail(52).to_html(index=False, classes='data-table')
        else:
            html_content += "<p><em>Code frequency data not available.</em></p>"

        # --- 5. Collaboration ---
        html_content += "<h2>5. Collaboration and Organizations</h2>"
        
        html_content += "<h3>Organization Memberships</h3>"
        if report['user_orgs']:
            org_df = pd.DataFrame([{'Organization': org.get('login')} for org in report['user_orgs']])
            html_content += org_df.to_html(index=False, classes='data-table')
        else:
            html_content += "<p><em>No public organization memberships found.</em></p>"

        html_content += "<h3>Top External Projects Contributed To (Recent Events)</h3>"
        if report['top_external_projects']:
            ext_data = []
            for owner, activity in report['top_external_projects']:
                total = sum(activity.values())
                pushes = activity.get('PushEvent', 0)
                prs = activity.get('PullRequestEvent', 0)
                ext_data.append({
                    'Project Owner': owner,
                    'Total Events': total,
                    'Push Events': activity.get('PushEvent', 0),
                    'PR/Issue Events': activity.get('PullRequestEvent', 0) + activity.get('IssuesEvent', 0)
                })
            ext_df = pd.DataFrame(ext_data)
            html_content += ext_df.to_html(index=False, classes='data-table')
        else:
            html_content += "<p><em>No external contributions detected in recent public events.</em></p>"
            
    html_content += "</body></html>"
        
    return html_content

# --- 3. STREAMLIT VISUALIZATION FUNCTIONS (Used for App Display only) ---

def display_repo_star_report(owner, repo, df_plot, df_raw):
    """Displays the repository star history report."""
    st.subheader(f"Star History for `{owner}/{repo}`")
    
    if df_plot.empty:
        st.warning(f"Repo Status: **0 Stars** or no star data available for {owner}/{repo}.")
        return

    col1, col2, col3 = st.columns(3)
    
    total_stars = df_plot['cumulative_stars'].iloc[-1]
    first_star_date = df_plot['starred_at'].min().strftime('%Y-%m-%d')
    latest_star_date = df_plot['starred_at'].max().strftime('%Y-%m-%d')

    col1.metric("Total Stars", total_stars, delta_color="off")
    col2.metric("First Star Date", first_star_date)
    col3.metric("Latest Star Date", latest_star_date)
    
    st.markdown("### Cumulative Star Growth Over Time")
    
    # Graph is still shown in the app, but not in the export
    fig = px.line(
        df_plot, 
        x='starred_at', 
        y='cumulative_stars', 
        title=f"Cumulative Stars for {owner}/{repo}",
        labels={'starred_at': 'Date Starred', 'cumulative_stars': 'Total Stars'},
        markers=True
    )
    fig.update_traces(line=dict(color='#ff6347'))
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Export Section for Repo Star History ---
    st.markdown("---")
    st.subheader("⬇️ Export Analysis")
    
    export_cols = ['Starred At (UTC)', 'Cumulative Stars']
    df_raw.columns = export_cols
    csv_data = df_raw.to_csv(index=False).encode('utf-8')
    
    col_csv, col_pdf, col_json = st.columns(3)

    # 1. CSV Download
    col_csv.download_button(
        label="Download Star History **(CSV)**",
        data=csv_data,
        file_name=f"{owner}_{repo}_star_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        help="Downloads a file containing the raw timestamp and cumulative star count data, suitable for spreadsheets."
    )

    # 2. Auto-PDF HTML Download
    report_html = generate_html_report({}, star_df_raw=df_raw)
    col_pdf.download_button(
        label="Download Star History (Auto-PDF HTML)",
        data=report_html,
        file_name=f"{owner}_{repo}_star_history_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html",
        help="Downloads an HTML file that automatically opens the Print-to-PDF dialog."
    )
    
    # 3. JSON (Report data is simple enough to skip a dedicated button for now, CSV is primary data export)

def pandas_json_encoder(obj):
    """
    Custom function for json.dumps to handle non-standard objects.
    Specifically converts Pandas DataFrames into a JSON-friendly list of records.
    """
    # Check if the object is a Pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        # Convert the DataFrame to a list of dictionaries (one dictionary per row),
        # which is easily serializable to JSON.
        return obj.to_dict(orient='records')

    # Optional: Handle datetime objects if they are also causing issues
    # import datetime
    # if isinstance(obj, datetime.datetime):
    #     return obj.isoformat()

    # Raise the standard TypeError for any other unhandled type
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def display_user_report(report):
    """Displays the comprehensive user analysis report."""
    st.title(f"👤 GitHub User Report: {report['user_id']}")
    
    # --- AI ANALYSIS SECTION ---
    st.markdown("---")
    st.subheader("🤖 AI Profile Analysis & Recommendations")
    
    # Run the AI analysis and display it at the top
    # The fix in get_ai_analysis_markdown handles the DataFrames conversion
    with st.spinner("Running AI analysis and generating recommendations..."):
        ai_markdown = get_ai_analysis_markdown(report)
        st.markdown(ai_markdown)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    # Left Column: Profile Card & Extended Metrics
    with col1:
        st.header(f"{report['name']}")
        st.markdown(f"**Member Since:** {report['member_since'][:10]}")
        st.markdown(f"**Location:** {report['location']}")
        st.markdown(f"**Hirable:** {'Yes' if report['hireable'] else 'No/Unknown'} 💼")

        st.markdown("---")
        st.metric("Followers", report['followers'])
        st.metric("Public Repositories", report['public_repos_count'])
        st.metric("Total Public Gists", report['public_gists_count'], help="Total public code snippets (Gists) owned.")
        st.markdown("---")
        st.metric("Top Language", report['top_language'])

    # Right Column: Core Metrics & Activity Summary
    with col2:
        st.markdown("### Core Metrics (Across All Owned Public Repos)")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Stars Owned", report['total_stars_owned'], delta_color="off")
        m2.metric("Total Forks Owned", report['total_forks_owned'], delta_color="off")
        m3.metric("Repos with Recent Activity", report['repos_with_recent_activity'], 
                     help="Repositories with a push event in the last 12 months.")

        m4, m5 = st.columns(2)
        m4.metric("Total Open Issues", report['total_open_issues'], help="Sum of open issues across all owned public repos.")
        m5.metric("Total Watchers/Subscribers", report['total_watchers'], help="Sum of watchers across all owned public repos.")

        st.markdown(f"""
        ### Activity Summary
        - **Activity Level (Based on Recent Events):** {generate_activity_level(report['recent_activity_score'])}
        - **Most Active Commit Time (UTC):** `{report['most_active_time']}`
        - **Most Popular Repo:** `{report['most_popular_repo']['name']}` (⭐ {report['most_popular_repo']['stars']} / 🍴 {report['most_popular_repo']['forks']})
        """)
        
        st.markdown("---")
        
        
    st.markdown("## 📊 Repository and Coding Analysis")
    
    c1, c2 = st.columns(2)
    
    # Language Distribution (Repo Count) - Shown as graph in app
    with c1:
        st.markdown("### Language Distribution (by Repo Count)")
        lang_df = pd.DataFrame(list(report['repo_languages'].items()), columns=['Language', 'Count'])
        if not lang_df.empty:
            lang_fig = px.pie(lang_df, values='Count', names='Language', title='Repository Language Distribution', hole=.3)
            lang_fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(lang_fig, use_container_width=True)
        else:
            st.info("No language data available from owned repositories.")

    # Top Starred Repositories Bar Chart - Shown as graph in app
    with c2:
        st.markdown("### Top Starred Repositories")
        star_df = pd.DataFrame(list(report['repo_star_counts'].items()), columns=['Repository', 'Stars'])
        star_df = star_df.sort_values('Stars', ascending=False).head(10)
        
        if not star_df.empty:
            star_fig = px.bar(
                star_df, x='Stars', y='Repository', orientation='h',
                title='Top 10 Starred Repos', color_discrete_sequence=['#1f77b4']
            )
            star_fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(star_fig, use_container_width=True)
        else:
            st.info("No starred repositories found.")
            
    # --- Deep Dive on Most Popular Repo ---
    st.markdown("---")
    st.markdown(f"## 🔎 Deep Dive: `{report['most_popular_repo']['name']}`")
    
    # Repo Metadata: Topics, Pages, Wiki
    if report['top_repo_data']:
        st.markdown(f"### Repository Metadata")
        topics_str = ", ".join([f"`{t}`" for t in report['top_repo_data'].get('topics', [])])
        
        m_col1, m_col2 = st.columns(2)
        m_col1.markdown(f"""
        - **GitHub Pages:** {':green[Active] ✅' if report['top_repo_data']['has_pages'] else ':red[Inactive] ❌'}
        - **Wiki Enabled:** {':green[Yes] ✅' if report['top_repo_data']['has_wiki'] else ':red[No] ❌'}
        """)
        m_col2.metric("Est. Oldest Issue Lifespan (Days)", report['top_repo_data'].get('oldest_open_issue_lifespan_days', 'N/A'), 
                      help="Max days an open issue/PR could have existed, based on repo creation.")
        
        st.markdown(f"**Repository Topics:** {topics_str or 'N/A'}")
        
    col_code_1, col_code_2 = st.columns(2)

    # Detailed Language Breakdown Chart - Shown as graph in app
    with col_code_1:
        st.markdown("### Code Volume (Bytes)")
        detailed_langs = report['detailed_languages_top_repo']
        if detailed_langs and sum(detailed_langs.values()) > 0:
            
            total_bytes = sum(detailed_langs.values())
            lang_data = [{'Language': lang, 'Bytes': bytes} 
                         for lang, bytes in detailed_langs.items()]
            lang_df_detailed = pd.DataFrame(lang_data)

            lang_fig_detailed = px.pie(lang_df_detailed, values='Bytes', names='Language', title=f"Code Volume by Language", hole=.3)
            lang_fig_detailed.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(lang_fig_detailed, use_container_width=True)
        else:
            st.info("Detailed language data not available.")

    # Contributor Participation
    with col_code_2:
        st.markdown("### Contributor Participation (Last Year)")
        participation = report['repo_participation_stats']
        if participation:
            total_owner = sum(participation.get('owner', []))
            total_all = sum(participation.get('all', []))
            total_others = total_all - total_owner
            
            st.markdown(f"""
            - **Owner's Commits:** **{total_owner}**
            - **External Commits:** **{total_others}**
            """)
            if total_all > 0:
                st.progress(total_owner / total_all, text=f"{int(total_owner/total_all*100)}% of total commits are from the owner.")
            else:
                st.info("No commits detected in the last year for this repository.")
        else:
            st.info("Contributor participation stats not available.")
            
    # Weekly Commit Activity - Shown as graph in app
    st.markdown("### Weekly Commit Activity (Last 52 Weeks)")
    commit_df = report['repo_commit_activity_df']
    if not commit_df.empty:
        commit_fig = px.bar(
            commit_df, x='Week Start Date', y='Total Commits',
            title='Total Commits by Week', color_discrete_sequence=['#ff6347']
        )
        st.plotly_chart(commit_fig, use_container_width=True)
    else:
        st.info("Commit activity stats not available for the top repository.")

    # Code Frequency - Shown as graph in app
    st.markdown("### Code Frequency (Weekly Additions vs. Deletions)")
    code_df = report['repo_code_frequency_df']
    if not code_df.empty:
        code_df['Week Start Date'] = pd.to_datetime(code_df['Week Start Date'])
        
        fig = px.line(
            code_df, x='Week Start Date', y=['Additions', 'Deletions'], 
            title='Weekly Lines of Code Added vs. Deleted',
            labels={'value': 'Lines of Code', 'variable': 'Action'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Code frequency stats (Additions/Deletions) not available for the top repository.")

    # --- Organizations and Event Breakdown ---
    st.markdown("---")
    st.markdown("## 🌐 Collaboration and Community")

    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown("### Organization Memberships")
        orgs = report['user_orgs']
        if orgs:
            for org in orgs:
                st.markdown(f"- **{org.get('login')}**")
        else:
            st.info("No public organization memberships found.")

    with e2:
        st.markdown("### Top External Contributions")
        if report['top_external_projects']:
            for owner, activity in report['top_external_projects']:
                total = sum(activity.values())
                pushes = activity.get('PushEvent', 0)
                prs = activity.get('PullRequestEvent', 0)
                st.markdown(f"""
                - **{owner}** (Total: **{total}** events)
                    - Pushes: `{pushes}` | PR/Issues: `{prs}`
                """)
        else:
            st.info("No external contributions detected in recent public events.")

    with e3:
        st.markdown("### Recent Event Types")
        event_df = pd.DataFrame(list(report['recent_event_types'].items()), columns=['Event Type', 'Count'])
        event_df = event_df.sort_values('Count', ascending=False)
        st.dataframe(event_df, hide_index=True, use_container_width=True)
    
    # Commit Time Analysis - Shown as graph in app
    st.markdown("### Commit Hour Analysis (Recent Pushes, UTC)")
    push_df = pd.DataFrame(list(report['push_hours'].items()), columns=['Hour', 'Pushes'])
    push_df = push_df.sort_values('Hour')
    if not push_df.empty:
        all_hours = pd.DataFrame({'Hour': range(24)})
        push_df = pd.merge(all_hours, push_df, on='Hour', how='left').fillna(0)
        
        time_fig = px.bar(
            push_df, x='Hour', y='Pushes', 
            title='Push Events by Hour of Day (UTC)',
            color_discrete_sequence=['#2ca02c']
        )
        st.plotly_chart(time_fig, use_container_width=True)
        
    # --- Export Section ---
    st.markdown("---")
    st.subheader("⬇️ Export Analysis")
    
    col_csv, col_pdf, col_json = st.columns(3)
    
    # 1. CSV Download (Core Metrics and Top Repos)
    csv_data = consolidate_user_metrics_to_csv(report)
    col_csv.download_button(
        label="Download Core Data **(CSV)**",
        data=csv_data,
        file_name=f"{report['user_id']}_core_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        help="Downloads a file containing profile metrics and top 10 repos, suitable for spreadsheets."
    )
    
    # 2. Auto-PDF HTML Download (Tables Only)
    report_html = generate_html_report(report)
    col_pdf.download_button(
        label="Download Full Analysis (Auto-PDF HTML)",
        data=report_html,
        file_name=f"{report['user_id']}_github_analysis_tables_only_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html",
        help="Downloads an HTML file that automatically opens the Print-to-PDF dialog."
    )
    
    # 3. JSON Download (Full Report)
    report_json = json.dumps(report, indent=4, default=pandas_json_encoder)
    col_json.download_button(
        label="Download Full Report **(JSON)**",
        data=report_json,
        file_name=f"{report['user_id']}_full_report_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
        help="Downloads the complete analysis as a structured JSON object for programmatic use."
    )


# --- 4. MAIN STREAMLIT APPLICATION LOGIC ---

def main_app():
    """Main function to run the Streamlit app."""
    st.title("GitHub Analysis Tool ⭐")
    st.markdown("Enter a **GitHub Repository URL** (e.g., `github.com/streamlit/streamlit`) for star history, or a **User Profile URL/Username** (e.g., `github.com/torvalds` or `torvalds`) for a detailed profile analysis.")
    
    user_input = st.text_input(
        "GitHub URL or Username", 
        placeholder="Enter URL or Username here...", 
        key="github_input"
    )

    if not user_input:
        st.stop()

    REPO_OWNER, REPO_NAME = parse_github_url(user_input)

    if not REPO_OWNER:
        st.error("Invalid GitHub URL or Username format.")
        st.stop()
    
    st.divider()

    if REPO_NAME:
        # --- Repository Analysis Mode ---
        st.info(f"Analyzing Repository: **{REPO_OWNER}/{REPO_NAME}**")
        raw_star_data = fetch_repo_details(REPO_OWNER, REPO_NAME)
        
        if raw_star_data is not None:
            # df_plot is for the Streamlit chart, df_raw is the tables-only data for export
            df_plot, df_raw = process_star_data(raw_star_data)
            display_repo_star_report(REPO_OWNER, REPO_NAME, df_plot, df_raw)
        else:
            st.error(f"🛑 Could not retrieve data for {REPO_OWNER}/{REPO_NAME}. Check URL and rate limits.")
            
    else:
        # --- User Profile Analysis Mode ---
        st.info(f"Analyzing User Profile: **{REPO_OWNER}**")
        
        with st.spinner("Fetching user profile, repositories, organizations, and events..."):
            profile_data = fetch_user_profile(REPO_OWNER)
            repos_data = fetch_user_repos(REPO_OWNER)
            events_data = fetch_user_events(REPO_OWNER)

        if profile_data and repos_data is not None:
            user_report = analyze_user_data(REPO_OWNER, profile_data, repos_data, events_data)
            display_user_report(user_report)
        else:
            st.error(f"🛑 Failed to fetch full data for user **{REPO_OWNER}**. Check the username or if the user exists.")


if __name__ == "__main__":
    main_app()
