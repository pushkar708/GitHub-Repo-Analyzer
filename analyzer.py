import requests
import pandas as pd
import json
import os
import re
from collections import defaultdict
import operator
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")


def _make_github_request(url, params=None):
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.star+json, application/json",
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None


def fetch_repo_details(owner, repo):
    star_data = []
    page = 1
    per_page = 100
    base_url = f"https://api.github.com/repos/{owner}/{repo}/stargazers"

    print(f"Fetching star data from: {base_url}")

    while True:
        params = {"page": page, "per_page": per_page}
        response = _make_github_request(base_url, params=params)
        if response is None:
            return None

        current_page_data = response.json()
        star_data.extend(current_page_data)

        link_header = response.headers.get('Link')
        if 'rel="next"' in str(link_header):
            page += 1
        else:
            print(f"Fetched last page ({page}). Total starrer entries: {len(star_data)}")
            break

    return star_data


def fetch_user_profile(owner):
    url = f"https://api.github.com/users/{owner}"
    response = _make_github_request(url)
    return response.json() if response else None


def fetch_user_repos(owner):
    repos_data = []
    page = 1
    per_page = 100
    url = f"https://api.github.com/users/{owner}/repos"

    while True:
        params = {"page": page, "per_page": per_page, "type": "owner"}
        response = _make_github_request(url, params=params)
        if response is None:
            return None

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


def fetch_user_events(owner):
    url = f"https://api.github.com/users/{owner}/events/public"
    response = _make_github_request(url, params={"per_page": 100})
    return response.json() if response else None


def process_star_data(raw_star_data):
    if not raw_star_data:
        return pd.DataFrame()

    stars = [{'starred_at': entry['starred_at']} for entry in raw_star_data]
    df = pd.DataFrame(stars)
    if df.empty:
        return df

    df['starred_at'] = pd.to_datetime(df['starred_at'], utc=True)
    df = df.sort_values('starred_at').reset_index(drop=True)
    df['cumulative_stars'] = df.index + 1
    return df


def fetch_user_followers(owner):
    followers_data = []
    page = 1
    url = f"https://api.github.com/users/{owner}/followers"

    while True:
        response = _make_github_request(url, params={"page": page, "per_page": 100})
        if response is None:
            return None

        current_page_data = response.json()
        if not current_page_data:
            break

        followers_data.extend([user['login'] for user in current_page_data])
        link_header = response.headers.get('Link')
        if 'rel="next"' in str(link_header):
            page += 1
        else:
            break

    return followers_data


def fetch_user_following(owner):
    following_data = []
    page = 1
    url = f"https://api.github.com/users/{owner}/following"

    while True:
        response = _make_github_request(url, params={"page": page, "per_page": 100})
        if response is None:
            return None

        current_page_data = response.json()
        if not current_page_data:
            break

        following_data.extend([user['login'] for user in current_page_data])
        link_header = response.headers.get('Link')
        if 'rel="next"' in str(link_header):
            page += 1
        else:
            break

    return following_data


def analyze_user_data(owner, profile_data, repos_data, events_data):
    report = {
        'user_id': profile_data.get('login'),
        'name': profile_data.get('name', 'N/A'),
        'company': profile_data.get('company', 'N/A'),
        'location': profile_data.get('location', 'N/A'),
        'member_since': profile_data.get('created_at', 'N/A'),
        'followers': profile_data.get('followers', 0),
        'following': profile_data.get('following', 0),
        'public_repos_count': profile_data.get('public_repos', 0),
        'repo_url_names': {repo['name'] for repo in repos_data},
    }

    total_stars_gained = 0
    total_forks_gained = 0
    repo_languages = defaultdict(int)
    most_popular_repo = {'name': 'N/A', 'stars': 0, 'forks': 0}
    active_repos = 0
    activity_threshold = (pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=1)).isoformat()

    for repo in repos_data:
        stars = repo.get('stargazers_count', 0)
        forks = repo.get('forks_count', 0)
        total_stars_gained += stars
        total_forks_gained += forks

        if stars > most_popular_repo['stars']:
            most_popular_repo.update({'name': repo['name'], 'stars': stars, 'forks': forks})

        if repo.get('language'):
            repo_languages[repo['language']] += 1

        if repo.get('pushed_at', '1970-01-01T00:00:00Z') > activity_threshold:
            active_repos += 1

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
            repo_owner, repo_name = event['repo']['name'].split('/')
            if repo_owner != owner or repo_name not in report['repo_url_names']:
                external_contributions[repo_owner][event_type] += 1
            try:
                event_time = pd.to_datetime(event['created_at']).tz_convert('Asia/Kolkata')
                push_hours[event_time.hour] += 1
            except:
                pass

    report.update({
        'total_stars_owned': total_stars_gained,
        'total_forks_owned': total_forks_gained,
        'most_popular_repo': most_popular_repo,
        'top_language': max(repo_languages.items(), key=operator.itemgetter(1))[0] if repo_languages else 'N/A',
        'repos_with_recent_activity': active_repos,
        'recent_activity_score': recent_activity_score,
        'recent_event_types': dict(event_counts),
        'external_contributions': external_contributions,
        'top_external_projects': sorted(
            external_contributions.items(),
            key=lambda item: sum(item[1].values()),
            reverse=True
        )[:3],
        'most_active_time': 'N/A'
    })

    if push_hours:
        most_active_hour = max(push_hours.items(), key=operator.itemgetter(1))[0]
        start_time = f"{most_active_hour:02}:00"
        end_time = f"{(most_active_hour + 1) % 24:02}:00"
        report['most_active_time'] = f"{start_time} - {end_time} UTC Hour"

    return report


def generate_activity_level(score):
    if score > 500:
        return "**Extremely Active** 🚀"
    elif score > 200:
        return "**Highly Active** 🔥"
    elif score > 50:
        return "**Moderately Active** ✅"
    else:
        return "Low/Inconsistent Activity 🕰️"


def print_user_report(report):
    print("\n" + "=" * 50)
    print(f"🌟 GitHub User Report: {report['user_id']} 🌟")
    print("=" * 50)

    print("## 👤 User Summary")
    print(f"* **Name:** {report['name']}")
    print(f"* **Location:** {report['location']}")
    print(f"* **GitHub Member Since:** {report['member_since'][:10]}")
    print(f"* **Followers:** {report['followers']} / **Following:** {report['following']}")
    print(f"* **Public Repositories Owned:** {report['public_repos_count']}")
    print(f"* **Top Programming Language:** **{report['top_language']}**")

    print("\n---")
    print("## 🎯 Recruiter-Relevant Metrics")
    print(f"* **Overall Activity Level (Score: {report['recent_activity_score']}):** {generate_activity_level(report['recent_activity_score'])}")
    print(f"* **Active Repositories (Last 12 Months):** {report['repos_with_recent_activity']}")
    print(f"* **Most Active Hour:** **{report['most_active_time']}**")

    print("\n### Open Source Contributions (Recent):")
    if report['top_external_projects']:
        for owner, activity in report['top_external_projects']:
            total = sum(activity.values())
            pushes = activity.get('PushEvent', 0)
            prs = activity.get('PullRequestEvent', 0)
            print(f"> **{owner}** (Total: {total})")
            print(f"  - Pushed: {pushes} | PRs: {prs}")
    else:
        print("  - No external contributions detected.")

    print("\n---")
    print("## ✨ Achievement & Popularity")
    print(f"* **Total Stars:** {report['total_stars_owned']}")
    print(f"* **Total Forks:** {report['total_forks_owned']}")
    print(f"\n### Most Popular Repository:")
    print(f"> **{report['most_popular_repo']['name']}**")
    print(f"> ⭐ Stars: {report['most_popular_repo']['stars']} | 🍴 Forks: {report['most_popular_repo']['forks']}")

    print("\n### Recent Events Breakdown:")
    if report['recent_event_types']:
        print("  - " + " | ".join([f"**{k}**: {v}" for k, v in report['recent_event_types'].items()]))
    else:
        print("  - No significant public events.")
    print("=" * 50)


def print_repo_star_report(owner, repo, df):
    print("\n" + "=" * 50)
    print(f"⭐ Repository Star Report: {owner}/{repo} ⭐")
    print("=" * 50)
    if df.empty:
        print("Repo Status: **0 Stars**")
    else:
        print(f"**Total Stars:** {df['cumulative_stars'].iloc[-1]}")
        print(f"**First Star Date:** {df['starred_at'].min().strftime('%Y-%m-%d')}")
        print(f"**Latest Star Date:** {df['starred_at'].max().strftime('%Y-%m-%d')}")
        print("\n*Star history data is ready for plotting.*")
    print("=" * 50)


def parse_github_url(url):
    pattern = r'(?:https?://github\.com/|git@github\.com:)?([\w\-]+)/?([\w\-.]+)?'
    match = re.search(pattern, url, re.IGNORECASE)
    if match:
        owner = match.group(1)
        repo = match.group(2)
        if repo and repo.endswith('.git'):
            repo = repo[:-4]
        return owner, repo or None
    return None, None


if __name__ == "__main__":
    if not GITHUB_TOKEN:
        print("FATAL ERROR: GITHUB_TOKEN environment variable not set.")
        exit()

    while True:
        user_input = input("Enter a GitHub Repository URL or a User Profile URL: ").strip()
        if not user_input:
            print("Input cannot be empty. Please try again.")
            continue
        REPO_OWNER, REPO_NAME = parse_github_url(user_input)
        if not REPO_OWNER:
            print("Invalid GitHub URL. Please enter a valid one.")
        else:
            break

    print("\n" + "-" * 50)

    if REPO_NAME:
        print(f"Mode: Repository Analysis for **{REPO_OWNER}/{REPO_NAME}**")
        raw_star_data = fetch_repo_details(REPO_OWNER, REPO_NAME)
        if raw_star_data is not None:
            processed_star_data = process_star_data(raw_star_data)
            print_repo_star_report(REPO_OWNER, REPO_NAME, processed_star_data)
        else:
            print(f"🛑 Error: Could not retrieve star data for {REPO_OWNER}/{REPO_NAME}.")
    else:
        print(f"Mode: User Profile Analysis for **{REPO_OWNER}**")
        profile_data = fetch_user_profile(REPO_OWNER)
        repos_data = fetch_user_repos(REPO_OWNER)
        events_data = fetch_user_events(REPO_OWNER)

        if profile_data and repos_data is not None:
            user_report = analyze_user_data(REPO_OWNER, profile_data, repos_data, events_data)
            print_user_report(user_report)
        else:
            print(f"🛑 Error: Failed to fetch full data for user {REPO_OWNER}.")
    print("\n" + "-" * 50)
