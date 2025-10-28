import os
import re
import json
import time
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
import operator

load_dotenv()
gh_token = os.environ.get("GITHUB_TOKEN", "")
ai_key = os.environ.get("GEMINI_API_KEY", "")

st.set_page_config(page_title="GitHub Profile & Repo Analyzer", page_icon="⭐", layout="wide")

@st.cache_data(ttl=3600)
def _request_github(endpoint, params=None):
    headers = {"Accept": "application/vnd.github.v3.star+json, application/json"}
    if gh_token:
        headers["Authorization"] = f"token {gh_token}"
    else:
        st.warning("⚠️ No GITHUB_TOKEN set. API calls are rate-limited.")
    try:
        resp = requests.get(endpoint, headers=headers, params=params)
        limit = resp.headers.get("X-RateLimit-Limit")
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining and int(remaining) < 10:
            st.warning(f"GitHub Rate Limit Low: {remaining}/{limit} remaining requests.")
        resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        if resp is not None and resp.status_code == 404:
            st.error(f"Error 404: Resource not found: {endpoint}")
        elif resp is not None and resp.status_code == 403:
            st.error(f"Error 403: Forbidden or rate-limited. Remaining: {remaining}")
        else:
            st.error(f"GitHub request failed: {e}")
        return None

def _split_github_path(path):
    pattern = r'(?:https?://github\.com/|git@github\.com:)?([\w\-\.]+)/?([\w\-\.]+)?'
    m = re.search(pattern, path, re.IGNORECASE)
    if m:
        owner = m.group(1)
        repo = m.group(2)
        if repo and repo.endswith(".git"):
            repo = repo[:-4]
        return owner, repo or None
    return None, None

@st.cache_data(ttl=3600)
def fetch_stargazers(owner, repo):
    collected = []
    page = 1
    per_page = 100
    base = f"https://api.github.com/repos/{owner}/{repo}/stargazers"
    with st.spinner(f"Fetching stars for {owner}/{repo}..."):
        while True:
            params = {"page": page, "per_page": per_page, "media": "application/vnd.github.v3.star+json"}
            r = _request_github(base, params=params)
            if r is None:
                return None
            chunk = r.json()
            collected.extend(chunk)
            link = r.headers.get("Link")
            if 'rel="next"' in str(link):
                page += 1
            else:
                break
    return collected

@st.cache_data(ttl=3600)
def fetch_repo_langs(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/languages"
    r = _request_github(url)
    return r.json() if r else None

@st.cache_data(ttl=3600)
def fetch_repo_stats(owner, repo):
    base = f"https://api.github.com/repos/{owner}/{repo}/stats/"
    out = {}
    with st.spinner(f"Fetching deep stats for {owner}/{repo}..."):
        r = _request_github(base + "commit_activity")
        out["commit_activity"] = r.json() if r else None
        r = _request_github(base + "code_frequency")
        out["code_frequency"] = r.json() if r else None
        r = _request_github(base + "participation")
        out["participation"] = r.json() if r else None
    return out

@st.cache_data(ttl=3600)
def fetch_orgs(owner):
    url = f"https://api.github.com/users/{owner}/orgs"
    r = _request_github(url)
    return r.json() if r else None

@st.cache_data(ttl=3600)
def fetch_profile(owner):
    url = f"https://api.github.com/users/{owner}"
    r = _request_github(url)
    return r.json() if r else None

@st.cache_data(ttl=3600)
def fetch_repos(owner):
    out = []
    page = 1
    per_page = 100
    url = f"https://api.github.com/users/{owner}/repos"
    while True:
        params = {"page": page, "per_page": per_page, "type": "owner"}
        r = _request_github(url, params=params)
        if r is None:
            break
        page_chunk = r.json()
        if not page_chunk:
            break
        out.extend(page_chunk)
        link = r.headers.get("Link")
        if 'rel="next"' in str(link):
            page += 1
        else:
            break
    return out

@st.cache_data(ttl=3600)
def fetch_events(owner):
    url = f"https://api.github.com/users/{owner}/events/public"
    r = _request_github(url, params={"per_page": 100})
    return r.json() if r else None

def build_star_data(raw):
    if not raw:
        return pd.DataFrame()
    events = [{'starred_at': item.get('starred_at')} for item in raw if item.get('starred_at')]
    df = pd.DataFrame(events)
    if df.empty:
        return df
    df['starred_at'] = pd.to_datetime(df['starred_at'], utc=True)
    df = df.sort_values('starred_at').reset_index(drop=True)
    df['cumulative'] = df.index + 1
    df_plot = df.set_index('starred_at')['cumulative'].resample('D').max().ffill().reset_index()
    df_raw = df[['starred_at', 'cumulative']].copy()
    df_raw['starred_at'] = df_raw['starred_at'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    return df_plot, df_raw

def to_csv_bytes(report):
    metrics = {
        'Metric': [
            'User ID', 'Name', 'Hirable Status', 'Location', 'Member Since', 'Followers',
            'Public Repositories Count', 'Public Gists Count', 'Total Stars Owned',
            'Total Forks Owned', 'Total Open Issues', 'Total Watchers',
            'Top Language', 'Activity Score', 'Active Repos Count'
        ],
        'Value': [
            report['login'], report['display_name'], 'Yes' if report['hireable'] else 'No/Unknown',
            report['location'], report['created'][:10], report['followers'],
            report['repo_count'], report['gists_count'], report['stars_total'],
            report['forks_total'], report['open_issues_total'], report['watchers_total'],
            report['top_lang'], report['activity_score'], report['active_repos']
        ]
    }
    df_metrics = pd.DataFrame(metrics)
    star_df = pd.DataFrame(list(report['repo_stars'].items()), columns=['Repo', 'Stars'])
    top_repos = star_df.sort_values('Stars', ascending=False).head(10)
    out = "--- CORE PROFILE METRICS ---\n"
    out += df_metrics.to_csv(index=False)
    out += "\n--- TOP 10 STARRED REPOSITORIES ---\n"
    out += top_repos.to_csv(index=False)
    return out.encode('utf-8')

def analyze_account(owner, profile, repos, events):
    open_issues = 0
    watchers = 0
    lang_count = defaultdict(int)
    repo_stars = defaultdict(int)
    top_repo = {'name': 'N/A', 'stars': 0, 'forks': 0}
    active = 0
    threshold = (pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=1)).isoformat()
    stars_sum = 0
    forks_sum = 0
    for r in repos:
        s = r.get('stargazers_count', 0)
        f = r.get('forks_count', 0)
        stars_sum += s
        forks_sum += f
        open_issues += r.get('open_issues_count', 0)
        watchers += r.get('watchers_count', 0)
        if s > 0:
            repo_stars[r['name']] = s
        if s > top_repo['stars']:
            top_repo.update({'name': r['name'], 'stars': s, 'forks': f})
        if r.get('language'):
            lang_count[r['language']] += 1
        if r.get('pushed_at', '1970-01-01T00:00:00Z') > threshold:
            active += 1

    detailed_langs = None
    repo_stats = None
    commit_df = pd.DataFrame()
    code_df = pd.DataFrame()
    top_name = top_repo['name']
    if top_name != 'N/A':
        full = next((x for x in repos if x['name'] == top_name), None)
        top_meta = {}
        if full:
            top_meta['topics'] = full.get('topics', [])
            top_meta['pages'] = full.get('has_pages', False)
            top_meta['wiki'] = full.get('has_wiki', False)
            oldest_days = "N/A"
            if full.get('open_issues_count', 0) > 0:
                created = pd.to_datetime(full.get('created_at'))
                oldest_days = (pd.Timestamp.now(tz='UTC') - created).days
            top_meta['oldest_issue_days'] = oldest_days
        else:
            top_meta = {'topics': [], 'pages': False, 'wiki': False, 'oldest_issue_days': "N/A"}

        detailed_langs = fetch_repo_langs(owner, top_name)
        repo_stats = fetch_repo_stats(owner, top_name)
        if repo_stats and repo_stats.get('commit_activity'):
            dates = [datetime.fromtimestamp(w['week'], tz=pd.Timestamp.now(tz='UTC').tz).strftime('%Y-%m-%d') for w in repo_stats['commit_activity']]
            commits = [w['total'] for w in repo_stats['commit_activity']]
            commit_df = pd.DataFrame({'Week Start Date': dates, 'Total Commits': commits})
        if repo_stats and repo_stats.get('code_frequency'):
            payload = []
            for week in repo_stats['code_frequency']:
                start = datetime.fromtimestamp(week[0], tz=pd.Timestamp.now(tz='UTC').tz).strftime('%Y-%m-%d')
                payload.append({'Week Start Date': start, 'Additions': week[1], 'Deletions': abs(week[2]), 'Net': week[1] + week[2]})
            code_df = pd.DataFrame(payload)
    orgs = fetch_orgs(owner)

    score = 0
    events_count = defaultdict(int)
    external = defaultdict(lambda: defaultdict(int))
    hour_buckets = defaultdict(int)
    for ev in events or []:
        t = ev.get('type')
        events_count[t] += 1
        if t == 'PushEvent':
            score += 5
        elif t in ('PullRequestEvent', 'IssuesEvent', 'CreateEvent'):
            score += 3
        elif t == 'ForkEvent':
            score += 1
        if t in ('PushEvent', 'PullRequestEvent'):
            try:
                repo_owner = ev['repo']['name'].split('/')[0]
                if repo_owner.lower() != owner.lower():
                    external[repo_owner][t] += 1
            except Exception:
                pass
            try:
                et = pd.to_datetime(ev['created_at']).tz_convert('UTC')
                hour_buckets[et.hour] += 1
            except Exception:
                pass

    most_active = 'N/A'
    if hour_buckets:
        h = max(hour_buckets.items(), key=operator.itemgetter(1))[0]
        most_active = f"{h:02}:00 - {(h+1)%24:02}:00 UTC"

    report = {
        'login': profile.get('login'),
        'display_name': profile.get('name', 'N/A'),
        'hireable': profile.get('hireable', False),
        'location': profile.get('location', 'N/A'),
        'created': profile.get('created_at', 'N/A'),
        'followers': profile.get('followers', 0),
        'repo_count': profile.get('public_repos', 0),
        'gists_count': profile.get('public_gists', 0),
        'stars_total': stars_sum,
        'forks_total': forks_sum,
        'open_issues_total': open_issues,
        'watchers_total': watchers,
        'most_starred': top_repo,
        'langs': lang_count,
        'repo_stars': repo_stars,
        'detailed_langs_top': detailed_langs,
        'top_meta': top_meta if top_name != 'N/A' else {},
        'repo_commit_activity_df': commit_df,
        'repo_code_frequency_df': code_df,
        'participation': repo_stats.get('participation') if repo_stats else None,
        'orgs': orgs,
        'top_lang': max(lang_count.items(), key=operator.itemgetter(1))[0] if lang_count else 'N/A',
        'active_repos': active,
        'activity_score': score,
        'recent_events': dict(events_count),
        'top_external': sorted(external.items(), key=lambda it: sum(it[1].values()), reverse=True)[:5],
        'most_active_time': most_active,
        'push_hours': hour_buckets
    }
    return report

def human_activity_label(n):
    if n > 500:
        return "**Extremely Active** 🚀"
    if n > 200:
        return "**Highly Active** 🔥"
    if n > 50:
        return "**Moderately Active** ✅"
    return "Low/Inconsistent Activity 🕰️"

def html_report_from_data(data, star_raw=None):
    # html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Report - {data.get('login','Repo')}</title><style>body{{font-family:Arial;margin:20px;color:#333}}h1,h2,h3{{color:#2C3E50;border-bottom:2px solid #ECF0F1;padding-bottom:5px;margin-top:25px}}.metric-table{{width:100%;border-collapse:collapse;margin-bottom:20px}}.metric-table td{{padding:10px;border:1px solid #ddd}}.metric-table tr:nth-child(even){{background-color:#f9f9f9}}.metric-table td:first-child{{font-weight:bold;width:30%;background-color:#ECF0F1}}table.data-table{{width:100%;border-collapse:collapse;margin-bottom:20px;font-size:12px}}table.data-table th,table.data-table td{{padding:8px;border:1px solid #ddd;text-align:left}}table.data-table th{{background-color:#3498DB;color:white}}@media print{{body{{margin:0;padding:0;font-size:10pt}}h1{{font-size:18pt}}h2{{font-size:16pt}}h3{{font-size:14pt}}.metric-table td,table.data-table th,table.data-table td{{padding:5px}}}}</style><script>window.onload=function(){setTimeout(function(){window.print();},500);};</script></head><body>"""
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Report - {data.get('login','Repo')}</title><style>body{{{{font-family:Arial;margin:20px;color:#333}}}}h1,h2,h3{{{{color:#2C3E50;border-bottom:2px solid #ECF0F1;padding-bottom:5px;margin-top:25px}}}}.metric-table{{{{width:100%;border-collapse:collapse;margin-bottom:20px}}}}.metric-table td{{{{padding:10px;border:1px solid #ddd}}}}.metric-table tr:nth-child(even){{{{background-color:#f9f9f9}}}}.metric-table td:first-child{{{{font-weight:bold;width:30%;background-color:#ECF0F1}}}}table.data-table{{{{width:100%;border-collapse:collapse;margin-bottom:20px;font-size:12px}}}}table.data-table th,table.data-table td{{{{padding:8px;border:1px solid #ddd;text-align:left}}}}table.data-table th{{{{background-color:#3498DB;color:white}}}}@media print{{{{body{{{{margin:0;padding:0;font-size:10pt}}}}h1{{{{font-size:18pt}}}}h2{{{{font-size:16pt}}}}h3{{{{font-size:14pt}}}}.metric-table td,table.data-table th,table.data-table td{{{{padding:5px}}}}}}}}</style><script>window.onload=function(){{{{setTimeout(function(){{{{window.print();}}}},500);}}}};</script></head><body>"""

    if star_raw is not None:
        html += f"<h1>Star History</h1><p><em>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p><hr>"
        if not star_raw.empty:
            df = star_raw.copy()
            df.columns = ['Starred At (UTC)', 'Cumulative Stars']
            html += f"<h3>Star Events ({len(df)} Total)</h3>"
            html += df.to_html(index=False, classes='data-table')
        else:
            html += "<p><em>No star data available.</em></p>"
    elif data.get('login'):
        html += f"<h1>GitHub User Analysis: {data['login']}</h1><p><em>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p><hr>"
        html += "<h2>1. Profile and Core Metrics</h2><table class='metric-table'>"
        html += f"<tr><td>User ID</td><td>{data['login']}</td></tr>"
        html += f"<tr><td>Name</td><td>{data['display_name']}</td></tr>"
        html += f"<tr><td>Hirable</td><td>{'Yes' if data['hireable'] else 'No/Unknown'}</td></tr>"
        html += f"<tr><td>Location</td><td>{data['location']}</td></tr>"
        html += f"<tr><td>Member Since</td><td>{data['created'][:10]}</td></tr>"
        html += f"<tr><td>Followers</td><td>{data['followers']}</td></tr>"
        html += f"<tr><td>Public Repos</td><td>{data['repo_count']}</td></tr>"
        html += f"<tr><td>Total Stars</td><td>{data['stars_total']}</td></tr>"
        html += f"<tr><td>Total Forks</td><td>{data['forks_total']}</td></tr>"
        html += f"<tr><td>Total Open Issues</td><td>{data['open_issues_total']}</td></tr>"
        html += f"<tr><td>Total Watchers</td><td>{data['watchers_total']}</td></tr>"
        html += f"<tr><td>Top Language</td><td>{data['top_lang']}</td></tr>"
        html += "</table>"
        html += "<h2>2. Activity and Engagement</h2>"
        html += f"<p><strong>Activity Level:</strong> {human_activity_label(data['activity_score'])}</p>"
        html += f"<p><strong>Most Active Time:</strong> {data['most_active_time']}</p>"
        html += f"<p><strong>Repos with Recent Activity:</strong> {data['active_repos']}</p>"
        html += "<h3>Recent Event Types</h3>"
        ev_df = pd.DataFrame(list(data['recent_events'].items()), columns=['Event Type', 'Count'])
        html += ev_df.to_html(index=False, classes='data-table')
        html += "<h3>Commit Hour Analysis</h3>"
        ph = pd.DataFrame(list(data['push_hours'].items()), columns=['Hour (UTC)', 'Pushes']).sort_values('Hour (UTC)')
        if not ph.empty:
            html += ph.to_html(index=False, classes='data-table')
        else:
            html += "<p><em>No push data.</em></p>"
        html += "<h2>3. Repository Analysis</h2>"
        star_df = pd.DataFrame(list(data['repo_stars'].items()), columns=['Repository', 'Stars']).sort_values('Stars', ascending=False).head(10)
        html += star_df.to_html(index=False, classes='data-table')
        lang_df = pd.DataFrame(list(data['langs'].items()), columns=['Language', 'Repository Count'])
        html += lang_df.to_html(index=False, classes='data-table')
        top = data['most_starred']['name']
        html += f"<h2>4. Deep Dive: {top}</h2>"
        html += "<h3>Metadata</h3><table class='metric-table'>"
        html += f"<tr><td>Stars</td><td>{data['most_starred']['stars']}</td></tr>"
        html += f"<tr><td>Forks</td><td>{data['most_starred']['forks']}</td></tr>"
        html += f"<tr><td>Pages</td><td>{'Yes' if data['top_meta'].get('pages') else 'No'}</td></tr>"
        html += f"<tr><td>Wiki</td><td>{'Yes' if data['top_meta'].get('wiki') else 'No'}</td></tr>"
        html += f"<tr><td>Estimated Issue Lifespan (Days)</td><td>{data['top_meta'].get('oldest_issue_days','N/A')}</td></tr>"
        html += f"<tr><td>Topics</td><td>{', '.join(data['top_meta'].get('topics',[])) or 'N/A'}</td></tr>"
        html += "</table>"
        if data['detailed_langs_top']:
            df_lang_d = pd.DataFrame(list(data['detailed_langs_top'].items()), columns=['Language', 'Bytes'])
            html += df_lang_d.to_html(index=False, classes='data-table')
        else:
            html += "<p><em>No detailed language data.</em></p>"
        html += "<h3>Contributor Participation (Last Year)</h3>"
        part = data.get('participation')
        if part:
            total_owner = sum(part.get('owner', []))
            total_all = sum(part.get('all', []))
            total_others = total_all - total_owner
            p_df = pd.DataFrame({'Metric': ['Total Commits (All)', 'Owner Commits', 'External Commits'], 'Value': [total_all, total_owner, total_others]})
            html += p_df.to_html(index=False, classes='data-table')
        html += "<h3>Weekly Commit Activity</h3>"
        if not data['repo_commit_activity_df'].empty:
            html += data['repo_commit_activity_df'].tail(52).to_html(index=False, classes='data-table')
        else:
            html += "<p><em>No commit activity data.</em></p>"
        html += "<h3>Code Frequency</h3>"
        if not data['repo_code_frequency_df'].empty:
            html += data['repo_code_frequency_df'].tail(52).to_html(index=False, classes='data-table')
        else:
            html += "<p><em>No code frequency data.</em></p>"
        html += "<h2>5. Collaboration</h2><h3>Organizations</h3>"
        if data['orgs']:
            orgs_df = pd.DataFrame([{'Organization': o.get('login')} for o in data['orgs']])
            html += orgs_df.to_html(index=False, classes='data-table')
        else:
            html += "<p><em>No public orgs.</em></p>"
        html += "<h3>Top External Projects</h3>"
        if data['top_external']:
            rows = []
            for o, act in data['top_external']:
                t = sum(act.values())
                rows.append({'Project Owner': o, 'Total Events': t, 'Push Events': act.get('PushEvent',0), 'PR/Issue Events': act.get('PullRequestEvent',0) + act.get('IssuesEvent',0)})
            html += pd.DataFrame(rows).to_html(index=False, classes='data-table')
        else:
            html += "<p><em>No external contributions detected.</em></p>"
    html += "</body></html>"
    return html

def _json_encoder(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    raise TypeError(f"Type {obj.__class__.__name__} not serializable")

def ai_summarize(data_dict):
    payload = json.dumps(data_dict, default=_json_encoder, indent=4)
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
    system_txt = """You are a world-class GitHub Profile Analyst. Your task is to interpret the provided JSON data about a user's GitHub profile and provide a concise, professional analysis. The analysis MUST contain the following three sections: 1. Executive Summary (3 sentences, covering their specialty and activity level). 2. Key Strengths (3 bullet points, highlighting the best aspects of the profile). 3. Actionable Recommendations (3 concrete, next-step suggestions for improvement). Present the response strictly in Markdown format, using headings. Use emojis for emphasis."""
    user_txt = f"Analyze the following user data and provide the requested summary, strengths, and recommendations.\n\n{payload}"
    body = {
        "contents": [{"parts": [{"text": user_txt}]}],
        "system_instruction": {"parts": [{"text": system_txt}]},
    }
    max_attempts = 3
    for i in range(max_attempts):
        try:
            headers = {"Content-Type": "application/json"}
            r = requests.post(f"{api_url}?key={ai_key}", headers=headers, json=body)
            r.raise_for_status()
            res = r.json()
            if 'candidates' in res and len(res['candidates']) > 0 and 'parts' in res['candidates'][0].get('content', {}):
                return res['candidates'][0]['content']['parts'][0].get('text', "AI returned empty.")
            return "AI Analysis: Could not parse response."
        except requests.exceptions.RequestException as e:
            if i < max_attempts - 1:
                time.sleep(2 ** i)
            else:
                return f"AI Analysis Failed: {e}"
    return "AI Analysis Failed: Unknown error."

def show_repo_stars(owner, repo, df_plot, df_raw):
    st.subheader(f"Star History for `{owner}/{repo}`")
    if df_plot.empty:
        st.warning(f"No star data for {owner}/{repo}.")
        return
    c1, c2, c3 = st.columns(3)
    total = df_plot['cumulative'].iloc[-1]
    first = df_plot['starred_at'].min().strftime('%Y-%m-%d')
    last = df_plot['starred_at'].max().strftime('%Y-%m-%d')
    c1.metric("Total Stars", total, delta_color="off")
    c2.metric("First Star Date", first)
    c3.metric("Latest Star Date", last)
    st.markdown("### Cumulative Star Growth")
    fig = px.line(df_plot, x='starred_at', y='cumulative', title=f"Cumulative Stars for {owner}/{repo}", labels={'starred_at': 'Date Starred', 'cumulative': 'Total Stars'}, markers=True)
    fig.update_traces(line=dict(color='#ff6347'))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("⬇️ Export")
    df_export = df_raw.copy()
    df_export.columns = ['Starred At (UTC)', 'Cumulative Stars']
    csv_bytes = df_export.to_csv(index=False).encode('utf-8')
    col_csv, col_pdf, col_json = st.columns(3)
    col_csv.download_button(label="Download Star History (CSV)", data=csv_bytes, file_name=f"{owner}_{repo}_star_history_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    html_blob = html_report_from_data({}, star_raw=df_export)
    col_pdf.download_button(label="Download Star History (HTML)", data=html_blob, file_name=f"{owner}_{repo}_star_history_{datetime.now().strftime('%Y%m%d')}.html", mime="text/html")

def show_user_report(report):
    st.title(f"👤 GitHub User Report: {report['login']}")
    st.markdown("---")
    st.subheader("🤖 AI Profile Analysis & Recommendations")
    with st.spinner("Running AI analysis..."):
        ai_md = ai_summarize(report)
        st.markdown(ai_md)
    st.markdown("---")
    left, right = st.columns([1, 2])
    with left:
        st.header(report['display_name'])
        st.markdown(f"**Member Since:** {report['created'][:10]}")
        st.markdown(f"**Location:** {report['location']}")
        st.markdown(f"**Hirable:** {'Yes' if report['hireable'] else 'No/Unknown'} 💼")
        st.markdown("---")
        st.metric("Followers", report['followers'])
        st.metric("Public Repositories", report['repo_count'])
        st.metric("Public Gists", report['gists_count'])
        st.markdown("---")
        st.metric("Top Language", report['top_lang'])
    with right:
        st.markdown("### Core Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Stars", report['stars_total'], delta_color="off")
        m2.metric("Total Forks", report['forks_total'], delta_color="off")
        m3.metric("Active Repos (12mo)", report['active_repos'])
        m4, m5 = st.columns(2)
        m4.metric("Open Issues", report['open_issues_total'])
        m5.metric("Watchers", report['watchers_total'])
        st.markdown(f"### Activity Summary\n- **Activity Level:** {human_activity_label(report['activity_score'])}\n- **Most Active Time (UTC):** `{report['most_active_time']}`\n- **Top Repo:** `{report['most_starred']['name']}` (⭐ {report['most_starred']['stars']} / 🍴 {report['most_starred']['forks']})")
    st.markdown("## 📊 Repository and Coding Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Language Distribution (by Repo Count)")
        lang_df = pd.DataFrame(list(report['langs'].items()), columns=['Language', 'Count'])
        if not lang_df.empty:
            fig = px.pie(lang_df, values='Count', names='Language', title='Repository Language Distribution', hole=.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No language data available.")
    with c2:
        st.markdown("### Top Starred Repositories")
        star_df = pd.DataFrame(list(report['repo_stars'].items()), columns=['Repository', 'Stars']).sort_values('Stars', ascending=False).head(10)
        if not star_df.empty:
            bar = px.bar(star_df, x='Stars', y='Repository', orientation='h', title='Top 10 Starred Repos')
            bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(bar, use_container_width=True)
        else:
            st.info("No starred repositories found.")
    st.markdown("---")
    st.markdown(f"## 🔎 Deep Dive: `{report['most_starred']['name']}`")
    if report.get('top_meta'):
        st.markdown("### Repository Metadata")
        topics = ", ".join([f"`{t}`" for t in report['top_meta'].get('topics', [])])
        a, b = st.columns(2)
        a.markdown(f"- **GitHub Pages:** {'Active ✅' if report['top_meta'].get('pages') else 'Inactive ❌'}\n- **Wiki Enabled:** {'Yes ✅' if report['top_meta'].get('wiki') else 'No ❌'}")
        b.metric("Est. Oldest Issue Lifespan (Days)", report['top_meta'].get('oldest_issue_days', 'N/A'))
        st.markdown(f"**Repository Topics:** {topics or 'N/A'}")
    code_col1, code_col2 = st.columns(2)
    with code_col1:
        st.markdown("### Code Volume (Bytes)")
        dl = report['detailed_langs_top']
        if dl and sum(dl.values()) > 0:
            data = [{'Language': k, 'Bytes': v} for k, v in dl.items()]
            df_dl = pd.DataFrame(data)
            fig = px.pie(df_dl, values='Bytes', names='Language', title='Code Volume by Language', hole=.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No detailed language data available.")
    with code_col2:
        st.markdown("### Contributor Participation (Last Year)")
        part = report.get('participation')
        if part:
            owner_commits = sum(part.get('owner', []))
            total_commits = sum(part.get('all', []))
            external_commits = total_commits - owner_commits
            st.markdown(f"- **Owner's Commits:** **{owner_commits}**\n- **External Commits:** **{external_commits}**")
            if total_commits > 0:
                st.progress(owner_commits / total_commits, text=f"{int(owner_commits/total_commits*100)}% owner commits")
        else:
            st.info("No participation stats available.")
    st.markdown("### Weekly Commit Activity (Last 52 Weeks)")
    if not report['repo_commit_activity_df'].empty:
        fig_commits = px.bar(report['repo_commit_activity_df'], x='Week Start Date', y='Total Commits', title='Total Commits by Week')
        st.plotly_chart(fig_commits, use_container_width=True)
    else:
        st.info("No commit activity data.")
    st.markdown("### Code Frequency (Weekly Additions vs Deletions)")
    if not report['repo_code_frequency_df'].empty:
        dfc = report['repo_code_frequency_df'].copy()
        dfc['Week Start Date'] = pd.to_datetime(dfc['Week Start Date'])
        fig_cf = px.line(dfc, x='Week Start Date', y=['Additions', 'Deletions'], title='Weekly Lines of Code Added vs Deleted', labels={'value':'Lines of Code','variable':'Action'})
        st.plotly_chart(fig_cf, use_container_width=True)
    else:
        st.info("No code frequency data.")
    st.markdown("---")
    st.markdown("## 🌐 Collaboration and Community")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("### Organizations")
        if report['orgs']:
            for o in report['orgs']:
                st.markdown(f"- **{o.get('login')}**")
        else:
            st.info("No public orgs.")
    with e2:
        st.markdown("### Top External Contributions")
        if report['top_external']:
            for owner, activity in report['top_external']:
                total = sum(activity.values())
                pushes = activity.get('PushEvent', 0)
                prs = activity.get('PullRequestEvent', 0)
                st.markdown(f"- **{owner}** (Total: **{total}** events)\n    - Pushes: `{pushes}` | PR/Issues: `{prs}`")
        else:
            st.info("No external contributions detected.")
    with e3:
        st.markdown("### Recent Event Types")
        edf = pd.DataFrame(list(report['recent_events'].items()), columns=['Event Type', 'Count']).sort_values('Count', ascending=False)
        st.dataframe(edf, hide_index=True, use_container_width=True)
    st.markdown("### Commit Hour Analysis (UTC)")
    ph = pd.DataFrame(list(report['push_hours'].items()), columns=['Hour', 'Pushes']).sort_values('Hour')
    if not ph.empty:
        all_hours = pd.DataFrame({'Hour': range(24)})
        merged = pd.merge(all_hours, ph, on='Hour', how='left').fillna(0)
        fig_hours = px.bar(merged, x='Hour', y='Pushes', title='Push Events by Hour of Day (UTC)')
        st.plotly_chart(fig_hours, use_container_width=True)
    st.markdown("---")
    st.subheader("⬇️ Export Analysis")
    col_csv, col_pdf, col_json = st.columns(3)
    csv_bytes = to_csv_bytes(report)
    col_csv.download_button(label="Download Core Data (CSV)", data=csv_bytes, file_name=f"{report['login']}_core_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    html_blob = html_report_from_data(report)
    col_pdf.download_button(label="Download Full Analysis (HTML)", data=html_blob, file_name=f"{report['login']}_github_analysis_{datetime.now().strftime('%Y%m%d')}.html", mime="text/html")
    report_json = json.dumps(report, indent=4, default=_json_encoder)
    col_json.download_button(label="Download Full Report (JSON)", data=report_json, file_name=f"{report['login']}_full_report_{datetime.now().strftime('%Y%m%d')}.json", mime="application/json")

def main():
    st.title("GitHub Analysis Tool ⭐")
    st.markdown("Enter a GitHub Repository URL for star history or a Username for a profile analysis.")
    user_input = st.text_input("GitHub URL or Username", placeholder="github.com/owner/repo or owner", key="gh_input")
    if not user_input:
        st.stop()
    owner, repo = _split_github_path(user_input)
    if not owner:
        st.error("Invalid GitHub URL or username.")
        st.stop()
    st.divider()
    if repo:
        st.info(f"Analyzing Repository: **{owner}/{repo}**")
        raw = fetch_stargazers(owner, repo)
        if raw is not None:
            df_plot, df_raw = build_star_data(raw)
            show_repo_stars(owner, repo, df_plot, df_raw)
        else:
            st.error(f"Could not retrieve data for {owner}/{repo}.")
    else:
        st.info(f"Analyzing User Profile: **{owner}**")
        with st.spinner("Fetching profile, repos and events..."):
            profile = fetch_profile(owner)
            repos = fetch_repos(owner)
            events = fetch_events(owner)
        if profile and repos is not None:
            user_report = analyze_account(owner, profile, repos, events)
            show_user_report(user_report)
        else:
            st.error(f"Failed to fetch data for user {owner}.")

if __name__ == "__main__":
    main()
