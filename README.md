# GitHub Activity and Star Reporter

## 🌟 Project Status: Complete (Initial Version) ✅

This project is a functional Python command-line utility for fetching and analyzing public GitHub data from either a user profile or a repository URL.

## 💡 Overview

This utility serves as a powerful command-line interface (CLI) tool for fetching and generating detailed statistics from the GitHub API. It operates in two main modes:

1.  **Repository Mode:** Analyzes a given repository URL to provide a time series of its star acquisition history.

2.  **User Mode:** Analyzes a developer's profile to generate a comprehensive report covering contribution metrics, primary language, overall activity level, and external project involvement, providing a data-driven overview of their GitHub presence.

<p align="center">
    <video src="YOUR_GITHUB_ASSET_URL_HERE" width="700" controls autoplay muted loop></video>
</p>

## ✨ Features

  * **Repository Star History:** Fetches the complete star history for any public repository, handling API pagination automatically.

  * **Comprehensive User Analysis:** Generates a detailed report on a user's activity, including name, location, followers, and public repository count.

  * **Activity Level Score:** Calculates an overall activity score based on recent events (Pushes, Pull Requests, Issues, etc.) and categorizes it (e.g., Highly Active).

  * **Top Language Identification:** Automatically determines the user's most frequently used programming language across their public repositories.

  * **Most Active Time:** Calculates the user's most active committing hour (based on PushEvents), adjusted to the Asia/Kolkata timezone.

  * **External Contribution Tracking:** Identifies and summarizes contributions (Pushes, Pull Requests) made to projects outside of the user's ownership.

  * **Data Processing:** Uses the **Pandas** library for efficient time series and data manipulation.

## 🛠️ Tech Stack

This project is built using the following technologies:

| Category | Technology |
| ----- | ----- |
| **Language** | **Python 3.x** |
| **API Calls** | `requests` |
| **Data Analysis** | `pandas` |
| **Configuration** | `python-dotenv` |

## 🚀 Getting Started

### Prerequisites

You must have the following installed on your machine:

  * **Python 3.8+**

  * A **GitHub Personal Access Token** (Highly recommended to avoid strict API rate limiting, even for public data).

### Installation

1.  **Clone the repository:**

    ```
    git clone https://github.com/pushkar708/GitHub-Repo-Analyzer.git
    cd GitHub-Repo-Analyzer

    ```

2.  **Install dependencies:**

    ```
    pip install requests pandas python-dotenv

    ```

3.  **Setup Authentication (.env):**
    The script requires a GitHub token to function correctly. Create a file named **`.env`** in the root directory and add your token:

    ```
    GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    ```

    (Note: Replace the placeholder value with your actual token.)

### Usage

Run the main script and enter the target GitHub URL when prompted:

```
python analyzer.py

```

**Example Interactions:**

  * To analyze a **repository**: Enter `https://github.com/requests/requests`

  * To analyze a **user profile**: Enter `https://github.com/torvalds`

## 🤝 Contributing

We welcome contributions\! If you have suggestions or want to report a bug, please open an issue first. If you'd like to submit a patch:

1.  Fork the repository.

2.  Create a new branch (`git checkout -b feature/add-time-filtering`).

3.  Commit your changes (`git commit -m 'Feat: Allow filtering star history by date range'`).

4.  Push to the branch (`git push origin feature/add-time-filtering`).

5.  Open a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

*Project created by \[Pushkar Agarwal\].*