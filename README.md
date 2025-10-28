# 🔍 GitHub Repository Analyzer Dashboard

## 🚀 Project Status: Complete (Streamlit + Gemini Integration) ✅

This project is an interactive **Streamlit web application** that analyzes GitHub repositories and users, generating detailed insights and AI-powered summaries. It combines **GitHub API analytics**, **data visualization**, and **Gemini AI** for intelligent summaries.

---

## 💡 Overview

The **GitHub Repository Analyzer Dashboard** allows users to:

1. **Analyze Repositories:** Visualize repository metrics such as commits, pull requests, issues, stars, and forks over time.  
2. **Analyze User Profiles:** Get insights into a developer’s activity, repositories, and engagement metrics.  
3. **Generate AI Summaries:** Automatically produce concise summaries of repository or user performance using **Gemini AI**.  
4. **Export Reports:** Generate printable and shareable HTML reports for documentation or presentation.

---

## ✨ Features

- 📊 **Dynamic Analytics Dashboard:** Real-time stats from the GitHub API.  
- 🧠 **Gemini AI Summaries:** Intelligent overviews of user or repo data.  
- 📈 **Visual Metrics:** Interactive charts for commits, stars, forks, and issues.  
- 🧾 **Printable Reports:** Generate an HTML summary ready for export.  
- ⚙️ **Rate-Limit Handling:** Retries with exponential backoff for stable API access.  
- 🧮 **Data Handling:** Efficient data management with Pandas and caching via Streamlit.

---

## 🛠️ Tech Stack

| Category | Technology |
|-----------|-------------|
| **Frontend** | [Streamlit](https://streamlit.io) |
| **Backend API** | [GitHub REST API](https://docs.github.com/en/rest) |
| **AI Model** | [Gemini API (Google Generative AI)](https://ai.google.dev/) |
| **Language** | Python 3.8+ |
| **Libraries** | `requests`, `pandas`, `matplotlib`, `streamlit` |
| **Environment Management** | `python-dotenv` |

---

## ⚙️ Environment Variables

Create a `.env` file in your project root with:

```

GITHUB_TOKEN=your_github_token_here
GEMINI_API_KEY=your_gemini_api_key_here

````

You can generate:
- **GitHub Token:** From [GitHub → Settings → Developer Settings → Personal Access Tokens](https://github.com/settings/tokens)  
- **Gemini API Key:** From [Google AI Studio](https://aistudio.google.com/)

---

## 🧩 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/pushkar708/GitHub-Repo-Analyzer.git
cd GitHub-Repo-Analyzer
````

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:

```bash
pip install streamlit requests pandas matplotlib python-dotenv
```

3. **Set Up Environment**

Create a `.env` file in the root directory and add your tokens (as shown above).

---

## 🧠 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the link shown in your terminal (usually `http://localhost:8501`).

### In the App:

* Enter a **GitHub Repository URL** (e.g., `https://github.com/streamlit/streamlit`)
* Or enter a **GitHub Username** (e.g., `torvalds`)
* Wait for the analytics and **AI summary** to generate
* Optionally, **download or print** the full report

---

## 📊 Sample Output

The dashboard provides:

* Repository metrics (commits, forks, issues, stars)
* Contributor breakdowns
* Star growth visualization
* AI-generated insights (via Gemini)
* Printable reports with clean formatting

---

## 🎥 Live Demo

<p align="center">
    <video src="https://github.com/user-attachments/assets/8b9efb48-31aa-4e08-aacc-42c613da147d" width="700" controls autoplay muted loop></video>
</p>

> 🎬 The video above showcases the live working of the Streamlit GitHub Analyzer App.

---

## 🧱 Project Structure

```
GitHub-Repo-Analyzer/
│
├── app.py                # Main Streamlit application
├── .env                  # API keys (GitHub & Gemini)
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork this repository
2. Create a new branch: `git checkout -b feature/add-improvement`
3. Make your changes and commit: `git commit -m "Add new visualization"`
4. Push the branch: `git push origin feature/add-improvement`
5. Open a Pull Request 🎉

---

## 📜 License

Distributed under the **MIT License**.
See `LICENSE` for more information.

---

**Project by [Pushkar Agarwal](https://github.com/pushkar708)**
Built with ❤️ using Streamlit, GitHub API, and Gemini AI.