# GCP Terraform Chatbot (Agentic Jira-Driven Version)

## Overview

This project is an **agentic DevOps assistant** that automates GCP infrastructure changes using **Jira tickets as the sole trigger**. The system is designed for a human-in-the-loop, auditable workflow where:
- Creating a Jira ticket in "To Do" triggers the agent to propose and PR Terraform changes.
- Creating a **sub-task** for a ticket in "In Review" triggers a follow-up agentic workflow for that parent ticket.
- All actions (status transitions, PRs, summaries) are logged to **Google Cloud Logging**.

**No manual chat or web UI is used in production. All workflows are driven by Jira.**

---

## Architecture

- **Backend:** FastAPI app that listens for Jira webhooks, runs the agentic workflow, manages GitHub PRs, and logs to GCP Logging.
- **Jira:** The only user interface for requesting and tracking infrastructure changes.
- **GitHub:** Stores Terraform code and receives automated PRs.
- **Vertex AI:** Generates Terraform code changes from natural language (using Gemini 2.5 Pro).
- **GCP Logging:** Stores all logs for traceability and debugging.

---

## Setup

### 1. **Jira Setup**

#### a. **Create a Jira API Token**
- Go to https://id.atlassian.com/manage-profile/security/api-tokens
- Click **Create API token**, label it, and copy the token.

#### b. **Create a Jira Webhook**
- Go to Jira Settings → System → Webhooks
- Click **Create a Webhook**
- **URL:** Use your public FastAPI endpoint (see ngrok below for local dev)
- **Events:** Select **Issue Created**
- **Status:** Enabled

#### c. **Jira Permissions**
- The API user must have permission to transition issues and add comments in the relevant project.

---

### 2. **GCP Logging Setup**

- The service account used by your backend must have the **Logs Writer** role (`roles/logging.logWriter`).
- Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your service account key JSON.
- [IAM & Admin > IAM](https://console.cloud.google.com/iam-admin/iam) → Add role to your service account.

---

### 3. **ngrok for Local Development**

If running locally, expose your FastAPI server to the internet using ngrok:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
ngrok http 8000
```
- Use the HTTPS forwarding URL from ngrok (e.g., `https://abcd1234.ngrok.io/webhook/jira`) as your Jira webhook URL.

---

### 4. **Environment Variables (.env Example)**

Create a `.env` file in your backend directory:

```env
PROJECT_ID=your-gcp-project-id
REGION=us-central1
GITHUB_TOKEN=your_github_token
GITHUB_REPO=your_github_username/your_terraform_repo_name
DEFAULT_BRANCH=main
JIRA_URL=https://yourdomain.atlassian.net
JIRA_USER=your-email@example.com
JIRA_API_TOKEN=your-jira-api-token
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account-key.json
JIRA_PROJECT_KEY=your-project-id
```

---

### 5. **Install Dependencies**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Agentic Usage (Jira-Driven Only)

### **A. Propose a New Change**
1. **Create a Jira ticket** in the configured project, in the "To Do" status, describing your infrastructure change.
2. The agent will:
   - Move the ticket to **In Progress**
   - Propose and commit Terraform changes in a new branch
   - Open a PR in GitHub
   - Move the ticket to **In Review** and comment with a summary and PR link
   - Log all actions to GCP Logging

### **B. Propose a Follow-up Change (for tickets already In Review)**
1. **Create a sub-task** for any ticket that is currently in "In Review".
2. The agent will:
   - Move the parent ticket to **In Progress**
   - Use the sub-task's summary/description as the new prompt
   - Propose and commit Terraform changes in a new branch
   - Open a PR in GitHub
   - Move the parent ticket back to **In Review** and comment with a summary and PR link
   - Log all actions to GCP Logging

**Note:** Only sub-tasks can trigger follow-up changes for tickets in "In Review". Comments and other ticket types are ignored.

---

## Robustness & Debugging

- **Status Checks:** Before processing, the backend always fetches the latest status of the ticket (and parent, for sub-tasks) from Jira. Tickets are only processed if they are still in "To Do" (for normal tickets) or the parent is in "In Review" (for sub-tasks). Deleted or completed tickets are ignored.
- **Debug Endpoint:**
  - `POST /debug/clear_cache` — Clears all in-memory state (pending user Terraform changes and context) without restarting the backend.
- **Logs:** All actions are logged to GCP Logging for traceability and debugging.

---

## Technical Highlights
- **Vertex AI Gemini 2.5 Pro** is used for all LLM tasks.
- **Patch-by-block** logic ensures robust Terraform file updates.
- **All workflows are triggered and managed via Jira tickets and sub-tasks.**
- **No manual chat or web UI is used in production.**

---

## Troubleshooting
- **Permission Denied for Logging:**
  - Ensure your service account has `roles/logging.logWriter`.
  - Make sure `GOOGLE_APPLICATION_CREDENTIALS` is set and points to a valid key.
- **Jira Comment/Transition Fails:**
  - Ensure your Jira API user has permission to transition issues and add comments.
  - Double-check your Jira status names (case and whitespace sensitive).
- **ngrok Not Working:**
  - Make sure ngrok is running and you are using the HTTPS forwarding URL in Jira.
- **No PR Created:**
  - Check GCP Logging for errors.
  - Ensure the ticket is created in the "To Do" status or the parent is in "In Review" for sub-tasks.
- **Old/Deleted Tickets Processed:**
  - The backend always checks the latest status before processing, but you can also clear the in-memory cache with the debug endpoint or by restarting the backend.

---

For issues or contributions, please open an issue or PR on GitHub.