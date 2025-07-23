from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict
import os
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel
from github import Github
import datetime
import difflib
from unidiff import PatchSet
import io
import re
import json
import requests
from google.cloud import logging as gcp_logging
from apscheduler.schedulers.background import BackgroundScheduler
import threading
from fastapi.responses import JSONResponse
import tempfile
import shutil
import subprocess

# Load environment variables from .env file
load_dotenv()

# Initialize GCP Logging
try:
    gcp_logging_client = gcp_logging.Client()
    gcp_logging_client.setup_logging()
    print("[GCP LOGGING] Initialized GCP logging client.")
except Exception as e:
    print(f"[GCP LOGGING] Failed to initialize: {e}")

app = FastAPI()

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
DEFAULT_BRANCH = "main"
JIRA_URL = os.getenv("JIRA_URL")
JIRA_USER = os.getenv("JIRA_USER")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")

# In-memory store for generated Terraform change and context per user
user_terraform_change = {}
user_terraform_context = {}

class ChatRequest(BaseModel):
    message: str
    user_id: str

class ApprovalRequest(BaseModel):
    user_id: str
    action: str  # 'approve' or 'reject'

class SummarizeRequest(BaseModel):
    user_id: str


def fetch_terraform_files():
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(GITHUB_REPO)
    branch = repo.get_branch(DEFAULT_BRANCH)
    tree = repo.get_git_tree(branch.commit.sha, recursive=True).tree
    tf_files = [f for f in tree if (f.path.endswith('.tf') or f.path.endswith('.tfvars')) and f.type == 'blob']
    # Limit to 25 files, 100 KB each
    files = []
    for f in tf_files[:25]:
        blob = repo.get_git_blob(f.sha)
        content = blob.content
        import base64
        decoded = base64.b64decode(content)
        if len(decoded) <= 100 * 1024:
            files.append({"path": f.path, "content": decoded.decode(errors='replace')})
    return files

def call_vertex_ai(prompt: str) -> str:
    vertexai.init(project=PROJECT_ID, location=REGION)
    # Upgraded to Gemini 2.5 Pro (as of July 2024)
    model = GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text

def initial_summary_and_diff(user_prompt: str, files: list) -> str:
    context = "\n\n".join([f"File: {f['path']}\n{f['content']}" for f in files])
    full_prompt = (
        f"You are an expert DevOps assistant. Here is the current state of the infrastructure as Terraform files:\n"
        f"{context}\n\n"
        f"IMPORTANT: First determine if this is actually a request for infrastructure changes or just casual conversation. If it is casual conversation, respond with a friendly message\n"
        f"User request: {user_prompt}\n\n"
        f"For each file that needs to be changed, output only the full, updated content for each changed block (resource/module/variable/etc.), with clear file and block identifiers.\n"
        f"Use this format for each change:\n"
        f"File: <filename>\nBlock: <block identifier or resource name>\n```hcl\n<full new block content>\n```\n"
        f"Repeat for each changed block in each file.\n"
        f"Do NOT include explanations, comments, or extra text."
    )
    return call_vertex_ai(full_prompt)

def cleanup_diff(diff: str) -> str:
    prompt = (
        f"Here is a proposed unified diff. Clean it up to ensure it is a valid, patchable unified diff, with correct headers, context lines, and no extra text. Output only the cleaned diff.\n"
        f"```diff\n{diff}\n```"
    )
    return call_vertex_ai(prompt)

def validate_and_fix_diff(diff: str, files: list) -> str:
    context = "\n\n".join([f"File: {f['path']}\n{f['content']}" for f in files])
    prompt = (
        f"Here is the current state of the infrastructure as Terraform files:\n"
        f"{context}\n\n"
        f"Here is a unified diff:\n````diff\n{diff}\n````\n"
        f"STRICT: Output ONLY a valid, patchable unified diff for all changed files. Do NOT include any 'File: ...' blocks, explanations, or extra text. Only the diff. Ensure there are blank lines between file diffs. Double-check that all hunk headers, line numbers, and context match the current file content exactly. If you are unsure, output a full file diff that replaces the entire file, with correct hunk headers and context."
    )
    return call_vertex_ai(prompt)

def apply_diff_to_files(files, diff_text):
    """
    Apply a unified diff to a list of files (dicts with 'path' and 'content').
    Returns a dict of updated file contents {path: new_content}.
    Handles multi-file diffs robustly: tries PatchSet on the whole diff, falls back to per-file splitting if needed.
    """
    print("[DEBUG] Raw validated diff_text to be applied:")
    print(diff_text)
    # Remove markdown code block markers if present
    diff_text = re.sub(r'^```diff\s*|```$', '', diff_text.strip(), flags=re.MULTILINE)
    # Remove '\ No newline at end of file' lines
    diff_text = '\n'.join(line for line in diff_text.splitlines() if line.strip() != '\ No newline at end of file')
    file_map = {f['path']: f['content'].splitlines(keepends=True) for f in files}
    updated_files = {path: ''.join(lines) for path, lines in file_map.items()}
    print(f"[DEBUG] Files in repo context: {list(file_map.keys())}")

    try:
        patch = PatchSet(io.StringIO(diff_text))
        print(f"[DEBUG] PatchSet parsed {len(patch)} files from unified diff.")
        for patched_file in patch:
            # Remove a/ or b/ prefix for matching
            path = patched_file.path
            if path.startswith('a/') or path.startswith('b/'):
                path = path[2:]
            file_exists = path in file_map
            print(f"[DEBUG] Processing file: {path} (exists in repo: {file_exists})")
            if file_exists:
                print(f"[DEBUG] File '{path}' length: {len(file_map[path])} lines.")
            else:
                print(f"[DEBUG] File '{path}' does not exist in repo context. Will be created if diff applies.")
            for hunk in patched_file:
                print(f"[DEBUG] Hunk header for {path}: source_start={hunk.source_start}, source_length={hunk.source_length}, target_start={hunk.target_start}, target_length={hunk.target_length}")
            if not file_exists:
                print(f"[DEBUG] Creating new file from diff: {path}")
                # New file: build content from added and context lines in the diff
                new_lines = []
                for hunk in patched_file:
                    for line in hunk:
                        if line.is_added or line.is_context:
                            new_lines.append(line.value)
                updated_files[path] = ''.join(new_lines)
                print(f"[DEBUG] New file created: {path}")
                continue
            original = file_map[path]
            new_lines = []
            i = 0
            try:
                for hunk in patched_file:
                    print(f"[DEBUG] Applying hunk to {path}: file length={len(original)}, hunk source_start={hunk.source_start}, hunk source_length={hunk.source_length}")
                    # Add unchanged lines before the hunk
                    while i < hunk.source_start - 1:
                        new_lines.append(original[i])
                        i += 1
                    # Apply hunk
                    for line in hunk:
                        if line.is_added:
                            new_lines.append(line.value)
                        elif line.is_context:
                            new_lines.append(original[i])
                            i += 1
                        elif line.is_removed:
                            i += 1
                    # After hunk, i is at the next line to process
                # Add any remaining lines after the last hunk
                new_lines.extend(original[i:])
                updated_files[path] = ''.join(new_lines)
                print(f"[DEBUG] Updated file: {path}")
            except IndexError as e:
                print(f"[ERROR] IndexError applying hunk in file {path}: {e}")
                print(f"[ERROR] Falling back to full file replacement for {path} using all added/context lines from diff.")
                fallback_lines = []
                for hunk in patched_file:
                    for line in hunk:
                        # Only include actual code lines, not diff headers or file markers
                        if (line.is_added or line.is_context) and not (
                            line.value.strip().startswith('--- a/') or
                            line.value.strip().startswith('+++ b/') or
                            line.value.strip().startswith('@@') or
                            line.value.strip().startswith('File:')
                        ):
                            fallback_lines.append(line.value)
                updated_files[path] = ''.join(fallback_lines)
                print(f"[DEBUG] Fallback content for {path} (first 500 chars):\n{updated_files[path][:500]}")
        print(f"[DEBUG] Updated files to be returned: {list(updated_files.keys())}")
        return updated_files
    except Exception as e:
        print(f"[ERROR] PatchSet failed on whole diff: {e}")
        print("[DEBUG] Falling back to per-file diff splitting.")
        # Fallback: Pre-process and split diff into per-file chunks
        file_diffs = re.split(r'(?=^--- a/)', diff_text, flags=re.MULTILINE)
        for file_diff in file_diffs:
            file_diff = file_diff.strip()
            if not file_diff:
                continue
            print(f"[DEBUG] Processing file diff chunk:\n{file_diff[:500]}\n--- END CHUNK ---")
            try:
                patch = PatchSet(io.StringIO(file_diff))
            except Exception as e:
                print(f"[ERROR] PatchSet parse error for chunk: {e}")
                continue
            for patched_file in patch:
                # Remove a/ or b/ prefix for matching
                path = patched_file.path
                if path.startswith('a/') or path.startswith('b/'):
                    path = path[2:]
                file_exists = path in file_map
                print(f"[DEBUG] (Fallback) Processing file: {path} (exists in repo: {file_exists})")
                if file_exists:
                    print(f"[DEBUG] (Fallback) File '{path}' length: {len(file_map[path])} lines.")
                else:
                    print(f"[DEBUG] (Fallback) File '{path}' does not exist in repo context. Will be created if diff applies.")
                for hunk in patched_file:
                    print(f"[DEBUG] (Fallback) Hunk header for {path}: source_start={hunk.source_start}, source_length={hunk.source_length}, target_start={hunk.target_start}, target_length={hunk.target_length}")
                if not file_exists:
                    print(f"[DEBUG] (Fallback) Creating new file from diff: {path}")
                    new_lines = []
                    for hunk in patched_file:
                        for line in hunk:
                            if line.is_added or line.is_context:
                                new_lines.append(line.value)
                    updated_files[path] = ''.join(new_lines)
                    print(f"[DEBUG] (Fallback) New file created: {path}")
                    continue
                original = file_map[path]
                new_lines = []
                i = 0
                try:
                    for hunk in patched_file:
                        print(f"[DEBUG] (Fallback) Applying hunk to {path}: file length={len(original)}, hunk source_start={hunk.source_start}, hunk source_length={hunk.source_length}")
                        while i < hunk.source_start - 1:
                            new_lines.append(original[i])
                            i += 1
                        for line in hunk:
                            if line.is_added:
                                new_lines.append(line.value)
                            elif line.is_context:
                                new_lines.append(original[i])
                                i += 1
                            elif line.is_removed:
                                i += 1
                    new_lines.extend(original[i:])
                    updated_files[path] = ''.join(new_lines)
                    print(f"[DEBUG] (Fallback) Updated file: {path}")
                except IndexError as e:
                    print(f"[ERROR] (Fallback) IndexError applying hunk in file {path}: {e}")
                    print(f"[ERROR] (Fallback) Falling back to full file replacement for {path} using all added/context lines from diff.")
                    fallback_lines = []
                    for hunk in patched_file:
                        for line in hunk:
                            # Only include actual code lines, not diff headers or file markers
                            if (line.is_added or line.is_context) and not (
                                line.value.strip().startswith('--- a/') or
                                line.value.strip().startswith('+++ b/') or
                                line.value.strip().startswith('@@') or
                                line.value.strip().startswith('File:')
                            ):
                                fallback_lines.append(line.value)
                    updated_files[path] = ''.join(fallback_lines)
                    print(f"[DEBUG] (Fallback) Fallback content for {path} (first 500 chars):\n{updated_files[path][:500]}")
        print(f"[DEBUG] (Fallback) Updated files to be returned: {list(updated_files.keys())}")
        return updated_files

def parse_changed_files_and_summary(response: str):
    """
    Parse the model's response and extract (summary, {filename: new_content})
    Expects format:
    Summary: ...\nFile: <filename>\n```terraform\n<new file content>\n```\n(Repeat for each changed file)
    """
    summary_match = re.search(r'^Summary:(.*)$', response, re.MULTILINE)
    summary = summary_match.group(1).strip() if summary_match else None
    files = {}
    # Strictly match: File: <filename>\n```(terraform)?\n<content>\n```
    file_blocks = re.findall(r'^File: (.*?)\n```(?:terraform)?\n([\s\S]*?)\n```', response, re.MULTILINE)
    for filename, content in file_blocks:
        files[filename.strip()] = content.strip()
    return summary, files

# Helper: parse model response for block changes
def parse_block_changes(response: str):
    """
    Parse model response for block changes in the format:
    File: <filename>\nBlock: <block identifier>\n```hcl\n<block content>\n```\n
    Returns: dict {filename: list of (block_id, block_content)}
    """
    changes = {}
    pattern = r'File: (.*?)\nBlock: (.*?)\n```hcl\n([\s\S]*?)```'
    for match in re.finditer(pattern, response):
        filename = match.group(1).strip()
        block_id = match.group(2).strip()
        block_content = match.group(3).strip()
        if filename not in changes:
            changes[filename] = []
        changes[filename].append((block_id, block_content))
    return changes

def find_block_span(file_content, block_header):
    """
    Returns (start, end) indices of the block in file_content, or None if not found.
    """
    import re
    header_pattern = re.compile(re.escape(block_header) + r'\s*\{', re.MULTILINE)
    match = header_pattern.search(file_content)
    if not match:
        return None
    start = match.start()
    i = match.end()  # position after the opening brace
    depth = 1
    while i < len(file_content):
        if file_content[i] == '{':
            depth += 1
        elif file_content[i] == '}':
            depth -= 1
            if depth == 0:
                return (start, i + 1)
        i += 1
    return None  # Block not closed properly


def replace_or_insert_block(file_content, block_id, new_block):
    import re
    print(f"[DEBUG] Attempting to match block_id: {block_id}")
    print(f"[DEBUG] File content preview:\n{file_content[:200]}")
    # 1. Try to match assignment: block_id = [ or block_id = {
    assign_pattern = re.compile(rf'^{re.escape(block_id)}\s*=\s*([\[\{{])', re.MULTILINE)
    assign_match = assign_pattern.search(file_content)
    if assign_match:
        open_bracket = assign_match.group(1)
        close_bracket = ']' if open_bracket == '[' else '}'
        start = assign_match.start()
        i = assign_match.end()
        depth = 1
        while i < len(file_content):
            if file_content[i] == open_bracket:
                depth += 1
            elif file_content[i] == close_bracket:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    new_content = file_content[:start] + new_block + '\n' + file_content[end:]
                    print(f"[DEBUG] Replaced assignment '{block_id}' in file (bracket-counting for assignment).")
                    return new_content
            i += 1
        print(f"[DEBUG] Assignment header found but not closed properly for '{block_id}'. Appending at end.")
        return file_content.rstrip() + '\n\n' + new_block + '\n'
    # 2. Try to match block header: block_id { (as before)
    block_header_pattern = re.compile(rf'^{re.escape(block_id)}\s*\{{', re.MULTILINE)
    match = block_header_pattern.search(file_content)
    if match:
        start = match.start()
        i = match.end()
        depth = 1
        while i < len(file_content):
            if file_content[i] == '{':
                depth += 1
            elif file_content[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    new_content = file_content[:start] + new_block + '\n' + file_content[end:]
                    print(f"[DEBUG] Replaced block '{block_id}' in file (bracket-counting, robust header match).")
                    return new_content
            i += 1
        print(f"[DEBUG] Block header found but block not closed properly for '{block_id}'. Appending at end.")
        return file_content.rstrip() + '\n\n' + new_block + '\n'
    else:
        print(f"[DEBUG] Block or assignment header for '{block_id}' not found, inserting at end of file.")
        return file_content.rstrip() + '\n\n' + new_block + '\n'

# Main patch-by-block logic
def apply_block_changes(files, block_changes):
    """
    files: list of dicts with 'path' and 'content'
    block_changes: dict {filename: list of (block_id, block_content)}
    Returns: dict {filename: new_content}
    """
    updated_files = {f['path']: f['content'] for f in files}
    for filename, changes in block_changes.items():
        if filename not in updated_files:
            print(f"[DEBUG] File '{filename}' not found in repo context, skipping.")
            continue
        content = updated_files[filename]
        for block_id, new_block in changes:
            print(f"[DEBUG] Applying block change: file={filename}, block_id={block_id}")
            content = replace_or_insert_block(content, block_id, new_block)
        updated_files[filename] = content
    print(f"[DEBUG] Updated files after block changes: {list(updated_files.keys())}")
    return updated_files

# Helper to transition a Jira issue
def jira_transition_issue(issue_key, transition_name):
    print(f"[JIRA] Transitioning {issue_key} to '{transition_name}'...")
    # Get all transitions
    url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}/transitions"
    auth = (JIRA_USER, JIRA_API_TOKEN)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    resp = requests.get(url, auth=auth, headers=headers)
    if resp.status_code != 200:
        print(f"[JIRA] Failed to get transitions: {resp.text}")
        return False
    transitions = resp.json().get("transitions", [])
    tid = None
    for t in transitions:
        if t["name"].lower() == transition_name.lower():
            tid = t["id"]
            break
    if not tid:
        print(f"[JIRA] Transition '{transition_name}' not found for {issue_key}.")
        return False
    # Do the transition
    resp = requests.post(url, auth=auth, headers=headers, json={"transition": {"id": tid}})
    print(f"[JIRA] Transition response: {resp.status_code} {resp.text}")
    return resp.status_code == 204

# Helper to comment on a Jira issue
def jira_comment_issue(issue_key, comment):
    print(f"[JIRA] Commenting on {issue_key}...")
    url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}/comment"
    auth = (JIRA_USER, JIRA_API_TOKEN)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    # Jira Cloud requires Atlassian Document Format (ADF)
    body = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {
                            "type": "text",
                            "text": comment
                        }
                    ]
                }
            ]
        }
    }
    resp = requests.post(url, auth=auth, headers=headers, json=body)
    print(f"[JIRA] Comment response: {resp.status_code} {resp.text}")
    return resp.status_code == 201

def get_existing_suggestions():
    """
    Fetch all existing suggestion issues in the Suggestions column from Jira.
    Returns a set of suggestion summaries (for deduplication).
    """
    url = f"{JIRA_URL}/rest/api/3/search"
    auth = (JIRA_USER, JIRA_API_TOKEN)
    headers = {"Accept": "application/json"}
    jql = f'status = "Suggestions" AND project = {JIRA_PROJECT_KEY}'
    params = {"jql": jql, "fields": "summary", "maxResults": 100}
    resp = requests.get(url, auth=auth, headers=headers, params=params)
    if resp.status_code != 200:
        print(f"[JIRA] Failed to fetch existing suggestions: {resp.text}")
        return set()
    issues = resp.json().get("issues", [])
    return set(issue["fields"]["summary"].strip() for issue in issues)

def create_jira_suggestion_issue(suggestion):
    url = f"{JIRA_URL}/rest/api/3/issue"
    auth = (JIRA_USER, JIRA_API_TOKEN)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    data = {
        "fields": {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": suggestion[:100],
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": suggestion
                            }
                        ]
                    }
                ]
            },
            "issuetype": {"name": "Task"},  # Or "Suggestion" if you have a custom type
            "labels": ["suggestion"]
        }
    }
    resp = requests.post(url, auth=auth, headers=headers, json=data)
    print(f"[JIRA] Created suggestion issue: {resp.status_code} {resp.text}")
    if resp.status_code == 201:
        issue_key = resp.json().get("key")
        if issue_key:
            jira_transition_issue(issue_key, "Suggestions")

def generate_and_post_suggestions():
    print("[SUGGESTIONS] Running daily suggestion generation...")
    files = fetch_terraform_files()
    prompt = (
        "You are an expert code reviewer. Suggest 2-3 improvements for this Terraform codebase. "
        "Be specific and actionable. Do not repeat previous suggestions.\n\n"
        + "\n\n".join([f"File: {f['path']}\n{f['content']}" for f in files])
    )
    suggestions = call_vertex_ai(prompt)
    print(f"[SUGGESTIONS] Suggestions generated:\n{suggestions}")
    existing = get_existing_suggestions()
    for suggestion in suggestions.split("\n"):
        suggestion = suggestion.strip("-•1234567890. ").strip()
        if suggestion and suggestion not in existing:
            create_jira_suggestion_issue(suggestion)

# Start the scheduler in a background thread after FastAPI app is created
scheduler = BackgroundScheduler()
scheduler.add_job(generate_and_post_suggestions, 'cron', hour=3)  # Runs daily at 3am
scheduler_thread = threading.Thread(target=scheduler.start)
scheduler_thread.daemon = True
scheduler_thread.start()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        files = fetch_terraform_files()
        print(f"[DEBUG] Files fetched for context: {[f['path'] for f in files]}")
        response = initial_summary_and_diff(req.message, files)
        print(f"[DEBUG] Initial model response (block changes):\n{response}")
        # Store the full response for approval
        user_terraform_change[req.user_id] = response
        user_terraform_context[req.user_id] = files
        return {"response": response}
    except Exception as e:
        print(f"[ERROR] Exception in /chat: {e}")
        return {"response": f"Error: {str(e)}"}

@app.post("/approve")
def approve(req: ApprovalRequest):
    print(f"[DEBUG] user_terraform_change keys: {list(user_terraform_change.keys())}")
    print(f"[DEBUG] user_terraform_context keys: {list(user_terraform_context.keys())}")
    print(f"[DEBUG] user_terraform_change for user {req.user_id}: {user_terraform_change.get(req.user_id)}")
    print(f"[DEBUG] user_terraform_context for user {req.user_id}: {user_terraform_context.get(req.user_id)}")
    if req.action == "approve":
        response = user_terraform_change.get(req.user_id)
        files = user_terraform_context.get(req.user_id)
        if not response or not files:
            print("[DEBUG] No change or context found for this user.")
            return {"result": "No change or context found for this user. Please generate a change first."}
        print(f"[DEBUG] Model response for user {req.user_id} (block changes):\n{response}")
        try:
            block_changes = parse_block_changes(response)
            if not block_changes:
                print("[DEBUG] No block changes parsed from model response.")
                return {"result": "No block changes found in model response."}
            updated_files = apply_block_changes(files, block_changes)
        except Exception as e:
            print(f"[ERROR] Error applying block changes: {e}")
            return {"result": f"Error applying block changes: {e}"}
        # Only push if any file content actually changed
        changed = False
        for f in files:
            orig = f['content']
            updated = updated_files.get(f['path'], orig)
            if orig != updated:
                changed = True
                break
        if not changed:
            print("[DEBUG] No actual file changes detected. Not creating PR.")
            return {"result": "No files were changed. The block changes could not be applied or resulted in no changes."}
        try:
            print(f"[DEBUG] Preparing to push changes to GitHub...")
            g = Github(GITHUB_TOKEN)
            repo = g.get_repo(GITHUB_REPO)
            base = repo.get_branch(DEFAULT_BRANCH)
            branch_name = f"infra-change-{req.user_id}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base.commit.sha)
            commit_message = f"Apply infrastructure change for user {req.user_id} via chatbot"
            for path, content in updated_files.items():
                print(f"[DEBUG] Committing file: {path}")
                # If file exists, update; else, create
                try:
                    f = repo.get_contents(path, ref=branch_name)
                    repo.update_file(path, commit_message, content, f.sha, branch=branch_name)
                except Exception:
                    repo.create_file(path, commit_message, content, branch=branch_name)
            pr = repo.create_pull(
                title=f"Infra change for user {req.user_id}",
                body="Automated PR from GCP Terraform Chatbot",
                head=branch_name,
                base=DEFAULT_BRANCH
            )
            print(f"[DEBUG] Pull request created: {pr.html_url}")

            # Run terraform checks and post results
            repo_url = f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com/{GITHUB_REPO}.git"
            tf_results = run_terraform_checks(repo_url, branch_name)
            tf_results_str = format_terraform_check_results(tf_results)
            # Post as PR comment
            pr.create_issue_comment(f"Terraform checks (init/validate/plan) results:\n\n{tf_results_str}")
            # Optionally, post to Jira if user_id is a Jira ticket key
            if re.match(r"^[A-Z]+-\d+$", req.user_id):
                jira_comment_issue(req.user_id, f"Terraform checks (init/validate/plan) results:\n\n{tf_results_str}")

            return {"result": f"Pull request created: {pr.html_url}"}
        except Exception as e:
            print(f"[ERROR] Error creating PR: {e}")
            return {"result": f"Error creating PR: {str(e)}"}
    else:
        user_terraform_change.pop(req.user_id, None)
        user_terraform_context.pop(req.user_id, None)
        print("[DEBUG] Request rejected and change discarded.")
        return {"result": "Request rejected and change discarded."}

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    response = user_terraform_change.get(req.user_id)
    if not response:
        return {"summary": "No change found for this user."}
    prompt = (
        f"You are an expert DevOps assistant. Here is a set of Terraform block changes, each with a file and block name. "
        f"Summarize the overall infrastructure change in 1-2 sentences, focusing on what is being added, removed, or modified. "
        f"Do NOT include code, only a human-readable summary.\n\n"
        f"{response}"
    )
    summary = call_vertex_ai(prompt)
    return {"summary": summary.strip()}

# Temporarily allow GET for Jira webhook validation
@app.api_route("/webhook/jira", methods=["POST", "GET"])
async def jira_webhook(request: Request):
    if request.method == "GET":
        # TEMP: Allow GET for Jira webhook validation. Remove after webhook is saved.
        return JSONResponse({"status": "ok"})
    print("[JIRA WEBHOOK] Endpoint hit!")
    import logging
    logger = logging.getLogger("jira-webhook")
    payload = await request.json()
    headers = dict(request.headers)
    logger.info({"event": "webhook_received", "payload": payload, "headers": headers})
    print("[JIRA WEBHOOK] Received payload:", json.dumps(payload, indent=2))
    print(f"[JIRA WEBHOOK] Headers: {headers}")

    event_type = payload.get("webhookEvent")
    logger.info({"event": "event_type_parsed", "event_type": event_type})
    print(f"[JIRA WEBHOOK] Event type: {event_type}")

    if event_type == "jira:issue_created":
        issue = payload.get("issue", {})
        key = issue.get("key")
        fields = issue.get("fields", {})
        summary = fields.get("summary")
        description = fields.get("description")
        reporter = fields.get("reporter", {}).get("displayName")
        status_name = fields.get("status", {}).get("name", "")
        issue_type = fields.get("issuetype", {}).get("name", "")
        parent = fields.get("parent")
        logger.info({"event": "ticket_info_extracted", "key": key, "summary": summary, "status": status_name, "reporter": reporter, "issue_type": issue_type, "parent": parent})
        print(f"[JIRA WEBHOOK] Issue key: {key}")
        print(f"[JIRA WEBHOOK] Summary: {summary}")
        print(f"[JIRA WEBHOOK] Description: {description}")
        print(f"[JIRA WEBHOOK] Reporter: {reporter}")
        print(f"[JIRA WEBHOOK] Issue status (raw): '{status_name}'")
        print(f"[JIRA WEBHOOK] Issue type: {issue_type}")
        # Always fetch latest ticket status from Jira
        url = f"{JIRA_URL}/rest/api/3/issue/{key}"
        auth = (JIRA_USER, JIRA_API_TOKEN)
        headers_jira = {"Accept": "application/json"}
        resp = requests.get(url, auth=auth, headers=headers_jira)
        if resp.status_code != 200:
            print(f"[JIRA WEBHOOK] Ticket {key} no longer exists. Skipping.")
            return {"status": "ignored", "reason": "ticket deleted"}
        latest_fields = resp.json().get("fields", {})
        latest_status = latest_fields.get("status", {}).get("name", "")
        print(f"[JIRA WEBHOOK] Latest status for {key}: {latest_status}")
        # If this is a sub-task, check parent status
        if issue_type.lower() == "sub-task" and parent:
            parent_key = parent.get("key")
            # Fetch parent issue to get its status
            url = f"{JIRA_URL}/rest/api/3/issue/{parent_key}"
            resp = requests.get(url, auth=auth, headers=headers_jira)
            if resp.status_code != 200:
                print(f"[JIRA WEBHOOK] Parent ticket {parent_key} no longer exists. Skipping.")
                return {"status": "ignored", "reason": "parent ticket deleted"}
            parent_fields = resp.json().get("fields", {})
            parent_status = parent_fields.get("status", {}).get("name", "")
            print(f"[JIRA WEBHOOK] Parent {parent_key} status: {parent_status}")
            if parent_status.strip().lower() == "in review":
                # Move parent to In Progress
                jira_transition_issue(parent_key, "In Progress")
                user_prompt = summary or ""
                if description:
                    user_prompt += f"\n{description}"
                print(f"[JIRA WEBHOOK] Using user_prompt from sub-task: {user_prompt}")
                files = fetch_terraform_files()
                logger.info({"event": "files_fetched", "key": parent_key, "files": [f['path'] for f in files]})
                print(f"[JIRA WEBHOOK] Files fetched for context: {[f['path'] for f in files]}")
                response = initial_summary_and_diff(user_prompt, files)
                logger.info({"event": "model_response", "key": parent_key, "response": response})
                print(f"[JIRA WEBHOOK] Model response (block changes):\n{response}")
                user_terraform_change[parent_key] = response
                user_terraform_context[parent_key] = files
                # --- Apply changes and create PR (same as /approve logic) ---
                block_changes = parse_block_changes(response)
                if not block_changes:
                    logger.info({"event": "no_block_changes", "key": parent_key})
                    print("[JIRA WEBHOOK] No block changes parsed from model response.")
                    return {"status": "no_changes", "reason": "No block changes found in model response."}
                updated_files = apply_block_changes(files, block_changes)
                changed = False
                for f in files:
                    orig = f['content']
                    updated = updated_files.get(f['path'], orig)
                    if orig != updated:
                        changed = True
                        break
                if not changed:
                    logger.info({"event": "no_actual_changes", "key": parent_key})
                    print("[JIRA WEBHOOK] No actual file changes detected. Not creating PR.")
                    return {"status": "no_changes", "reason": "No files were changed. The block changes could not be applied or resulted in no changes."}
                print(f"[JIRA WEBHOOK] Preparing to push changes to GitHub...")
                g = Github(GITHUB_TOKEN)
                repo = g.get_repo(GITHUB_REPO)
                base = repo.get_branch(DEFAULT_BRANCH)
                branch_name = f"infra-change-{parent_key}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base.commit.sha)
                commit_message = f"Apply infrastructure change for Jira ticket {parent_key} via chatbot (from sub-task {key})"
                for path, content in updated_files.items():
                    print(f"[JIRA WEBHOOK] Committing file: {path}")
                    try:
                        f = repo.get_contents(path, ref=branch_name)
                        repo.update_file(path, commit_message, content, f.sha, branch=branch_name)
                    except Exception:
                        repo.create_file(path, commit_message, content, branch=branch_name)
                pr = repo.create_pull(
                    title=f"Infra change for Jira ticket {parent_key} (from sub-task {key})",
                    body=f"Automated PR from GCP Terraform Chatbot for Jira ticket {parent_key} (from sub-task {key})",
                    head=branch_name,
                    base=DEFAULT_BRANCH
                )
                logger.info({"event": "pr_created", "key": parent_key, "pr_url": pr.html_url, "branch": branch_name})
                print(f"[JIRA WEBHOOK] Pull request created: {pr.html_url}")
                # Run terraform checks and post results
                repo_url = f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com/{GITHUB_REPO}.git"
                tf_results = run_terraform_checks(repo_url, branch_name)
                tf_results_str = format_terraform_check_results(tf_results)
                pr.create_issue_comment(f"Terraform checks (init/validate/plan) results:\n\n{tf_results_str}")
                jira_comment_issue(parent_key, f"Terraform checks (init/validate/plan) results:\n\n{tf_results_str}")
                gcp_logging_client.logger("terraform-checks").log_text(f"Terraform checks for {parent_key} (sub-task {key}):\n{tf_results_str}")
                # Move parent back to In Review and comment with summary and PR link
                summary_text = None
                try:
                    prompt = (
                        f"You are an expert DevOps assistant. Here is a set of Terraform block changes, each with a file and block name. "
                        f"Summarize the overall infrastructure change in 1-2 sentences, focusing on what is being added, removed, or modified. "
                        f"Do NOT include code, only a human-readable summary.\n\n"
                        f"{response}"
                    )
                    summary_text = call_vertex_ai(prompt)
                    logger.info({"event": "summary_generated", "key": parent_key, "summary": summary_text})
                    print(f"[JIRA WEBHOOK] Summary for comment: {summary_text}")
                except Exception as e:
                    logger.error({"event": "summary_error", "key": parent_key, "error": str(e)})
                    print(f"[JIRA WEBHOOK] Error getting summary: {e}")
                    summary_text = "(Could not generate summary)"
                jira_transition_issue(parent_key, "In Review")
                comment = (
                    f"Automated infrastructure change proposed for this ticket (from sub-task {key}).\n\n"
                    f"**Jira Ticket:** {parent_key} (from sub-task {key})\n\n"
                    f"**Summary of changes:**\n{summary_text}\n\n"
                    f"**Review the proposed changes in this PR:** {pr.html_url}\n\n"
                    f"If you have feedback or require changes, please create another sub-task."
                )
                jira_comment_issue(parent_key, comment)
                logger.info({"event": "in_review_transitioned_and_commented", "key": parent_key})
                return {
                    "status": "pr_created_from_subtask",
                    "pr_url": pr.html_url,
                    "issue_key": parent_key,
                    "summary": summary,
                    "description": description,
                    "reporter": reporter
                }
            else:
                logger.info({"event": "ignored_subtask_parent_status", "key": key, "parent_key": parent_key, "parent_status": parent_status})
                print(f"[JIRA WEBHOOK] Parent {parent_key} is not in 'In Review'. Skipping sub-task workflow.")
                return {"status": "ignored", "reason": "parent not in In Review"}
        # Otherwise, fall through to original logic for normal tickets
        if latest_status.strip().lower() != "to do":
            logger.info({"event": "ignored_status", "key": key, "status": latest_status})
            print(f"[JIRA WEBHOOK] Ticket {key} is no longer in 'To Do' (now '{latest_status}'). Skipping agentic workflow.")
            return {"status": "ignored", "reason": f"not in To Do (now {latest_status})"}

        try:
            logger.info({"event": "workflow_triggered", "key": key})
            # Move ticket to In Progress
            jira_transition_issue(key, "In Progress")

            user_prompt = summary or ""
            if description:
                user_prompt += f"\n{description}"
            print(f"[JIRA WEBHOOK] Using user_prompt: {user_prompt}")
            files = fetch_terraform_files()
            logger.info({"event": "files_fetched", "key": key, "files": [f['path'] for f in files]})
            print(f"[JIRA WEBHOOK] Files fetched for context: {[f['path'] for f in files]}")
            response = initial_summary_and_diff(user_prompt, files)
            logger.info({"event": "model_response", "key": key, "response": response})
            print(f"[JIRA WEBHOOK] Model response (block changes):\n{response}")
            user_terraform_change[key] = response
            user_terraform_context[key] = files

            # --- Apply changes and create PR (same as /approve logic) ---
            block_changes = parse_block_changes(response)
            if not block_changes:
                logger.info({"event": "no_block_changes", "key": key})
                print("[JIRA WEBHOOK] No block changes parsed from model response.")
                return {"status": "no_changes", "reason": "No block changes found in model response."}
            updated_files = apply_block_changes(files, block_changes)
            changed = False
            for f in files:
                orig = f['content']
                updated = updated_files.get(f['path'], orig)
                if orig != updated:
                    changed = True
                    break
            if not changed:
                logger.info({"event": "no_actual_changes", "key": key})
                print("[JIRA WEBHOOK] No actual file changes detected. Not creating PR.")
                return {"status": "no_changes", "reason": "No files were changed. The block changes could not be applied or resulted in no changes."}
            print(f"[JIRA WEBHOOK] Preparing to push changes to GitHub...")
            g = Github(GITHUB_TOKEN)
            repo = g.get_repo(GITHUB_REPO)
            base = repo.get_branch(DEFAULT_BRANCH)
            branch_name = f"infra-change-{key}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base.commit.sha)
            commit_message = f"Apply infrastructure change for Jira ticket {key} via chatbot"
            for path, content in updated_files.items():
                print(f"[JIRA WEBHOOK] Committing file: {path}")
                try:
                    f = repo.get_contents(path, ref=branch_name)
                    repo.update_file(path, commit_message, content, f.sha, branch=branch_name)
                except Exception:
                    repo.create_file(path, commit_message, content, branch=branch_name)
            pr = repo.create_pull(
                title=f"Infra change for Jira ticket {key}",
                body="Automated PR from GCP Terraform Chatbot",
                head=branch_name,
                base=DEFAULT_BRANCH
            )
            logger.info({"event": "pr_created", "key": key, "pr_url": pr.html_url, "branch": branch_name})
            print(f"[JIRA WEBHOOK] Pull request created: {pr.html_url}")
            # Run terraform checks and post results
            repo_url = f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com/{GITHUB_REPO}.git"
            tf_results = run_terraform_checks(repo_url, branch_name)
            tf_results_str = format_terraform_check_results(tf_results)
            pr.create_issue_comment(f"Terraform checks (init/validate/plan) results:\n\n{tf_results_str}")
            jira_comment_issue(key, f"Terraform checks (init/validate/plan) results:\n\n{tf_results_str}")
            gcp_logging_client.logger("terraform-checks").log_text(f"Terraform checks for {key}:\n{tf_results_str}")

            # Move ticket to In Review and comment with summary and PR link
            summary_text = None
            try:
                prompt = (
                    f"You are an expert DevOps assistant. Here is a set of Terraform block changes, each with a file and block name. "
                    f"Summarize the overall infrastructure change in 1-2 sentences, focusing on what is being added, removed, or modified. "
                    f"Do NOT include code, only a human-readable summary.\n\n"
                    f"{response}"
                )
                summary_text = call_vertex_ai(prompt)
                logger.info({"event": "summary_generated", "key": key, "summary": summary_text})
                print(f"[JIRA WEBHOOK] Summary for comment: {summary_text}")
            except Exception as e:
                logger.error({"event": "summary_error", "key": key, "error": str(e)})
                print(f"[JIRA WEBHOOK] Error getting summary: {e}")
                summary_text = "(Could not generate summary)"
            jira_transition_issue(key, "In Review")
            comment = (
                f"Automated infrastructure change proposed for this ticket.\n\n"
                f"**Jira Ticket:** {key} - {summary}\n\n"
                f"**Summary of changes:**\n{summary_text}\n\n"
                f"**Review the proposed changes in this PR:** {pr.html_url}\n\n"
                f"If you have feedback or require changes, please comment here."
            )
            jira_comment_issue(key, comment)
            logger.info({"event": "in_review_transitioned_and_commented", "key": key})

            return {
                "status": "pr_created",
                "pr_url": pr.html_url,
                "issue_key": key,
                "summary": summary,
                "description": description,
                "reporter": reporter
            }
        except Exception as e:
            logger.error({"event": "error", "key": key, "error": str(e)})
            print(f"[JIRA WEBHOOK] Error in agentic workflow: {e}")
            return {"status": "error", "error": str(e)}
    else:
        logger.info({"event": "ignored_event", "reason": "not issue_created", "event_type": event_type})
        print("[JIRA WEBHOOK] Ignoring non-issue_created event.")
        return {"status": "ignored", "reason": "not issue_created"}

# Debug endpoint to clear in-memory cache
@app.post("/debug/clear_cache")
def clear_cache():
    user_terraform_change.clear()
    user_terraform_context.clear()
    return {"status": "cleared"}

@app.post("/suggestions/generate")
def manual_generate_suggestions():
    generate_and_post_suggestions()
    return {"status": "manual suggestion generation triggered"}

def run_terraform_checks(repo_url, branch_name, tf_dirs=None):
    """
    Clone the repo, checkout the branch, run terraform init/validate/plan in the 'terraform' directory only.
    Returns a dict: {dir: {init: ..., validate: ..., plan: ...}}
    """
    results = {}
    tempdir = tempfile.mkdtemp(prefix="tfcheck-")
    try:
        # Clone repo
        subprocess.run(["git", "clone", repo_url, tempdir], check=True, capture_output=True)
        subprocess.run(["git", "checkout", branch_name], cwd=tempdir, check=True, capture_output=True)
        tf_dir = os.path.join(tempdir, "terraform")
        if not os.path.isdir(tf_dir):
            results["error"] = f"No 'terraform' directory found in repo root."
            return results
        res = {}
        for cmd in ["init", "validate", "plan"]:
            try:
                if cmd == "plan":
                    proc = subprocess.run(["terraform", cmd, "-no-color"], cwd=tf_dir, capture_output=True, timeout=120)
                else:
                    proc = subprocess.run(["terraform", cmd, "-no-color"], cwd=tf_dir, capture_output=True, timeout=60)
                res[cmd] = proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace")
            except Exception as e:
                res[cmd] = f"Error running terraform {cmd}: {e}"
        results["terraform"] = res
    except Exception as e:
        results["error"] = str(e)
    finally:
        shutil.rmtree(tempdir)
    return results

def format_terraform_check_results(results):
    if not results:
        return "No terraform results."
    if "error" in results:
        return f"Error running terraform checks: {results['error']}"
    out = []
    for d, res in results.items():
        out.append(f"### Terraform checks for `{d}`\n")
        for cmd in ["init", "validate", "plan"]:
            if cmd in res:
                snippet = res[cmd]
                if len(snippet) > 3000:
                    snippet = snippet[:3000] + "\n... (truncated) ..."
                out.append(f"**terraform {cmd}:**\n```")
                out.append(snippet)
                out.append("```")
    return "\n\n".join(out)
