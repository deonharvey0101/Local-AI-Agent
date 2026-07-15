# Local Email Agent — Phase 1

A personal email assistant that runs entirely on your own machine. It talks to
a local LLM (via [Ollama](https://ollama.com)) in a terminal chat loop, and
uses tool-calling to read your Gmail inbox, compose and send emails, and
manage a small saved-contacts address book — all backed by SQLite.

Nothing leaves your machine except the actual Gmail API calls needed to read
or send mail. The model itself runs locally.

## Features

- **Read email** — pulls the most recent messages from your Primary inbox or
  your Sent folder.
- **Send email** — composes a draft and shows you an exact preview before
  anything goes out; a real send only happens after you explicitly confirm.
- **Contacts** — save, look up, list, and rename contacts by name or email,
  with duplicate and ambiguous-name handling (e.g. two contacts sharing the
  same name).
- **Streaming replies** — the agent's responses print as they're generated,
  not all at once at the end.

## How it works

- `main.py` — defines the agent's tools (email + contacts), wires up the
  local model and prompt, and runs the interactive chat loop.
- `database.py` — a small SQLite-backed address book (save / find / rename /
  list contacts).
- `system_prompt.txt` — the instructions that shape how the model uses its
  tools and responds.

The email-sending flow is deliberately split into two tools — `prepare_email`
(composes and previews, never sends) and `send_pending_email` (sends exactly
what was already previewed, takes no arguments) — so a real send can never
happen without an explicit, separate confirmation step.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally, with a
  tool-calling-capable model pulled:
  ```
  ollama pull llama3.1:8b
  ```
- A Gmail account with an [App Password](https://myaccount.google.com/apppasswords)
  (regular account passwords won't work with Gmail's SMTP/IMAP over
  `smtplib`/`imaplib`)

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your Gmail address and app
   password:
   ```
   GMAIL_USER=your_email@gmail.com
   GMAIL_PASSWORD=your_app_password
   ```
3. Run it:
   ```
   python main.py
   ```

A `local_agent.db` SQLite file is created automatically on first run to store
contacts.

## Example session

```
You: read my emails
Agent: Found 3 email(s):
1. From: ... | Subject: ...
...

You: email Alex and say hello
Agent: Ready to send to alex@example.com:
Subject: (none)
Body: hello
Reply to confirm sending, or say what to change.

You: yes
Agent: Email sent successfully!
```

## Notes

- This assistant is designed around models that support native tool-calling.
  Smaller or non-tool-tuned models may not reliably call the right tool or
  extract the full intended message content — `llama3.1:8b` has proven
  reliable in testing.
- `.env` and `local_agent.db` are git-ignored — never commit real credentials
  or personal contact data.
