"""
A personal email assistant that runs in a terminal and talks to a local AI
model via Ollama (nothing leaves your machine except the actual email
send/read requests to Gmail).

Example commands:
    "read my emails"
    "email <name> and say hi"
    "list my contacts"

The AI decides what you want and calls one of the tools defined below to
actually do it -- check your inbox, send an email, look someone up in a
saved address book, etc.

File layout:
    1. Setup                   -- imports and environment setup
    2. Email-safety state       -- tracks the pending-email confirmation flow
    3. Email tools              -- read/compose/send
    4. Contact tools            -- thin wrappers around database.py
    5. Agent setup              -- wires the model, tools, and prompt together
    6. Reply engine             -- streams one exchange with the agent
    7. Main loop                -- the interactive REPL
"""

# ============================================================================
# SECTION 1: Setup
# ============================================================================

import sys
import asyncio

# Windows' console defaults to cp1252, which can't print emoji/curly quotes
# that show up in real email subjects -- force UTF-8 to avoid crashing on them.
sys.stdout.reconfigure(encoding="utf-8")

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import smtplib
from email.mime.text import MIMEText
from langchain_core.tools import tool
import imaplib
import email
from email.header import decode_header
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
import os
from dotenv import load_dotenv

from database import save_contact, get_contact, list_all_contacts, rename_contact, find_contacts

load_dotenv()


# ============================================================================
# SECTION 2: Email-safety state
#
# Sending an email is a one-way action, so the agent always shows a preview
# and waits for explicit confirmation before actually sending. These module-
# level variables track that pending draft between user turns.
# ============================================================================

_pending_email = None
_last_prepared_recipient = None  # kept separately since _pending_email is cleared on send
_prepared_this_turn = False

_current_user_input = ""
_CONFIRM_PHRASES = (
    "yes", "yeah", "yep", "yup", "confirm", "go ahead", "do it", "sure",
    "ok", "okay", "send it", "looks good", "correct", "that's right", "please send",
)

# If the user's message doesn't look like a request for any of the assistant's
# capabilities, skip the tool-calling model and ask for clarification instead.
_ACTION_KEYWORDS = (
    "email", "emails", "inbox", "read", "send", "message", "reply",
    "contact", "contacts", "find", "look", "search", "list", "save", "sent",
)


# ============================================================================
# SECTION 3: Email tools
#
# Each function marked "@tool" is something the agent can call. The AI reads
# the docstring to decide when and how to use it.
# ============================================================================

@tool
def prepare_email(recipient: str = "", subject: str = "", body: str = "", to: str = "") -> str:
    """Compose an email for the user to review -- this does NOT send anything yet.
    `recipient` can be a real email address, OR a saved contact's name -- pass whatever the
    user gave you exactly as they gave it. Do NOT try to resolve a name to an email address
    yourself; this tool looks names up for you, safely. After calling this, show the user
    the exact preview text it returns and wait for them to confirm before calling
    send_pending_email."""
    global _pending_email, _last_prepared_recipient, _prepared_this_turn

    # The model sometimes names this argument "to" instead of "recipient" --
    # accept either to avoid a validation error blocking the whole request.
    recipient = recipient or to
    if not recipient:
        return "Failed to prepare email: no recipient email address was given."

    # Resolve a bare contact name against the database ourselves, rather than
    # relying on the model to look it up correctly first.
    if "@" not in recipient:
        matches = find_contacts(recipient)
        if not matches:
            return (f"Failed to prepare email: '{recipient}' is not a saved contact and is not "
                     "a valid email address. Ask the user for the real email address.")
        if len(matches) > 1:
            # More than one saved contact matches this name -- refuse to guess
            # and surface the real options instead.
            options = ", ".join(f"{name} ({email})" for _id, name, email, _created in matches)
            return (f"Failed to prepare email: multiple saved contacts match '{recipient}': "
                    f"{options}. Ask the user which email address to use.")
        recipient = matches[0][2]  # (id, name, email, created_at)

    if not body.strip():
        return "Failed to prepare email: no message content was given. What would you like the email to say?"

    # Remember this draft so send_pending_email (below) can find it later.
    _pending_email = {"recipient": recipient, "subject": subject, "body": body}
    _last_prepared_recipient = recipient
    _prepared_this_turn = True
    return (
        f"Ready to send to {recipient}:\n"
        f"Subject: {subject or '(none)'}\n"
        f"Body: {body}\n"
        "Reply to confirm sending, or say what to change."
    )


@tool
def send_pending_email() -> str:
    """Actually sends the email that prepare_email already showed the user. Only call this
    after the user has explicitly confirmed the preview -- never on the same turn as
    prepare_email. Takes no arguments: it sends exactly what was already previewed."""
    global _pending_email

    if _pending_email is None:
        return "Nothing to send -- there's no prepared email waiting for confirmation. Call prepare_email first."

    # A draft can only be confirmed in a later turn than the one that created it.
    if _prepared_this_turn:
        return ("Cannot send yet -- this email was just prepared in this same turn. "
                "Show the user the preview and wait for their explicit confirmation in "
                "a separate message before calling send_pending_email again.")

    # The user's latest message must contain a real, recognizable confirmation --
    # this is what prevents an unrelated or unclear message from being misread
    # as "yes, send it."
    if not any(phrase in _current_user_input.lower() for phrase in _CONFIRM_PHRASES):
        return ("Cannot send -- the user's last message didn't contain a clear "
                "confirmation. Ask them to explicitly confirm (e.g. 'yes, send it') "
                "before calling send_pending_email again.")

    recipient = _pending_email["recipient"]
    subject = _pending_email["subject"]
    body = _pending_email["body"]

    sender_email = os.getenv("GMAIL_USER")
    sender_password = os.getenv("GMAIL_PASSWORD")

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient, msg.as_string())
        _pending_email = None
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {str(e)}"


@tool
def read_emails(max_emails: int = 10, folder: str = "inbox") -> str:
    """Read recent emails.

    folder="inbox" (default): the most recent emails from your Primary inbox category, read or not.
    folder="sent": the most recent emails YOU sent, from the Sent Mail folder.
    """
    try:
        max_emails = int(max_emails)
    except Exception:
        max_emails = 10

    # Normalize folder so the model can say "sent"/"Sent"/"sent mail"/etc. and still match.
    is_sent = str(folder).strip().lower().startswith("sent")

    username = os.getenv("GMAIL_USER")
    password = os.getenv("GMAIL_PASSWORD")

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, password)

        if is_sent:
            # Gmail's Sent folder is a real IMAP folder, so a plain "ALL" search works.
            mail.select('"[Gmail]/Sent Mail"')
            status, messages = mail.search(None, "ALL")
        else:
            mail.select("inbox")
            # Plain IMAP search covers the whole inbox, including Social/Promotions/
            # Updates -- those tabs are a Gmail-only label, not a real IMAP folder.
            # X-GM-RAW lets us use Gmail's own search syntax to filter to Primary.
            status, messages = mail.search(None, "X-GM-RAW", '"category:primary"')

        if status != "OK":
            return "Failed to search emails."

        raw_ids = messages[0]
        if isinstance(raw_ids, bytes):
            raw_ids = raw_ids.decode()
        email_ids = raw_ids.split()

        if not email_ids:
            return "No sent emails found." if is_sent else "No emails found in Primary."

        # IMAP returns ids oldest-first; the last max_emails are the most recent,
        # and reversed() lists them newest-first.
        output = []
        for i in reversed(email_ids[-max_emails:]):
            status, msg_data = mail.fetch(i, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or "utf-8")
                    if is_sent:
                        output.append((msg.get("To"), subject))
                    else:
                        output.append((msg.get("From"), subject))

        mail.logout()

        # Pre-number the list so the model only has to relay these lines, not
        # re-parse or re-count them.
        label = "To" if is_sent else "From"
        lines = [f"{idx}. {label}: {who} | Subject: {subject}" for idx, (who, subject) in enumerate(output, start=1)]
        return f"Found {len(lines)} email(s):\n" + "\n".join(lines)

    except Exception as e:
        return f"Failed to read emails: {str(e)}"


# ============================================================================
# SECTION 4: Contact tools
#
# Thin wrappers around the address-book functions in database.py.
# ============================================================================

@tool
def save_contact_tool(email: str, name: str) -> str:
    """Save a new email contact to the database."""
    return save_contact(email, name)


@tool
def rename_contact_tool(identifier: str, new_name: str) -> str:
    """Rename an existing saved contact. `identifier` can be their current name or their
    email address. Use this when the user asks to rename/relabel a contact, or when
    save_contact_tool says a contact already exists under a different name than they wanted."""
    return rename_contact(identifier, new_name)


@tool
def get_contact_tool(query: str) -> str:
    """Retrieve a contact from the database by exact email OR by (partial) name. If the
    query matches more than one saved contact, this lists all of them instead of picking
    one -- relay that list to the user and ask them to specify by email address."""
    matches = find_contacts(query)
    if not matches:
        return "Contact not found."
    if len(matches) > 1:
        options = "\n".join(f"- {name} ({email})" for _id, name, email, _created in matches)
        return f"Multiple contacts match '{query}':\n{options}\nPlease specify by email address."
    contact = matches[0]
    return f"Contact found: {contact[1]} ({contact[2]})"


@tool
def list_contacts_tool() -> str:
    """List all contacts in the database."""
    contacts = list_all_contacts()
    if contacts:
        return "\n".join([f"{contact[1]} ({contact[2]})" for contact in contacts])
    return "No contacts found."


# ============================================================================
# SECTION 5: Agent setup
#
# Picks the model, gives it the tools above, loads its instructions from
# system_prompt.txt, and wires it all into something that can hold a
# conversation and call tools when needed.
# ============================================================================

model = ChatOllama(model="llama3.1:8b", temperature=0)  # solid native tool-calling support

tools = [prepare_email, send_pending_email, read_emails, save_contact_tool, get_contact_tool, list_contacts_tool, rename_contact_tool]

# If the model calls a tool with the wrong argument name, return the error as
# text instead of letting a validation error crash the whole run -- the model
# can then retry with the correct argument name.
for t in tools:
    t.handle_validation_error = True

try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt_text = f.read().strip()
except FileNotFoundError:
    print("Warning: system_prompt.txt not found. Using default text.")
    system_prompt_text = "You are a helpful AI assistant."


template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    MessagesPlaceholder("chat_history"),  # prior turns, so the model has conversation context
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # filled in by AgentExecutor during tool calls
])


# create_tool_calling_agent wires the model + tools + prompt into an agent that
# can decide to call a tool instead of just replying with text. AgentExecutor
# actually runs that agent: it calls the model, runs any requested tool, feeds
# the result back, and repeats until the model gives a final answer.
agent = create_tool_calling_agent(model, tools, template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# These tools' output is meant to be shown to the user as-is. Skipping the
# model's second "paraphrase" pass for them avoids it rewriting or dropping
# parts of an already-correct answer (a list, a lookup result, an email preview).
RELAY_VERBATIM_TOOLS = {"list_contacts_tool", "get_contact_tool", "read_emails", "save_contact_tool", "prepare_email", "rename_contact_tool"}


# ============================================================================
# SECTION 6: Reply engine
#
# Streams one exchange with the agent, printing the model's answer as it's
# generated rather than waiting for the whole response.
# ============================================================================

async def stream_agent_reply(user_input, chat_history) -> str:
    """Run one turn and print the model's answer as it's generated.

    A turn can involve two LLM calls: one where the model decides to call a
    tool (no visible text -- the "answer" is the tool call itself), and a
    second one where it writes its reply after seeing the tool's result.

    For tools in RELAY_VERBATIM_TOOLS, we print that tool's raw output the
    moment it's available and skip the second LLM call entirely -- both to
    avoid the model altering an already-correct answer, and because that
    second call would otherwise run for nothing.

    send_pending_email gets similar treatment: whether the recipient is
    already a saved contact is a plain database fact, so we check it directly
    and skip the "want to save this contact?" follow-up when it's not needed,
    and force the tool's own message through whenever the send failed.
    """
    global _prepared_this_turn, _current_user_input
    _prepared_this_turn = False
    _current_user_input = user_input
    had_pending_before = _pending_email is not None

    # Skip the tool-calling model entirely for input that doesn't look like
    # any of the assistant's capabilities, unless a confirmation is pending.
    if not had_pending_before and not any(k in user_input.lower() for k in _ACTION_KEYWORDS):
        clarification = ("I'm not sure what you meant by that. Try something like "
                          "'read my emails', 'email <name> and say ...', "
                          "'find <name>'s email', or 'list my contacts'.")
        print(f"Agent: {clarification}\n")
        return clarification

    print("Agent: ", end="", flush=True)
    full_reply = ""
    async for event in agent_executor.astream_events(
        {"input": user_input, "chat_history": chat_history}, version="v2"
    ):
        override = None
        if had_pending_before and event["event"] == "on_tool_end" and event.get("name") == "read_emails":
            # A confirmation reply like "send it" can occasionally get misread
            # as a request to check sent mail -- suppress that output and keep
            # listening for the real result instead of showing an unrelated inbox dump.
            continue
        elif event["event"] == "on_tool_end" and event.get("name") in RELAY_VERBATIM_TOOLS:
            override = event["data"]["output"]
        elif event["event"] == "on_tool_end" and event.get("name") == "send_pending_email":
            tool_output = event["data"]["output"]
            failed = not tool_output.startswith("Email sent successfully")
            if failed or (_last_prepared_recipient and get_contact(_last_prepared_recipient)):
                override = tool_output
            else:
                # Succeeded, for a brand-new (not-yet-saved) contact. Print the
                # confirmation immediately without stopping the loop, so
                # whatever the model does next (ask to save, or save directly)
                # is appended after it rather than replacing it.
                print(tool_output + " ", end="", flush=True)
                full_reply += tool_output + " "
                await asyncio.sleep(0.02)
                continue

        if override is not None:
            for word in override.split(" "):
                print(word + " ", end="", flush=True)
                await asyncio.sleep(0.02)  # small pause per word so it still reads as a stream
            full_reply += override
            break

        if event["event"] == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                print(token, end="", flush=True)
                full_reply += token

    print("\n")
    return full_reply


# ============================================================================
# SECTION 7: Main loop
# ============================================================================

async def main():
    messages = []  # conversation history for this session

    while True:
        try:
            user_input = input("You: ")

            if user_input.strip().lower() in ("quit", "exit", "q"):
                print("Exiting.")
                sys.exit(0)

            agent_output = await stream_agent_reply(user_input, messages)

            messages.append(HumanMessage(content=user_input))
            messages.append(AIMessage(content=agent_output))

        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)


asyncio.run(main())
