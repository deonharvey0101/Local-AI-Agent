"""
A small SQLite-backed address book stored in "local_agent.db". Supports:
  1. Saving a new contact (name + email)
  2. Looking up a contact by name or email
  3. Renaming a contact that's already saved
  4. Listing every saved contact

main.py calls into these functions whenever it needs to touch the address
book -- nothing else in the project talks to the database file directly.
"""

import re
import sqlite3

# ============================================================================
# SECTION 1: Connect to the database
# ============================================================================

# check_same_thread=False: the agent's async event loop can run tool calls on
# a different thread than the one that opened this connection, and SQLite
# blocks cross-thread use by default. Only one query ever runs at a time here,
# so this is safe.
connection = sqlite3.connect("local_agent.db", check_same_thread=False)
cursor = connection.cursor()

# id: auto-incrementing primary key
# name: the contact's saved display name
# email: UNIQUE so the same address can't be saved twice
# created_at: when the contact was added
cursor.execute("""
    CREATE TABLE IF NOT EXISTS recipients(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")


# ============================================================================
# SECTION 2: Saving a new contact
# ============================================================================

def save_contact(email, name):
    """Add a new name + email pair to the address book."""
    try:
        cursor.execute("""
            INSERT INTO recipients (email, name) VALUES (?, ?)
        """, (email, name))
        connection.commit()
        return f"Contact saved: {name} ({email})"
    except sqlite3.IntegrityError:
        # That email is already saved, possibly under a different name --
        # surface the existing name and point toward rename_contact instead
        # of a dead-end "already exists" message.
        existing = get_contact(email)
        existing_name = existing[1] if existing else "unknown"
        return (f"Contact already exists: {existing_name} ({email}). "
                f"Say 'rename it to ...' if you'd like to change the saved name.")
    except Exception as e:
        return f"Error saving contact: {str(e)}"


# ============================================================================
# SECTION 3: Renaming a contact
# ============================================================================

def rename_contact(identifier, new_name):
    """Change the saved name for a contact found by their current name or
    email address -- their email address itself never changes."""
    matches = find_contacts(identifier)
    if not matches:
        return f"No contact found matching '{identifier}'."
    if len(matches) > 1:
        # Names aren't unique (email is), so a name search can match more than
        # one contact. Refuse to guess which one and list the real options.
        options = ", ".join(f"{name} ({email})" for _id, name, email, _created in matches)
        return (f"Multiple contacts match '{identifier}': {options}. "
                f"Please specify which one by email address instead.")
    email = matches[0][2]
    cursor.execute("UPDATE recipients SET name = ? WHERE email = ?", (new_name, email))
    connection.commit()
    return f"Contact renamed: {new_name} ({email})"


# ============================================================================
# SECTION 4: Looking up contacts
# ============================================================================

# Matches a real email address anywhere inside a longer string -- this lets a
# caller pass something like "Alex (alex@example.com)" and have it resolve to
# that one contact, instead of falling through to a fuzzy name search.
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def find_contacts(query):
    """Return every contact matching `query` -- zero, one, or several.

    If `query` contains an email address anywhere in it, match on that email
    alone (it's unique, so never ambiguous). Otherwise fall back to a fuzzy,
    case-insensitive match against the name, which can return more than one
    contact if two people share a name."""
    email_match = _EMAIL_PATTERN.search(query)
    if email_match:
        cursor.execute("SELECT * FROM recipients WHERE email = ?", (email_match.group(0),))
        return cursor.fetchall()
    cursor.execute("""
        SELECT * FROM recipients WHERE email = ? OR name LIKE ?
    """, (query, f"%{query}%"))
    return cursor.fetchall()


def get_contact(query):
    """Find one contact, for callers that only care about the unambiguous
    case. Returns None both when nothing matches and when more than one
    contact matches -- use find_contacts directly if you need to tell those
    two cases apart."""
    matches = find_contacts(query)
    return matches[0] if len(matches) == 1 else None


# ============================================================================
# SECTION 5: Listing every saved contact
# ============================================================================

def list_all_contacts():
    """Return every contact that's been saved."""
    cursor.execute("SELECT * FROM recipients")
    return cursor.fetchall()
