# admin_dashboard.py
# Streamlit admin dashboard for managing the KARE FAQ dataset.
#
# Run from the project root:
#   streamlit run backend/admin_dashboard.py
#
# Features
# --------
# - View / search / filter all FAQ entries
# - Add new entries
# - Edit existing entries inline
# - Delete entries with confirmation
# - Persist changes to kare_faq.json (atomic write + backup)
# - Trigger ChromaDB reindexing after modifications

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the backend package root is importable
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import FAQ_FILE, VECTOR_DB_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File helpers — atomic save with backup
# ---------------------------------------------------------------------------


def _load_faq(path: str = FAQ_FILE) -> list[dict]:
    """Load the FAQ dataset from disk."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_faq(entries: list[dict], path: str = FAQ_FILE) -> None:
    """
    Persist *entries* to disk **atomically**.

    1. Write to a temp file in the same directory.
    2. Back up the current file (``kare_faq.json.bak``).
    3. Rename the temp file over the original.

    This prevents half-written files if the process is interrupted.
    """
    directory = os.path.dirname(path)
    backup_path = path + ".bak"

    # Step 1 — write to a temporary file (same directory for same-device rename)
    fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
            json.dump(entries, tmp_f, indent=2, ensure_ascii=False)

        # Step 2 — create backup of current file
        if os.path.exists(path):
            shutil.copy2(path, backup_path)

        # Step 3 — atomic rename (same filesystem)
        shutil.move(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _next_id(entries: list[dict]) -> int:
    """Return the next available entry ID."""
    if not entries:
        return 1
    return max(e["id"] for e in entries) + 1


# ---------------------------------------------------------------------------
# ChromaDB reindexing helper
# ---------------------------------------------------------------------------


def _trigger_reindex() -> str:
    """
    Delete the existing ChromaDB collection and reindex from the
    knowledge base.  Returns a status message.
    """
    try:
        import chromadb
        from chromadb.config import Settings

        if os.path.exists(VECTOR_DB_DIR):
            client = chromadb.PersistentClient(
                path=VECTOR_DB_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
            # Delete all collections so next app startup reindexes
            for col in client.list_collections():
                client.delete_collection(col.name)
            return "ChromaDB collections cleared. Embeddings will be regenerated on next backend startup."
        return "ChromaDB directory not found — nothing to clear."
    except Exception as exc:
        return f"Reindex error: {exc}"


# ---------------------------------------------------------------------------
# Streamlit helpers
# ---------------------------------------------------------------------------

_CATEGORIES = [
    "General Information",
    "Admissions",
    "Courses & Departments",
    "Fees & Scholarships",
    "Hostel & Facilities",
    "Placements & Internships",
    "Research & Publications",
    "Sports & Extracurricular",
    "Student Life & Clubs",
    "Transportation & Connectivity",
]


def _get_categories(entries: list[dict]) -> list[str]:
    """Return sorted unique categories from entries + known defaults."""
    cats = {e.get("category", "") for e in entries if e.get("category")}
    cats.update(_CATEGORIES)
    return sorted(cats)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="KARE FAQ Admin",
    page_icon="⚙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state — load entries once
# ---------------------------------------------------------------------------

if "faq_entries" not in st.session_state:
    st.session_state.faq_entries = _load_faq()

if "dirty" not in st.session_state:
    st.session_state.dirty = False

entries: list[dict] = st.session_state.faq_entries


# ---------------------------------------------------------------------------
# Sidebar — stats & actions
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ KARE FAQ Admin")
    st.metric("Total entries", len(entries))

    st.divider()

    # Save button
    if st.button("💾  Save to disk", type="primary", use_container_width=True):
        _save_faq(entries)
        st.session_state.dirty = False
        st.success("Saved successfully!")

    if st.session_state.dirty:
        st.warning("You have unsaved changes.")

    st.divider()

    # Reindex button
    if st.button("🔄  Reindex ChromaDB", use_container_width=True):
        msg = _trigger_reindex()
        st.info(msg)

    st.divider()

    # Reload from disk
    if st.button("📂  Reload from disk", use_container_width=True):
        st.session_state.faq_entries = _load_faq()
        st.session_state.dirty = False
        st.rerun()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_view, tab_add, tab_edit, tab_delete = st.tabs([
    "📋 View / Search", "➕ Add Entry", "✏️ Edit Entry", "🗑️ Delete Entry",
])

# ========================== VIEW / SEARCH ==================================

with tab_view:
    st.subheader("Browse FAQ Entries")

    col_search, col_cat = st.columns([2, 1])
    with col_search:
        search_term = st.text_input(
            "Search", placeholder="Type to filter by question or answer…"
        )
    with col_cat:
        cat_filter = st.selectbox(
            "Category", ["All"] + _get_categories(entries)
        )

    filtered = entries
    if cat_filter != "All":
        filtered = [e for e in filtered if e.get("category") == cat_filter]
    if search_term:
        term_lower = search_term.lower()
        filtered = [
            e for e in filtered
            if term_lower in e.get("question", "").lower()
            or term_lower in e.get("answer", "").lower()
        ]

    st.caption(f"Showing {len(filtered)} of {len(entries)} entries")

    for entry in filtered:
        with st.expander(f"**#{entry['id']}** — {entry.get('question', 'N/A')}", expanded=False):
            st.markdown(f"**Category:** {entry.get('category', 'N/A')}")
            st.markdown(f"**Answer:** {entry.get('answer', 'N/A')}")
            st.markdown(f"**Source:** {entry.get('source_reference', 'N/A')}")


# ========================== ADD ENTRY ======================================

with tab_add:
    st.subheader("Add New FAQ Entry")

    with st.form("add_form", clear_on_submit=True):
        new_cat = st.selectbox("Category", _get_categories(entries), key="add_cat")
        new_question = st.text_area("Question", key="add_q")
        new_answer = st.text_area("Answer", key="add_a")
        new_source = st.text_input(
            "Source Reference", value="", key="add_src"
        )
        submitted = st.form_submit_button("Add Entry", type="primary")

    if submitted:
        if not new_question.strip() or not new_answer.strip():
            st.error("Question and Answer are required.")
        else:
            new_entry = {
                "id": _next_id(entries),
                "category": new_cat,
                "question": new_question.strip(),
                "answer": new_answer.strip(),
                "source_reference": new_source.strip() or "Admin Dashboard",
            }
            entries.append(new_entry)
            st.session_state.dirty = True
            st.success(f"Entry #{new_entry['id']} added!")
            st.rerun()


# ========================== EDIT ENTRY =====================================

with tab_edit:
    st.subheader("Edit Existing Entry")

    if not entries:
        st.info("No entries to edit.")
    else:
        id_options = {f"#{e['id']} — {e.get('question', '')[:60]}": e["id"] for e in entries}
        selected_label = st.selectbox("Select entry", list(id_options.keys()), key="edit_select")
        selected_id = id_options[selected_label]

        entry_to_edit = next(e for e in entries if e["id"] == selected_id)
        idx = entries.index(entry_to_edit)

        with st.form("edit_form"):
            edit_cat = st.selectbox(
                "Category",
                _get_categories(entries),
                index=_get_categories(entries).index(entry_to_edit.get("category", _CATEGORIES[0]))
                if entry_to_edit.get("category") in _get_categories(entries)
                else 0,
                key="edit_cat",
            )
            edit_question = st.text_area(
                "Question",
                value=entry_to_edit.get("question", ""),
                key="edit_q",
            )
            edit_answer = st.text_area(
                "Answer",
                value=entry_to_edit.get("answer", ""),
                key="edit_a",
            )
            edit_source = st.text_input(
                "Source Reference",
                value=entry_to_edit.get("source_reference", ""),
                key="edit_src",
            )
            save_edit = st.form_submit_button("Save Changes", type="primary")

        if save_edit:
            if not edit_question.strip() or not edit_answer.strip():
                st.error("Question and Answer cannot be empty.")
            else:
                entries[idx] = {
                    "id": selected_id,
                    "category": edit_cat,
                    "question": edit_question.strip(),
                    "answer": edit_answer.strip(),
                    "source_reference": edit_source.strip() or entry_to_edit.get("source_reference", ""),
                }
                st.session_state.dirty = True
                st.success(f"Entry #{selected_id} updated!")
                st.rerun()


# ========================== DELETE ENTRY ===================================

with tab_delete:
    st.subheader("Delete Entry")

    if not entries:
        st.info("No entries to delete.")
    else:
        del_options = {f"#{e['id']} — {e.get('question', '')[:60]}": e["id"] for e in entries}
        del_label = st.selectbox("Select entry to delete", list(del_options.keys()), key="del_select")
        del_id = del_options[del_label]
        del_entry = next(e for e in entries if e["id"] == del_id)

        st.warning(f"**You are about to delete entry #{del_id}:**")
        st.markdown(f"> **Q:** {del_entry.get('question', 'N/A')}")
        st.markdown(f"> **A:** {del_entry.get('answer', 'N/A')}")

        col1, col2 = st.columns(2)
        with col1:
            confirm = st.checkbox("I confirm this deletion", key="del_confirm")
        with col2:
            if st.button("🗑️ Delete", type="primary", disabled=not confirm):
                entries[:] = [e for e in entries if e["id"] != del_id]
                st.session_state.dirty = True
                st.success(f"Entry #{del_id} deleted.")
                st.rerun()
