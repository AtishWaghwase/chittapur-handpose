"""
Native macOS UI via AppleScript (osascript). Avoids Python's Tk on macOS, which can abort
with Tcl/Tk framework version checks (e.g. "macOS 26 or later required").
"""

from __future__ import annotations

import subprocess
from typing import List, Optional


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def choose_from_list(
    items: List[str],
    prompt: str,
    *,
    default_first: bool = True,
    default_item: Optional[str] = None,
) -> Optional[str]:
    """Show a Cocoa list dialog. Returns selected string, None if cancelled or empty."""
    if not items:
        return None
    quoted = ", ".join(f'"{_escape(x)}"' for x in items)
    default_clause = ""
    if default_item is not None and default_item in items:
        default_clause = f'default items {{"{_escape(default_item)}"}}'
    elif default_first:
        default_clause = f'default items {{"{_escape(items[0])}"}}'
    script = f'''
set theList to {{{quoted}}}
set c to choose from list theList with prompt "{_escape(prompt)}" {default_clause}
if c is false then
  return ""
end if
return item 1 of c
'''
    out = _run_osascript(script)
    return out if out else None


def choose_image_files(prompt: str = "Select training images") -> List[str]:
    """Multi-file open dialog; returns POSIX paths. Empty if cancelled."""
    script = f'''
try
  set theFiles to choose file with prompt "{_escape(prompt)}" with multiple selections allowed without invisibles
on error
  return ""
end try
set out to ""
repeat with f in theFiles
  set out to out & POSIX path of f & (ASCII character 10)
end repeat
return out
'''
    out = _run_osascript(script)
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def alert(message: str, title: str = "Chittapur") -> None:
    script = f'''
display alert "{_escape(title)}" message "{_escape(message)}" as informational
'''
    subprocess.run(["osascript", "-e", script], capture_output=True, check=False)


def _run_osascript(script: str) -> str:
    r = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        return ""
    return r.stdout.strip()
