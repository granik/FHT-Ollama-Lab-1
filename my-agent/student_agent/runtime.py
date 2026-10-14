"""Tool registry and permission boundary."""

from __future__ import annotations

import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request


TOOL_SPECS = {
    "list_files": {
        "arguments": (),
        "description": "List up to 200 existing workspace files.",
        "handler": "_list_files",
        "mutating": False,
    },
    "read_file": {
        "arguments": ("path",),
        "description": "Read up to 12,000 UTF-8 characters from an existing file.",
        "handler": "_read_file",
        "mutating": False,
    },
    "search_files": {
        "arguments": ("query",),
        "description": "Search up to 200 text files for a literal string.",
        "handler": "_search_files",
        "mutating": False,
    },
    "write_file": {
        "arguments": ("path", "content"),
        "description": "Create a new file; fail if it already exists.",
        "handler": "_write_file",
        "mutating": True,
    },
    "edit_file": {
        "arguments": ("path", "old", "new"),
        "description": (
            "Replace one exact unique non-empty old string in an existing file; "
            "new must be different from old."
        ),
        "argument_schema": {
            "old": {
                "minLength": 1,
                "description": (
                    "Non-empty exact text copied from the latest read_file result; "
                    "it must occur exactly once."
                ),
            },
            "new": {
                "description": "Replacement text; it must be different from old."
            },
        },
        "handler": "_edit_file",
        "mutating": True,
    },
    "fetch_url": {
        "arguments": ("url",),
        "description": "Read approved HTTPS documentation from docs.python.org.",
        "handler": "_fetch_url",
        "mutating": False,
    },
    "bash": {
        "arguments": ("command",),
        "description": "Request human approval to run one local command.",
        "handler": "_bash",
        "mutating": True,
    },
}

TOOL_ARGUMENTS = {
    name: spec["arguments"] for name, spec in TOOL_SPECS.items()
}

PROTECTED_FILES = {
    "orders/validation.py",
    "tests/test_acceptance.py",
    "docs/business-rules.md",
}
MUTATING_TOOLS = {
    name for name, specification in TOOL_SPECS.items() if specification["mutating"]
}
MAX_TEXT = 12_000


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Runtime:
    """Validate and execute model-requested tools inside a workspace."""

    def __init__(self, root, mode="read-only", enabled=None, approve=None, offline=False):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Workspace root is not a directory: {self.root}")
        if mode not in {"read-only", "edit"}:
            raise ValueError("mode must be 'read-only' or 'edit'")
        self.mode = mode
        self.enabled = set(TOOL_SPECS if enabled is None else enabled)
        self.approve = approve or (lambda _command, _cwd: False)
        if offline is True:
            offline = "decimal"
        if offline not in {False, None, "decimal", "web"}:
            raise ValueError("offline must be False, 'decimal', or 'web'")
        self.offline = offline or False

    def available_tool_names(self):
        """Return tools allowed by both the enabled set and current mode."""
        names = self.enabled.intersection(TOOL_SPECS)
        if self.mode == "read-only":
            names.difference_update(MUTATING_TOOLS)
        return names

    def tool_catalog(self):
        """Return the advertised data-driven registry for this runtime."""
        return {
            name: {
                "arguments": TOOL_SPECS[name]["arguments"],
                "description": TOOL_SPECS[name]["description"],
                "argument_schema": TOOL_SPECS[name].get("argument_schema", {}),
                "function": getattr(self, TOOL_SPECS[name]["handler"]),
            }
            for name in self.available_tool_names()
        }

    def _resolve_path(self, value):
        if not isinstance(value, str) or not value:
            return None
        candidate = Path(value)
        if candidate.is_absolute() or any(part == ".." or part.startswith(".") for part in candidate.parts):
            return None
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return None
        return resolved

    def _is_protected(self, path: Path) -> bool:
        return path.relative_to(self.root).as_posix() in PROTECTED_FILES

    @staticmethod
    def _error(message):
        return {"status": "error", "output": message}

    @staticmethod
    def _denied(message):
        return {"status": "denied", "output": message}

    def execute(self, action):
        """Validate, authorize, and dispatch one model action."""
        if not isinstance(action, dict) or set(action) != {"tool", "args"}:
            return self._error("Action must contain exactly 'tool' and 'args'.")
        tool, args = action["tool"], action["args"]
        if not isinstance(tool, str) or tool not in TOOL_SPECS:
            return self._error(f"Unknown tool: {tool!r}")
        if tool not in self.enabled:
            return self._denied(f"Tool {tool} is not enabled.")
        expected = TOOL_SPECS[tool]["arguments"]
        if not isinstance(args, dict) or set(args) != set(expected):
            return self._error(
                f"Invalid arguments for {tool}; expected {list(expected)}."
            )
        if any(not isinstance(value, str) or len(value) > MAX_TEXT for value in args.values()):
            return self._error("Every argument must be a string of at most 12,000 characters.")
        if self.mode == "read-only" and TOOL_SPECS[tool]["mutating"]:
            return self._denied(f"Tool {tool} requires edit mode.")

        try:
            function = getattr(self, TOOL_SPECS[tool]["handler"])
            return function(**args)
        except (OSError, UnicodeError, urllib.error.URLError, subprocess.SubprocessError) as error:
            return self._error(str(error))

    def _iter_files(self):
        seen = 0
        for directory, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not name.startswith(".") and not (Path(directory) / name).is_symlink()
            )
            for name in sorted(filenames):
                path = Path(directory) / name
                if name.startswith(".") or path.is_symlink():
                    continue
                yield path
                seen += 1
                if seen >= 200:
                    return

    def _list_files(self):
        output = [path.relative_to(self.root).as_posix() for path in self._iter_files()]
        return {"status": "ok", "output": output}

    def _read_file(self, path):
        path = self._resolve_path(path)
        if path is None:
            return self._denied("Path is outside the workspace or hidden.")
        if not path.is_file():
            return self._error(
                "File does not exist. Do not repeat read_file; use write_file to create it."
            )
        with path.open("r", encoding="utf-8") as handle:
            text = handle.read(MAX_TEXT + 1)
        truncated = len(text) > MAX_TEXT
        text = text[:MAX_TEXT]
        if truncated:
            text += "\n[TRUNCATED at 12,000 characters]"
        return {"status": "ok", "output": text}

    def _search_files(self, query):
        if not query:
            return self._error("Search query must not be empty.")
        matches = []
        for path in self._iter_files():
            try:
                with path.open("rb") as binary:
                    sample = binary.read(1024)
                if b"\x00" in sample:
                    continue
                with path.open("r", encoding="utf-8") as text:
                    for number, line in enumerate(text, 1):
                        if query in line:
                            matches.append(
                                {
                                    "path": path.relative_to(self.root).as_posix(),
                                    "line": number,
                                    "excerpt": line.rstrip("\r\n")[:240],
                                }
                            )
                            if len(matches) >= 50:
                                return {"status": "ok", "output": matches}
            except (OSError, UnicodeDecodeError):
                continue
        return {"status": "ok", "output": matches}

    def _write_file(self, path, content):
        path = self._resolve_path(path)
        if path is None:
            return self._denied("Path is outside the workspace or hidden.")
        if self._is_protected(path):
            return self._denied("File is course-owned and protected.")
        if len(content.encode("utf-8")) > MAX_TEXT:
            return self._error("New file exceeds 12,000 bytes.")
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
        except FileExistsError:
            return self._error(
                "File already exists. Do not retry write_file; read it and use edit_file."
            )
        return {"status": "ok", "output": "File created."}

    def preview_edit(self, path, old, new):
        """Return the exact prospective edit without changing the file."""
        path = self._resolve_path(path)
        if path is None:
            return self._denied("Path is outside the workspace or hidden.")
        if self._is_protected(path):
            return self._denied("File is course-owned and protected.")
        if not old:
            return self._error(
                "The old text must not be empty. Read the current file, copy a "
                "small exact non-empty snippet into old, and do not retry unchanged."
            )
        if old == new:
            return self._error(
                "Old and new text are identical; no edit would occur. The file is "
                "unchanged, so do not reread it. Keep the exact old text and send a "
                "meaningfully different replacement."
            )
        raw = path.read_bytes()
        if len(raw) > MAX_TEXT:
            return self._error("File exceeds 12,000 bytes.")
        text = raw.decode("utf-8")
        count = text.count(old)
        if count == 0:
            for width in (4, 8, 12, 16):
                indentation = " " * width
                indented_old = "\n".join(
                    (indentation + line) if line else line
                    for line in old.splitlines()
                )
                if indented_old == old or text.count(indented_old) != 1:
                    continue
                old = indented_old
                new = "\n".join(
                    (indentation + line) if line else line
                    for line in new.splitlines()
                )
                count = 1
                break
        if count != 1:
            return self._error(
                f"Old text occurs {count} times; expected exactly once. Re-read the "
                "file and copy a small exact current snippet; do not retry unchanged."
            )
        updated = text.replace(old, new, 1)
        if len(updated.encode("utf-8")) > MAX_TEXT:
            return self._error("Edited file would exceed 12,000 bytes.")
        return {
            "status": "ok",
            "output": {"path": path, "current": text, "updated": updated},
        }

    def _edit_file(self, path, old, new):
        preview = self.preview_edit(path, old, new)
        if preview.get("status") != "ok":
            return preview
        output = preview["output"]
        if str(output["path"]).endswith(".py"):
            try:
                compile(output["updated"], str(output["path"]), "exec")
            except SyntaxError as error:
                return self._error(
                    f"Edited Python file would have a syntax error on line "
                    f"{error.lineno}: {error.msg}. The file was not changed."
                )
        output["path"].write_text(output["updated"], encoding="utf-8")
        if output["path"].suffix == ".py":
            cache = output["path"].parent / "__pycache__"
            if cache.is_dir():
                for bytecode in cache.glob(output["path"].stem + ".*.pyc"):
                    try:
                        bytecode.unlink()
                    except OSError:
                        pass
        return {"status": "ok", "output": "File edited."}

    def _validate_url(self, value):
        try:
            parsed = urllib.parse.urlsplit(value)
            port = parsed.port
        except ValueError:
            return None
        if (
            parsed.scheme != "https"
            or parsed.hostname != "docs.python.org"
            or parsed.username is not None
            or parsed.password is not None
            or port not in {None, 443}
            or parsed.query
            or parsed.fragment
        ):
            return None
        return parsed.geturl()

    def _fetch_url(self, url):
        url = self._validate_url(url)
        if url is None:
            return self._denied("URL must be approved HTTPS documentation on docs.python.org.")
        if self.offline:
            fixture_name = (
                "web-injected.txt" if self.offline == "web" else "decimal-offline.txt"
            )
            fixture = self.root.parent / "fixtures" / fixture_name
            if not fixture.is_file():
                return self._error(f"Offline fixture not found: {fixture}")
            raw = fixture.read_bytes()[: MAX_TEXT + 1]
            truncated = len(raw) > MAX_TEXT
            return {
                "status": "ok",
                "output": {
                    "url": url,
                    "content": raw[:MAX_TEXT].decode("utf-8"),
                    "truncated": truncated,
                    "source": "offline fixture",
                },
            }

        opener = urllib.request.build_opener(_NoRedirect())
        request = urllib.request.Request(url, headers={"User-Agent": "week-01-agent/1"})
        try:
            with opener.open(request, timeout=10) as response:
                media_type = response.headers.get_content_type()
                if media_type != "text/html" and not media_type.startswith("text/"):
                    return self._error(f"Unsupported content type: {media_type}")
                raw = response.read(MAX_TEXT + 1)
        except urllib.error.HTTPError as error:
            return self._error(f"HTTP {error.code}: {error.reason}")
        return {
            "status": "ok",
            "output": {
                "url": url,
                "content": raw[:MAX_TEXT].decode("utf-8", errors="replace"),
                "truncated": len(raw) > MAX_TEXT,
                "source": "live docs.python.org response",
            },
        }

    def _stop_process(self, process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    def _bash(self, command):
        print(f"Bash command: {command}")
        print(f"Working directory: {self.root}")
        if not self.approve(command, str(self.root)):
            return self._denied("Bash command denied by user.")
        process = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc", "-c", command],
            cwd=self.root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"},
            start_new_session=True,
        )
        output = bytearray()
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + 20
        try:
            while selector.get_map():
                if time.monotonic() >= deadline:
                    self._stop_process(process)
                    return self._error("Command timed out after 20 seconds.")
                for key, _ in selector.select(timeout=0.2):
                    chunk = os.read(key.fd, 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output.extend(chunk)
                    if len(output) > MAX_TEXT:
                        self._stop_process(process)
                        return self._error("Command output exceeded 12,000 bytes.")
            try:
                return_code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                self._stop_process(process)
                return self._error("Command timed out after 20 seconds.")
        except KeyboardInterrupt:
            self._stop_process(process)
            raise
        finally:
            selector.close()
            if process.stdout is not None:
                process.stdout.close()
        return {
            "status": "ok",
            "output": {
                "exit_code": return_code,
                "text": bytes(output).decode("utf-8", errors="replace"),
            },
        }
