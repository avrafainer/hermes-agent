"""Local OpenRouter budget checks, not a guaranteed account-wide hard cap.

Provider reporting can lag and concurrent requests have no shared reservation.
The historical production verification exception was suppressed; its cause is unknown.
No model IDs or credentials are rewritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from hermes_constants import get_hermes_home

VERSION = "2026-10-30-v4-unattended-only"

# Legacy recovery readers remain only so historical recovery records can be
# inspected by older tooling. No enforcement or CLI path can activate them.
RECOVERY_FILE = "cache/interactive-recovery.json"
RECOVERY_ALLOWANCE_USD = 2.0
RECOVERY_ZONE = ZoneInfo("America/Toronto")
RECOVERY_SCOPE = "retired interactive recovery mechanism"


class SpendGuardBlocked(RuntimeError):
    """Terminal local rejection; never retry or activate a paid fallback."""


class BaselineUnavailable(ValueError):
    """Allowlisted diagnostic classification; contains no provider response."""

    def __init__(self, reason_code):
        self.reason_code = reason_code
        super().__init__(reason_code)


class MissingCredentials(ValueError):
    """No accounting credential was found by the canonical secret loader."""


DIAGNOSTIC_FILE = "cache/spend-guard-diagnostic.jsonl"


def _diagnose(operation, exc):
    """Append metadata only; never serialize exception text, context or responses."""
    try:
        path = _shared_home() / DIAGNOSTIC_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
        fd = os.open(path, flags, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                return False
            os.fchmod(fd, 0o600)
            record = {"time": datetime.now().astimezone().isoformat(timespec="seconds"),
                      "operation": operation, "exception_type": type(exc).__name__,
                      "pid": os.getpid(), "ppid": os.getppid(), "version": VERSION,
                      "module": str(Path(__file__).resolve()),
                      "profile_home": str(get_hermes_home()), "shared_home": str(_shared_home()),
                      "expected_date": today(),
                      "reason_code": getattr(exc, "reason_code", "verification-failed")}
            payload = (json.dumps(record) + "\n").encode("utf-8")
            return os.write(fd, payload) == len(payload)
        finally:
            os.close(fd)
    except OSError:
        return False


def _verification_failure(operation, exc):
    reasons = {
        "settings": "budget configuration is invalid or unreadable",
        "budget-lock": "budget lock is invalid or unreadable",
        "credentials": "accounting credentials could not be loaded",
        "account-baseline": "today's trusted account-wide baseline is missing, stale or invalid",
        "key-usage": "key usage accounting is unavailable or invalid",
        "account-usage": "account-wide usage accounting is unavailable or invalid",
        "settings-write": "budget configuration could not be saved",
    }
    reason = ("OpenRouter accounting credentials are missing" if isinstance(exc, MissingCredentials)
              else reasons[operation])
    logged = _diagnose(operation, exc)
    detail = (f"Diagnostic recorded in {_shared_home() / DIAGNOSTIC_FILE} (guard {VERSION}, pid {os.getpid()})." if logged
              else "The restricted diagnostic log could not be written.")
    return SpendGuardBlocked(f"Paid request stopped: {reason}. No model request sent. "
                             f"{detail} Codex subscription and local models remain available.")


def _verified(operation, fn, *args):
    try:
        return fn(*args)
    except SpendGuardBlocked:
        raise
    except Exception as exc:
        raise _verification_failure(operation, exc) from None


def _shared_home():
    """The home that holds the one global budget.

    Dispatched workers run under a profile home (<root>/profiles/<name>) which
    has its own .env and cache but no budget settings. Spending is account-wide,
    tracked against one OpenRouter key, so the budget, the usage baseline and
    the lock all live in the root home. Resolving to it here is what keeps a
    worker from being refused with "settings could not be verified" — the
    failure mode that silently stopped every dispatched task.
    """
    home = get_hermes_home()
    return home.parent.parent if home.parent.name == "profiles" else home


def _recovery_path():
    return _shared_home() / RECOVERY_FILE


def _process_row(pid):
    """Return (parent pid, command) for a live process without invoking a shell."""
    output = subprocess.check_output(
        ["ps", "-o", "ppid=,command=", "-p", str(int(pid))],
        text=True, timeout=2, stderr=subprocess.DEVNULL,
    ).strip()
    parent, command = output.split(None, 1)
    return int(parent), command


def _is_hermes_desktop_command(command):
    normalized = str(command or "").replace("\\", "/")
    return "Hermes.app/Contents/MacOS/Hermes" in normalized


def _is_desktop_backend_process():
    """True only in the Python backend whose direct parent is Hermes Desktop.

    A caller-supplied source label cannot satisfy this process-lineage check.
    Tool subprocesses, tests, cron, workers and delegated agents have a Python
    or shell parent instead and are therefore ineligible.
    """
    try:
        _, parent_command = _process_row(os.getppid())
        return _is_hermes_desktop_command(parent_command)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _session_identity_fields():
    try:
        from gateway.session_context import get_session_env
        source = str(get_session_env("HERMES_SESSION_SOURCE", "") or "").strip()
        session_id = str(get_session_env("HERMES_SESSION_ID", "") or "").strip()
        ui_session_id = str(get_session_env("HERMES_UI_SESSION_ID", "") or "").strip()
        cron = str(get_session_env("HERMES_CRON_SESSION", "") or "").strip()
    except Exception:
        source = str(os.environ.get("HERMES_SESSION_SOURCE") or "").strip()
        session_id = str(os.environ.get("HERMES_SESSION_ID") or "").strip()
        ui_session_id = str(os.environ.get("HERMES_UI_SESSION_ID") or "").strip()
        cron = str(os.environ.get("HERMES_CRON_SESSION") or "").strip()
    return source, session_id, ui_session_id, cron


def _has_unattended_marker(cron=""):
    if cron or any(str(os.environ.get(name) or "").strip() for name in (
        "HERMES_KANBAN_TASK", "HERMES_CRON_SESSION", "HERMES_DELEGATED_CHILD_CONTEXT",
    )):
        return True
    try:
        from agent.delegation_context import is_delegated_child_process_context
        return bool(is_delegated_child_process_context())
    except Exception:
        # Identity attribution must fail closed when its delegated-context
        # authority cannot be loaded.
        return True


def _current_interactive_identity():
    """Server-attested identity for the exact attended Desktop turn, or None."""
    source, session_id, ui_session_id, cron = _session_identity_fields()
    from agent.interactive_spending import is_attended_desktop_provider_request

    if (source != "desktop" or not session_id or not ui_session_id or cron
            or not is_attended_desktop_provider_request()):
        return None
    return {"source": source, "session_id": session_id, "ui_session_id": ui_session_id,
            "pid": os.getpid()}


def _operator_desktop_identity():
    """Resolve the Desktop backend ancestor for an operator's in-chat tool call.

    The activation command is intentionally unusable from an ordinary shell:
    it must be a descendant of the live Desktop backend and carry the exact
    Desktop session/window identity bridged by that backend.
    """
    source, session_id, ui_session_id, cron = _session_identity_fields()
    if source != "desktop" or not session_id or not ui_session_id or _has_unattended_marker(cron):
        return None
    try:
        pid = os.getpid()
        for _ in range(8):
            parent, _ = _process_row(pid)
            _grandparent, parent_command = _process_row(parent)
            if _is_hermes_desktop_command(parent_command):
                return {"source": source, "session_id": session_id, "ui_session_id": ui_session_id,
                        "pid": pid, "ppid": parent}
            if parent <= 1 or parent == pid:
                break
            pid = parent
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def _atomic_private_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _read_private_json(path):
    flags = os.O_RDONLY | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != 1 or info.st_mode & 0o077):
            raise ValueError("Recovery authorization file is not private and regular")
        with os.fdopen(fd, "r") as stream:
            fd = -1
            value = json.load(stream)
    finally:
        if fd >= 0:
            os.close(fd)
    if not isinstance(value, dict):
        raise ValueError("Recovery authorization must be an object")
    return value


def _recovery_expiry():
    now = datetime.now(RECOVERY_ZONE)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def _parse_recovery_time(value):
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("Recovery time must include a timezone")
    return parsed


def authorize_recovery():
    """Create the one-shot operator window from a fresh account-total snapshot."""
    identity = _current_interactive_identity() or _operator_desktop_identity()
    if identity is None:
        raise SpendGuardBlocked(
            "Recovery authorization refused: this is not the live attended Hermes Desktop conversation."
        )
    key = _verified("credentials", _key)
    baseline = _verified("account-usage", _account_total, key)
    now = datetime.now(RECOVERY_ZONE)
    record = {
        "version": 1, "date": now.strftime("%Y-%m-%d"),
        "authorized_usd": RECOVERY_ALLOWANCE_USD,
        "baseline_total_usage": baseline,
        "authorized_at": now.isoformat(timespec="seconds"),
        "expires_at": _recovery_expiry(),
        "identity": identity, "scope": RECOVERY_SCOPE,
        "notice": "Incremental recovery allowance only; not proof the original daily cap was respected.",
    }
    _atomic_private_json(_recovery_path(), record)
    return record


def _validated_recovery_for_current():
    identity = _current_interactive_identity()
    if identity is None:
        return None
    try:
        record = _read_private_json(_recovery_path())
    except FileNotFoundError:
        return None
    except Exception:
        raise SpendGuardBlocked(
            "Paid request stopped: recovery authorization is invalid. No model request sent."
        ) from None
    if record.get("identity") != identity:
        return None
    try:
        allowance = float(record["authorized_usd"])
        baseline = float(record["baseline_total_usage"])
        expiry = _parse_recovery_time(record["expires_at"])
        if (record.get("version") != 1 or record.get("scope") != RECOVERY_SCOPE
                or record.get("date") != datetime.now(RECOVERY_ZONE).strftime("%Y-%m-%d")
                or not math.isfinite(allowance) or allowance != RECOVERY_ALLOWANCE_USD
                or not math.isfinite(baseline) or baseline < 0):
            raise ValueError("Invalid recovery authorization")
        if datetime.now(RECOVERY_ZONE) >= expiry.astimezone(RECOVERY_ZONE):
            raise PermissionError("expired")
    except PermissionError:
        raise SpendGuardBlocked(
            "Paid request stopped: interactive recovery authorization expired. No model request sent."
        ) from None
    except Exception:
        raise SpendGuardBlocked(
            "Paid request stopped: recovery authorization is invalid. No model request sent."
        ) from None
    return record


def _recovery_reading_or_none():
    record = _validated_recovery_for_current()
    if record is None:
        return None
    try:
        key = _key()
        total = _account_total(key)
    except Exception:
        raise SpendGuardBlocked(
            "Paid request stopped: recovery account-wide accounting is unavailable or invalid. "
            "No model request sent."
        ) from None
    baseline = float(record["baseline_total_usage"])
    if not math.isfinite(total) or total < 0:
        raise SpendGuardBlocked(
            "Paid request stopped: recovery account-wide accounting is unavailable or invalid. "
            "No model request sent."
        )
    if total < baseline:
        raise SpendGuardBlocked(
            "Paid request stopped: recovery account total regressed and cannot be trusted. "
            "No model request sent."
        )
    used = total - baseline
    allowance = float(record["authorized_usd"])
    remaining = max(0.0, allowance - used)
    return {"baseline_total_usage": baseline, "account_total_usage": total,
            "authorized_usd": allowance, "used_usd": used, "remaining_usd": remaining,
            "expires_at": record["expires_at"]}


def recovery_reading():
    reading = _recovery_reading_or_none()
    if reading is None:
        raise SpendGuardBlocked(
            "Paid request stopped: this process/conversation is not authorized for interactive recovery."
        )
    return reading


def today():
    return datetime.now(RECOVERY_ZONE).strftime("%Y-%m-%d")


def amount(value):
    n = float(value)
    if not math.isfinite(n) or n <= 0:
        raise ValueError("Cap must be a finite positive dollar amount")
    return n


def interactive_grace(data):
    """Extra headroom granted only to an attended Desktop conversation.

    The daily cap is a hard ceiling for unattended work (workers, cron,
    delegated agents). A live conversation with Avra is not that: blocking her
    mid-sentence to protect a number is the wrong trade. Interactive turns are
    allowed to run past the cap by this grace amount, then stop.
    """
    raw = data.get("interactive_grace_usd", DEFAULT_INTERACTIVE_GRACE)
    try:
        grace = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_INTERACTIVE_GRACE
    return grace if math.isfinite(grace) and grace >= 0 else DEFAULT_INTERACTIVE_GRACE


def read_settings():
    data = json.loads((_shared_home() / "spend-budget.json").read_text())
    base = amount(data["daily_cap_usd"])
    override = data.get("temporary") or {}
    effective = max(base, amount(override["cap_usd"])) if override.get("date") == today() else base
    return data, base, effective


def save_settings(data):
    path = _shared_home() / "spend-budget.json"
    fd, name = tempfile.mkstemp(prefix=".spend-budget-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def route_kind(provider=None, base_url=None):
    prov = str(provider or "").lower().strip()
    url = urlparse(str(base_url or ""))
    host = (url.hostname or "").lower()
    if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        return "openrouter"
    # Actual destination takes precedence over a provider label.
    if prov == "openrouter" or prov.startswith("openrouter-"):
        return "openrouter" if not host else "untracked"
    if host == "chatgpt.com" and url.scheme == "https" and url.path.startswith("/backend-api/codex"):
        return "subscription"
    if prov == "openai-codex" and not host:
        return "subscription"
    local_host = host in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}
    if prov in {"ollama", "lmstudio", "llamacpp", "local", "vllm", "mlx"} and (local_host or not host):
        return "local"
    # Ollama's standard loopback endpoint is also used under the custom provider.
    if local_host and url.port == 11434:
        return "local"
    return "untracked"


def _key():
    from agent.secret_scope import load_env_file
    key = load_env_file(_shared_home() / ".env").get("OPENROUTER_API_KEY")
    if not key:
        raise MissingCredentials("OpenRouter usage credentials unavailable")
    return key


# Day boundary: LOCAL midnight (America/Toronto), because Avra's day is her day.
# OpenRouter's key usage_daily resets at UTC midnight, which is 20:00 EDT the
# previous evening, so it is NOT used as the day boundary. Key-scoped usage
# alone must never authorize paid calls.
#
# One baseline, one meaning: cache/spend-baseline.json holds the account
# total_usage snapshot taken at the first watchdog tick after local midnight.
# It is written atomically (temp + rename) and is valid only for its own date.
BASELINE_FILE = "cache/spend-baseline.json"
# Attended Desktop conversations may exceed the daily cap by this much before
# they are stopped. Unattended work never gets this allowance.
DEFAULT_INTERACTIVE_GRACE = 10.0
BASELINE_WINDOW_MINUTES = 90  # snapshot only within 90 min after local midnight


def minutes_since_local_midnight():
    now = datetime.now(RECOVERY_ZONE)
    return now.hour * 60 + now.minute


def _baseline_path():
    return _shared_home() / BASELINE_FILE


def _account_total(key=None):
    key = key or _key()
    request = Request("https://openrouter.ai/api/v1/credits",
                      headers={"Authorization": "Bearer " + key})
    with urlopen(request, timeout=8) as response:
        total = float(json.load(response)["data"]["total_usage"])
    if not math.isfinite(total) or total < 0:
        raise ValueError("Invalid account total")
    return total


def read_baseline():
    """Return today's valid baseline or None; unreadable/malformed files raise."""
    try:
        data = json.loads(_baseline_path().read_text())
        start = float(data["day_start_usage"])
    except FileNotFoundError:
        return None
    if data.get("date") != today() or not math.isfinite(start) or start < 0:
        return None
    return {"date": data["date"], "day_start_usage": start}


def snapshot_baseline():
    """Write today's local-midnight baseline once. Idempotent within a day.

    Called by the watchdog tick. Late ticks return None. Read failures raise;
    no baseline is written if the account total cannot be read.
    """
    existing = read_baseline()
    if existing:
        return existing
    # Only a tick near local midnight may define the day. A snapshot taken at
    # 15:00 would hide the whole morning's spend, so later ticks write nothing
    # and paid calls stay blocked until a trusted baseline is available.
    if minutes_since_local_midnight() >= BASELINE_WINDOW_MINUTES:
        return None
    total = _account_total()
    path = _baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"date": today(), "day_start_usage": total,
              "source": "watchdog-tick-after-local-midnight",
              "ts": datetime.now(RECOVERY_ZONE).isoformat(timespec="seconds")}
    fd, name = tempfile.mkstemp(prefix=".spend-baseline-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return read_baseline()


def _required_baseline():
    baseline = read_baseline()
    if baseline is not None:
        return baseline
    # Self-heal at the moment of the first paid call after local midnight.  Until
    # 2026-11-03 this only raised, so a request arriving in the gap between midnight
    # and the next watchdog tick (every 30 minutes) was refused outright: Avra was
    # locked out of her own assistant at 00:02.  snapshot_baseline() is idempotent
    # and still refuses outside the 90 minute window, so the day can never be
    # back-dated to hide morning spend.
    healed = snapshot_baseline()
    if healed is not None:
        return healed
    raise BaselineUnavailable("baseline-missing-or-stale")


def _key_daily(key):
    request = Request("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + key})
    with urlopen(request, timeout=8) as response:
        daily = float(json.load(response)["data"]["usage_daily"])
    if not math.isfinite(daily) or daily < 0:
        raise ValueError("Invalid usage")
    return daily


def spend_reading():
    """Account delta from today's baseline, conservatively floored by key usage.

    Missing/stale/invalid baselines block; no key-only authorization or implicit
    baseline creation. Reporting lag and concurrent in-flight costs remain.
    """
    baseline = _verified("account-baseline", _required_baseline)
    key = _verified("credentials", _key)
    daily = _verified("key-usage", _key_daily, key)
    total = _verified("account-usage", _account_total, key)
    if total < baseline["day_start_usage"]:
        raise _verification_failure("account-baseline", BaselineUnavailable("account-total-regressed")) from None
    # OpenRouter's key usage_daily resets at UTC midnight, which is 20:00 EDT the
    # previous evening. Between 20:00 and 04:00 local, that figure still carries
    # yesterday evening's spend, so using it as a floor inflates the current
    # day's number by however much was spent after 8pm. Only within the window
    # where the UTC day and the local day agree (04:00-20:00) is the floor
    # trustworthy; outside it, the account delta alone is the correct measure.
    delta = total - baseline["day_start_usage"]
    hour = datetime.now(RECOVERY_ZONE).hour
    utc_aligned = 4 <= hour < 20
    spend = max(daily, delta) if utc_aligned else delta
    return {"spend": spend, "basis": "account",
            "degraded": False, "reason": "",
            "detail": {"account_delta": delta, "key_daily": daily, "utc_aligned": utc_aligned}}


def live_spend():
    return spend_reading()["spend"]


def enforce(provider=None, base_url=None, **_context):
    kind = route_kind(provider, base_url)
    if kind in {"subscription", "local"}:
        return
    # Authenticated Desktop turns are isolated onto their own provider-limited
    # key at the final wire boundary. Local accounting is intentionally not a
    # dependency for them; unattended work remains fail-closed below.
    if kind == "openrouter" and _current_interactive_identity() is not None:
        return
    operation = "settings"
    try:
        _, _, cap = read_settings()
        if kind != "openrouter":
            _diagnose("route-check", SpendGuardBlocked())
            raise SpendGuardBlocked("Paid request stopped: this route has no verified spending meter. "
                                    "Select OpenRouter, OpenAI Codex subscription, or a local model.")
        operation = "account-usage"
        spend = live_spend()
        operation = "budget-lock"
        lock_path = _shared_home() / "cache/budget-lock.json"
        try:
            lock_text = lock_path.read_text()
        except FileNotFoundError:
            lock_text = None
        if lock_text is not None:
            lock = json.loads(lock_text)
            if lock.get("date") == today():
                locked_spend = float(lock["spend"])
                if not math.isfinite(locked_spend) or locked_spend < 0:
                    raise ValueError("Invalid budget lock")
                spend = max(spend, locked_spend)
        if spend >= cap:
            _diagnose("budget-exhausted", SpendGuardBlocked())
            raise SpendGuardBlocked(
                f"OpenRouter unattended daily cap reached: ${spend:.2f} spent / ${cap:.2f} limit. "
                "Use hermes-budget status, hermes-budget today AMOUNT, or choose Codex subscription/local.")
    except SpendGuardBlocked:
        raise
    except Exception as exc:
        raise _verification_failure(operation, exc) from None


def main():
    parser = argparse.ArgumentParser(description="View/set the OpenRouter daily cap in USD. No model calls.")
    parser.add_argument("action", nargs="?", default="status",
                        choices=["status", "set", "today", "clear"])
    parser.add_argument("amount", nargs="?", type=amount)
    args = parser.parse_args()
    data, base, cap = _verified("settings", read_settings)
    if args.action in {"set", "today"} and args.amount is None:
        parser.error("Supply the total daily limit, for example: hermes-budget today 20")
    if args.action == "set":
        data["daily_cap_usd"] = args.amount
    elif args.action == "today":
        if args.amount < base:
            parser.error("A temporary raise must be at least the base cap")
        data["temporary"] = {"date": today(), "cap_usd": args.amount}
    elif args.action == "clear":
        data.pop("temporary", None)
    if args.action != "status":
        _verified("settings-write", save_settings, data)
    _, base, cap = _verified("settings", read_settings)
    print(f"OpenRouter base: ${base:.2f}/day; effective today ({today()}): ${cap:.2f}")
    print("Codex subscription and recognized local models bypass this cap. Today-only raises expire at local midnight.")
    print("Attended Desktop OpenRouter requests use the dedicated provider-limited interactive key; "
          "this local cap applies only to unattended work.")
    if args.action == "status":
        try:
            reading = spend_reading()
            spend = reading["spend"]
            print(f"Live tracked spending: ${spend:.2f}; remaining: ${max(0, cap-spend):.2f}")
            # Show the composition so the number is legible: it is account-wide
            # OpenRouter usage since the local-midnight baseline, not this chat.
            try:
                baseline = read_baseline() or {}
                total = _account_total()
                print(f"  basis: account-wide usage since {baseline.get('date','?')} local midnight "
                      f"(baseline ${baseline.get('day_start_usage', 0):.4f}, account total ${total:.4f})")
                print("  note: includes Honcho derivation/embeddings and any other profile using this key.")
            except Exception:
                pass
            if reading["degraded"]:
                print(f"DEGRADED: {reading['reason']}")
        except SpendGuardBlocked as exc:
            print(str(exc))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
