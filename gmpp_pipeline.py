#!/usr/bin/env python3
"""Quarterly GMPP data poll, layout check, optional clean, train and upload.

This script is safe by default:
  * --dry-run only detects releases. It does not write into the download
    folder, does not train, and does not upload a model.
  * A new file with an unknown column layout stops before cleaning or training.
  * The live Hugging Face model is never replaced during a dry run or during
    setup. A scheduled --run may upload only after a known (or approved)
    layout and a successful local training run, and only if a token is already
    available. The previous Space model is backed up locally first.

It never writes into an existing Desktop/gmpp-data folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from known_layouts import match_known_layout, normalized_columns

try:
    from clean_gmpp import clean_file, read_raw_spreadsheet
except ImportError:  # pragma: no cover
    clean_file = None
    read_raw_spreadsheet = None

LONDON = ZoneInfo("Europe/London")
USER_AGENT = "ErnestJenkins-GMPP-pipeline/1.0 (quarterly academic check; low volume)"
COLLECTION_API = "https://www.gov.uk/api/content/government/collections/major-projects-data"
CONTENT_API = "https://www.gov.uk/api/content"
SEARCH_API = "https://www.gov.uk/api/search.json"
HF_REPO = "Legumchim/gmpp-dca-predict"
HF_REPO_TYPE = "space"
HF_MODEL_NAME = "best_model.joblib"

# National annual report pages, not departmental GMPP returns.
ANNUAL_TITLE_RE = re.compile(
    r"(NISTA|IPA|Infrastructure and Projects Authority).{0,100}"
    r"(Annual Report|annual report)",
    re.I,
)
EXCLUDE_TITLE_RE = re.compile(
    r"\b(PFI|PF2|assurance|toolkit|pipeline|routemap|mandate|Green Book|"
    r"SRO appointment|transparency policy|how to publish|"
    r"Government Major Projects Portfolio data,)\b",
    re.I,
)
FY_IN_TEXT_RE = re.compile(r"(20\d{2})\s*[-/]\s*(20)?(\d{2})")
MARCH_FILE_RE = re.compile(r"March[_\s-](20\d{2})", re.I)
DATA_EXT = {".csv", ".xlsx", ".xls", ".ods"}
SHEET_EXT_PRIORITY = (".csv", ".xlsx", ".xls", ".ods")

# Official attachments already stored as gmpp-YYYY-YY.csv (byte-identical
# check on 12 November 2026).
SEED_RELEASES = [
    {
        "year": "2025-26",
        "url": "https://assets.publishing.service.gov.uk/media/6a4fb28da6586e258d371bfd/NISTA_Major_Projects_Annual_Report_2025-2026_Data.csv",
        "sha256": "434e3667308157be0539f4ce85f208e7b69af373a2638df457f40c1197d8524e",
        "publication": "https://www.gov.uk/government/publications/nista-major-projects-annual-report-2025-26",
        "title": "NISTA Major Projects Annual Report 2025-26 data",
    },
    {
        "year": "2024-25",
        "url": "https://assets.publishing.service.gov.uk/media/6895fbb63080e72710b2e2ef/nista_annual_report_data_2425.csv",
        "sha256": "cac14ed4e6a90a81486bdc24295ea40996e7fd6fcd3834e3ffad8a2685f8c8b3",
        "publication": "https://www.gov.uk/government/publications/nista-annual-report-2024-2025",
        "title": "NISTA Annual Report 2024-25 data",
    },
    {
        "year": "2023-24",
        "url": "https://assets.publishing.service.gov.uk/media/6787e8ee1124a2c3ceb646be/Government_Major_Projects_Portofolio_AR_Data_March_2024.csv",
        "sha256": "29596242964ff850351b9d091ae7db2584e1c8cdec3b7c8e53ca661479db614b",
        "publication": "https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2023-24",
        "title": "GMPP Government Major Projects Portfolio AR Data March 2024",
    },
    {
        "year": "2022-23",
        "url": "https://assets.publishing.service.gov.uk/media/64b79c5171749c001389ee41/GMPP_Government_Major_Projects_Portofolio_AR_Data_March_2023.csv",
        "sha256": "ecd3756718a71e6f52fb25050caba535d64e7f5b21f6db361cfa10859a4364ac",
        "publication": "https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2022-23",
        "title": "GMPP Government Major Projects Portfolio AR Data March 2023",
    },
    {
        "year": "2021-22",
        "url": "https://assets.publishing.service.gov.uk/media/62d6c047e90e071e753d6936/GMPP_Government_Major_Projects_Portfolio_AR_Data_March_2022.csv",
        "sha256": "f34d32e4ed93a34cb9e7500491bba4cfa9568999c7dfabcbe74e0cc4cf538f2c",
        "publication": "https://www.gov.uk/government/publications/infrastructure-and-projects-authority-annual-report-2022",
        "title": "GMPP Government Major Projects Portfolio AR Data March 2022",
    },
]


def now_london() -> datetime:
    return datetime.now(LONDON)


def stamp() -> str:
    return now_london().strftime("%Y-%m-%d %H:%M:%S %Z")


def iso_now() -> str:
    return now_london().isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

def pipeline_root() -> Path:
    env = os.environ.get("GMPP_PIPELINE_ROOT")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent


def default_config(root: Path) -> dict:
    return {
        "download_dir": "auto",
        "existing_data_dir": "auto",
        "upload_to_hf": True,
        "hf_repo_id": HF_REPO,
        "hf_repo_type": HF_REPO_TYPE,
        "hf_path_in_repo": HF_MODEL_NAME,
        "user_agent": USER_AGENT,
    }


def load_config(root: Path) -> dict:
    cfg = default_config(root)
    path = root / "config.json"
    if path.exists():
        cfg.update(json.loads(path.read_text()))
    return cfg


def resolve_download_dir(cfg: dict, home: Path) -> Path:
    setting = cfg.get("download_dir") or "auto"
    if setting != "auto":
        return Path(setting).expanduser()
    dropbox_candidates = [
        home / "Dropbox" / "GMPP" / "raw",
        home / "Library" / "CloudStorage" / "Dropbox" / "GMPP" / "raw",
    ]
    for p in dropbox_candidates:
        if p.parent.parent.exists() or (p.parents[2] if len(p.parts) > 3 else None and Path(str(p).split("/GMPP")[0]).exists()):
            base = p
            # only use if the Dropbox root exists
    if (home / "Dropbox").exists():
        return home / "Dropbox" / "GMPP" / "raw"
    cloud = home / "Library" / "CloudStorage"
    if cloud.exists():
        for child in sorted(cloud.glob("Dropbox*")):
            return child / "GMPP" / "raw"
        for child in sorted(cloud.glob("OneDrive*")):
            return child / "GMPP" / "raw"
    return home / "GMPP" / "raw"


def resolve_existing_data_dir(cfg: dict, home: Path) -> Path | None:
    setting = cfg.get("existing_data_dir") or "auto"
    if setting != "auto":
        path = Path(setting).expanduser()
        return path if path.exists() else None
    for cand in (
        home / "Desktop" / "gmpp-data",
        home / "Documents" / "gmpp-data",
        home / "GMPP" / "existing-data",
    ):
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------
# Logging, status, notifications
# ---------------------------------------------------------------------------

def append_log(root: Path, message: str) -> None:
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    line = f"{stamp()} {message}"
    with (log_dir / "pipeline.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def write_status(root: Path, payload: dict) -> Path:
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    path = state / "status.json"
    payload = dict(payload)
    payload["updated_at"] = iso_now()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    text = state / "status.txt"
    lines = [
        f"GMPP pipeline status ({payload['updated_at']})",
        f"Result: {payload.get('result', '')}",
        f"Detail: {payload.get('detail', '')}",
    ]
    if payload.get("next_step"):
        lines.append(f"Next step: {payload['next_step']}")
    text.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def notify_macos(title: str, message: str) -> None:
    """Show a macOS notification. Silently skipped on other systems."""
    if sys.platform != "darwin":
        return
    def q(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{q(message)}" with title "{q(title)}"'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=15,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        pass


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def manifest_path(root: Path) -> Path:
    return root / "state" / "seen_manifest.json"


def empty_manifest() -> dict:
    return {"version": 1, "items": []}


def load_manifest(root: Path) -> dict:
    path = manifest_path(root)
    if not path.exists():
        return empty_manifest()
    return json.loads(path.read_text())


def save_manifest(root: Path, manifest: dict) -> None:
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def seed_manifest_if_needed(root: Path) -> dict:
    manifest = load_manifest(root)
    known_urls = {item.get("url") for item in manifest.get("items", [])}
    added = 0
    for rel in SEED_RELEASES:
        if rel["url"] in known_urls:
            continue
        manifest.setdefault("items", []).append({
            "url": rel["url"],
            "sha256": rel["sha256"],
            "year": rel["year"],
            "title": rel["title"],
            "publication": rel["publication"],
            "seen_at": iso_now(),
            "source": "seeded-from-known-official-files",
        })
        added += 1
    if added or not manifest_path(root).exists():
        save_manifest(root, manifest)
    return manifest


def seen_url(manifest: dict, url: str) -> bool:
    return any(item.get("url") == url for item in manifest.get("items", []))


def seen_hash(manifest: dict, digest: str) -> bool:
    return any(item.get("sha256") == digest for item in manifest.get("items", []))


def year_sort_key(year: str | None) -> int:
    if not year:
        return -1
    try:
        return int(str(year).split("-")[0])
    except ValueError:
        return -1


def latest_manifest_year(manifest: dict) -> str | None:
    years = [item.get("year") for item in manifest.get("items", []) if item.get("year")]
    if not years:
        return None
    return max(years, key=year_sort_key)


def is_newer_year(year: str | None, latest: str | None) -> bool:
    """Unknown years are treated as new. Older historical years are ignored."""
    if not latest or not year:
        return True
    return year_sort_key(year) > year_sort_key(latest)


# ---------------------------------------------------------------------------
# GOV.UK API
# ---------------------------------------------------------------------------

def http_json(url: str, user_agent: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_bytes(url: str, user_agent: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def is_annual_report_title(title: str) -> bool:
    if not title:
        return False
    if EXCLUDE_TITLE_RE.search(title) and not ANNUAL_TITLE_RE.search(title):
        return False
    if EXCLUDE_TITLE_RE.search(title):
        # Departmental "XYZ Government Major Projects Portfolio data, 2024"
        if "Government Major Projects Portfolio data," in title:
            return False
    return bool(ANNUAL_TITLE_RE.search(title))


def year_from_text(*parts: str) -> str | None:
    text = " ".join(p for p in parts if p)
    # Prefer a full financial year in the title or filename.
    matches = list(FY_IN_TEXT_RE.finditer(text))
    if matches:
        start, maybe, end = matches[-1].groups()
        end2 = end if maybe else end
        # 2025-26 or 2024-2025 -> 2024-25 / 2025-26
        start_i = int(start)
        end_i = int(end2) if len(end2) == 2 else int(end2)
        if end_i < 100:
            end_i = (start_i // 100) * 100 + end_i
        if end_i in {start_i, start_i + 1}:
            return f"{start_i}-{end_i % 100:02d}"
    march = MARCH_FILE_RE.search(text)
    if march:
        y = int(march.group(1))
        return f"{y - 1}-{y % 100:02d}"
    # Bare "annual report 2022" published in 2022 usually holds 2021-22 data.
    m = re.search(r"annual report(?: on major projects)?(?: for)?\s+(20\d{2})\b", text, re.I)
    if m and not FY_IN_TEXT_RE.search(text):
        y = int(m.group(1))
        return f"{y - 1}-{y % 100:02d}"
    return None


def attachment_filename(att: dict) -> str:
    name = att.get("filename") or ""
    if name:
        return name
    url = att.get("url") or ""
    return urllib.parse.unquote(url.rsplit("/", 1)[-1].split("?")[0])


def attachment_ext(att: dict) -> str:
    name = attachment_filename(att).lower()
    for ext in DATA_EXT:
        if name.endswith(ext):
            return ext
    url = (att.get("url") or "").lower()
    for ext in DATA_EXT:
        if url.split("?")[0].endswith(ext):
            return ext
    return ""


def collect_publication_paths(user_agent: str) -> list[dict]:
    """Return candidate national annual-report publications."""
    found: dict[str, dict] = {}

    def add(title: str, base_path: str, published=None):
        if not base_path or not is_annual_report_title(title or ""):
            return
        if not base_path.startswith("/"):
            return
        found[base_path] = {
            "title": title,
            "base_path": base_path,
            "published": published,
        }

    # 1. The official collection (verified live).
    try:
        col = http_json(COLLECTION_API, user_agent)
        for doc in (col.get("links") or {}).get("documents") or []:
            add(doc.get("title") or "", doc.get("base_path") or "",
                doc.get("public_updated_at") or doc.get("first_published_at"))
    except Exception as exc:
        # Collection failure is not fatal if search works.
        found["_collection_error"] = {"title": "ERROR", "base_path": "", "published": str(exc)}

    collection_error = found.pop("_collection_error", None)

    # 2. Search API backup for NISTA and the former IPA.
    queries = [
        SEARCH_API + "?q=NISTA+Major+Projects+Annual+Report&count=20&order=-public_timestamp",
        SEARCH_API + "?filter_organisations=national-infrastructure-and-service-transformation-authority"
                     "&filter_content_store_document_type=transparency&count=30",
        SEARCH_API + "?q=%22Annual+Report+on+Major+Projects%22"
                     "&filter_organisations=infrastructure-and-projects-authority"
                     "&count=20&order=-public_timestamp",
    ]
    for url in queries:
        try:
            data = http_json(url, user_agent)
        except Exception:
            continue
        for res in data.get("results") or []:
            add(res.get("title") or "", res.get("link") or "", res.get("public_timestamp"))

    items = list(found.values())
    if collection_error:
        # Keep a breadcrumb for logs; not a publication.
        pass
    return items, collection_error


def list_data_attachments(pub: dict, user_agent: str) -> list[dict]:
    path = pub["base_path"]
    page = http_json(CONTENT_API + path, user_agent)
    atts = (page.get("details") or {}).get("attachments") or []
    out = []
    for att in atts:
        ext = attachment_ext(att)
        if not ext:
            continue
        filename = attachment_filename(att)
        # Skip SRO name lists and similar sidecars.
        lower = f"{att.get('title') or ''} {filename}".lower()
        if "senior responsible" in lower and "data" not in lower:
            continue
        if "sro" in lower and "list" in lower:
            continue
        url = att.get("url") or ""
        if url.startswith("/"):
            url = "https://www.gov.uk" + url
        year = year_from_text(pub.get("title") or "", att.get("title") or "", filename)
        out.append({
            "title": att.get("title") or pub.get("title"),
            "filename": filename,
            "url": url,
            "ext": ext,
            "year": year,
            "publication_title": page.get("title") or pub.get("title"),
            "publication": "https://www.gov.uk" + path,
            "published": page.get("first_published_at") or pub.get("published"),
        })
    return out


def preferred_attachment(cands: list[dict]) -> dict | None:
    if not cands:
        return None
    by_ext = {c["ext"]: c for c in cands}
    for ext in SHEET_EXT_PRIORITY:
        if ext in by_ext:
            return by_ext[ext]
    return cands[0]


def discover_releases(user_agent: str, log) -> list[dict]:
    pubs, col_err = collect_publication_paths(user_agent)
    if col_err:
        log(f"Collection API warning: {col_err.get('published')}")
    log(f"Found {len(pubs)} national annual-report publication page(s).")
    grouped: dict[tuple, list[dict]] = {}
    for pub in pubs:
        try:
            atts = list_data_attachments(pub, user_agent)
        except Exception as exc:
            log(f"Could not read {pub.get('base_path')}: {type(exc).__name__}: {exc}")
            continue
        for att in atts:
            key = (att["publication"], att.get("year") or att["filename"])
            grouped.setdefault(key, []).append(att)
    releases = []
    for items in grouped.values():
        chosen = preferred_attachment(items)
        if chosen:
            releases.append(chosen)
    # Prefer newer years when sorting, unknown years last.
    def sort_key(item):
        return item.get("year") or "", item.get("published") or ""
    releases.sort(key=sort_key, reverse=True)
    return releases


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def approved_layouts_path(root: Path) -> Path:
    return root / "state" / "approved_layouts.json"


def load_approved_layouts(root: Path) -> list[dict]:
    path = approved_layouts_path(root)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("layouts") or []


def save_approved_layouts(root: Path, layouts: list[dict]) -> None:
    path = approved_layouts_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"layouts": layouts}, indent=2) + "\n", encoding="utf-8")


def check_layout(path: Path, root: Path) -> dict:
    df = read_raw_spreadsheet(path)
    columns = list(df.columns)
    status, label, detail = match_known_layout(columns, extra_layouts=load_approved_layouts(root))
    return {
        "status": status,
        "label": label,
        "detail": detail,
        "columns": columns,
        "normalized": normalized_columns(columns),
        "row_count": int(len(df)),
    }


def pending_layout_path(root: Path) -> Path:
    return root / "state" / "pending_layout.json"


# ---------------------------------------------------------------------------
# Hugging Face (optional, never during dry-run)
# ---------------------------------------------------------------------------

def hf_token_available() -> bool:
    """Return True if a token appears to be present. Never log or return it."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    home = Path.home()
    for p in (
        home / ".cache" / "huggingface" / "token",
        home / ".huggingface" / "token",
        home / ".config" / "huggingface" / "token",
    ):
        if p.exists() and p.stat().st_size > 0:
            return True
    return False


def backup_and_upload_model(root: Path, model_path: Path, cfg: dict, log) -> dict:
    repo_id = cfg.get("hf_repo_id") or HF_REPO
    path_in_repo = cfg.get("hf_path_in_repo") or HF_MODEL_NAME
    repo_type = cfg.get("hf_repo_type") or HF_REPO_TYPE
    backup_dir = root / "model-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp_name = now_london().strftime("%Y%m%d-%H%M%S")
    result = {"uploaded": False, "backup": None, "error": None}

    if not hf_token_available():
        result["error"] = "No Hugging Face token is available. Run huggingface-cli login (or hf auth login) in Terminal before an upload can work. The new model has been left in the candidate folder."
        log(result["error"])
        return result

    from huggingface_hub import hf_hub_download, upload_file

    try:
        previous = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type=repo_type,
        )
        backup = backup_dir / f"best_model-before-{stamp_name}.joblib"
        shutil.copy2(previous, backup)
        result["backup"] = str(backup)
        log(f"Backed up the current Space model to {backup}")
    except Exception as exc:
        log(f"Could not download the current Space model for backup: {type(exc).__name__}: {exc}")
        # Continue only if we still want to be conservative? Better stop than replace without a backup.
        result["error"] = f"Upload skipped because the previous model could not be backed up ({type(exc).__name__})."
        return result

    try:
        upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Update best_model.joblib from GMPP pipeline {stamp_name}",
        )
        result["uploaded"] = True
        log(f"Uploaded {model_path.name} to {repo_id}:{path_in_repo}")
    except Exception as exc:
        result["error"] = f"Upload failed: {type(exc).__name__}: {exc}"
        log(result["error"])
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(root: Path, existing_data: Path | None, new_cleaned: Path, year: str, log) -> Path:
    candidate = root / "candidates" / now_london().strftime("%Y%m%d-%H%M%S") / f"gmpp-{year}"
    cleaned_dir = candidate / "cleaned"
    output_dir = candidate / "training-output"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy existing cleaned files, then add (or replace) this year's cleaned file
    # in the candidate folder only.
    if existing_data and existing_data.exists():
        for src in sorted(existing_data.glob("cleaned-gmpp-*.csv")):
            shutil.copy2(src, cleaned_dir / src.name)
    shutil.copy2(new_cleaned, cleaned_dir / f"cleaned-gmpp-{year}.csv")

    env = os.environ.copy()
    env["GMPP_CLEANED_DIR"] = str(cleaned_dir)
    env["GMPP_OUTPUT_DIR"] = str(output_dir)
    script = root / "train_models.py"
    log(f"Starting training from {script} into {output_dir}")
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"train_models.py exited with code {proc.returncode}")
    model = output_dir / "best_model.joblib"
    if not model.exists():
        raise RuntimeError("Training finished but best_model.joblib was not created.")
    log(f"Training finished. Candidate model: {model}")
    return model


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_dry_run(root: Path, cfg: dict, probe_temp: bool = True) -> int:
    log = lambda m: append_log(root, m)
    manifest = seed_manifest_if_needed(root)
    log("Dry run: detecting releases only. No download into the data folder, no training, no upload.")
    releases = discover_releases(cfg.get("user_agent") or USER_AGENT, log)
    latest = latest_manifest_year(manifest)
    new_items = []
    already = []
    older = []
    for rel in releases:
        if seen_url(manifest, rel["url"]):
            already.append(rel)
        elif not is_newer_year(rel.get("year"), latest):
            older.append(rel)
        else:
            new_items.append(rel)
    log(f"Attachments considered: {len(releases)}. Already in the manifest: {len(already)}. Older than {latest} and ignored: {len(older)}. New URLs: {len(new_items)}.")
    for rel in already[:8]:
        log(f"  seen  {rel.get('year') or '?'}  {rel['url']}")
    for rel in new_items:
        log(f"  NEW   {rel.get('year') or '?'}  {rel['url']}")

    # Optional: download new (or, if none, the latest known) file into a temp
    # folder only, to prove the API and the layout check work. Never write to
    # the user's data folder.
    probe_note = None
    if probe_temp:
        target = new_items[0] if new_items else (already[0] if already else None)
        if target:
            tmp = Path(tempfile.mkdtemp(prefix="gmpp-dry-run-"))
            dest = tmp / (target.get("filename") or "download.csv")
            try:
                data = http_bytes(target["url"], cfg.get("user_agent") or USER_AGENT)
                dest.write_bytes(data)
                digest = sha256_bytes(data)
                layout = check_layout(dest, root)
                probe_note = {
                    "temp_file": str(dest),
                    "bytes": len(data),
                    "sha256": digest,
                    "already_known_hash": seen_hash(manifest, digest),
                    "layout": {k: layout[k] for k in ("status", "label", "detail", "row_count")},
                }
                log(
                    f"Temp download (not in your data folder): {dest} "
                    f"({len(data)} bytes, sha256 {digest[:12]}..., "
                    f"layout {layout['status']} {layout.get('label') or ''}, "
                    f"{layout['row_count']} rows)."
                )
            except Exception as exc:
                probe_note = {"error": f"{type(exc).__name__}: {exc}"}
                log(f"Temp download failed: {probe_note['error']}")

    if new_items:
        write_status(root, {
            "result": "dry-run-new-urls",
            "detail": f"{len(new_items)} new attachment URL(s) found. Nothing was saved to the download folder.",
            "new": new_items,
            "probe": probe_note,
        })
    else:
        write_status(root, {
            "result": "dry-run-nothing-new",
            "detail": "No new annual GMPP spreadsheet URL was found. Manifest already lists the known official files.",
            "probe": probe_note,
        })
        log("No new file this quarter (dry run). Exiting quietly.")
    return 0


def _save_new_release(root, cfg, rel, dest_dir: Path, log) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    year = rel.get("year") or "unknown"
    ext = rel.get("ext") or Path(rel.get("filename") or "data.csv").suffix
    dest = dest_dir / f"gmpp-{year}{ext}"
    # Never overwrite a different existing file silently.
    if dest.exists():
        dest = dest_dir / f"gmpp-{year}-{now_london().strftime('%Y%m%d-%H%M%S')}{ext}"
    data = http_bytes(rel["url"], cfg.get("user_agent") or USER_AGENT)
    dest.write_bytes(data)
    log(f"Downloaded {rel['url']} -> {dest} ({len(data)} bytes)")
    return dest


def cmd_run(root: Path, cfg: dict, upload: bool, approve_pending: bool = False) -> int:
    log = lambda m: append_log(root, m)
    home = Path.home()
    download_dir = resolve_download_dir(cfg, home)
    existing = resolve_existing_data_dir(cfg, home)
    manifest = seed_manifest_if_needed(root)

    if approve_pending:
        return cmd_approve_layout(root, cfg, upload)

    log("Starting GMPP quarterly check.")
    releases = discover_releases(cfg.get("user_agent") or USER_AGENT, log)
    latest = latest_manifest_year(manifest)
    new_items = []
    for rel in releases:
        if seen_url(manifest, rel["url"]):
            continue
        if not is_newer_year(rel.get("year"), latest):
            continue
        new_items.append(rel)

    if not new_items:
        write_status(root, {
            "result": "nothing-new",
            "detail": "No new annual GMPP spreadsheet was found this quarter.",
        })
        log("No new file this quarter. Exiting quietly.")
        return 0

    # Process one new annual file at a time, newest first.
    rel = new_items[0]
    log(f"New release detected: year={rel.get('year')} url={rel['url']}")
    dest = _save_new_release(root, cfg, rel, download_dir, log)
    digest = sha256_file(dest)
    if seen_hash(manifest, digest):
        log("This file's SHA-256 already appears in the manifest (same bytes, new URL). Treating it as already seen.")
        manifest["items"].append({
            "url": rel["url"], "sha256": digest, "year": rel.get("year"),
            "title": rel.get("title"), "publication": rel.get("publication"),
            "seen_at": iso_now(), "source": "duplicate-hash", "path": str(dest),
        })
        save_manifest(root, manifest)
        write_status(root, {"result": "nothing-new", "detail": "New URL, but the file bytes were already in the manifest."})
        log("No new file this quarter (duplicate content). Exiting quietly.")
        return 0

    layout = check_layout(dest, root)
    log(f"Layout check: {layout['status']} {layout.get('label') or ''} ({layout['row_count']} rows).")
    if layout["status"] == "unknown":
        pending = {
            "file": str(dest),
            "url": rel["url"],
            "sha256": digest,
            "year": rel.get("year"),
            "release": rel,
            "columns": layout["columns"],
            "normalized": layout["normalized"],
            "detected_at": iso_now(),
        }
        pending_layout_path(root).write_text(json.dumps(pending, indent=2) + "\n", encoding="utf-8")
        write_status(root, {
            "result": "NEW LAYOUT: human review needed",
            "detail": (
                f"The new file {dest} does not match the known 2021-22 to 2025-26 "
                "column layouts. Cleaning and training were not started. Open the "
                "file, check the headers, then run: "
                f"python3 {root / 'gmpp_pipeline.py'} --approve-layout"
            ),
            "file": str(dest),
            "columns": layout["normalized"],
            "next_step": f'"{sys.executable}" "{root / "gmpp_pipeline.py"}" --approve-layout',
        })
        log("NEW LAYOUT: human review needed. Stopped before cleaning or training.")
        notify_macos("GMPP pipeline", "NEW LAYOUT: human review needed")
        return 2

    # Known or previously approved layout: clean, then train. Record the file
    # in the manifest only after a successful clean, so a failed clean can be retried.
    year = rel.get("year") or layout.get("label") or "unknown"
    cleaned_dir = download_dir.parent / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = cleaned_dir / f"cleaned-gmpp-{year}.csv"
    cleaned, stats = clean_file(dest, cleaned_path)
    log(
        f"Cleaned {stats['cleaned_rows']} of {stats['raw_rows']} rows "
        f"(dropped {stats['dropped_rows']}; SRO fallback {stats['used_sro_fallback']}) "
        f"-> {cleaned_path}"
    )

    manifest["items"].append({
        "url": rel["url"], "sha256": digest, "year": year,
        "title": rel.get("title"), "publication": rel.get("publication"),
        "seen_at": iso_now(), "source": "downloaded", "path": str(dest),
        "cleaned": str(cleaned_path), "layout": layout.get("label"),
    })
    save_manifest(root, manifest)

    try:
        model = run_training(root, existing, cleaned_path, year, log)
    except Exception as exc:
        write_status(root, {
            "result": "training-failed",
            "detail": f"{type(exc).__name__}: {exc}",
            "cleaned": str(cleaned_path),
        })
        log(f"Training failed: {type(exc).__name__}: {exc}")
        notify_macos("GMPP pipeline", "Training failed. See logs/pipeline.log")
        return 1

    upload_result = {"uploaded": False, "skipped": True}
    if upload and cfg.get("upload_to_hf", True):
        upload_result = backup_and_upload_model(root, model, cfg, log)
        upload_result["skipped"] = False
    else:
        log("Upload skipped (disabled for this run). Candidate model left locally for review.")

    if upload_result.get("uploaded"):
        write_status(root, {
            "result": "success-uploaded",
            "detail": f"New {year} data cleaned, model trained, and uploaded to {cfg.get('hf_repo_id')}. Backup: {upload_result.get('backup')}",
            "model": str(model),
        })
        notify_macos("GMPP pipeline", f"New {year} model trained and uploaded.")
    elif upload_result.get("error"):
        write_status(root, {
            "result": "trained-upload-skipped",
            "detail": upload_result["error"],
            "model": str(model),
            "backup": upload_result.get("backup"),
        })
        notify_macos("GMPP pipeline", "Model trained locally. Upload did not run. See status.txt")
    else:
        write_status(root, {
            "result": "trained-local-only",
            "detail": f"New {year} model is in the candidate folder. The live Hugging Face model was not changed.",
            "model": str(model),
        })
        notify_macos("GMPP pipeline", f"New {year} model trained and saved locally.")
    return 0


def cmd_approve_layout(root: Path, cfg: dict, upload: bool) -> int:
    log = lambda m: append_log(root, m)
    pending_path = pending_layout_path(root)
    if not pending_path.exists():
        log("No pending layout to approve.")
        write_status(root, {"result": "no-pending-layout", "detail": "There is no state/pending_layout.json file."})
        return 0
    pending = json.loads(pending_path.read_text())
    layouts = load_approved_layouts(root)
    new_id = f"approved-{pending.get('year') or 'unknown'}-{now_london().strftime('%Y%m%d-%H%M%S')}"
    layouts.append({
        "id": new_id,
        "columns": pending.get("columns") or pending.get("normalized"),
        "notes": "Approved by the user with --approve-layout",
        "approved_at": iso_now(),
        "source_file": pending.get("file"),
        "url": pending.get("url"),
    })
    save_approved_layouts(root, layouts)
    log(f"Approved layout {new_id} for {pending.get('file')}. Resuming clean and train.")

    dest = Path(pending["file"])
    if not dest.exists():
        log(f"Pending file is missing: {dest}")
        return 1
    home = Path.home()
    download_dir = dest.parent
    existing = resolve_existing_data_dir(cfg, home)
    year = pending.get("year") or "unknown"
    cleaned_dir = download_dir.parent / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = cleaned_dir / f"cleaned-gmpp-{year}.csv"
    cleaned, stats = clean_file(dest, cleaned_path)
    log(
        f"Cleaned {stats['cleaned_rows']} of {stats['raw_rows']} rows "
        f"-> {cleaned_path}"
    )
    manifest = load_manifest(root)
    if not seen_url(manifest, pending.get("url") or ""):
        manifest["items"].append({
            "url": pending.get("url"), "sha256": pending.get("sha256"),
            "year": year, "seen_at": iso_now(), "source": "approved-layout",
            "path": str(dest), "cleaned": str(cleaned_path), "layout": new_id,
        })
        save_manifest(root, manifest)
    try:
        model = run_training(root, existing, cleaned_path, year, log)
    except Exception as exc:
        write_status(root, {"result": "training-failed", "detail": f"{type(exc).__name__}: {exc}"})
        notify_macos("GMPP pipeline", "Training failed after layout approval.")
        return 1
    pending_path.unlink(missing_ok=True)
    upload_result = {"uploaded": False}
    if upload and cfg.get("upload_to_hf", True):
        upload_result = backup_and_upload_model(root, model, cfg, log)
    write_status(root, {
        "result": "approved-and-trained",
        "detail": f"Layout approved. Candidate model at {model}",
        "model": str(model),
        "uploaded": bool(upload_result.get("uploaded")),
    })
    notify_macos("GMPP pipeline", "Approved layout cleaned and trained.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GMPP quarterly download pipeline")
    p.add_argument("--dry-run", action="store_true",
                   help="Detect only. Do not save into the data folder, train, or upload.")
    p.add_argument("--run", action="store_true",
                   help="Full quarterly run: download new files, check layout, clean, train, maybe upload.")
    p.add_argument("--approve-layout", action="store_true",
                   help="After you have inspected a new layout, record it as approved and resume.")
    p.add_argument("--no-upload", action="store_true",
                   help="Even on --run, leave the new model in the candidate folder only.")
    p.add_argument("--print-paths", action="store_true",
                   help="Show the folders this install will use, then exit.")
    return p


def cmd_print_paths(root: Path, cfg: dict) -> int:
    home = Path.home()
    download_dir = resolve_download_dir(cfg, home)
    existing = resolve_existing_data_dir(cfg, home)
    print(f"pipeline_root: {root}")
    print(f"download_dir:  {download_dir}")
    print(f"cleaned_dir:   {download_dir.parent / 'cleaned'}")
    print(f"existing_data: {existing}")
    print(f"logs:          {root / 'logs' / 'pipeline.log'}")
    print(f"status:        {root / 'state' / 'status.txt'}")
    print(f"manifest:      {manifest_path(root)}")
    print(f"candidates:    {root / 'candidates'}")
    print(f"model_backups: {root / 'model-backups'}")
    print(f"hf_token:      {'present' if hf_token_available() else 'missing'}")
    print(f"upload_to_hf:  {cfg.get('upload_to_hf', True)}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = pipeline_root()
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    cfg = load_config(root)
    upload = not args.no_upload
    if args.print_paths:
        return cmd_print_paths(root, cfg)
    if args.dry_run:
        return cmd_dry_run(root, cfg)
    if args.approve_layout:
        return cmd_approve_layout(root, cfg, upload)
    if args.run:
        return cmd_run(root, cfg, upload)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
