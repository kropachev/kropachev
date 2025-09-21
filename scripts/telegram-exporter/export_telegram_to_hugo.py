#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export all posts from a public Telegram channel to Hugo-ready Markdown bundles.
Improved version with error handling, retry logic, and better stability.

Usage:
  python export_telegram_channel_to_hugo.py --channel <username_or_link> --api-id <id> --api-hash <hash> [options]

Options:
  --out PATH              Output root (default: ./content/posts)
  --limit N               Number of posts to fetch (latest first)
  --no-images             Skip downloading images/media
  --remove-hashtags       Remove #tags from body (default: on)
  --slug-maxlen N         Max slug length (default: 60)
  --include-empty         Export even empty/whitespace messages
  --tmp PATH              Temp dir for media (default: ./.telegram_tmp)
  --dry-run               Show what would be written, do not create files
  --api-id, --api-hash    Telegram API credentials (https://my.telegram.org)
  --session FILE          Telethon session file (default: telegram_export.session)
  --delete-session        Delete session file(s) after run
  --max-retries N         Max retries for media download (default: 3)
  --verbose               Enable verbose logging

Examples:
  python export_telegram_channel_to_hugo.py --channel kropachevDigital --api-id 123456789 --api-hash abcdef0123456789
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import unicodedata
import shutil
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    from telethon import TelegramClient, errors
    from telethon.errors import UsernameNotOccupiedError, FloodWaitError, ChatAdminRequiredError
    from telethon.tl.types import MessageEntityBold
    from tqdm import tqdm
except ImportError as e:
    missing = str(e).split("'")[1] if "'" in str(e) else str(e)
    if missing == "tqdm":
        print("tqdm is required for progress bar. Install with: pip install tqdm", file=sys.stderr)
    else:
        print(f"Required package missing: {missing}. Install with: pip install telethon tqdm", file=sys.stderr)
    raise

# Setup logging
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='[%(levelname)s %(asctime)s] %(name)s: %(message)s',
        level=level
    )
    # Suppress telethon debug logs unless verbose
    if not verbose:
        logging.getLogger('telethon').setLevel(logging.WARNING)

EM_DASH = "—"
YO = "ё"

RU_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "ё": "e",
}
RU_TO_LAT.update({k.upper(): v.title() for k, v in list(RU_TO_LAT.items())})

def translit_ru_to_lat(s: str) -> str:
    return "".join(RU_TO_LAT.get(ch, ch) for ch in s)

def ascii_slugify(value: str, maxlen: int = 60) -> str:
    value = translit_ru_to_lat(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    words = re.split(r"\s+", value)
    if words:
        value = "-".join([w for w in words if w][:3])
    if value.startswith("pyatnichnoe"):
        value = value.replace("pyatnichnoe", "tgif", 1)
    value = re.sub(r"[^a-z0-9-]+", "-", value).strip("-")
    if len(value) > maxlen:
        value = value[:maxlen].rstrip("-")
    return value or "post"

def clean_text(text: str) -> str:
    """Clean text from problematic characters and encoding issues."""
    if not text:
        return ""
    
    # Handle encoding issues with emojis and special characters
    try:
        # Normalize unicode and handle potential encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        # Fallback for problematic characters
        text = ''.join(char for char in text if ord(char) < 65536)
    
    return text

def normalize_text(s: str) -> str:
    """Improved text normalization with better error handling."""
    if not s:
        return ""
    
    # Clean text first
    s = clean_text(s)
    
    s = s.replace(EM_DASH, "-").replace(YO, "е")
    s = s.replace("«", "\"").replace("»", "\"")
    s = re.sub(r"\r\n?|\n", "\n", s)
    s = re.sub(r"[ \t]+$", "", s, flags=re.MULTILINE)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"(?m)^([ \t]*)•\s*", lambda m: m.group(1) + "* ", s)
    return s.strip()

# --- bold detection from Telegram entities ---

def _build_utf16_index(s: str) -> List[int]:
    offsets = [0]
    u = 0
    for ch in s:
        u += 1 if ord(ch) <= 0xFFFF else 2
        offsets.append(u)
    return offsets

def _u16_to_py_index(u16_map: List[int], u16_pos: int) -> int:
    import bisect
    return bisect.bisect_left(u16_map, u16_pos)

def bold_ranges_from_entities(text: str, entities) -> List[Tuple[int, int]]:
    if not entities:
        return []
    u16 = _build_utf16_index(text)
    out: List[Tuple[int, int]] = []
    for e in entities:
        if isinstance(e, MessageEntityBold):
            start = _u16_to_py_index(u16, e.offset)
            end = _u16_to_py_index(u16, e.offset + e.length)
            if start < end:
                out.append((start, end))
    return out

def convert_bold_lines_to_h2(text: str, bold_ranges: List[Tuple[int, int]]) -> str:
    if not bold_ranges:
        return text
    bold_ranges = sorted(bold_ranges)
    merged: List[List[int]] = []
    for s, e in bold_ranges:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    def covered(a: int, b: int) -> bool:
        for s, e in merged:
            if a >= s and b <= e:
                return True
        return False

    out, pos = [], 0
    for line in text.splitlines(True):
        ln = line.rstrip("\n")
        start = pos
        end = pos + len(ln)
        pos += len(line)
        m = re.match(r"^[^\w\d#А-Яа-я]+\s*", ln)
        trim = m.end() if m else 0
        a = start + trim
        b = end
        if ln.strip() and covered(a, b):
            out.append("## " + ln.strip() + "\n")
        else:
            out.append(line)
    return "".join(out)

# --- parse title/description ---

def extract_title_and_description(text: str) -> Tuple[str, str, str]:
    """
    Берём первую непустую строку как title (без '#'),
    и ОСТАВЛЯЕМ её в теле как H1: '# {title}'.
    Описание — первая непустая "абзацная" секция после заголовка; из него убираем #хештеги.
    """
    if not text or not text.strip():
        return ("", "", "")
    
    lines = text.splitlines()
    # индекс первой непустой строки
    idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if idx is None:
        return ("", "", text.strip())

    # Заголовок без ведущих '#'
    title = lines[idx].lstrip("#").strip()

    # Описание: первый непустой абзац после title
    i = idx + 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    desc_lines: List[str] = []
    while i < len(lines) and lines[i].strip():
        desc_lines.append(lines[i].strip())
        i += 1
    description = " ".join(desc_lines).strip()
    if description:
        # убрать из description хеш-теги вида #tag
        description = re.sub(r"(?<!\w)#([\w\-]+)", r"\1", description)

    # В ТЕЛЕ оставляем заголовок как ровно один H1
    if idx < len(lines):
        lines[idx] = f"# {title}"

    body = "\n".join(lines).strip()
    return (title, description, body)

def parse_tags(text: str) -> List[str]:
    if not text:
        return []
    tags = set()
    for m in re.finditer(r"(?<!\w)#([\w\-]+)", text, flags=re.UNICODE):
        tags.add(m.group(1))
    return sorted(tags)

@dataclass
class Post:
    date: datetime
    text: str
    tags: List[str]
    image_path: Optional[Path]
    id: int
    bold_ranges: List[Tuple[int, int]]

async def download_media_with_retry(client, message, tmp_dir: Path, max_retries: int = 3) -> Optional[Path]:
    """Download media with retry logic and better error handling."""
    if not (message.photo or getattr(message, "media", None)):
        return None
    
    for attempt in range(max_retries):
        try:
            # Create unique filename with proper extension
            base_name = f"{message.id}_attempt_{attempt}"
            file_path = await client.download_media(message, file=tmp_dir / base_name)
            if file_path and Path(file_path).exists():
                return Path(file_path)
        except FloodWaitError as e:
            logging.warning(f"FloodWaitError during media download (attempt {attempt + 1}): waiting {e.seconds}s")
            await asyncio.sleep(e.seconds)
        except (OSError, IOError) as e:
            logging.warning(f"IO error during media download (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logging.warning(f"Unexpected error during media download (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
    
    logging.error(f"Failed to download media for message {message.id} after {max_retries} attempts")
    return None

async def handle_flood_wait(func, *args, max_wait: int = 300, **kwargs):
    """Handle FloodWaitError with automatic retry."""
    while True:
        try:
            return await func(*args, **kwargs)
        except FloodWaitError as e:
            if e.seconds > max_wait:
                logging.error(f"FloodWait too long: {e.seconds}s (max: {max_wait}s)")
                raise
            logging.info(f"FloodWaitError: sleeping for {e.seconds}s")
            await asyncio.sleep(e.seconds)
        except Exception:
            raise

async def fetch_posts(args) -> List[Post]:
    # Validate API credentials
    api_id = args.api_id or os.environ.get("TG_API_ID")
    api_hash = args.api_hash or os.environ.get("TG_API_HASH")
    
    if not api_id or not api_hash:
        raise SystemExit("Telegram api_id/api_hash are required. Set via arguments or environment variables TG_API_ID/TG_API_HASH")
    
    try:
        api_id = int(api_id)
    except ValueError:
        raise SystemExit("api_id must be a valid integer")
    
    session = args.session or "telegram_export.session"
    
    # Configure client with better error handling
    client = TelegramClient(session, api_id, api_hash)
    
    # Set flood sleep threshold for automatic handling of small waits
    client.flood_sleep_threshold = 60  # Auto-sleep for waits under 60s
    
    try:
        await client.start()
        logging.info("Connected to Telegram")
    except Exception as e:
        raise SystemExit(f"Failed to connect to Telegram: {e}")

    try:
        # Get entity with error handling
        try:
            entity = await handle_flood_wait(client.get_entity, args.channel)
            logging.info(f"Found channel: {getattr(entity, 'title', 'Unknown')} ({args.channel})")
        except UsernameNotOccupiedError:
            raise SystemExit(f"Channel not found: {args.channel}")
        except ChatAdminRequiredError:
            raise SystemExit(f"No access to private channel: {args.channel}")
        except Exception as e:
            raise SystemExit(f"Error accessing channel {args.channel}: {e}")

        posts: List[Post] = []
        tmp_dir = Path(args.tmp or ".telegram_tmp")
        tmp_dir.mkdir(exist_ok=True)
        
        # Count total messages first for progress bar
        try:
            total_messages = await client.get_messages(entity, limit=1)
            total_count = total_messages.total if hasattr(total_messages, 'total') else args.limit or 1000
            if args.limit:
                total_count = min(total_count, args.limit)
        except Exception:
            total_count = args.limit or 1000
        
        logging.info(f"Processing up to {total_count} messages...")
        
        with tqdm(total=total_count, desc="Fetching messages") as pbar:
            processed = 0
                    # Determine iteration strategy based on parameters
        if args.message_ids:
            # Export specific message IDs
            message_ids = [int(id.strip()) for id in args.message_ids.split(',') if id.strip().isdigit()]
            logging.info(f"Targeting specific message IDs: {message_ids}")

            for msg_id in tqdm(message_ids, desc="Processing specific messages"):
                try:
                    msg = await client.get_messages(entity, ids=msg_id)
                    if msg and not msg.empty:
                        processed += 1
                        pbar.update(1) if 'pbar' in locals() else None

                        if not getattr(msg, "date", None):
                            continue

                        msg_dt = msg.date
                        if msg_dt.tzinfo is None:
                            msg_dt = msg_dt.replace(tzinfo=timezone.utc)

                        raw_text = (msg.message or "")
                        raw_text = clean_text(raw_text)
                        bold_ranges = bold_ranges_from_entities(raw_text, getattr(msg, "entities", None))
                        text_with_h2 = convert_bold_lines_to_h2(raw_text, bold_ranges)
                        text = normalize_text(text_with_h2)

                        if not text and not args.include_empty:
                            continue

                        tags = parse_tags(text)

                        # Download media with retry logic
                        image_path: Optional[Path] = None
                        if not args.no_images:
                            image_path = await download_media_with_retry(
                                client, msg, tmp_dir, args.max_retries or 3
                            )

                        posts.append(Post(
                            date=msg_dt,
                            text=text,
                            tags=tags,
                            image_path=image_path,
                            id=msg.id,
                            bold_ranges=bold_ranges
                        ))

                except Exception as e:
                    logging.warning(f"Failed to process message {msg_id}: {e}")
                    continue

        elif args.min_id and args.max_id:
            # Export messages in ID range
            logging.info(f"Exporting messages from ID {args.min_id} to {args.max_id}")
            async for msg in client.iter_messages(entity, min_id=args.min_id-1, max_id=args.max_id):
                if processed >= total_count:
                    break
                processed += 1
                pbar.update(1)

                if not getattr(msg, "date", None):
                    continue

                msg_dt = msg.date
                if msg_dt.tzinfo is None:
                    msg_dt = msg_dt.replace(tzinfo=timezone.utc)

                raw_text = (msg.message or "")
                raw_text = clean_text(raw_text)
                bold_ranges = bold_ranges_from_entities(raw_text, getattr(msg, "entities", None))
                text_with_h2 = convert_bold_lines_to_h2(raw_text, bold_ranges)
                text = normalize_text(text_with_h2)

                if not text and not args.include_empty:
                    continue

                tags = parse_tags(text)

                # Download media with retry logic
                image_path: Optional[Path] = None
                if not args.no_images:
                    image_path = await download_media_with_retry(
                        client, msg, tmp_dir, args.max_retries or 3
                    )

                posts.append(Post(
                    date=msg_dt,
                    text=text,
                    tags=tags,
                    image_path=image_path,
                    id=msg.id,
                    bold_ranges=bold_ranges
                ))

                # Small delay to be nice to the API
                if processed % 10 == 0:
                    await asyncio.sleep(0.1)
        else:
            # Default: export by limit (fallback to original behavior)
            async for msg in client.iter_messages(entity, limit=args.limit or None):
                if processed >= total_count:
                    break
                    
                processed += 1
                pbar.update(1)
                
                if not getattr(msg, "date", None):
                    continue
                    
                msg_dt = msg.date
                if msg_dt.tzinfo is None:
                    msg_dt = msg_dt.replace(tzinfo=timezone.utc)

                raw_text = (msg.message or "")
                
                # Handle potential encoding issues
                raw_text = clean_text(raw_text)
                
                bold_ranges = bold_ranges_from_entities(raw_text, getattr(msg, "entities", None))
                text_with_h2 = convert_bold_lines_to_h2(raw_text, bold_ranges)
                text = normalize_text(text_with_h2)
                
                if not text and not args.include_empty:
                    continue

                tags = parse_tags(text)

                # Download media with retry logic
                image_path: Optional[Path] = None
                if not args.no_images:
                    image_path = await download_media_with_retry(
                        client, msg, tmp_dir, args.max_retries or 3
                    )

                posts.append(Post(
                    date=msg_dt, 
                    text=text, 
                    tags=tags, 
                    image_path=image_path, 
                    id=msg.id, 
                    bold_ranges=bold_ranges
                ))
                
                # Small delay to be nice to the API
                if processed % 10 == 0:
                    await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        logging.info("Export cancelled by user")
        return []
    except Exception as e:
        logging.error(f"Error during export: {e}")
        raise
    finally:
        try:
            await client.disconnect()
            logging.info("Disconnected from Telegram")
        except Exception:
            pass

    return list(reversed(posts))

def toml_front_matter(title: str, date: datetime, description: str, tags: List[str], image_rel: Optional[str]) -> str:
    """Generate TOML front matter with proper escaping."""
    def esc(s: str) -> str:
        if not s:
            return ""
        return s.replace("\\", r"\\").replace('"', r'\"').replace("\n", "\\n").replace("\r", "\\r")
    
    lines = ["+++"]
    lines.append(f'title = "{esc(title)}"')
    lines.append(f'date = "{date.astimezone(timezone.utc).isoformat()}"')
    if description:
        lines.append(f'description = "{esc(description)}"')
    if tags:
        taglist = ", ".join([f'"{esc(t)}"' for t in tags])
        lines.append(f"tags = [{taglist}]")
#    lines.append('categories = ["short"]')
    if image_rel:
        lines.append(f'image = "{esc(image_rel)}"')
    lines.append("toc = true")
    lines.append("readingTime = true")
    lines.append("draft = false")
    lines.append("+++")
    return "\n".join(lines) + "\n\n"

def _remove_trailing_hashtag_block(body: str) -> str:
    if not body:
        return ""
    lines = body.splitlines()
    def is_tagline(ln: str) -> bool:
        return re.fullmatch(r"(?:\s*#[\w\-]+)+\s*", ln) is not None
    while lines and (not lines[-1].strip() or is_tagline(lines[-1])):
        lines.pop()
    return "\n".join(lines)

def insert_dividers(body: str) -> str:
    """
    Вставляет разделитель '---' только между абзацами, когда:
    — предыдущая непустая строка НЕ является заголовком (не начинается с '#'), и
    — следующая непустая строка НЕ является заголовком.
    Во всех остальных случаях последовательность пустых строк сохраняется как есть.
    """
    if not body:
        return ""
    
    lines = body.splitlines()
    out: List[str] = []
    i = 0

    def is_header(s: str) -> bool:
        return s.lstrip().startswith("#")

    while i < len(lines):
        if lines[i].strip() != "":
            out.append(lines[i])
            i += 1
            continue

        # Последовательность пустых строк
        start = i
        while i < len(lines) and lines[i].strip() == "":
            i += 1

        # Ищем соседние непустые строки
        prev_nonempty = None
        for j in range(len(out) - 1, -1, -1):
            if out[j].strip() != "":
                prev_nonempty = out[j]
                break
        next_nonempty = lines[i] if i < len(lines) else None

        if (
            prev_nonempty is not None
            and next_nonempty is not None
            and not is_header(prev_nonempty)
            and not is_header(next_nonempty)
        ):
            # Стандартизированная вставка разделителя
            out.append("")
            out.append("---")
            out.append("")
        else:
            # Сохраняем исходное количество пустых строк
            out.extend(lines[start:i])

    return "\n".join(out)

def delete_session_files(session_path: str) -> None:
    """Best-effort remove Telethon SQLite session file and aux files (-journal, -wal, -shm)."""
    base = Path(session_path)
    candidates = [
        base,
        Path(str(base) + "-journal"),
        Path(str(base) + "-wal"),
        Path(str(base) + "-shm"),
    ]
    for pth in candidates:
        try:
            if pth.exists():
                pth.unlink()
                logging.info(f"Deleted session file: {pth}")
        except Exception as e:
            logging.warning(f"Could not delete {pth}: {e}")

def write_bundle(out_dir: Path, post: Post, slug_maxlen: int = 60, remove_hashtags: bool = True) -> Path:
    try:
        title, description, body = extract_title_and_description(post.text or f"post-{post.id}")

        base_title = title or f"post-{post.id}"
        words = re.findall(r"[\w\-\u0400-\u04FF]+", base_title)
        first_three = " ".join(words[:3]) if words else base_title
        slug = ascii_slugify(first_three, maxlen=slug_maxlen)

        year = post.date.strftime('%Y')
        month = post.date.strftime('%m')
        day = post.date.strftime('%d')
        bundle = out_dir / year / month / f"{day}-{slug}"
        bundle.mkdir(parents=True, exist_ok=True)

        image_rel = None
        if post.image_path and post.image_path.exists():
            try:
                ext = post.image_path.suffix or ".jpg"
                dest = bundle / f"{bundle.name}{ext}"
                dest.write_bytes(post.image_path.read_bytes())
                image_rel = dest.name
                post.image_path.unlink(missing_ok=True)  # cleanup temp file
            except Exception as e:
                logging.warning(f"Error processing image for post {post.id}: {e}")

        body = _remove_trailing_hashtag_block(body)
        if remove_hashtags:
            body = re.sub(r"(?<!\w)#([\w\-]+)", lambda m: m.group(1), body)

        body = re.sub(r"(?m)^\s*\*\*(.+?)\*\*\s*$", lambda m: "## " + m.group(1), body)
        body = insert_dividers(body)  # вставляем '---' только между абзацами, не рядом с заголовками
        body = "\n".join([ln if ln.endswith("  ") else (ln + "  ") for ln in body.splitlines()])

        fm = toml_front_matter(title=title or slug, date=post.date, description=description, tags=post.tags, image_rel=image_rel)
        (bundle / "index.md").write_text(fm + body + "\n", encoding="utf-8")
        return bundle
    except Exception as e:
        logging.error(f"Error writing bundle for post {post.id}: {e}")
        raise

def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Export Telegram channel posts to Hugo bundles",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--channel", required=True, help="Channel username or link")
    p.add_argument("--out", default="./content/posts", help="Output directory")
    p.add_argument("--limit", type=int, help="Limit number of posts to fetch")
    p.add_argument("--no-images", action="store_true", help="Skip downloading images")
    p.add_argument("--remove-hashtags", action="store_true", default=True, help="Remove hashtags from body")
    p.add_argument("--slug-maxlen", type=int, default=60, help="Maximum slug length")
    p.add_argument("--include-empty", action="store_true", help="Include empty messages")
    p.add_argument("--tmp", help="Temporary directory for media")
    p.add_argument("--dry-run", action="store_true", help="Show what would be done")
    p.add_argument("--api-id", help="Telegram API ID")
    p.add_argument("--api-hash", help="Telegram API hash")
    p.add_argument("--session", help="Session file path")
    p.add_argument("--delete-session", action="store_true", help="Delete session after run")
    p.add_argument("--max-retries", type=int, default=3, help="Max retries for media download")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    p.add_argument("--message-ids", help="Comma-separated list of specific message IDs to export")
    p.add_argument("--min-id", type=int, help="Minimum message ID to export")  
    p.add_argument("--max-id", type=int, help="Maximum message ID to export")
    p.add_argument("--update-ids", help="Comma-separated list of update IDs to process")

    args = p.parse_args(argv)
    
    # Setup logging
    setup_logging(args.verbose)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        posts = asyncio.run(fetch_posts(args))
        if not posts:
            logging.info("No posts to export.")
            return 0

        logging.info(f"Exporting {len(posts)} posts...")
        
        with tqdm(posts, desc="Writing bundles") as pbar:
            for post in pbar:
                if args.dry_run:
                    title, _, _ = extract_title_and_description(post.text)
                    words = re.findall(r"[\w\-\u0400-\u04FF]+", title)
                    first_three = " ".join(words[:3]) if words else title
                    slug = ascii_slugify(first_three or f"post-{post.id}", maxlen=args.slug_maxlen)
                    bundle_path = out_dir / post.date.strftime('%Y') / post.date.strftime('%m') / (post.date.strftime('%d') + '-' + slug)
                    logging.info(f"Would write {bundle_path}/index.md")
                else:
                    try:
                        bundle = write_bundle(out_dir, post, slug_maxlen=args.slug_maxlen, remove_hashtags=args.remove_hashtags)
                        pbar.set_description(f"Wrote {bundle.name}")
                    except Exception as e:
                        logging.error(f"Failed to write post {post.id}: {e}")
                        continue

        # Cleanup
        tmp_dir = Path(args.tmp or ".telegram_tmp")
        if tmp_dir.exists():
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                logging.info("Cleaned up temporary directory")
            except Exception as e:
                logging.warning(f"Could not cleanup temp dir: {e}")

        if args.delete_session:
            session_path = args.session or "telegram_export.session"
            delete_session_files(session_path)

        logging.info(f"Export completed successfully! Exported {len(posts)} posts.")
        return 0
        
    except KeyboardInterrupt:
        logging.info("Export cancelled by user")
        return 1
    except Exception as e:
        logging.error(f"Export failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())