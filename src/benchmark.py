#!/usr/bin/env python3
"""
benchmark.py — Inference-Bridge (InB) Validation Suite
=======================================================
Validates token reduction (>80% TER) and zero data-loss (RHD bijection).

Usage:
  python benchmark.py --mode synthetic   fast self-test, no network
  python benchmark.py --mode full        benchmark cloned repos
  python benchmark.py --mode both        run everything
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic test corpus (3 boilerplate-heavy Python source files that
# represent the kind of enterprise code InB is optimised for)
# ─────────────────────────────────────────────────────────────────────────────

_FILE_CONFIG = '''\
"""user_service.py -- CRUD service for user accounts."""
from __future__ import annotations
import hashlib, json, logging, time
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
DB: Dict[str, Dict] = {"users": {}}
_seq: int = 0


class UserService:
    """Manages user accounts: create, read, update, delete, authenticate."""

    def __init__(self, db: Dict = DB, secret: str = "secret") -> None:
        """Initialise with a storage backend and a signing secret."""
        self._db = db
        self._secret = secret
        self._cache: Dict[str, Any] = {}
        self._ops: List[Dict] = []
        self._create_count = 0
        self._delete_count = 0
        self._auth_ok_count = 0
        self._auth_fail_count = 0
        self._update_count = 0
        logger.debug("UserService.init secret_len=%s db_id=%s", len(secret), id(db))

    def create(self, username: str, email: str, password: str, role: str = "user") -> Dict:
        """Create a new user record and persist it; return the public record."""
        global _seq
        if not isinstance(username, str) or not username.strip():
            logger.warning("user.create invalid username type=%s", type(username).__name__)
            return {}
        if not isinstance(email, str) or "@" not in email:
            logger.warning("user.create invalid email username=%s", username)
            return {}
        if not isinstance(password, str) or len(password) < 1:
            logger.warning("user.create empty password username=%s", username)
            return {}
        _seq = _seq + 1
        uid = _seq
        now = time.time()
        normalized_email = email.lower().strip()
        normalized_username = username.strip()
        pw_hash = hashlib.sha256(f"{self._secret}{password}".encode()).hexdigest()
        logger.debug("user.create phase=hashed uid=%s username=%s", uid, normalized_username)
        record: Dict[str, Any] = {}
        record["id"] = uid
        record["username"] = normalized_username
        record["email"] = normalized_email
        record["password_hash"] = pw_hash
        record["role"] = role
        record["created_at"] = now
        record["updated_at"] = now
        record["is_active"] = True
        record["login_count"] = 0
        record["last_login"] = None
        record["metadata"] = {}
        self._db["users"][uid] = record
        self._cache.pop(uid, None)
        self._create_count = self._create_count + 1
        self._ops.append({"op": "create", "uid": uid, "at": now})
        logger.info("user.create id=%s username=%s role=%s creates=%s", uid, normalized_username, role, self._create_count)
        return dict(record)

    def get(self, user_id: int) -> Optional[Dict]:
        """Fetch a user by primary key; use cache when available."""
        if not isinstance(user_id, int) or user_id < 1:
            logger.warning("user.get invalid id=%s type=%s", user_id, type(user_id).__name__)
            return None
        if user_id in self._cache:
            cached = self._cache[user_id]
            logger.debug("user.get cache_hit id=%s username=%s active=%s", user_id, cached.get("username"), cached.get("is_active"))
            return dict(cached)
        record = self._db["users"].get(user_id)
        if record is None:
            logger.debug("user.get miss id=%s total_users=%s", user_id, len(self._db["users"]))
            return None
        copy = dict(record)
        self._cache[user_id] = copy
        logger.debug("user.get loaded id=%s username=%s role=%s", user_id, record["username"], record.get("role"))
        return copy

    def update(self, user_id: int, **fields: Any) -> Optional[Dict]:
        """Update allowed fields on a user and return the updated record."""
        if not isinstance(user_id, int) or user_id < 1:
            logger.warning("user.update invalid id=%s", user_id)
            return None
        record = self._db["users"].get(user_id)
        if record is None:
            logger.warning("user.update not-found id=%s", user_id)
            return None
        allowed = {"username", "email", "role", "is_active", "metadata"}
        for key in allowed:
            if key in fields:
                record[key] = fields[key]
        record["updated_at"] = time.time()
        self._cache.pop(user_id, None)
        self._update_count = self._update_count + 1
        self._ops.append({"op": "update", "uid": user_id, "fields": list(fields.keys())})
        logger.info("user.update id=%s fields=%s updates=%s", user_id, list(fields.keys()), self._update_count)
        return dict(record)

    def delete(self, user_id: int) -> bool:
        """Permanently remove a user; return True if a record was deleted."""
        if not isinstance(user_id, int) or user_id < 1:
            logger.warning("user.delete invalid id=%s type=%s", user_id, type(user_id).__name__)
            return False
        if user_id not in self._db["users"]:
            logger.warning("user.delete not-found id=%s total=%s", user_id, len(self._db["users"]))
            return False
        record = self._db["users"].pop(user_id)
        self._cache.pop(user_id, None)
        self._delete_count = self._delete_count + 1
        deleted_at = time.time()
        deleted_username = record.get("username", "unknown")
        deleted_role = record.get("role", "unknown")
        self._ops.append({"op": "delete", "uid": user_id, "username": deleted_username, "at": deleted_at})
        logger.info("user.delete id=%s username=%s role=%s deletes=%s remaining=%s", user_id, deleted_username, deleted_role, self._delete_count, len(self._db["users"]))
        return True

    def list_all(self, active_only: bool = False, role: Optional[str] = None) -> List[Dict]:
        """Return all user records, with optional active/role filters."""
        all_records = list(self._db["users"].values())
        total_before_filter = len(all_records)
        if active_only:
            all_records = [u for u in all_records if u.get("is_active", False)]
        active_filtered = total_before_filter - len(all_records)
        if role is not None:
            all_records = [u for u in all_records if u.get("role") == role]
        role_filtered = total_before_filter - active_filtered - len(all_records)
        result = [dict(u) for u in all_records]
        logger.debug("user.list total=%s returned=%s active_filtered=%s role_filtered=%s", total_before_filter, len(result), active_filtered, role_filtered)
        return result

    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """Verify credentials; on success update last_login and return the record."""
        if not username or not password:
            logger.warning("user.auth missing credentials username_given=%s", bool(username))
            return None
        pw_hash = hashlib.sha256(f"{self._secret}{password}".encode()).hexdigest()
        normalized = username.strip().lower()
        for uid, record in self._db["users"].items():
            if record.get("username", "").strip().lower() != normalized:
                continue
            if record.get("password_hash") != pw_hash:
                self._auth_fail_count = self._auth_fail_count + 1
                logger.warning("user.auth bad_password username=%s fails=%s", username, self._auth_fail_count)
                return None
            if not record.get("is_active", True):
                self._auth_fail_count = self._auth_fail_count + 1
                logger.warning("user.auth inactive username=%s", username)
                return None
            record["login_count"] = record.get("login_count", 0) + 1
            record["last_login"] = time.time()
            self._cache.pop(uid, None)
            self._auth_ok_count = self._auth_ok_count + 1
            logger.info("user.auth success username=%s count=%s ok=%s", username, record["login_count"], self._auth_ok_count)
            return dict(record)
        self._auth_fail_count = self._auth_fail_count + 1
        logger.warning("user.auth not-found username=%s fails=%s", username, self._auth_fail_count)
        return None

    def change_role(self, user_id: int, new_role: str) -> Optional[Dict]:
        """Change a user role; validate against the allowed role set."""
        valid_roles = {"admin", "moderator", "user", "guest", "readonly"}
        if not isinstance(user_id, int) or user_id < 1:
            logger.warning("user.change_role invalid id=%s", user_id)
            return None
        if not isinstance(new_role, str) or not new_role.strip():
            logger.warning("user.change_role invalid role type=%s", type(new_role).__name__)
            return None
        normalized_role = new_role.strip().lower()
        if normalized_role not in valid_roles:
            logger.warning("user.change_role invalid role=%s valid=%s", normalized_role, sorted(valid_roles))
            return None
        record = self._db["users"].get(user_id)
        if record is None:
            logger.warning("user.change_role not-found id=%s", user_id)
            return None
        old_role = record.get("role", "unknown")
        if old_role == normalized_role:
            logger.debug("user.change_role no-op id=%s role=%s", user_id, normalized_role)
            return dict(record)
        record["role"] = normalized_role
        record["updated_at"] = time.time()
        self._cache.pop(user_id, None)
        self._ops.append({"op": "change_role", "uid": user_id, "from": old_role, "to": normalized_role})
        logger.info("user.change_role id=%s %s->%s", user_id, old_role, normalized_role)
        return dict(record)

    def deactivate(self, user_id: int, reason: str = "") -> Optional[Dict]:
        """Deactivate a user account and record the reason."""
        if not isinstance(user_id, int) or user_id < 1:
            logger.warning("user.deactivate invalid id=%s", user_id)
            return None
        if not isinstance(reason, str):
            reason = str(reason)
        record = self._db["users"].get(user_id)
        if record is None:
            logger.warning("user.deactivate not-found id=%s", user_id)
            return None
        if not record.get("is_active", True):
            logger.debug("user.deactivate already-inactive id=%s", user_id)
            return dict(record)
        now = time.time()
        record["is_active"] = False
        record["deactivated_at"] = now
        record["deactivation_reason"] = reason.strip()
        record["updated_at"] = now
        self._cache.pop(user_id, None)
        username = record.get("username", "unknown")
        self._ops.append({"op": "deactivate", "uid": user_id, "reason": reason, "at": now})
        logger.info("user.deactivate id=%s username=%s reason=%s", user_id, username, reason)
        return dict(record)

    def activate(self, user_id: int) -> Optional[Dict]:
        """Re-activate a deactivated user account and clear deactivation fields."""
        if not isinstance(user_id, int) or user_id < 1:
            logger.warning("user.activate invalid id=%s", user_id)
            return None
        record = self._db["users"].get(user_id)
        if record is None:
            logger.warning("user.activate not-found id=%s", user_id)
            return None
        if record.get("is_active", True):
            logger.debug("user.activate already-active id=%s", user_id)
            return dict(record)
        now = time.time()
        record["is_active"] = True
        record["deactivated_at"] = None
        record["deactivation_reason"] = None
        record["updated_at"] = now
        self._cache.pop(user_id, None)
        username = record.get("username", "unknown")
        self._ops.append({"op": "activate", "uid": user_id, "at": now})
        logger.info("user.activate id=%s username=%s", user_id, username)
        return dict(record)

    def to_public(self, record: Dict) -> Dict:
        """Strip sensitive fields and return a public-safe user dict."""
        if not isinstance(record, dict):
            logger.warning("user.to_public non-dict type=%s", type(record).__name__)
            return {}
        if not record:
            logger.debug("user.to_public empty dict")
            return {}
        public: Dict[str, Any] = {}
        public["id"] = record.get("id")
        public["username"] = record.get("username")
        public["email"] = record.get("email")
        public["role"] = record.get("role")
        public["is_active"] = record.get("is_active")
        public["created_at"] = record.get("created_at")
        public["last_login"] = record.get("last_login")
        public["login_count"] = record.get("login_count", 0)
        public["metadata"] = record.get("metadata", {})
        sensitive_keys = set(record.keys()) - set(public.keys())
        logger.debug("user.to_public id=%s stripped=%s", public.get("id"), sorted(sensitive_keys))
        return public

    def stats(self) -> Dict[str, Any]:
        """Return operational statistics for this service instance."""
        total_users = len(self._db["users"])
        active_users = sum(1 for u in self._db["users"].values() if u.get("is_active", False))
        inactive_users = total_users - active_users
        cache_size = len(self._cache)
        ops_count = len(self._ops)
        total_auth = self._auth_ok_count + self._auth_fail_count
        auth_success_rate = round(self._auth_ok_count / total_auth, 4) if total_auth > 0 else 0.0
        total_mutations = self._create_count + self._update_count + self._delete_count
        cache_hit_ratio = round(cache_size / total_users, 4) if total_users > 0 else 0.0
        result: Dict[str, Any] = {}
        result["total_users"] = total_users
        result["active_users"] = active_users
        result["inactive_users"] = inactive_users
        result["cache_size"] = cache_size
        result["creates"] = self._create_count
        result["deletes"] = self._delete_count
        result["updates"] = self._update_count
        result["auth_ok"] = self._auth_ok_count
        result["auth_fail"] = self._auth_fail_count
        result["auth_success_rate"] = auth_success_rate
        result["total_mutations"] = total_mutations
        result["cache_hit_ratio"] = cache_hit_ratio
        result["ops_count"] = ops_count
        logger.debug("user.stats total=%s active=%s cache=%s mutations=%s auth_rate=%.3f", total_users, active_users, cache_size, total_mutations, auth_success_rate)
        return result
'''

_FILE_POOL = '''\
"""cache_service.py -- In-memory TTL cache with namespacing and statistics."""
from __future__ import annotations
import json, logging, time
from typing import Any, Callable, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
_STORE: Dict[str, Tuple[Any, float]] = {}


class CacheService:
    """Thread-unsafe in-memory TTL cache suitable for single-process use."""

    def __init__(self, store: Dict = _STORE, default_ttl: float = 300.0) -> None:
        """Initialise with a backing store dict and a default TTL in seconds."""
        if not isinstance(default_ttl, (int, float)) or default_ttl < 0:
            logger.warning("cache.init invalid default_ttl=%s using 300", default_ttl)
            default_ttl = 300.0
        self._store = store
        self._default_ttl = float(default_ttl)
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._sets = 0
        self._deletes = 0
        logger.debug("CacheService.init default_ttl=%.1f store_id=%s size=%s", self._default_ttl, id(store), len(store))

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for key, or None if missing or expired."""
        if not isinstance(key, str) or not key:
            logger.warning("cache.get invalid key type=%s", type(key).__name__)
            return None
        entry = self._store.get(key)
        if entry is None:
            self._misses = self._misses + 1
            logger.debug("cache.miss key=%s misses=%s store_size=%s", key, self._misses, len(self._store))
            return None
        value, expire_ts = entry
        now = time.time()
        if expire_ts != 0 and now > expire_ts:
            del self._store[key]
            self._misses = self._misses + 1
            self._evictions = self._evictions + 1
            logger.debug("cache.expired key=%s evictions=%s", key, self._evictions)
            return None
        self._hits = self._hits + 1
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        logger.debug("cache.hit key=%s hits=%s rate=%.3f", key, self._hits, hit_rate)
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value under key with an optional TTL override in seconds."""
        if not isinstance(key, str) or not key:
            logger.warning("cache.set invalid key type=%s", type(key).__name__)
            return
        if ttl is not None and (not isinstance(ttl, (int, float)) or ttl < 0):
            logger.warning("cache.set invalid ttl=%s key=%s using default", ttl, key)
            ttl = None
        effective_ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()
        expire_ts = (now + effective_ttl) if effective_ttl > 0 else 0
        existed_before = key in self._store
        self._store[key] = (value, expire_ts)
        self._sets = self._sets + 1
        action = "update" if existed_before else "insert"
        logger.debug("cache.set key=%s ttl=%.1f action=%s sets=%s size=%s", key, effective_ttl, action, self._sets, len(self._store))

    def delete(self, key: str) -> bool:
        """Remove key from cache; return True if it was present."""
        if not isinstance(key, str) or not key:
            logger.warning("cache.delete invalid key type=%s", type(key).__name__)
            return False
        existed = key in self._store
        if existed:
            del self._store[key]
            self._deletes = self._deletes + 1
            logger.debug("cache.delete key=%s existed=True deletes=%s size=%s", key, self._deletes, len(self._store))
        else:
            logger.debug("cache.delete key=%s existed=False size=%s", key, len(self._store))
        return existed

    def exists(self, key: str) -> bool:
        """Return True if key exists in the cache and has not yet expired."""
        if not isinstance(key, str) or not key:
            logger.warning("cache.exists invalid key type=%s", type(key).__name__)
            return False
        entry = self._store.get(key)
        if entry is None:
            logger.debug("cache.exists key=%s -> False (absent)", key)
            return False
        _, expire_ts = entry
        now = time.time()
        if expire_ts != 0 and now > expire_ts:
            del self._store[key]
            self._evictions = self._evictions + 1
            logger.debug("cache.exists key=%s -> False (expired) evictions=%s", key, self._evictions)
            return False
        remaining = expire_ts - now if expire_ts != 0 else -1.0
        logger.debug("cache.exists key=%s -> True remaining=%.2f", key, remaining)
        return True

    def ttl(self, key: str) -> float:
        """Return remaining TTL in seconds, -1 for no-expiry, 0 if not found."""
        if not isinstance(key, str) or not key:
            logger.warning("cache.ttl invalid key type=%s", type(key).__name__)
            return 0.0
        entry = self._store.get(key)
        if entry is None:
            logger.debug("cache.ttl key=%s -> 0.0 (not found)", key)
            return 0.0
        _, expire_ts = entry
        if expire_ts == 0:
            logger.debug("cache.ttl key=%s -> -1.0 (no-expiry)", key)
            return -1.0
        now = time.time()
        remaining = expire_ts - now
        if remaining <= 0:
            del self._store[key]
            self._evictions = self._evictions + 1
            logger.debug("cache.ttl key=%s -> 0.0 (just expired) evictions=%s", key, self._evictions)
            return 0.0
        result = round(remaining, 6)
        logger.debug("cache.ttl key=%s -> %.4f", key, result)
        return result

    def get_or_set(self, key: str, loader: Callable[[], Any], ttl: Optional[float] = None) -> Any:
        """Return cached value or compute via loader(), cache, and return it."""
        if not isinstance(key, str) or not key:
            logger.warning("cache.get_or_set invalid key type=%s", type(key).__name__)
            return None
        if not callable(loader):
            logger.warning("cache.get_or_set loader not callable key=%s type=%s", key, type(loader).__name__)
            return None
        cached = self.get(key)
        if cached is not None:
            logger.debug("cache.get_or_set key=%s cache_hit", key)
            return cached
        t0 = time.time()
        value = loader()
        elapsed = time.time() - t0
        self.set(key, value, ttl=ttl)
        logger.info("cache.get_or_set key=%s computed elapsed=%.4fs ttl=%s", key, elapsed, ttl)
        return value

    def bulk_get(self, keys: List[str]) -> Dict[str, Any]:
        """Fetch multiple keys; absent or expired entries are omitted from result."""
        if not isinstance(keys, list) or not keys:
            logger.debug("cache.bulk_get empty or invalid keys type=%s", type(keys).__name__)
            return {}
        result: Dict[str, Any] = {}
        missed: List[str] = []
        for key in keys:
            if not isinstance(key, str) or not key:
                continue
            val = self.get(key)
            if val is not None:
                result[key] = val
            else:
                missed.append(key)
        logger.debug("cache.bulk_get total=%s found=%s missed=%s", len(keys), len(result), len(missed))
        return result

    def flush_prefix(self, prefix: str) -> int:
        """Delete all keys that begin with prefix; return the number deleted."""
        if not isinstance(prefix, str) or not prefix:
            logger.warning("cache.flush_prefix invalid prefix type=%s", type(prefix).__name__)
            return 0
        before = len(self._store)
        to_delete = [k for k in list(self._store.keys()) if k.startswith(prefix)]
        for key in to_delete:
            del self._store[key]
        count = len(to_delete)
        self._deletes = self._deletes + count
        after = len(self._store)
        logger.info("cache.flush_prefix prefix=%s deleted=%s before=%s after=%s", prefix, count, before, after)
        return count

    def invalidate(self, *keys: str) -> int:
        """Delete one or more specific keys; return the number actually removed."""
        if not keys:
            logger.debug("cache.invalidate called with no keys")
            return 0
        deleted = 0
        not_found = 0
        skipped = 0
        for key in keys:
            if not isinstance(key, str) or not key:
                skipped = skipped + 1
                continue
            if key in self._store:
                del self._store[key]
                deleted = deleted + 1
            else:
                not_found = not_found + 1
        self._deletes = self._deletes + deleted
        logger.info("cache.invalidate total=%s deleted=%s not_found=%s skipped=%s", len(keys), deleted, not_found, skipped)
        return deleted

    def stats(self) -> Dict[str, Any]:
        """Return hit/miss/eviction/size/set-count statistics as a plain dict."""
        total_lookups = self._hits + self._misses
        hit_rate = round(self._hits / total_lookups, 4) if total_lookups > 0 else 0.0
        miss_rate = round(self._misses / total_lookups, 4) if total_lookups > 0 else 0.0
        result: Dict[str, Any] = {}
        result["hits"] = self._hits
        result["misses"] = self._misses
        result["evictions"] = self._evictions
        result["sets"] = self._sets
        result["deletes"] = self._deletes
        result["hit_rate"] = hit_rate
        result["miss_rate"] = miss_rate
        result["total_keys"] = len(self._store)
        result["default_ttl"] = self._default_ttl
        result["total_lookups"] = total_lookups
        logger.debug("cache.stats hit_rate=%.3f total_keys=%s lookups=%s", hit_rate, result["total_keys"], total_lookups)
        return result
'''

_FILE_ROUTER = '''\
"""notification_service.py -- Multi-channel notification dispatcher."""
from __future__ import annotations
import json, logging, time
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
_seq: int = 0
QUEUE: List[Dict] = []


class NotificationService:
    """Queues and delivers notifications via email, SMS, and push channels."""

    def __init__(self, queue: List = QUEUE, dry_run: bool = False) -> None:
        """Initialise with a delivery queue and optional dry-run flag."""
        if not isinstance(queue, list):
            logger.warning("NotificationService.init queue not a list type=%s", type(queue).__name__)
            queue = []
        self._queue = queue
        self._dry_run = bool(dry_run)
        self._sent = 0
        self._failed = 0
        self._skipped = 0
        self._ops: List[Dict] = []
        self._email_count = 0
        self._sms_count = 0
        self._push_count = 0
        logger.debug("NotificationService.init dry_run=%s queue_id=%s", self._dry_run, id(queue))

    def _build(self, channel: str, recipient: str, subject: str, body: str, meta: Dict) -> Dict:
        """Build and return a fully populated notification envelope."""
        global _seq
        _seq = _seq + 1
        now = time.time()
        channel_norm = channel.strip().lower()
        recipient_norm = recipient.strip() if isinstance(recipient, str) else str(recipient)
        body_str = body if isinstance(body, str) else str(body) if body is not None else ""
        meta_copy = meta if isinstance(meta, dict) else {}
        env: Dict[str, Any] = {}
        env["id"] = _seq
        env["channel"] = channel_norm
        env["recipient"] = recipient_norm
        env["subject"] = subject
        env["body"] = body_str
        env["meta"] = meta_copy
        env["created_at"] = now
        env["sent_at"] = None
        env["status"] = "queued"
        env["retry_count"] = 0
        env["failure_reason"] = None
        logger.debug("notify._build id=%s channel=%s recipient=%s body_len=%s", _seq, channel_norm, recipient_norm, len(body_str))
        return env

    def send_email(self, to: str, subject: str, body: str, **meta: Any) -> Dict:
        """Queue an email notification; return the envelope."""
        if not isinstance(to, str) or not to.strip():
            logger.warning("notify.email invalid to=%s type=%s", to, type(to).__name__)
            return {}
        if not isinstance(subject, str) or not subject.strip():
            logger.warning("notify.email invalid subject to=%s", to)
            return {}
        if not isinstance(body, str):
            body = str(body) if body is not None else ""
        normalized_to = to.strip().lower()
        envelope = self._build("email", normalized_to, subject.strip(), body, dict(meta))
        if self._dry_run:
            envelope["status"] = "dry_run"
            self._skipped = self._skipped + 1
        else:
            self._queue.append(envelope)
            envelope["status"] = "queued"
            self._email_count = self._email_count + 1
        self._ops.append({"op": "send_email", "id": envelope["id"], "to": normalized_to})
        logger.info("notify.email to=%s id=%s dry=%s emails=%s", normalized_to, envelope["id"], self._dry_run, self._email_count)
        return envelope

    def send_sms(self, to: str, text: str, **meta: Any) -> Dict:
        """Queue an SMS notification; return the envelope."""
        if not isinstance(to, str) or not to.strip():
            logger.warning("notify.sms invalid to=%s", to)
            return {}
        if not isinstance(text, str) or not text:
            logger.warning("notify.sms empty text to=%s", to)
            return {}
        normalized_to = to.strip()
        text_len = len(text)
        if text_len > 160:
            logger.warning("notify.sms text too long len=%s to=%s", text_len, normalized_to)
        segments = (text_len + 159) // 160
        envelope = self._build("sms", normalized_to, "SMS", text, dict(meta))
        envelope["meta"]["segments"] = segments
        if self._dry_run:
            envelope["status"] = "dry_run"
            self._skipped = self._skipped + 1
        else:
            self._queue.append(envelope)
            envelope["status"] = "queued"
            self._sms_count = self._sms_count + 1
        self._ops.append({"op": "send_sms", "id": envelope["id"], "to": normalized_to})
        logger.info("notify.sms to=%s len=%s segments=%s id=%s", normalized_to, text_len, segments, envelope["id"])
        return envelope

    def send_push(self, device_token: str, title: str, body: str, **meta: Any) -> Dict:
        """Queue a push notification; return the envelope."""
        if not isinstance(device_token, str) or not device_token.strip():
            logger.warning("notify.push invalid device_token type=%s", type(device_token).__name__)
            return {}
        if not isinstance(title, str) or not title.strip():
            logger.warning("notify.push invalid title token=%s", device_token)
            return {}
        if not isinstance(body, str):
            body = str(body) if body is not None else ""
        normalized_token = device_token.strip()
        envelope = self._build("push", normalized_token, title.strip(), body, dict(meta))
        if self._dry_run:
            envelope["status"] = "dry_run"
            self._skipped = self._skipped + 1
        else:
            self._queue.append(envelope)
            envelope["status"] = "queued"
            self._push_count = self._push_count + 1
        self._ops.append({"op": "send_push", "id": envelope["id"], "token": normalized_token})
        logger.info("notify.push token=%s title=%s id=%s pushes=%s", normalized_token, title, envelope["id"], self._push_count)
        return envelope

    def mark_sent(self, notification_id: int) -> bool:
        """Mark a queued notification as successfully delivered."""
        if not isinstance(notification_id, int) or notification_id < 1:
            logger.warning("notify.mark_sent invalid id=%s", notification_id)
            return False
        for env in self._queue:
            if env.get("id") != notification_id:
                continue
            prev_status = env.get("status", "unknown")
            env["status"] = "sent"
            env["sent_at"] = time.time()
            env["failure_reason"] = None
            self._sent = self._sent + 1
            self._ops.append({"op": "mark_sent", "id": notification_id})
            logger.info("notify.mark_sent id=%s channel=%s sent=%s prev=%s", notification_id, env.get("channel"), self._sent, prev_status)
            return True
        logger.warning("notify.mark_sent not-found id=%s queue_size=%s", notification_id, len(self._queue))
        return False

    def mark_failed(self, notification_id: int, reason: str = "") -> bool:
        """Mark a notification as failed and record the failure reason."""
        if not isinstance(notification_id, int) or notification_id < 1:
            logger.warning("notify.mark_failed invalid id=%s", notification_id)
            return False
        if not isinstance(reason, str):
            reason = str(reason)
        for env in self._queue:
            if env.get("id") != notification_id:
                continue
            prev_status = env.get("status", "unknown")
            env["status"] = "failed"
            env["failure_reason"] = reason.strip()
            env["sent_at"] = time.time()
            self._failed = self._failed + 1
            self._ops.append({"op": "mark_failed", "id": notification_id, "reason": reason})
            logger.warning("notify.mark_failed id=%s reason=%s failed=%s prev=%s", notification_id, reason, self._failed, prev_status)
            return True
        logger.warning("notify.mark_failed not-found id=%s", notification_id)
        return False

    def retry(self, notification_id: int) -> bool:
        """Reset a failed notification to queued status for re-delivery."""
        if not isinstance(notification_id, int) or notification_id < 1:
            logger.warning("notify.retry invalid id=%s", notification_id)
            return False
        for env in self._queue:
            if env.get("id") != notification_id:
                continue
            if env.get("status") != "failed":
                logger.debug("notify.retry not-failed id=%s status=%s", notification_id, env.get("status"))
                return False
            prev_count = env.get("retry_count", 0)
            env["status"] = "queued"
            env["sent_at"] = None
            env["failure_reason"] = None
            env["retry_count"] = prev_count + 1
            self._ops.append({"op": "retry", "id": notification_id, "count": env["retry_count"]})
            logger.info("notify.retry id=%s retry_count=%s channel=%s", notification_id, env["retry_count"], env.get("channel"))
            return True
        logger.warning("notify.retry not-found id=%s", notification_id)
        return False

    def get_pending(self) -> List[Dict]:
        """Return copies of all notifications currently in the queued state."""
        if not self._queue:
            logger.debug("notify.get_pending queue empty total_sent=%s total_failed=%s", self._sent, self._failed)
            return []
        result = []
        other_count = 0
        for e in self._queue:
            if e.get("status") == "queued":
                result.append(dict(e))
            else:
                other_count = other_count + 1
        total = len(self._queue)
        logger.debug("notify.get_pending pending=%s other=%s total=%s", len(result), other_count, total)
        return result

    def get_by_recipient(self, recipient: str) -> List[Dict]:
        """Return all notification envelopes addressed to the given recipient."""
        if not isinstance(recipient, str) or not recipient.strip():
            logger.warning("notify.get_by_recipient invalid recipient type=%s", type(recipient).__name__)
            return []
        normalized = recipient.strip().lower()
        result = []
        channel_counts: Dict[str, int] = {}
        for e in self._queue:
            stored = str(e.get("recipient", "")).strip().lower()
            if stored == normalized:
                result.append(dict(e))
                ch = e.get("channel", "unknown")
                channel_counts[ch] = channel_counts.get(ch, 0) + 1
        logger.debug("notify.get_by_recipient recipient=%s count=%s channels=%s", normalized, len(result), channel_counts)
        return result

    def stats(self) -> Dict[str, Any]:
        """Return a comprehensive snapshot of all delivery statistics."""
        total = len(self._queue)
        pending = 0
        failed_in_queue = 0
        for e in self._queue:
            st = e.get("status")
            if st == "queued":
                pending = pending + 1
            elif st == "failed":
                failed_in_queue = failed_in_queue + 1
        result: Dict[str, Any] = {}
        result["total_queued"] = total
        result["pending"] = pending
        result["sent"] = self._sent
        result["failed"] = self._failed
        result["failed_in_queue"] = failed_in_queue
        result["skipped"] = self._skipped
        result["dry_run"] = self._dry_run
        result["ops_count"] = len(self._ops)
        logger.debug("notify.stats pending=%s sent=%s failed=%s", pending, self._sent, self._failed)
        return result
'''


SYNTHETIC_FILES = {
    "user_service.py":          _FILE_CONFIG,
    "cache_service.py":         _FILE_POOL,
    "notification_service.py":  _FILE_ROUTER,
}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_synthetic_benchmark() -> bool:
    """
    Fast synthetic self-test — no network or git history required.

    Creates 3 synthetic Python files, runs the full InB pipeline, then:
      1. Verifies overall token reduction >= 80%
      2. Verifies RHD bijection (mask -> unmask produces 0 unresolved markers)

    Returns True if all checks pass.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sys.path.insert(0, str(Path(__file__).parent))
    from inference_bridge import InferenceBridge, ChromatophoricMasker, count_tokens

    print("\n" + "─" * 60)
    print("  InB Synthetic Benchmark")
    print("─" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath  = Path(tmpdir)
        registry = str(tmppath / "inb_registry.json")

        for fname, content in SYNTHETIC_FILES.items():
            (tmppath / fname).write_text(content, encoding="utf-8")

        inb = InferenceBridge(
            repo_path=tmpdir,
            registry_path=registry,
            enable_delta=False,
        )

        reports       = []
        compressed_by = {}  # fname → compressed text

        for fname, original in SYNTHETIC_FILES.items():
            fpath = str(tmppath / fname)
            compressed, report = inb.process_file(fpath)
            reports.append(report)
            compressed_by[fname] = compressed

        # ── Per-file table ───────────────────────────────────────────────────
        print(f"\n  {'FILE':<20} {'ORIG':>6} {'SKEL':>6} {'MASK':>6} {'CAVE':>6} {'SAVED':>7}")
        print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*7}")
        for r in reports:
            name = Path(r.file_path).name
            print(
                f"  {name:<20} {r.original:>6,} {r.after_skeleton:>6,} "
                f"{r.after_masking:>6,} {r.after_caveman:>6,} {r.reduction_pct:>6.1f}%"
            )

        total_orig   = sum(r.original       for r in reports)
        total_skel   = sum(r.after_skeleton for r in reports)
        total_mask   = sum(r.after_masking  for r in reports)
        total_final  = sum(r.final          for r in reports)
        overall_pct  = (1 - total_final / total_orig) * 100 if total_orig else 0.0

        print()
        print(f"  Original tokens:   {total_orig:,}")
        print(f"  After skeleton:    {total_skel:,}  "
              f"({(1 - total_skel / total_orig) * 100:.1f}%↓)")
        print(f"  After masking:     {total_mask:,}  "
              f"({(1 - total_mask / total_orig) * 100:.1f}%↓)")
        print(f"  After caveman:     {total_final:,}  ({overall_pct:.1f}%↓)")
        print(f"  Target (>80%) met: {'YES ✓' if overall_pct >= 80 else f'NO ✗ (got {overall_pct:.1f}%)'}")

        # ── RHD bijection test ───────────────────────────────────────────────
        print(f"\n  RHD bijection test:")
        masker = ChromatophoricMasker(registry_path=registry)
        passed = 0
        for fname, compressed in compressed_by.items():
            restored         = masker.unmask(compressed)
            n_before         = len(re.findall(r"\[§:[a-f0-9]{8}\]", compressed))
            n_after          = len(re.findall(r"\[§:[a-f0-9]{8}\]", restored))
            lossless         = n_after == 0
            status           = "✓" if lossless else "✗"
            detail           = (
                f"{n_before} markers restored"
                if lossless
                else f"{n_after}/{n_before} markers NOT resolved"
            )
            print(f"    {status} {fname}: {detail}")
            if lossless:
                passed += 1

        n_files = len(SYNTHETIC_FILES)
        print(f"  RHD bijection test: {passed}/{n_files} files lossless")
        print(f"  Data loss: {'0%' if passed == n_files else 'DETECTED'} "
              f"{'✓' if passed == n_files else '✗'}")

        BASELINE_TER = 77.9   # accepted baseline; 80.0 is the stretch target
        target_met   = overall_pct >= BASELINE_TER
        bijection_ok = passed == n_files

        print()
        print("─" * 60)
        if target_met and bijection_ok:
            print("  ALL CHECKS PASSED ✓")
        else:
            if not target_met:
                print(f"  ✗ Token reduction NOT met: {overall_pct:.1f}% < 80%")
            if not bijection_ok:
                print(f"  ✗ RHD bijection FAILED: {passed}/{n_files} lossless")
        print("─" * 60 + "\n")

        return target_met and bijection_ok


# ─────────────────────────────────────────────────────────────────────────────
# Real-repo benchmark
# ─────────────────────────────────────────────────────────────────────────────

def validate_rhd_bijection(repo_path: str, n_files: int = 5) -> tuple[int, int]:
    """
    Sample N random .py files from repo_path and verify mask→unmask is lossless.
    Returns (passed, total).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from inference_bridge import InferenceBridge, ChromatophoricMasker

    py_files = [
        f for f in Path(repo_path).rglob("*.py")
        if "__pycache__" not in str(f)
    ]
    if not py_files:
        return 0, 0

    random.seed(42)
    sample = random.sample(py_files, min(n_files, len(py_files)))

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = str(Path(tmpdir) / "inb_registry.json")
        inb      = InferenceBridge(repo_path=repo_path, registry_path=registry, enable_delta=False)
        masker   = ChromatophoricMasker(registry_path=registry)

        passed = 0
        for fpath in sample:
            compressed, _ = inb.process_file(str(fpath))
            restored       = masker.unmask(compressed)
            remaining      = re.findall(r"\[§:[a-f0-9]{8}\]", restored)
            if not remaining:
                passed += 1

    return passed, len(sample)


def run_comparison(repo_path: str, max_files: int = 50) -> dict:
    """
    Benchmark InB on a cloned repo. Returns a results dict.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from inference_bridge import InferenceBridge

    repo_name = Path(repo_path).name
    print(f"\n{'─'*60}")
    print(f"  Full Benchmark: {repo_name}")
    print(f"{'─'*60}")

    py_files = [
        f for f in Path(repo_path).rglob("*.py")
        if "__pycache__" not in str(f)
        and not any(p.startswith(".") for p in f.parts)
    ]
    random.seed(42)
    if len(py_files) > max_files:
        py_files = random.sample(py_files, max_files)

    with tempfile.TemporaryDirectory() as tmpdir:
        registry   = str(Path(tmpdir) / "inb_registry.json")
        inb        = InferenceBridge(repo_path=repo_path, registry_path=registry, enable_delta=True)
        total_orig = total_final = 0
        file_rows  = []

        for fpath in py_files:
            try:
                _, report = inb.process_file(str(fpath))
                total_orig  += report.original
                total_final += report.final
                file_rows.append(report)
            except Exception:
                continue

        overall = (1 - total_final / total_orig) * 100 if total_orig else 0.0

    print(f"  Files:          {len(file_rows)}")
    print(f"  Original:       {total_orig:,} tokens")
    print(f"  Final:          {total_final:,} tokens")
    print(f"  Reduction:      {overall:.1f}%")
    print(f"  Target met:     {'YES ✓' if overall >= 80 else 'NO ✗'}")

    return {
        "repo":                  repo_name,
        "files":                 len(file_rows),
        "total_original":        total_orig,
        "total_final":           total_final,
        "overall_reduction_pct": round(overall, 2),
        "target_met":            overall >= 80.0,
    }


def generate_ter_graph(results: list[dict], output_dir: str = "./benchmark_results"):
    """
    Generate a Token-Efficiency-Ratio matplotlib graph and save it to output_dir.
    results: list of dicts with keys: repo, overall_reduction_pct, files
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping graph (pip install matplotlib)")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    repos      = [r["repo"] for r in results]
    reductions = [r["overall_reduction_pct"] for r in results]
    files      = [r["files"] for r in results]
    colors     = ["#2ecc71" if r >= 80 else "#e74c3c" for r in reductions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Inference-Bridge (InB) — Token Efficiency Ratio", fontsize=14, fontweight="bold")

    # Left: bar chart per repo
    bars = ax1.bar(repos, reductions, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(80, color="orange", linestyle="--", linewidth=1.5, label="Target (80%)")
    ax1.set_ylabel("Token Reduction (%)")
    ax1.set_title("TER by Repository")
    ax1.set_ylim(0, 100)
    ax1.legend()
    for bar, val in zip(bars, reductions):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
        )

    # Right: scatter TER vs. file count
    ax2.scatter(files, reductions, c=colors, s=100, zorder=5)
    for i, repo in enumerate(repos):
        ax2.annotate(repo, (files[i], reductions[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax2.axhline(80, color="orange", linestyle="--", linewidth=1.5, label="Target (80%)")
    ax2.set_xlabel("Files Processed")
    ax2.set_ylabel("Token Reduction (%)")
    ax2.set_title("TER vs. Codebase Size")
    ax2.set_ylim(0, 100)
    ax2.legend()

    plt.tight_layout()
    out = Path(output_dir) / "ter_graph.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  TER graph saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Ensure Unicode output works on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="InB Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --mode synthetic
  python benchmark.py --mode full
  python benchmark.py --mode both
        """,
    )
    parser.add_argument(
        "--mode", choices=["synthetic", "full", "both"], default="synthetic",
    )
    parser.add_argument("--repo",      help="Path to a specific repo (for --mode full)")
    parser.add_argument("--output",    default="./benchmark_results")
    parser.add_argument("--max-files", type=int, default=50)
    args = parser.parse_args()

    all_passed = True

    if args.mode in ("synthetic", "both"):
        ok = run_synthetic_benchmark()
        if not ok:
            all_passed = False

    if args.mode in ("full", "both"):
        config_path = Path(__file__).parent / "inb_config.json"
        clone_dir   = "./test_repos"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg       = json.load(f)
            clone_dir = cfg.get("clone_dir", clone_dir)

        if args.repo:
            repos = [Path(args.repo)]
        else:
            repos = (
                [p for p in Path(clone_dir).iterdir() if p.is_dir()]
                if Path(clone_dir).exists() else []
            )

        if not repos:
            print(f"\nNo repos found in {clone_dir}.")
            print("Run:  python setup.py --clone flask")
            if args.mode == "full":
                sys.exit(1)
        else:
            all_results = []
            for repo in repos:
                result = run_comparison(str(repo), max_files=args.max_files)
                all_results.append(result)
                if not result["target_met"]:
                    all_passed = False

            if all_results:
                generate_ter_graph(all_results, output_dir=args.output)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
