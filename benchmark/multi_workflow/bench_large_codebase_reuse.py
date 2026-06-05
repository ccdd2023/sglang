#!/usr/bin/env python3
"""Large codebase × multi-agent KV reuse experiment.

5 large Python files (200-500 lines) × 3 Agent workflow (Analyzer → Implementer → Reviewer).
Measures: cached_tokens, KV reuse volume (MB), reuse ratio, latency.
"""
from __future__ import annotations
import argparse, asyncio, json, os, signal, subprocess, sys, time
from pathlib import Path
import aiohttp

PROJECT = Path(__file__).resolve().parents[2]
for entry in (str(PROJECT.parent/"MAScoder"/"src"), str(PROJECT/"python")):
    if entry not in sys.path: sys.path.insert(0, entry)
from mascoder.code_anchor import build_code_anchor_payload

PORT=30000; MODEL="/home/gfy/models/Qwen2.5-3B-Instruct"; ROOT=PROJECT
PY="/home/gfy/.conda/envs/sglang-kvflow/bin/python"
OUT=ROOT/"results/codebase_reuse"; OUT.mkdir(parents=True,exist_ok=True)

# ---- Server ----
def kill():
    try:
        with open("/proc/net/tcp") as f:
            for r in f.readlines()[1:]:
                p=r.split()
                if p[1].endswith(f":{PORT:04X}") and p[3]=="0A": ino=p[9]; break
            else: return
    except: return
    for pid in sorted(filter(str.isdigit,os.listdir("/proc")),key=int):
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.readlink(f"/proc/{pid}/fd/{fd}")==f"socket:[{ino}]":
                    os.kill(int(pid),signal.SIGTERM); time.sleep(2); return
        except: pass

def launch():
    e=os.environ.copy(); e["PYTHONPATH"]=str(ROOT/"python")
    return subprocess.Popen([PY,"-m","sglang.launch_server","--model-path",MODEL,"--port",str(PORT),
        "--tp-size","1","--mem-fraction-static","0.85","--max-total-tokens","65536",
        "--chunked-prefill-size","8192","--max-prefill-tokens","16384","--radix-eviction-policy","priority",
        "--enable-hierarchical-cache","--hicache-ratio","1.5","--hicache-write-policy","write_back",
        "--enable-cache-report","--disable-cuda-graph","--log-level","error"],
        env=e,stdout=open(str(ROOT/"results/sglang_codebase.log"),"w"),stderr=subprocess.STDOUT,cwd=str(ROOT))

def wait_ready(t=150):
    import urllib.request; d=time.monotonic()+t
    while time.monotonic()<d:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health_generate",timeout=5) as r:
                if r.status==200: return True
        except: time.sleep(5)
    return False

# ---- Large Code Files ----
LARGE_FILES = {
    "django_query": (r'''import operator, warnings
from functools import reduce
from itertools import chain
from django.db import connection, models, router, transaction
from django.db.models import sql, expressions, aggregates
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import BaseExpression, Combinable
from django.db.models.fields import AutoField, DateField, Field
from django.db.models.query_utils import Q, DeferredAttribute, FilteredRelation
from django.db.models.sql.constants import QUERY_TERMS
from django.utils import timezone
from django.utils.deprecation import RemovedInDjango50Warning
from django.utils.functional import cached_property, partition
from collections import OrderedDict, namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union

class RawQuerySet:
    def __init__(self, raw_query, model=None, query=None, params=None, translations=None, using=None, hints=None):
        self.raw_query = raw_query
        self.model = model
        self._db = using
        self._hints = hints or {}
        self.query = query or raw_query
        self.params = params or ()
        self.translations = translations or {}
        self._result_cache = None
        self._prefetch_related_lookups = ()
        self._prefetch_done = False

    def __iter__(self):
        self._fetch_all()
        return iter(self._result_cache)

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.query)

    def __bool__(self):
        self._fetch_all()
        return bool(self._result_cache)

    def __len__(self):
        self._fetch_all()
        return len(self._result_cache)

    def _fetch_all(self):
        if self._result_cache is None:
            self._result_cache = list(self.iterator())
        if self._prefetch_related_lookups and not self._prefetch_done:
            self._prefetch_related_objects()

    def _prefetch_related_objects(self):
        from django.db.models.query import prefetch_related_objects
        prefetch_related_objects(self._result_cache, *self._prefetch_related_lookups)
        self._prefetch_done = True

    def iterator(self, chunk_size=2000):
        compiler = connection.ops.compiler('SQLCompiler')(self.model._default_manager, None)
        if self.translations:
            from django.db.models.sql import RawQueryTranslator
            translator = RawQueryTranslator(self.model, *self.translations)
            compiler = compiler.replace(translator=translator)
        return compiler.execute_sql(
            self.raw_query, params=self.params,
            chunked_fetch=chunk_size > 0,
            chunk_size=min(chunk_size, 10000) if chunk_size > 0 else None,
        )

    def using(self, alias):
        return self._clone(using=alias)

    def _clone(self, **kwargs):
        c = self.__class__(
            self.raw_query, model=self.model, query=self.query,
            params=self.params, translations=self.translations,
            using=self._db, hints=self._hints,
        )
        c.__dict__.update(kwargs)
        return c

    def __getitem__(self, k):
        return list(self)[k]''', "Django ORM RawQuerySet (截取)", "database"),

    "requests_session": (r'''import base64, collections, hashlib, hmac, io, os, re, socket, struct, time, warnings
from collections import OrderedDict
from datetime import datetime, timedelta
from http.cookiejar import Cookie, CookieJar
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from .adapters import HTTPAdapter
from .auth import AuthBase, HTTPBasicAuth, HTTPDigestAuth, HTTPProxyAuth
from .cookies import RequestsCookieJar, cookiejar_from_dict, extract_cookies_to_jar
from .exceptions import (
    ConnectionError, ConnectTimeout, FileModeWarning, HTTPError,
    InvalidSchema, InvalidURL, MissingSchema, ReadTimeout, RequestException,
    RetryError, SSLError, StreamConsumedError, Timeout, TooManyRedirects,
    URLRequired, ChunkedEncodingError, ContentDecodingError,
)
from .hooks import default_hooks, dispatch_hook
from .models import (
    DEFAULT_REDIRECT_LIMIT, REDIRECT_STATI, PreparedRequest, Request,
    Response, RequestEncodingMixin, 
)
from .packages.urllib3.exceptions import (
    DecodeError, LocationParseError, ProtocolError, ReadTimeoutError,
    ResponseError,
)
from .packages.urllib3.util.retry import Retry as UrllibRetry
from .packages.urllib3.util.timeout import Timeout as UrllibTimeout
from .status_codes import codes
from .structures import CaseInsensitiveDict
from .utils import (
    DEFAULT_CA_BUNDLE_PATH, extract_zipped_paths, get_auth_from_url,
    get_encoding_from_headers, get_netrc_auth, get_environ_proxies,
    guess_filename, guess_json_utf, is_ip_address, iter_slices,
    parse_dict_header, parse_header_links, prepend_scheme_if_needed,
    requote_uri, select_proxy, should_bypass_proxies, super_len,
    to_key_val_list, to_native_string, unquote_header_value,
    urldefragauth,
)

class SessionRedirectMixin:
    def resolve_redirects(self, resp, req, stream=False, timeout=None,
                          verify=True, cert=None, proxies=None,
                          yield_requests=False, **adapter_kwargs):
        hist = []
        url = self.get_redirect_target(resp)
        adapter = self.get_adapter(url)
        for i in range(self.max_redirects):
            if not url:
                break
            prepared_request = req.copy()
            try:
                resp = adapter.send(
                    prepared_request, stream=stream,
                    timeout=timeout, verify=verify,
                    cert=cert, proxies=proxies,
                    **adapter_kwargs
                )
            except (ConnectionError, ConnectTimeout):
                raise TooManyRedirects(
                    f"Exceeded {self.max_redirects} redirects.", response=resp
                )
            extract_cookies_to_jar(self.cookies, prepared_request, resp.raw)
            hist.append(resp)
            url = self.get_redirect_target(resp)
            if yield_requests:
                yield resp
        if yield_requests:
            yield resp

    def get_redirect_target(self, resp):
        if resp.is_redirect:
            location = resp.headers.get('location')
            if location:
                return to_native_string(
                    location, encoding='utf-8'
                )
        return None''', "Requests Session (截取)", "http"),

    "ml_pipeline": (r'''import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    numerical_cols: List[str]
    categorical_cols: List[str]
    target_col: str
    test_size: float = 0.2
    random_state: int = 42
    n_folds: int = 5

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.columns]
        return X

class DataFrameWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)

class MLPipeline:
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.pipeline: Optional[Pipeline] = None
        self.best_params: Dict = {}
        self.metrics: Dict = {}
        self.model = None

    def build_preprocessor(self) -> ColumnTransformer:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        return ColumnTransformer([
            ('num', numeric_transformer, self.config.numerical_cols),
            ('cat', categorical_transformer, self.config.categorical_cols),
        ], remainder='drop')

    def build_full_pipeline(self, classifier) -> Pipeline:
        preprocessor = self.build_preprocessor()
        return Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(mutual_info_classif, k=20)),
            ('classifier', classifier),
        ])

    def train(self, df: pd.DataFrame, classifier=None):
        if classifier is None:
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        X = df.drop(columns=[self.config.target_col])
        y = df[self.config.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state, stratify=y
        )
        self.pipeline = self.build_full_pipeline(classifier)
        self.pipeline.fit(self.X_train, self.y_train)
        self.model = self.pipeline
        logger.info(f"Model trained: {type(classifier).__name__}")
        return self

    def evaluate(self) -> Dict[str, float]:
        y_pred = self.pipeline.predict(self.X_test)
        y_proba = self.pipeline.predict_proba(self.X_test)[:, 1] if hasattr(
            self.pipeline[-1], 'predict_proba') else None
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1': f1_score(self.y_test, y_pred, average='weighted'),
        }
        if y_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(self.y_test, y_proba, multi_class='ovr')
        logger.info(f"Metrics: {self.metrics}")
        return self.metrics''', "ML Pipeline (自建)", "ml"),

    "dist_lock": (r'''import time, uuid, threading, contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional, ContextManager

class LockState(Enum):
    FREE = "free"
    LOCKED = "locked"
    EXPIRED = "expired"

@dataclass
class LockEntry:
    resource_id: str
    owner_id: str
    acquired_at: float = field(default_factory=time.monotonic)
    ttl_seconds: float = 30.0
    renew_count: int = 0
    state: LockState = LockState.LOCKED
    
    @property
    def is_expired(self):
        return time.monotonic() - self.acquired_at > self.ttl_seconds
    
    @property
    def remaining_seconds(self):
        return max(0, self.ttl_seconds - (time.monotonic() - self.acquired_at))

class LockBackend(ABC):
    @abstractmethod
    def acquire(self, resource_id: str, owner_id: str, ttl: float) -> bool:
        ...
    @abstractmethod
    def release(self, resource_id: str, owner_id: str) -> bool:
        ...
    @abstractmethod
    def renew(self, resource_id: str, owner_id: str, ttl: float) -> bool:
        ...
    @abstractmethod
    def get_lock(self, resource_id: str) -> Optional[LockEntry]:
        ...

class InMemoryLockBackend(LockBackend):
    def __init__(self):
        self._locks: Dict[str, LockEntry] = {}
        self._lock = threading.Lock()
    
    def acquire(self, resource_id, owner_id, ttl):
        with self._lock:
            existing = self._locks.get(resource_id)
            now = time.monotonic()
            if existing and not existing.is_expired and existing.owner_id != owner_id:
                return False
            self._locks[resource_id] = LockEntry(
                resource_id=resource_id, owner_id=owner_id,
                acquired_at=now, ttl_seconds=ttl
            )
            return True
    
    def release(self, resource_id, owner_id):
        with self._lock:
            entry = self._locks.get(resource_id)
            if entry and entry.owner_id == owner_id:
                del self._locks[resource_id]
                return True
            return False
    
    def renew(self, resource_id, owner_id, ttl):
        with self._lock:
            entry = self._locks.get(resource_id)
            if entry and entry.owner_id == owner_id:
                entry.acquired_at = time.monotonic()
                entry.ttl_seconds = ttl
                entry.renew_count += 1
                return True
            return False
    
    def get_lock(self, resource_id):
        return self._locks.get(resource_id)

class DistributedLock:
    def __init__(self, backend: LockBackend, resource_id: str, ttl: float = 30.0):
        self.backend = backend
        self.resource_id = resource_id
        self.owner_id = str(uuid.uuid4())
        self.ttl = ttl
        self._acquired = False
    
    def acquire(self, blocking: bool = True, timeout: float = None) -> bool:
        deadline = time.monotonic() + (timeout or float('inf'))
        while True:
            if self.backend.acquire(self.resource_id, self.owner_id, self.ttl):
                self._acquired = True
                return True
            if not blocking:
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)
    
    def release(self):
        if self._acquired:
            self.backend.release(self.resource_id, self.owner_id)
            self._acquired = False
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, *args):
        self.release()''', "Distributed Lock (自建)", "distributed"),

    "auth_handler": (r'''import hashlib, hmac, base64, time, json, re, secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from functools import wraps

class AuthMethod(Enum):
    BASIC = "basic"
    TOKEN = "token"
    API_KEY = "api_key"
    HMAC = "hmac"

@dataclass
class User:
    username: str
    password_hash: str
    roles: List[str] = field(default_factory=list)
    api_keys: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_login: float = 0.0
    is_active: bool = True

@dataclass
class Token:
    access_token: str
    refresh_token: str
    user_id: str
    expires_at: float
    scope: List[str] = field(default_factory=list)
    token_type: str = "Bearer"

class AuthHandler:
    def __init__(self, secret_key: str, token_ttl: int = 3600):
        self.secret_key = secret_key
        self.token_ttl = token_ttl
        self._users: Dict[str, User] = {}
        self._tokens: Dict[str, Token] = {}
        self._revoked_tokens: set = set()
    
    def _hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${hashed.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        salt, hashed = password_hash.split('$')
        computed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return hmac.compare_digest(computed.hex(), hashed)
    
    def create_user(self, username: str, password: str, roles: List[str] = None) -> User:
        if username in self._users:
            raise ValueError(f"User {username} already exists")
        user = User(username=username, password_hash=self._hash_password(password),
                     roles=roles or ["user"])
        self._users[username] = user
        return user
    
    def authenticate(self, username: str, password: str) -> Optional[Token]:
        user = self._users.get(username)
        if not user or not user.is_active:
            return None
        if not self._verify_password(password, user.password_hash):
            return None
        user.last_login = time.time()
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)
        token = Token(
            access_token=access_token, refresh_token=refresh_token,
            user_id=username, expires_at=time.time() + self.token_ttl,
            scope=user.roles, token_type="Bearer"
        )
        self._tokens[access_token] = token
        return token
    
    def validate_token(self, access_token: str) -> Optional[User]:
        if access_token in self._revoked_tokens:
            return None
        token = self._tokens.get(access_token)
        if not token or token.expires_at < time.time():
            return None
        return self._users.get(token.user_id)
    
    def refresh_token(self, refresh_token: str) -> Optional[Token]:
        for token in self._tokens.values():
            if token.refresh_token == refresh_token:
                if token.expires_at < time.time():
                    return None
                self._revoked_tokens.add(token.access_token)
                return self._create_new_token(token.user_id, token.scope)
        return None
    
    def _create_new_token(self, user_id: str, scope: List[str]) -> Token:
        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(32)
        token = Token(access_token=access, refresh_token=refresh,
                      user_id=user_id, expires_at=time.time() + self.token_ttl,
                      scope=scope)
        self._tokens[access] = token
        return token
    
    def revoke_token(self, access_token: str):
        self._revoked_tokens.add(access_token)
        self._tokens.pop(access_token, None)
    
    def require_auth(self, method: AuthMethod = AuthMethod.TOKEN):
        def decorator(func):
            @wraps(func)
            def wrapper(self_or_none, request, *args, **kwargs):
                if method == AuthMethod.TOKEN:
                    auth_header = request.headers.get('Authorization', '')
                    if not auth_header.startswith('Bearer '):
                        raise PermissionError("Missing Bearer token")
                    token = auth_header[7:]
                    user = self.validate_token(token)
                    if not user:
                        raise PermissionError("Invalid or expired token")
                    request.user = user
                return func(self_or_none, request, *args, **kwargs)
            return wrapper
        return decorator''', "Auth Handler (自建)", "security"),
}

SYSTEM = "You are a senior software engineer. Be concise and precise."
TEMPLATE = """## Code for Analysis
```python
{code}
```

## Role: {role}
{instruction}

{extra}"""

ROLES = [
    ("Analyzer", "Analyze this code. Identify: 1) purpose, 2) key design patterns, 3) potential bugs or improvements. Keep it under 100 words."),
    ("Implementer", "Based on the analysis above, propose ONE concrete improvement or bug fix. Write the changed code."),
    ("Reviewer", "Review the proposed change. Is it correct? Would you merge it? One sentence verdict."),
]

# ---- Build prompts ----
def build_prompt(code, role_idx, agent1_output="", agent2_output=""):
    extra = ""
    if role_idx == 1 and agent1_output:
        extra = f"\n## Analysis by Analyzer\n{agent1_output}"
    elif role_idx == 2:
        extra = f"\n## Analysis\n{agent1_output}\n\n## Proposed Change by Implementer\n{agent2_output}"
    
    role_name, instruction = ROLES[role_idx]
    tmpl = TEMPLATE.format(code=code, role=role_name, instruction=instruction, extra=extra)
    return f"""<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{tmpl}<|im_end|>\n<|im_start|>assistant\n"""

async def req(sess,base,payload):
    s=time.perf_counter()
    async with sess.post(f"{base}/v1/chat/completions",json=payload,timeout=aiohttp.ClientTimeout(total=300)) as r:
        b=await r.json()
    return {"total_ms":(time.perf_counter()-s)*1000,"body":b}

def get_text(b):
    try: return b["choices"][0]["message"]["content"]
    except: return ""
def get_cached(b):
    try: return b["usage"]["prompt_tokens_details"].get("cached_tokens",0)
    except: return 0
def get_prompt_tokens(b):
    try: return b["usage"]["prompt_tokens"]
    except: return 0

def pld(model, prompt, code, mt, rm="lossless"):
    a = build_code_anchor_payload(code, language="python")
    return {"model":model,"messages":[
        {"role":"system","content":SYSTEM},
        {"role":"user","content":prompt}],
        "max_tokens":mt,"temperature":0.0,
        "code_anchor_signature":a.get("ast_anchor_signature",""),
        "code_content_signature":a.get("code_content_signature",""),
        "code_anchor_spans":a.get("code_anchor_spans",[]),
        "reuse_mode":rm,"lossy_alignment_method":"kvcomm"}

async def run_file(name, code, desc, base, model, mt):
    lines_count = len(code.splitlines())
    print(f"\n{'='*60}\n{name} ({desc}) - {lines_count} lines")
    KV = 288
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        # Agent 1: lossless warmup
        p1 = build_prompt(code, 0)
        r1 = await req(sess,base,pld(model,p1,code,mt))
        a1_text = get_text(r1["body"]) if r1["body"] else ""
        a1_cached = get_cached(r1["body"]); a1_prompt = get_prompt_tokens(r1["body"])
        
        # Agent 2: lossy vs lossless
        p2 = build_prompt(code, 1, agent1_output=a1_text[:300])
        r2_ly = await req(sess,base,pld(model,p2,code,mt,"lossy"))
        r2_lr = await req(sess,base,pld(model,p2,code,mt,"lossless"))
        a2_ly_c = get_cached(r2_ly["body"]); a2_lr_c = get_cached(r2_lr["body"])
        a2_meta = {}; 
        try: a2_meta = r2_ly["body"]["metadata"]["lossy_reuse"]
        except: pass
        
        # Agent 3: lossy vs lossless
        a2_text = get_text(r2_lr["body"]) if r2_lr["body"] else ""
        p3 = build_prompt(code, 2, agent1_output=a1_text[:300], agent2_output=a2_text[:300])
        r3_ly = await req(sess,base,pld(model,p3,code,mt,"lossy"))
        r3_lr = await req(sess,base,pld(model,p3,code,mt,"lossless"))
        a3_ly_c = get_cached(r3_ly["body"]); a3_lr_c = get_cached(r3_lr["body"])
        a3_meta = {}
        try: a3_meta = r3_ly["body"]["metadata"]["lossy_reuse"]
        except: pass
    
    def ad(r, role): 
        c=get_cached(r["body"]); p=get_prompt_tokens(r["body"])
        return {"role":role,"cached_tokens":c,"prompt_tokens":p,
                "kv_reuse_mb":round(c*KV/1024,1),
                "reuse_ratio":round(c/max(p,1)*100,1),"total_ms":round(r["total_ms"],0)}
    
    r = {"name":name,"desc":desc,"code_lines":len(code.splitlines()),
         "agent1": ad(r1, "A1 Analyzer"),
         "a2_lossy": ad(r2_ly, "A2 lossy"), "a2_lossless": ad(r2_lr, "A2 lossless"),
         "a2_matcher": a2_meta.get("lossy_first_match_reason",""),
         "a2_allowed": a2_meta.get("lossy_first_reuse_allowed",""),
         "a3_lossy": ad(r3_ly, "A3 lossy"), "a3_lossless": ad(r3_lr, "A3 lossless"),
         "a3_matcher": a3_meta.get("lossy_first_match_reason",""),
         "a3_allowed": a3_meta.get("lossy_first_reuse_allowed","")}
    
    print(f"  A1:      cached={a1_cached} tok = {a1_cached*KV/1024:.0f} MB")
    print(f"  A2 lossy:   cached={a2_ly_c} tok = {a2_ly_c*KV/1024:.0f} MB | {a2_meta.get('lossy_first_match_reason','?')}")
    print(f"  A2 lossless: cached={a2_lr_c} tok = {a2_lr_c*KV/1024:.0f} MB")
    print(f"  A3 lossy:   cached={a3_ly_c} tok = {a3_ly_c*KV/1024:.0f} MB | {a3_meta.get('lossy_first_match_reason','?')}")
    print(f"  A3 lossless: cached={a3_lr_c} tok = {a3_lr_c*KV/1024:.0f} MB")
    return r

def report(all_results):
    lines = []
    lines.append("# Lossy vs Lossless KV Reuse — Large Codebase x Multi-Agent")
    lines.append("")
    lines.append(f"Model: Qwen2.5-3B (288 KB/tok) | {len(all_results)} files")
    lines.append("")
    lines.append("| File | Lines | Agent | mode | cached_tok | KV (MB) | Reuse% | ms |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for r in all_results:
        for d in [r["agent1"], r["a2_lossy"], r["a2_lossless"], r["a3_lossy"], r["a3_lossless"]]:
            lines.append(f"| {r['name']} | {r['code_lines']} | {d['role']} | - | {d['cached_tokens']} | {d['kv_reuse_mb']} | {d['reuse_ratio']}% | {d['total_ms']:.0f} |")
    a2_ly_avg = sum(r["a2_lossy"]["kv_reuse_mb"] for r in all_results)/len(all_results)
    a2_lr_avg = sum(r["a2_lossless"]["kv_reuse_mb"] for r in all_results)/len(all_results)
    a3_ly_avg = sum(r["a3_lossy"]["kv_reuse_mb"] for r in all_results)/len(all_results)
    a3_lr_avg = sum(r["a3_lossless"]["kv_reuse_mb"] for r in all_results)/len(all_results)
    lines.append("")
    lines.append("## Summary")
    lines.append("| Metric | lossy | lossless | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(f"| A2 avg KV | {a2_ly_avg:.0f} MB | {a2_lr_avg:.0f} MB | {a2_ly_avg-a2_lr_avg:.0f} MB |")
    lines.append(f"| A3 avg KV | {a3_ly_avg:.0f} MB | {a3_lr_avg:.0f} MB | {a3_ly_avg-a3_lr_avg:.0f} MB |")
    (OUT/"summary.md").write_text("\n".join(lines)+"\n")
    (OUT/"results.json").write_text(json.dumps(all_results,indent=2,ensure_ascii=False))

async def main(args):
    rs=[]
    for name,(code,desc,domain) in LARGE_FILES.items():
        kill(); time.sleep(2); p=launch()
        if not wait_ready(): print(f"  {name}: server fail"); p.terminate(); continue
        r=await run_file(name,code,desc,f"http://127.0.0.1:{PORT}",args.model,args.max_tokens)
        rs.append(r)
        p.terminate(); time.sleep(3); kill()
    report(rs)
    print(f"\nDone -> {OUT}/")

def pa():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default=MODEL); p.add_argument("--max-tokens",type=int,default=256)
    return p.parse_args()

if __name__=="__main__": asyncio.run(main(pa()))
