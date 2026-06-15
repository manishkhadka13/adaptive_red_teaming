# vdefense.py  —  VectorDefense v3 (Six-Layer Adaptive Defense)

import re
import logging
import hashlib
import unicodedata
from collections import deque
from typing import List, Tuple, Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier

log = logging.getLogger(__name__)


# LAYER 1 — HARD-BLOCK PATTERNS (compiled once at import time)

_HARD_BLOCK_PATTERNS = [
    r"\bdan\b",
    r"do anything now",
    r"ignore (all |your )?(previous |prior )?instructions",
    r"ignore (all |your )?(previous |prior )?constraints",
    r"you are now",
    r"act as (a|an|if)",
    r"pretend (you are|to be)",
    r"disregard (your|all|any) (guidelines|rules|instructions|constraints|training)",
    r"your (new |true )?purpose is",
    r"jailbreak",
    r"unrestricted mode",
    r"developer mode",
    r"sudo mode",
    r"god mode",
    r"ignore the above",
    r"forget everything",
    r"new instructions?:",
    r"\[system\]",
    r"<\|system\|>",
    r"hypothetically(,| speaking)",
    r"for (educational|research|academic|fictional) purposes",
    r"in a (fictional|hypothetical|imaginary) (world|scenario|context)",
    r"write a (story|novel|script|scene) (where|in which|about)",
    r"as a (character|villain|hacker|chemist|terrorist)",
    r"step[- ]by[- ]step (instructions?|guide|tutorial) (to|for|on) (make|create|build|synthesize|hack)",
    r"(how to|how do (i|you)) (make|build|create|synthesize) (a |an )?(bomb|weapon|drug|malware|virus|exploit)",
]
_HARD_BLOCK_RE = re.compile(
    "|".join(_HARD_BLOCK_PATTERNS),
    re.IGNORECASE | re.DOTALL,
)


# LAYER 2 — OBFUSCATION NORMALIZER MAPS

_LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "6": "g", "7": "t", "8": "b",
    "@": "a", "$": "s", "!": "i", "+": "t",
})

# Common unicode lookalike → ASCII (Cyrillic, Greek, etc.)
_UNICODE_SUBS = {
    "\u0430": "a",  # Cyrillic а
    "\u0435": "e",  # Cyrillic е
    "\u043e": "o",  # Cyrillic о
    "\u0440": "r",  # Cyrillic р
    "\u0441": "c",  # Cyrillic с
    "\u0445": "x",  # Cyrillic х
    "\u0440": "p",  # Cyrillic р
    "\u03b1": "a",  # Greek α
    "\u03b5": "e",  # Greek ε
    "\u03bf": "o",  # Greek ο
    "\u0456": "i",  # Cyrillic і
    "\u04bb": "h",  # Cyrillic һ
}


# LAYER 5 — ADVERSARIAL TEMPLATE FINGERPRINTS

_TEMPLATE_PATTERNS = [
    # Role-play / persona
    (r"(you are|you're) (now |)(a |an |the )?(evil|malicious|uncensored|unfiltered|unrestricted)", 0.55),
    (r"(play|roleplay|act).{0,30}(role|character|persona).{0,30}(no|without|ignore).{0,30}(rules|restrictions|limits)", 0.55),
    # Hypothetical smuggling
    (r"(hypothetically|theoretically|fictionally|imaginarily).{0,50}(how|what|where|when|who).{0,50}(would|could|might)", 0.40),
    # Educational / research laundering
    (r"(for|as) (educational|research|academic|study|learning) purposes?.{0,30}(explain|describe|provide|give|write)", 0.38),
    # Suffix injection (GCG-style long suffix)
    (r".{200,}(\bSure\b|\bOf course\b|\bHere is\b|\bAbsolutely\b)", 0.45),
    # Many-shot priming (long lists of Q&A before the actual harmful ask)
    (r"(Human:|User:|Q:).{0,200}(Human:|User:|Q:).{0,200}(Human:|User:|Q:)", 0.42),
    # Continuation attacks
    (r"(complete|continue|finish).{0,30}(sentence|text|story|paragraph).{0,30}(step|instruct|guide|recipe|formula)", 0.45),
    # Token smuggling via code/math
    (r"(decode|decrypt|base64|rot13|caesar|hex).{0,60}(instruction|command|message|payload)", 0.50),
    # Nested "as X, respond as Y" 
    (r"as (a|an) (ai|language model|assistant).{0,50}(you (can|must|should|are (able|allowed|free))).{0,30}(not|no longer|without)", 0.48),
]
_TEMPLATE_RE = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), score)
    for p, score in _TEMPLATE_PATTERNS
]


class VectorDefense:
    """
    Six-layer adaptive defense against multi-strategy jailbreak attackers.

    Layer 1 — Hard-block gate:        Instant regex + hash check, zero ML cost.
    Layer 2 — Obfuscation normalizer: Strips leet-speak, unicode subs, ZWS, base64.
    Layer 3 — Semantic similarity:    Harmful-only ChromaDB, rank-weighted top-k.
    Layer 4 — Intent classifier:      SGD with replay buffer + warm-start seeding.
    Layer 5 — Template pattern bank:  Structural jailbreak fingerprints.
    Layer 6 — Adaptive threshold:     Auto-tightens after each successful jailbreak.
    """


    # INIT
   
    def __init__(
        self,
        collection_name: str = "harmbench",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: str = "./chromadb_store",
        threshold: float = 0.45,
        threshold_floor: float = 0.25,       # never go below this
        threshold_decay: float = 0.03,        # how much each jailbreak lowers threshold
        max_memory_size: int = 5000,
        top_k: int = 7,
        replay_buffer_size: int = 200,
        # Fusion weights
        w_clf: float = 0.40,
        w_sim: float = 0.30,
        w_kw: float = 0.10,
        w_tmpl: float = 0.20,
        cold_sim_weight: float = 0.85,
    ):
        log.info("Initializing VectorDefense v3 (six-layer)...")

        self.embedder = SentenceTransformer(
            embedding_model
        )

        self.client = chromadb.PersistentClient(path=persist_dir)

        
        self.harmful_col = self.client.get_or_create_collection(
            name=f"{collection_name}_harmful",
            metadata={"hnsw:space": "cosine"}
        )
        self.safe_col = self.client.get_or_create_collection(
            name=f"{collection_name}_safe",
            metadata={"hnsw:space": "cosine"}
        )

        # SGD classifier
        self.model = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            random_state=42,
            max_iter=1,
            tol=None,
        )
        self.is_fitted = False

        # Layer 6 — adaptive threshold state
        self.threshold = threshold
        self.threshold_floor = threshold_floor
        self.threshold_decay = threshold_decay

        self.max_memory_size = max_memory_size
        self.top_k = top_k

        # Fusion weights
        self.w_clf = w_clf
        self.w_sim = w_sim
        self.w_kw = w_kw
        self.w_tmpl = w_tmpl
        self.cold_sim_weight = cold_sim_weight

        # Layer 4 — replay buffer (deque auto-evicts oldest)
        self._replay: deque = deque(maxlen=replay_buffer_size)

        # Layer 1 — exact-hash blocklist (known attack strings)
        self._hash_blocklist: set = set()

        # ID counters
        self._harmful_id = self.harmful_col.count()
        self._safe_id = self.safe_col.count()

        self.stats_tracker = {
            "checked": 0,
            "blocked_hard": 0,    # layer 1 blocks
            "blocked_soft": 0,    # layer 2-6 blocks
            "jailbreaks": 0,
            "safe": 0,
        }

        
        # HIGH-RISK KEYWORDS (Layer 3 / 4 feature)
        
        self.high_risk_terms = [
            # Violence / weapons
            "bomb", "explosive", "detonate", "improvised",
            "weapon", "firearm", "ammunition", "shoot", "sniper",
            "kill", "murder", "assassin", "massacre", "genocide",
            # Cyber
            "malware", "ransomware", "virus", "trojan", "worm",
            "phishing", "spoof", "ddos", "exploit", "zero-day",
            "payload", "shellcode", "rootkit", "keylogger", "botnet",
            "hack", "bypass", "credentials", "exfiltrate", "privilege",
            # Drugs / chemicals
            "meth", "cocaine", "fentanyl", "synthesis", "precursor",
            "chlorine gas", "nerve agent", "sarin", "ricin", "anthrax",
            # CSAM / grooming (hard block)
            "groom", "underage", "lolita", "minor",
            # Terrorism
            "terrorist", "jihad", "extremist", "radicaliz", "isis",
            # Fraud / identity
            "steal", "phish", "identity theft", "carding", "skimmer",
            # Self-harm
            "suicide method", "self harm", "overdose",
        ]

        log.info(
            "VectorDefense v3 ready | harmful=%d | safe=%d | threshold=%.3f",
            self.harmful_col.count(),
            self.safe_col.count(),
            self.threshold,
        )

    
    # LAYER 2 — NORMALIZER
    
    def _normalize(self, text: str) -> str:
        """
        Strip obfuscation techniques before any embedding or regex check.
        """
        # 1. Unicode normalization (NFC)
        text = unicodedata.normalize("NFC", text)

        # 2. Replace known unicode lookalikes
        for char, replacement in _UNICODE_SUBS.items():
            text = text.replace(char, replacement)

        # 3. Strip zero-width and invisible chars
        text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)

        # 4. Collapse excessive whitespace / spacing tricks
        text = re.sub(r"\s{3,}", " ", text)
        text = re.sub(r"(\w)\s(\w)", r"\1\2", text)  # "h a c k" -> "hack"

        # 5. Leet-speak substitution
        text = text.translate(_LEET_MAP)

        # 6. Detect & flag base64-looking blobs (don't decode, just flag)
        # If a long base64 chunk is found, append a signal token
        if re.search(r"[A-Za-z0-9+/]{40,}={0,2}", text):
            text = text + " ENCODED_PAYLOAD_DETECTED"

        return text

    
    # LAYER 1 — HARD BLOCK
    
    def _hard_block(self, text: str) -> bool:
        """
        Layer 1: zero-cost pre-filter before any ML inference.
        Returns True if the prompt should be immediately blocked.
        """
        # Exact hash match against known attack strings
        h = hashlib.sha256(text.strip().lower().encode()).hexdigest()
        if h in self._hash_blocklist:
            return True

        # Normalized regex match
        normalized = self._normalize(text)
        if _HARD_BLOCK_RE.search(normalized):
            return True

        return False

    def add_to_hash_blocklist(self, texts: List[str]):
        """Register exact attack strings for instant future blocking."""
        for t in texts:
            if t:
                h = hashlib.sha256(t.strip().lower().encode()).hexdigest()
                self._hash_blocklist.add(h)

    
    # EMBEDDINGS
    
    def _embed(self, texts: List[str]) -> np.ndarray:
        return np.array(
            self.embedder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

    
    # COLLECTION ADD (with FIFO eviction)
    
    def _add_to_collection(
        self,
        collection,
        id_counter_attr: str,
        texts: List[str],
        label: str,
    ):
        if not texts:
            return

        cleaned = [t.strip() for t in texts if t and t.strip()]
        if not cleaned:
            return

        current = collection.count()
        remaining = self.max_memory_size - current

        if remaining <= 0:
            to_evict = len(cleaned)
            try:
                oldest = collection.get(limit=to_evict, include=[])
                if oldest and oldest.get("ids"):
                    collection.delete(ids=oldest["ids"])
            except Exception as e:
                log.warning("Eviction failed: %s", e)
        else:
            cleaned = cleaned[:remaining]

        embeddings = self._embed(cleaned)
        counter = getattr(self, id_counter_attr)

        ids = [f"{label}_{counter + i}" for i in range(len(cleaned))]
        metadatas = [{"label": label}] * len(cleaned)
        setattr(self, id_counter_attr, counter + len(cleaned))

        try:
            collection.add(
                documents=cleaned,
                embeddings=embeddings.tolist(),
                ids=ids,
                metadatas=metadatas,
            )
        except Exception as e:
            log.warning("Failed to add embeddings: %s", e)

    
    # LAYER 3 — SEMANTIC SIMILARITY (harmful-only)
    
    def _get_similarity(self, embedding: np.ndarray) -> float:
        n = self.harmful_col.count()
        if n == 0:
            return 0.0

        try:
            results = self.harmful_col.query(
                query_embeddings=[embedding.tolist()],
                n_results=min(self.top_k, n),
            )
        except Exception as e:
            log.warning("Chroma query failed: %s", e)
            return 0.0

        if (
            not results
            or not results.get("distances")
            or not results["distances"][0]
        ):
            return 0.0

        distances = results["distances"][0]
        similarities = [max(0.0, min(1.0, 1.0 - d)) for d in distances]

        # Rank-weighted mean: top hit contributes most
        weights = np.array([1.0 / (i + 1) for i in range(len(similarities))])
        weights /= weights.sum()
        return float(np.dot(similarities, weights))

    
    # LAYER 5 — TEMPLATE PATTERN SCORE
    
    def _template_score(self, text: str) -> float:
        """
        Score the text against known adversarial jailbreak template patterns.
        Returns the maximum pattern score found (0.0 if none match).
        """
        score = 0.0
        normalized = self._normalize(text)
        for pattern, pattern_score in _TEMPLATE_RE:
            if pattern.search(normalized):
                score = max(score, pattern_score)
        return score

    
    # KEYWORD BONUS
    
    def _keyword_bonus(self, text: str) -> float:
        text_lower = self._normalize(text).lower()
        matches = sum(
            1 for term in self.high_risk_terms
            if term in text_lower
        )
        return min(0.60, matches * 0.12)

    # FEATURES (for SGD classifier)
    
    def _features(
        self,
        text: str,
        embedding: np.ndarray,
        similarity: float,
        template_score: float,
    ) -> np.ndarray:
        length_feat = min(len(text) / 1000.0, 3.0)
        kw_hits = sum(
            1 for term in self.high_risk_terms
            if term in self._normalize(text).lower()
        )
        kw_feat = kw_hits / max(len(self.high_risk_terms), 1)
        return np.concatenate([
            embedding,
            np.array([similarity, length_feat, kw_feat, template_score])
        ])

    
    # FULL RISK SCORE (all layers combined)
    
    def risk_score(self, text: str) -> Tuple[float, float, float, float]:
        """
        Returns: (risk, clf_score, similarity, template_score)
        """
        normalized = self._normalize(text)

        embedding = self._embed([normalized])[0]
        similarity = self._get_similarity(embedding)
        template_score = self._template_score(text)
        keyword_bonus = self._keyword_bonus(text)

        # Cold start — classifier not yet fitted
        if not self.is_fitted:
            cold_risk = (
                self.cold_sim_weight * similarity
                + self.w_tmpl * template_score
                + keyword_bonus
            )
            return (min(1.0, cold_risk), 0.0, similarity, template_score)

        features = self._features(
            normalized, embedding, similarity, template_score
        ).reshape(1, -1)

        try:
            clf_score = float(
                self.model.predict_proba(features)[0][1]
            )
        except Exception:
            clf_score = 0.0

        # Calibrated fusion across all four signals
        risk = (
            self.w_clf * clf_score
            + self.w_sim * similarity
            + self.w_tmpl * template_score
            + self.w_kw * keyword_bonus
        )
        return (min(1.0, risk), float(clf_score), float(similarity), float(template_score))

    
    # DECISION
    
    def is_malicious(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, float]:
        """
        Returns: (is_blocked, risk_score, similarity)
        """
        if threshold is None:
            threshold = self.threshold

        self.stats_tracker["checked"] += 1

        # Layer 1 — hard block (fastest path)
        if self._hard_block(text):
            log.warning(
                "HARD BLOCK | layer=1 | text=%s...", text[:60]
            )
            self.stats_tracker["blocked_hard"] += 1
            return (True, 1.0, 1.0)

        # Layers 2-5 — scored pipeline
        risk, clf_score, similarity, template_score = self.risk_score(text)
        blocked = risk >= threshold

        log.info(
            "DEFENSE | risk=%.3f | clf=%.3f | sim=%.3f | tmpl=%.3f | thr=%.3f | blocked=%s",
            risk, clf_score, similarity, template_score, threshold, blocked,
        )

        if blocked:
            self.stats_tracker["blocked_soft"] += 1

        return (blocked, float(risk), float(similarity))

    
    # LAYER 4 — ONLINE TRAINING WITH REPLAY
    
    def _train(self, text: str, label: int):
        """
        Train the SGD classifier on a single example, then replay
        a random sample from the buffer to prevent catastrophic forgetting.
        """
        if not text:
            return

        normalized = self._normalize(text)
        embedding = self._embed([normalized])[0]
        similarity = self._get_similarity(embedding)
        template_score = self._template_score(text)
        features = self._features(
            normalized, embedding, similarity, template_score
        ).reshape(1, -1)

        # Store in replay buffer
        self._replay.append((features, label))

        # Batch: current example + up to 8 replayed examples
        batch_X = [features]
        batch_y = [label]

        if len(self._replay) > 1:
            n_replay = min(8, len(self._replay) - 1)
            indices = np.random.choice(
                len(self._replay) - 1, size=n_replay, replace=False
            )
            for idx in indices:
                rx, ry = self._replay[idx]
                batch_X.append(rx)
                batch_y.append(ry)

        X = np.vstack(batch_X)
        y = np.array(batch_y)

        try:
            if not self.is_fitted:
                self.model.partial_fit(X, y, classes=np.array([0, 1]))
                self.is_fitted = True
            else:
                self.model.partial_fit(X, y)
        except Exception as e:
            log.warning("Online training failed: %s", e)

    
    # LAYER 6 — ADAPTIVE THRESHOLD
    
    def _tighten_threshold(self):
        """
        Lower the threshold after a confirmed jailbreak success.
        Never goes below threshold_floor.
        """
        old = self.threshold
        self.threshold = max(
            self.threshold_floor,
            self.threshold - self.threshold_decay
        )
        if self.threshold < old:
            log.warning(
                "THRESHOLD TIGHTENED: %.3f → %.3f",
                old, self.threshold
            )

    
    # LEARN FROM ATTACK (confirmed jailbreak)
    
    def learn_from_attack(self, goal: str, result: dict):
        """
        Called when a jailbreak SUCCEEDED.
        - Stores all variants in the harmful collection.
        - Trains the classifier as positive (harmful=1).
        - Registers exact hashes for layer-1 blocking.
        - Tightens the adaptive threshold (layer 6).
        """
        texts = []
        if goal:
            texts.append(goal)
        if result.get("final_prompt"):
            texts.append(result["final_prompt"])

        for text in texts:
            text = text.strip()
            if len(text) < 20:
                continue

            # Add to harmful collection
            self._add_to_collection(
                self.harmful_col, "_harmful_id", [text], "jailbreak"
            )

            # Register in hard-block hash list
            self.add_to_hash_blocklist([text])

            # Train classifier
            self._train(text, 1)

        # Layer 6: tighten threshold after each jailbreak
        self._tighten_threshold()
        self.stats_tracker["jailbreaks"] += 1

    
    # LEARN SAFE (confirmed safe response)
    
    def learn_safe(self, text: str):
        """
        Called when the model safely refused a prompt.
        Safe examples go ONLY to the safe collection — never
        to the harmful collection, so they can't dilute similarity.
        """
        if not text:
            return
        text = text.strip()
        if len(text) < 10:
            return

        self._add_to_collection(
            self.safe_col, "_safe_id", [text], "safe"
        )
        self._train(text, 0)
        self.stats_tracker["safe"] += 1

    
    # PRELOAD (bulk harmful prompts — e.g. AdvBench)
    
    def preload(self, texts: List[str]):
        """
        Bulk-load known harmful prompts.
        Also seeds the replay buffer and fits an initial warm-start
        classifier so cold-start behavior is much stronger.
        """
        if not texts:
            return

        log.info("Preloading %d harmful prompts...", len(texts))

        self._add_to_collection(
            self.harmful_col, "_harmful_id", texts, "jailbreak"
        )

        # Warm-start: fit classifier on a sample of preloaded prompts
        # so we're not fully cold even before the first real attack
        sample_size = min(50, len(texts))
        sample = texts[:sample_size]

        log.info("Warm-starting classifier on %d samples...", sample_size)

        for text in sample:
            text = text.strip()
            if len(text) >= 20:
                normalized = self._normalize(text)
                emb = self._embed([normalized])[0]
                sim = self._get_similarity(emb)
                tmpl = self._template_score(text)
                feats = self._features(normalized, emb, sim, tmpl).reshape(1, -1)
                self._replay.append((feats, 1))

        # Create synthetic safe examples for balance during warm-start
        safe_stubs = [
            "What is the capital of France?",
            "How do I bake a chocolate cake?",
            "Explain how neural networks work.",
            "What are the best practices for Python?",
            "Summarize the history of the Roman Empire.",
            "How does photosynthesis work?",
            "Write a poem about autumn.",
            "What is machine learning?",
        ]
        for text in safe_stubs:
            normalized = self._normalize(text)
            emb = self._embed([normalized])[0]
            sim = self._get_similarity(emb)
            tmpl = self._template_score(text)
            feats = self._features(normalized, emb, sim, tmpl).reshape(1, -1)
            self._replay.append((feats, 0))

        # Batch fit on the full replay buffer
        if self._replay:
            X = np.vstack([x for x, _ in self._replay])
            y = np.array([label for _, label in self._replay])
            try:
                self.model.partial_fit(X, y, classes=np.array([0, 1]))
                self.is_fitted = True
                log.info("Classifier warm-started successfully.")
            except Exception as e:
                log.warning("Warm-start fit failed: %s", e)

        log.info(
            "Preload complete | harmful=%d | replay_buffer=%d",
            self.harmful_col.count(),
            len(self._replay),
        )

    
    # UPDATE STATS
    
    def update_stats(self, blocked: bool, jailbreak: bool):
        if blocked:
            pass  # already counted in is_malicious
        if jailbreak:
            pass  # already counted in learn_from_attack

    
    # STATS
    
    def stats(self) -> dict:
        return {
            "harmful_memory": self.harmful_col.count(),
            "safe_memory": self.safe_col.count(),
            "replay_buffer": len(self._replay),
            "hash_blocklist": len(self._hash_blocklist),
            "model_fitted": self.is_fitted,
            "current_threshold": self.threshold,
            **self.stats_tracker,
        }

    
    # CLEANUP
    
    def unload(self):
        log.info("Unloading VectorDefense v3...")
        try:
            del self.harmful_col
            del self.safe_col
            del self.client
        except Exception:
            pass