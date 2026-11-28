"""Immutable append-only hash-chained experiment registry.

Records every attempted strategy trial—including FAILED, REJECTED, and ABORTED
runs—into an append-only JSONL ledger where each entry commits to the SHA-256
hash of the previous entry. Any retroactive deletion, reordering, or mutation
breaks hash-chain verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from .models import (
    ContractValidationError,
    RegistryEntry,
    RegistryIntegrityError,
    TrialRecord,
    TrialStatus,
    canonical_json,
    sha256_text,
)

GENESIS_HASH = "0" * 64


def compute_entry_hash(prev_hash: str, record: TrialRecord) -> str:
    payload = {"prev_hash": prev_hash, "record": record.to_dict()}
    return sha256_text(canonical_json(payload))


class ImmutableExperimentRegistry:
    """Append-only trial ledger backed by a JSONL file."""

    def __init__(self, ledger_path: Path) -> None:
        self._path = Path(ledger_path)

    @property
    def ledger_path(self) -> Path:
        return self._path

    def initialize(self) -> None:
        """Create parent directories and empty ledger if absent."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    def load_entries(self) -> list[RegistryEntry]:
        """Load all raw entries without silently ignoring corrupt JSON."""
        if not self._path.exists():
            return []
        entries: list[RegistryEntry] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    entries.append(RegistryEntry.from_dict(payload))
                except Exception as exc:
                    raise RegistryIntegrityError(
                        f"Malformed registry entry at line {line_no} in {self._path}: {exc}"
                    ) from exc
        return entries

    def verify(self) -> tuple[bool, list[str]]:
        """Verify cryptographic hash chain, unique trial IDs, and monotonic order."""
        errors: list[str] = []
        if not self._path.exists():
            return False, [f"Registry file does not exist: {self._path}"]

        try:
            entries = self.load_entries()
        except RegistryIntegrityError as exc:
            return False, [str(exc)]

        expected_prev = GENESIS_HASH
        seen_trial_ids: set[str] = set()
        seen_sequences: list[int] = []

        for idx, entry in enumerate(entries):
            if entry.prev_hash != expected_prev:
                errors.append(
                    f"Entry {idx} (trial_id={entry.record.trial_id}): prev_hash mismatch "
                    f"(expected {expected_prev[:12]}..., got {entry.prev_hash[:12]}...)"
                )
            recomputed = compute_entry_hash(entry.prev_hash, entry.record)
            if entry.entry_hash != recomputed:
                errors.append(
                    f"Entry {idx} (trial_id={entry.record.trial_id}): entry_hash tamper detected "
                    f"(expected {recomputed[:12]}..., got {entry.entry_hash[:12]}...)"
                )
            if entry.record.trial_id in seen_trial_ids:
                errors.append(f"Duplicate trial_id in registry: {entry.record.trial_id}")
            seen_trial_ids.add(entry.record.trial_id)

            if seen_sequences and entry.record.sequence_number <= seen_sequences[-1]:
                errors.append(
                    f"Non-monotonic sequence_number at entry {idx}: "
                    f"{entry.record.sequence_number} <= {seen_sequences[-1]}"
                )
            seen_sequences.append(entry.record.sequence_number)
            expected_prev = entry.entry_hash

        return len(errors) == 0, errors

    def append(self, record: TrialRecord) -> RegistryEntry:
        """Append a single validated TrialRecord to the ledger.

        Refuses to append if the existing ledger is tampered, if trial_id already
        exists, or if sequence_number is not strictly increasing.
        """
        self.initialize()
        entries = self.load_entries()
        if entries:
            valid, errs = self.verify()
            if not valid:
                raise RegistryIntegrityError(
                    f"Cannot append to corrupted registry {self._path}: {'; '.join(errs)}"
                )
            if any(e.record.trial_id == record.trial_id for e in entries):
                raise ContractValidationError(f"trial_id '{record.trial_id}' already exists in registry")
            last_seq = entries[-1].record.sequence_number
            if record.sequence_number <= last_seq:
                raise ContractValidationError(
                    f"sequence_number must exceed last logged sequence ({last_seq}), got {record.sequence_number}"
                )
            prev_hash = entries[-1].entry_hash
        else:
            prev_hash = GENESIS_HASH

        entry_hash = compute_entry_hash(prev_hash, record)
        entry = RegistryEntry(prev_hash=prev_hash, record=record, entry_hash=entry_hash)

        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(entry.to_dict()) + "\n")

        return entry

    def append_many(self, records: Iterable[TrialRecord]) -> list[RegistryEntry]:
        """Append multiple records sequentially."""
        appended: list[RegistryEntry] = []
        for rec in records:
            appended.append(self.append(rec))
        return appended

    def list_records(self, status: Optional[TrialStatus] = None) -> list[TrialRecord]:
        entries = self.load_entries()
        records = [e.record for e in entries]
        if status is not None:
            return [r for r in records if r.status == status]
        return records

    def head_hash(self) -> str:
        entries = self.load_entries()
        return entries[-1].entry_hash if entries else GENESIS_HASH
