"""Canonical public-data snapshot for REGAL v2 reports.

The statistical likelihood remains implemented in :mod:`event_likelihood`.
This module is the reporting boundary: it loads the same validated
``PublicHistory`` object used by the model, fingerprints the exact source JSON,
and exposes a compact JSON-safe summary for versioned result bundles.

Keeping this adapter separate prevents the browser from maintaining a second,
hand-copied version of the public evidence.  It performs no statistical
calculation and cannot produce a REGAL forecast by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from event_likelihood import (
    REGAL_PUBLIC_HISTORY_PATH,
    CountObservation,
    PublicHistory,
    load_regal_public_history,
)


def _iso(value):
    return value.isoformat() if value is not None else None


def public_history_as_of(history):
    """Return the latest dated fact represented by ``history``.

    Observation, announcement, and source-publication dates are all included.
    The result therefore describes the evidence snapshot, not the wall-clock
    time at which a Monte Carlo run happened.
    """

    if not isinstance(history, PublicHistory):
        raise ValueError("history must be PublicHistory")
    dates = []
    for observation in history.enrollment_observations + history.event_observations:
        dates.append(observation.source.published_date)
        if observation.observation_date is not None:
            dates.append(observation.observation_date)
        if observation.announcement_date is not None:
            dates.append(observation.announcement_date)
    if not dates:
        raise ValueError("public history contains no dated evidence")
    return max(dates)


def _source_path_label(path):
    path = Path(path).resolve()
    root = Path(__file__).resolve().parent
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _observation_record(stream, observation):
    if not isinstance(observation, CountObservation):
        raise ValueError("observation must be CountObservation")
    return {
        "stream": stream,
        "id": observation.observation_id,
        "observation_type": observation.observation_type.value,
        "observation_date": _iso(observation.observation_date),
        "announcement_date": _iso(observation.announcement_date),
        "count": observation.count,
        "count_lower": observation.count_lower,
        "count_upper": observation.count_upper,
        "use_in_likelihood": observation.use_in_likelihood,
        "accrual_anchor": observation.accrual_anchor,
        "reporting_lag": {
            "distribution": observation.reporting_lag.distribution,
            "values": [
                {"days": days, "probability": probability}
                for days, probability in observation.reporting_lag.choices
            ],
        },
        "source": {
            "title": observation.source.title,
            "url": observation.source.url,
            "published_date": observation.source.published_date.isoformat(),
        },
    }


@dataclass(frozen=True)
class RegalDataSnapshot:
    """Validated public history plus an exact source-file fingerprint."""

    history: PublicHistory
    source_path: Path
    source_sha256: str
    as_of_date: date

    def __post_init__(self):
        if not isinstance(self.history, PublicHistory):
            raise ValueError("history must be PublicHistory")
        path = Path(self.source_path).resolve()
        digest = str(self.source_sha256).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("source_sha256 must be a 64-character hexadecimal digest")
        if not isinstance(self.as_of_date, date):
            raise ValueError("as_of_date must be a date")
        expected_as_of = public_history_as_of(self.history)
        if self.as_of_date != expected_as_of:
            raise ValueError("as_of_date does not match the public history")
        object.__setattr__(self, "source_path", path)
        object.__setattr__(self, "source_sha256", digest)

    def to_mapping(self):
        history = self.history
        observations = [
            _observation_record("enrollment", observation)
            for observation in history.enrollment_observations
        ] + [
            _observation_record("events", observation)
            for observation in history.event_observations
        ]
        return MappingProxyType(
            {
                "schema_version": history.schema_version,
                "registry_id": history.registry_id,
                "as_of_date": self.as_of_date.isoformat(),
                "study_start": history.study_start.isoformat(),
                "target_enrollment": history.target_enrollment,
                "interim_event_threshold": history.interim_event_threshold,
                "final_event_threshold": history.final_event_threshold,
                "source": {
                    "path": _source_path_label(self.source_path),
                    "sha256": self.source_sha256,
                },
                "observations": observations,
            }
        )


def load_regal_data_snapshot(path=REGAL_PUBLIC_HISTORY_PATH):
    """Load, validate, and fingerprint the canonical REGAL public-data file."""

    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    history = load_regal_public_history(source_path)
    return RegalDataSnapshot(
        history=history,
        source_path=source_path,
        source_sha256=sha256(raw).hexdigest(),
        as_of_date=public_history_as_of(history),
    )


__all__ = (
    "RegalDataSnapshot",
    "load_regal_data_snapshot",
    "public_history_as_of",
)
