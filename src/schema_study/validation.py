"""Shared validation-report structure."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationReport:
    name: str
    issues: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    def add(self, severity: str, code: str, message: str, count: int = 1, **extra) -> None:
        self.issues.append(
            {
                "severity": severity,
                "code": code,
                "message": message,
                "count": count,
                **extra,
            }
        )

    @property
    def error_count(self) -> int:
        return sum(item["count"] for item in self.issues if item["severity"] == "error")

    def as_dict(self) -> dict:
        status = "pass"
        if any(item["severity"] == "error" for item in self.issues):
            status = "fail"
        elif any(item["severity"] == "warning" for item in self.issues):
            status = "warning"
        return {
            "name": self.name,
            "status": status,
            "error_count": self.error_count,
            "counts": self.counts,
            "issues": self.issues,
        }
