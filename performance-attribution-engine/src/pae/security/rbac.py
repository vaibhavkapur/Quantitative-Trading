"""RBAC and dual-control separation of duties preventing unauthorized or self-approved reports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set

from pae.errors import AuthorizationError, WorkflowViolationError


class Role(str, Enum):
    VIEWER = "VIEWER"
    ANALYST = "ANALYST"
    REVIEWER = "REVIEWER"
    APPROVER = "APPROVER"
    ADMIN = "ADMIN"


class Permission(str, Enum):
    READ_REPORT = "READ_REPORT"
    READ_APPROVED_ONLY = "READ_APPROVED_ONLY"
    RUN_ATTRIBUTION = "RUN_ATTRIBUTION"
    REVISE_REPORT = "REVISE_REPORT"
    REVIEW_REPORT = "REVIEW_REPORT"
    APPROVE_REPORT = "APPROVE_REPORT"
    READ_AUDIT = "READ_AUDIT"


ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {Permission.READ_REPORT, Permission.READ_APPROVED_ONLY},
    Role.ANALYST: {
        Permission.READ_REPORT,
        Permission.RUN_ATTRIBUTION,
        Permission.REVISE_REPORT,
    },
    Role.REVIEWER: {
        Permission.READ_REPORT,
        Permission.REVIEW_REPORT,
        Permission.READ_AUDIT,
    },
    Role.APPROVER: {
        Permission.READ_REPORT,
        Permission.APPROVE_REPORT,
        Permission.READ_AUDIT,
    },
    Role.ADMIN: {
        Permission.READ_REPORT,
        Permission.RUN_ATTRIBUTION,
        Permission.REVISE_REPORT,
        Permission.REVIEW_REPORT,
        Permission.APPROVE_REPORT,
        Permission.READ_AUDIT,
    },
}


@dataclass(frozen=True)
class Actor:
    """Authenticated caller identity (credentials live outside code/repo)."""

    actor_id: str
    role: Role

    @classmethod
    def from_headers(cls, actor_id: str, role_str: str) -> "Actor":
        try:
            role = Role(role_str.upper())
        except ValueError as exc:
            raise AuthorizationError(
                f"Unknown role: {role_str!r}",
                details={"valid_roles": [r.value for r in Role]},
            ) from exc
        if not actor_id or not actor_id.strip():
            raise AuthorizationError("actor_id is required.")
        return cls(actor_id=actor_id.strip(), role=role)


def check_permission(actor: Actor, permission: Permission) -> None:
    """Assert that actor's role holds the required permission."""
    allowed = ROLE_PERMISSIONS.get(actor.role, set())
    if permission not in allowed:
        raise AuthorizationError(
            f"Actor {actor.actor_id!r} with role {actor.role.value} lacks permission {permission.value}.",
            details={"actor_id": actor.actor_id, "role": actor.role.value, "required": permission.value},
        )


def enforce_dual_control(
    *,
    action: str,
    actor: Actor,
    author_id: str,
    reviewer_id: Optional[str] = None,
    current_status: str = "DRAFT",
) -> None:
    """Enforce separation of duties and state transitions for report review/approval."""
    if action == "REVIEW":
        if current_status != "DRAFT":
            raise WorkflowViolationError(
                f"Cannot review report in status {current_status}; must be DRAFT.",
                details={"current_status": current_status},
            )
        if actor.actor_id == author_id:
            raise WorkflowViolationError(
                f"Dual-control violation: author {author_id!r} cannot review their own report.",
                details={"author_id": author_id, "actor_id": actor.actor_id},
            )
    elif action == "APPROVE":
        if current_status != "REVIEWED":
            raise WorkflowViolationError(
                f"Cannot approve report in status {current_status}; independent review is required first.",
                details={"current_status": current_status},
            )
        if actor.actor_id == author_id:
            raise WorkflowViolationError(
                f"Dual-control violation: author {author_id!r} cannot approve their own report.",
                details={"author_id": author_id, "actor_id": actor.actor_id},
            )
        if reviewer_id and actor.actor_id == reviewer_id and actor.role != Role.ADMIN:
            raise WorkflowViolationError(
                f"Dual-control violation: reviewer {reviewer_id!r} cannot also approve the same report.",
                details={"reviewer_id": reviewer_id, "actor_id": actor.actor_id},
            )
