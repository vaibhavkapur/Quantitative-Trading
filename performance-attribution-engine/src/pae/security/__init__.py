"""Role-Based Access Control (RBAC) and dual-control approval policy."""

from pae.security.rbac import Actor, Permission, Role, check_permission, enforce_dual_control

__all__ = ["Actor", "Permission", "Role", "check_permission", "enforce_dual_control"]
