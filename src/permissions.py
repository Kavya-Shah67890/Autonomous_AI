"""Permission gate for sensitive agent actions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class PermissionRequest:
    """Represents a sensitive action requiring user approval."""

    action: str
    target: str
    reason: str


PermissionCallback = Callable[[PermissionRequest], bool]


class PermissionManager:
    """Manages user consent for sensitive actions."""

    def __init__(
        self,
        require_approval: bool = True,
        callback: Optional[PermissionCallback] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.require_approval = require_approval
        self.callback = callback
        self.logger = logger or logging.getLogger("research_system.permissions")
        self._approved_actions: set[tuple[str, str]] = set()

    def request_permission(self, action: str, target: str, reason: str) -> bool:
        """Ask for permission before executing a sensitive action."""
        if not self.require_approval:
            return True

        key = (action, target)
        if key in self._approved_actions:
            return True

        req = PermissionRequest(action=action, target=target, reason=reason)
        if self.callback is None:
            self.logger.warning("Permission denied (no interactive callback): %s -> %s", action, target)
            return False

        approved = bool(self.callback(req))
        if approved:
            self._approved_actions.add(key)
            self.logger.info("Permission GRANTED for %s (%s)", action, target)
        else:
            self.logger.warning("Permission DENIED for %s (%s)", action, target)
        return approved
