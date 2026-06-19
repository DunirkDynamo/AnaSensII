"""Placeholder tab used for future Qt modules."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderTab(QWidget):
    """Display a roadmap note for future features."""

    def __init__(self, title: str, message: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        header = QLabel(title)
        header.setObjectName("sectionTitle")
        body = QLabel(message)
        body.setWordWrap(True)
        body.setObjectName("subtleBody")

        layout.addWidget(header)
        layout.addWidget(body)
        layout.addStretch(1)
