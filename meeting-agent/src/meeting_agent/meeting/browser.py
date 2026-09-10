"""Playwright-managed persistent Chromium session."""

from __future__ import annotations

from ..config import Settings


class BrowserSession:
    """Launches a persistent Chromium profile so login state survives across runs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pw = None
        self.context = None

    def start(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "Playwright is not installed. Run:\n"
                "  uv sync --extra meeting\n"
                "  uv run playwright install chromium"
            ) from exc

        user_data_dir = self.settings.chrome_user_data_dir
        user_data_dir.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        args = [
            # Auto-select this tab when getDisplayMedia is called, so tab-audio
            # capture works without a picker dialog (Chrome >= 116).
            "--auto-select-tab-capture-source=meeting-agent-capture",
            "--auto-select-desktop-capture-source=meeting-agent-capture",
            "--window-size=1280,800",
        ]
        try:
            self.context = self._pw.chromium.launch_persistent_context(
                str(user_data_dir),
                headless=self.settings.headless,
                args=args,
                no_viewport=True,
            )
        except Exception as exc:  # pragma: no cover - browser launch is environment specific
            self._pw.stop()
            self._pw = None
            raise RuntimeError(f"Failed to launch Chromium: {exc}") from exc

    def new_page(self):
        if self.context is None:
            raise RuntimeError("BrowserSession.start() must be called first")
        return self.context.new_page()

    def grant_media_permissions(self, page, permissions: list[str] | None = None) -> None:
        try:
            page.context.grant_permissions(permissions or ["microphone", "camera"], origin=page.url)
        except Exception:
            pass  # permission API is best-effort across sites

    def close(self) -> None:
        try:
            if self.context is not None:
                self.context.close()
        finally:
            if self._pw is not None:
                self._pw.stop()
                self._pw = None
            self.context = None
