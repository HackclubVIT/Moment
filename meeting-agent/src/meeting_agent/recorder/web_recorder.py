"""Browser-based audio recorder.

Injects a small MediaRecorder into the meeting tab:
- ``tab`` mode: captures the meeting's audio via ``getDisplayMedia`` with
  ``preferCurrentTab`` — the meeting participants' voices as played by the tab.
  Chromium is launched with ``--auto-select-tab-capture-source`` so no picker
  dialog appears (Chrome >= 116).
- ``mic`` mode: falls back to ``getUserMedia`` for the local microphone.

Output is a WebM/Opus file, so no ffmpeg is required.
"""

from __future__ import annotations

import base64
import re
from datetime import datetime
from pathlib import Path

_RECORDER_JS = r"""
(() => {
  if (window.__marc) return;
  const state = { recorder: null, chunks: [], stream: null, mimeType: 'audio/webm', blobB64: null, readIdx: 0, error: null, mode: null };
  async function pick(mode) {
    if (mode === 'mic') {
      return navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    }
    return navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 4, width: 320, height: 180 },
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      preferCurrentTab: true,
      selfBrowserSurface: 'include',
      systemAudio: 'include',
    });
  }
  window.__marc = {
    async start(mode) {
      if (state.recorder && state.recorder.state !== 'inactive') {
        return { ok: false, error: 'already recording' };
      }
      document.title = 'meeting-agent-capture';
      try {
        state.stream = await pick(mode || 'tab');
      } catch (err) {
        state.error = 'Capture denied or unavailable: ' + err.message;
        return { ok: false, error: state.error };
      }
      const audioTracks = state.stream.getAudioTracks();
      if (!audioTracks.length) {
        state.stream.getTracks().forEach(t => t.stop());
        state.stream = null;
        return { ok: false, error: 'no audio track available' };
      }
      state.mode = mode || 'tab';
      state.chunks = [];
      const mime = (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported('audio/webm;codecs=opus'))
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';
      state.mimeType = mime;
      state.recorder = new MediaRecorder(state.stream, { mimeType, audioBitsPerSecond: 64000 });
      state.recorder.ondataavailable = (e) => { if (e.data && e.data.size) state.chunks.push(e.data); };
      state.recorder.start(1000);
      return { ok: true, mimeType: mime };
    },
    status() {
      return {
        recording: !!(state.recorder && state.recorder.state === 'recording'),
        mode: state.mode,
        error: state.error,
      };
    },
    stop() {
      return new Promise((resolve) => {
        if (!state.recorder || state.recorder.state === 'inactive') {
          resolve({ ok: false, error: 'not recording' });
          return;
        }
        state.recorder.onstop = () => {
          const blob = new Blob(state.chunks, { type: state.mimeType });
          const reader = new FileReader();
          reader.onloadend = () => {
            const b64 = String(reader.result).split(',')[1] || '';
            state.blobB64 = b64;
            state.readIdx = 0;
            state.recorder = null;
            state.stream.getTracks().forEach(t => t.stop());
            state.stream = null;
            resolve({ ok: true, size: b64.length, mimeType: state.mimeType, mode: state.mode });
          };
          reader.readAsDataURL(blob);
        };
        state.recorder.stop();
      });
    },
    readChunk() {
      if (!state.blobB64) return { done: true, chunk: '' };
      const chunk = state.blobB64.slice(state.readIdx, state.readIdx + 1500000);
      state.readIdx += chunk.length;
      return { done: state.readIdx >= state.blobB64.length, chunk };
    },
  };
})();
"""

_CONTROL_JS = r"""
(() => {
  if (document.getElementById('__marc_btn')) return;
  const b = document.createElement('button');
  b.id = '__marc_btn';
  b.textContent = '🎙 Record';
  Object.assign(b.style, {
    position: 'fixed', right: '16px', bottom: '16px', zIndex: '2147483647',
    padding: '10px 14px', fontSize: '14px', fontWeight: 600, color: '#fff',
    background: '#c026d3', border: 'none', borderRadius: '8px',
    boxShadow: '0 2px 10px rgba(0,0,0,.4)', cursor: 'pointer',
  });
  b.onclick = async () => {
    const mode = window.__marcNextMode || 'tab';
    const r = await window.__marc.start(mode);
    if (r.ok) { b.textContent = '⏹ Stop capture'; b.style.background = '#dc2626'; }
    else { b.textContent = '⚠ ' + String(r.error || 'failed').slice(0, 40); }
  };
  document.body.appendChild(b);
})();
"""


class WebRecorder:
    """Records audio from inside a Playwright page."""

    def __init__(self, page, output_dir: str | Path):
        self.page = page
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def attach(self) -> None:
        self.page.evaluate(_RECORDER_JS)
        self.page.evaluate(_CONTROL_JS)

    def status(self) -> dict:
        try:
            return self.page.evaluate("window.__marc.status()")
        except Exception:
            return {"recording": False, "mode": None, "error": None}

    def start(self, mode: str = "tab") -> tuple[bool, str]:
        """Start recording. The in-page button is clicked so the browser treats
        the capture request as a trusted user gesture (required by getDisplayMedia)."""
        if mode not in ("tab", "mic"):
            mode = "tab"
        self.page.evaluate("(m) => { window.__marcNextMode = m; }", mode)
        try:
            self.page.click("#__marc_btn", timeout=15000)
        except Exception as exc:
            return False, f"Could not click the record button: {exc}"
        try:
            self.page.wait_for_function(
                "window.__marc && window.__marc.status().recording", timeout=20000
            )
        except Exception:
            info = self.status()
            if info.get("error"):
                return False, f"Recording failed: {info['error']}"
            return (
                False,
                "Recording did not start. If a capture picker appeared, select this tab "
                "and press share; or retry with mic mode (/record mic).",
            )
        return True, f"Recording started ({mode} mode). Meeting audio is being captured."

    def stop(self) -> Path:
        meta = self.page.evaluate("window.__marc.stop()")
        if not meta.get("ok"):
            raise RuntimeError(meta.get("error") or "not recording")
        chunks: list[str] = []
        while True:
            part = self.page.evaluate("window.__marc.readChunk()")
            chunks.append(part["chunk"])
            if part["done"]:
                break
        data = base64.b64decode("".join(chunks))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"recording_{stamp}.webm"
        # The mime type may carry parameters like ";codecs=opus"; keep the file safe.
        safe_name = re.sub(r"[^\w.-]", "", name) or "recording.webm"
        path = self.output_dir / safe_name
        path.write_bytes(data)
        return path
