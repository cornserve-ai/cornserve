"""Utility for playing streamed audio response."""

import contextlib
import queue
import threading
import time

import sounddevice as sd


class PCM16StreamPlayer:
    """Streams little-endian PCM16 mono to the default audio output."""

    def __init__(self, samplerate: int = 24000, channels: int = 1, prebuffer_seconds: float = 0.20):
        """Initialize the stream player."""
        self.samplerate = samplerate
        self.channels = channels

        # 2 bytes per int16 sample
        self.prebuffer_bytes = int(prebuffer_seconds * samplerate * channels * 2)

        self._q: queue.Queue[bytes] = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._stream = sd.RawOutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
            latency="low",
            blocksize=0,  # let PortAudio decide
        )
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)

        self._prebuffer = bytearray()
        self._stream_started = False

    def start(self) -> None:
        """Start the writer thread."""
        self._writer_thread.start()

    def feed(self, pcm_bytes: bytes) -> None:
        """Feed raw PCM16 bytes (LE). Thread-safe."""
        if not pcm_bytes:
            return
        if not self._stream_started:
            self._prebuffer.extend(pcm_bytes)
            if len(self._prebuffer) >= self.prebuffer_bytes:
                self._q.put(bytes(self._prebuffer))
                self._prebuffer.clear()
                self._stream_started = True
            return
        self._q.put(pcm_bytes)

    def close(self) -> None:
        """Finish playback cleanly, ensuring the last chunks are heard."""
        # If prebuffer never flushed (short audio), push it now
        if not self._stream_started and self._prebuffer:
            self._q.put(bytes(self._prebuffer))
            self._prebuffer.clear()
            self._stream_started = True

        # Tell writer to finish after consuming everything currently queued
        self._stop.set()

        # sentinel (will be processed after all queued audio)
        self._q.put(b"")

        # Wait for writer to drain and stop the stream
        self._writer_thread.join(timeout=30.0)

        # Writer thread performs the blocking stop(); here we just close.
        try:
            if self._stream:
                # A tiny grace period for some drivers after stop()
                time.sleep(0.01)
                self._stream.close()
        except Exception:
            pass

    def _writer_loop(self) -> None:
        first_write_done = False
        while True:
            try:
                chunk = self._q.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    # nothing more coming; if stream active, drain/stop
                    if self._stream.active:
                        with contextlib.suppress(Exception):
                            # blocking drain
                            self._stream.stop()
                    break
                continue

            # Sentinel indicates "no more chunks will arrive"; fall through to draining stop
            if chunk == b"" and self._stop.is_set():
                try:
                    if not first_write_done and not self._stream.active and self._stream_started:
                        # Start then write a tiny silence so stop() has something to drain
                        self._stream.start()
                        # ~ 10ms
                        self._stream.write(b"\x00" * (self.samplerate // 100))
                        first_write_done = True
                    if self._stream.active:
                        # blocking drain
                        self._stream.stop()
                except Exception:
                    pass
                break

            try:
                if not self._stream.active:
                    self._stream.start()

                # blocks until queued to PortAudio
                self._stream.write(chunk)

                first_write_done = True
            except sd.PortAudioError as e:
                print(f"[Audio warning] {e}")
                try:
                    if self._stream.active:
                        self._stream.abort()
                except Exception:
                    pass
                time.sleep(0.05)
                with contextlib.suppress(Exception):
                    self._stream.start()
