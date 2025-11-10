"""Utility for playing streamed audio responses."""

import contextlib
import queue
import threading
import time

import sounddevice as sd


class PCMStreamPlayer:
    """Streams raw PCM data to the default audio output."""

    # Map of format strings to sounddevice dtype and bytes per sample.
    PCM_FORMATS = {
        "pcm16": {"dtype": "int16", "bytes": 2},
        "pcm24": {"dtype": "int32", "bytes": 3},
        "pcm32": {"dtype": "int32", "bytes": 4},
    }

    DEFAULT_SAMPLE_RATE = 24000
    DEFAULT_CHANNELS = 1
    DEFAULT_PREBUFFER_SEC = 0.20
    DEFAULT_FORMAT = "pcm16"

    def __init__(
        self,
        sample_rate: int | None = None,
        channels: int | None = None,
        pcm_format: str | None = None,
        prebuffer_seconds: float | None = None,
    ):
        """Initialize the stream player."""
        self.sample_rate = self.DEFAULT_SAMPLE_RATE if sample_rate is None else sample_rate
        self.channels = self.DEFAULT_CHANNELS if channels is None else channels

        prebuffer_seconds = self.DEFAULT_PREBUFFER_SEC if prebuffer_seconds is None else prebuffer_seconds
        pcm_format_str = (self.DEFAULT_FORMAT if pcm_format is None else pcm_format).lower()

        # Get format details
        format_details = self.PCM_FORMATS.get(pcm_format_str)
        if not format_details:
            raise ValueError(f"Unsupported PCM format: {pcm_format}. Supported: {list(self.PCM_FORMATS.keys())}")

        self.dtype = format_details["dtype"]
        self.bytes_per_sample = format_details["bytes"]

        self.prebuffer_bytes = int(prebuffer_seconds * self.sample_rate * self.channels * self.bytes_per_sample)

        self._q: queue.Queue[bytes] = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
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
        """Feed raw PCM bytes. Thread-safe."""
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

                        # ~ 10ms of silence
                        num_samples_10ms = int(self.sample_rate * 0.01)
                        num_bytes_10ms = num_samples_10ms * self.channels * self.bytes_per_sample

                        # Ensure at least a small amount of silence is written
                        if num_bytes_10ms == 0:
                            num_bytes_10ms = max(
                                self.channels * self.bytes_per_sample,
                                1,
                            )

                        self._stream.write(b"\x00" * num_bytes_10ms)
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
