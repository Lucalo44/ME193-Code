"""Mic -> 1 s clip -> log-mel spectrogram, all in numpy (no torchaudio/librosa), plus the
voice-activated trigger shared by collect_voice.py and live_voice.py so training clips and
live clips are cut the same way.
"""
import threading
import wave
from collections import deque

import cv2
import numpy as np

import config as C

CLIP_N = int(C.SAMPLE_RATE * C.CLIP_S)


# ---------------- signal helpers ----------------
def resample(x, sr_in, sr_out):
    """Windowed-sinc low-pass (when downsampling) + linear interpolation."""
    x = np.asarray(x, np.float32)
    if sr_in == sr_out:
        return x
    if sr_out < sr_in:
        cutoff = 0.45 * sr_out / sr_in                       # cycles/sample at sr_in
        n = np.arange(-32, 33)
        h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(len(n))
        x = np.convolve(x, h / h.sum(), mode="same")
    t = np.arange(int(round(len(x) * sr_out / sr_in))) * (sr_in / sr_out)
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def fit_clip(x, n=CLIP_N):
    x = np.asarray(x, np.float32)[:n]
    return np.pad(x, (0, n - len(x))) if len(x) < n else x


def level_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9) if len(x) else -120.0


def write_wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(C.SAMPLE_RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return resample(x, sr, C.SAMPLE_RATE)


# ---------------- features ----------------
WIN = int(C.SAMPLE_RATE * C.WIN_MS / 1000)       # 400 samples = 25 ms
HOP = int(C.SAMPLE_RATE * C.HOP_MS / 1000)       # 160 samples = 10 ms
N_FFT = 512


def _mel_filterbank():
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = hz(np.linspace(mel(20), mel(C.SAMPLE_RATE / 2), C.N_MELS + 2))
    freqs = np.linspace(0, C.SAMPLE_RATE / 2, N_FFT // 2 + 1)
    fb = np.zeros((C.N_MELS, len(freqs)), np.float32)
    for i in range(C.N_MELS):
        lo, mid, hi = edges[i:i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return fb


_FB = _mel_filterbank()
_HANN = np.hanning(WIN).astype(np.float32)


def log_mel(x):
    """(N_MELS, frames) in dB. 1 s at 16 kHz -> 40 x 98."""
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])               # pre-emphasis: lift the consonants
    frames = np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP] * _HANN
    power = np.abs(np.fft.rfft(frames, N_FFT)) ** 2
    return (10 * np.log10(power @ _FB.T + 1e-10)).T.astype(np.float32)


def voice_array(x):
    """Waveform -> (1, N_MELS, frames) float32 net input.

    Subtracting each band's mean over the clip (cepstral mean normalization) removes the fixed
    coloring of whatever mic recorded it, so laptop-mic and webcam-mic clips look alike.
    """
    db = log_mel(fit_clip(x))
    db = (db - db.mean(axis=1, keepdims=True)) / 20.0
    return db[None]


def spec_image(x, w, h):
    """Colored spectrogram (low frequencies at the bottom) for the UI."""
    db = log_mel(fit_clip(x))
    img = np.clip((db - db.max() + 80) / 80 * 255, 0, 255).astype(np.uint8)[::-1]
    return cv2.applyColorMap(cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO)


# ---------------- live input ----------------
class MicStream:
    """Background recording into a ring buffer at the device's own rate; reads come back at
    SAMPLE_RATE. Opens at 16 kHz when the device allows it, otherwise its default rate."""

    def __init__(self, device, seconds=3.0):
        import sounddevice as sd
        try:
            sd.check_input_settings(device=device, channels=1, samplerate=C.SAMPLE_RATE)
            self.sr = C.SAMPLE_RATE
        except Exception:
            self.sr = int(sd.query_devices(device)["default_samplerate"])
        self.name = sd.query_devices(device)["name"]
        self.buf = np.zeros(int(self.sr * seconds), np.float32)
        self.pos = 0
        self.lock = threading.Lock()
        self.stream = sd.InputStream(device=device, channels=1, samplerate=self.sr,
                                     blocksize=int(self.sr * 0.01), callback=self._callback)
        self.stream.start()

    def _callback(self, indata, frames, t, status):
        x = indata[:, 0]
        with self.lock:
            idx = (self.pos + np.arange(len(x))) % len(self.buf)
            self.buf[idx] = x
            self.pos = (self.pos + len(x)) % len(self.buf)

    def _latest_native(self, seconds):
        n = min(len(self.buf), int(self.sr * seconds))
        with self.lock:
            return self.buf[(self.pos - n + np.arange(n)) % len(self.buf)].copy()

    def latest(self, seconds=C.CLIP_S):
        return fit_clip(resample(self._latest_native(seconds), self.sr, C.SAMPLE_RATE),
                        int(C.SAMPLE_RATE * seconds))

    def level_db(self, seconds=0.03):
        return level_db(self._latest_native(seconds))

    def close(self):
        self.stream.stop()
        self.stream.close()


class ClipTrigger:
    """When the level crosses the threshold, wait until the clip is full, then return the last
    CLIP_S seconds, so the word starts PREROLL_S into the clip.

    The threshold adapts to the room: it is trigger_db or TRIGGER_MARGIN_DB above the noise
    floor (the 20th-percentile level of the last 8 s), whichever is higher. A loud room or a
    hot mic (the UNO Q boots the webcam mic at +24 dB) then doesn't fire a clip every second.
    """

    def __init__(self, mic, trigger_db=C.TRIGGER_DB):
        self.mic, self.trigger_db = mic, trigger_db
        self.fire_at, self.quiet_until = None, 0.0
        self.levels = deque(maxlen=2000)
        self.floor, self.t_floor = None, 0.0
        self.clips = 0

    @property
    def armed(self):
        return self.fire_at is None

    @property
    def threshold(self):
        if self.floor is None:
            return self.trigger_db
        return max(self.trigger_db, self.floor + C.TRIGGER_MARGIN_DB)

    def poll(self, now):
        level = self.mic.level_db()
        if self.fire_at is None:                    # only learn the floor between clips
            self.levels.append((now, level))
        if now - self.t_floor > 0.5:
            recent = [l for t, l in self.levels if now - t <= 8.0]
            if len(recent) >= 10:
                self.floor = float(np.percentile(recent, 20))
            self.t_floor = now
        if self.fire_at is not None and now >= self.fire_at:
            self.fire_at, self.quiet_until = None, now + C.TRIGGER_COOLDOWN_S
            self.clips += 1
            return self.mic.latest(C.CLIP_S)
        if self.fire_at is None and now >= self.quiet_until and level > self.threshold:
            self.fire_at = now + C.CLIP_S - C.PREROLL_S
        return None


def draw_meter(img, x, y, w, h, level, trigger_db):
    """Horizontal level meter from -80 to 0 dBFS with the trigger line."""
    frac = lambda db: np.clip((db + 80) / 80, 0, 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    col = (60, 210, 60) if level > trigger_db else (200, 200, 200)
    cv2.rectangle(img, (x, y), (x + int(w * frac(level)), y + h), col, -1)
    tx = x + int(w * frac(trigger_db))
    cv2.line(img, (tx, y - 4), (tx, y + h + 4), (0, 170, 255), 2)
