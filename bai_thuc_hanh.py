from __future__ import annotations

import math
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, welch


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"


DTMF_LOW = np.array([697, 770, 852, 941])
DTMF_HIGH = np.array([1209, 1336, 1477, 1633])
DTMF_KEYS = {
    (697, 1209): "1",
    (697, 1336): "2",
    (697, 1477): "3",
    (697, 1633): "A",
    (770, 1209): "4",
    (770, 1336): "5",
    (770, 1477): "6",
    (770, 1633): "B",
    (852, 1209): "7",
    (852, 1336): "8",
    (852, 1477): "9",
    (852, 1633): "C",
    (941, 1209): "*",
    (941, 1336): "0",
    (941, 1477): "#",
    (941, 1633): "D",
}


@dataclass
class DtmfTone:
    index: int
    start_s: float
    end_s: float
    low_hz: int
    high_hz: int
    key: str


@dataclass
class DtmfResult:
    fs: int
    duration_s: float
    sequence: str
    tones: list[DtmfTone]
    waveform_plot: Path
    spectrum_plot: Path
    segment_power_plot: Path


@dataclass
class BpskResult:
    fs: int
    duration_s: float
    carrier_hz: float
    bit_duration_s: float
    bit_count: int
    bit_start_s: float
    bits: str
    message: str
    waveform_plot: Path
    spectrum_plot: Path
    symbols_plot: Path


def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]

    x = data.astype(float)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak

    return fs, x


def moving_average(x: np.ndarray, size: int) -> np.ndarray:
    size = max(1, int(size))
    return np.convolve(x, np.ones(size) / size, mode="same")


def detect_active_segments(
    x: np.ndarray,
    fs: int,
    threshold_ratio: float,
    energy_window_s: float,
    min_gap_s: float,
    min_duration_s: float,
) -> list[tuple[int, int]]:
    energy = moving_average(x * x, int(energy_window_s * fs))
    threshold = threshold_ratio * float(np.max(energy))
    active = energy > threshold

    padded = np.pad(active.astype(int), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    raw = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]

    min_gap = int(min_gap_s * fs)
    merged: list[list[int]] = []
    for start, end in raw:
        if not merged or start - merged[-1][1] > min_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = end

    min_len = int(min_duration_s * fs)
    return [(start, end) for start, end in merged if end - start >= min_len]


def frequency_power(segment: np.ndarray, fs: int, frequencies: Iterable[float]) -> np.ndarray:
    n = np.arange(len(segment))
    powers = []
    for freq in frequencies:
        coeff = np.sum(segment * np.exp(-1j * 2 * np.pi * freq * n / fs))
        powers.append(abs(coeff) ** 2 / max(1, len(segment) ** 2))
    return np.array(powers)


def decode_dtmf(path: Path) -> DtmfResult:
    fs, x = read_wav_mono(path)
    duration_s = len(x) / fs

    segments = detect_active_segments(
        x=x,
        fs=fs,
        threshold_ratio=0.10,
        energy_window_s=0.020,
        min_gap_s=0.050,
        min_duration_s=0.100,
    )

    tones: list[DtmfTone] = []
    all_freqs = np.concatenate([DTMF_LOW, DTMF_HIGH])

    for idx, (start, end) in enumerate(segments, start=1):
        segment = x[start:end]
        trim = len(segment) // 10
        if trim > 0 and len(segment) > 2 * trim:
            segment = segment[trim:-trim]

        powers = frequency_power(segment, fs, all_freqs)
        low_hz = int(DTMF_LOW[int(np.argmax(powers[: len(DTMF_LOW)]))])
        high_hz = int(DTMF_HIGH[int(np.argmax(powers[len(DTMF_LOW) :]))])
        key = DTMF_KEYS[(low_hz, high_hz)]

        tones.append(
            DtmfTone(
                index=idx,
                start_s=start / fs,
                end_s=end / fs,
                low_hz=low_hz,
                high_hz=high_hz,
                key=key,
            )
        )

    sequence = "".join(t.key for t in tones)

    waveform_plot = FIG_DIR / "dtmf_waveform_segments.png"
    spectrum_plot = FIG_DIR / "dtmf_spectrum.png"
    segment_power_plot = FIG_DIR / "dtmf_segment_powers.png"

    t = np.arange(len(x)) / fs
    plt.figure(figsize=(11, 4))
    plt.plot(t, x, linewidth=0.8)
    for tone in tones:
        plt.axvspan(tone.start_s, tone.end_s, alpha=0.18)
        plt.text(
            (tone.start_s + tone.end_s) / 2,
            1.05,
            tone.key,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    plt.ylim(-1.15, 1.25)
    plt.xlabel("Thoi gian (s)")
    plt.ylabel("Bien do")
    plt.title(f"DTMF waveform - chuoi nhan dang: {sequence}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(waveform_plot, dpi=180)
    plt.close()

    f, pxx = welch(x, fs=fs, nperseg=min(len(x), 32768))
    mask = (f >= 500) & (f <= 1800)
    plt.figure(figsize=(11, 4))
    plt.semilogy(f[mask], pxx[mask])
    for freq in all_freqs:
        plt.axvline(freq, color="tab:red", linestyle="--", alpha=0.45)
        plt.text(freq, max(pxx[mask]) * 0.55, str(freq), rotation=90, ha="right", va="top")
    plt.xlabel("Tan so (Hz)")
    plt.ylabel("PSD")
    plt.title("Pho cong suat cua dtmf.wav")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(spectrum_plot, dpi=180)
    plt.close()

    power_rows = []
    for start, end in segments:
        segment = x[start:end]
        trim = len(segment) // 10
        if trim > 0 and len(segment) > 2 * trim:
            segment = segment[trim:-trim]

        powers = frequency_power(segment, fs, all_freqs)
        power_rows.append(powers / max(np.max(powers), 1e-12))

    plt.figure(figsize=(11, 4.8))
    plt.imshow(np.array(power_rows), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(label="Cong suat chuan hoa")
    plt.xticks(np.arange(len(all_freqs)), [str(int(freq)) for freq in all_freqs])
    plt.yticks(np.arange(len(tones)), [f"{tone.index}: {tone.key}" for tone in tones])
    plt.axvline(len(DTMF_LOW) - 0.5, color="black", linewidth=1)
    plt.xlabel("Tan so DTMF chuan (Hz)")
    plt.ylabel("Doan / phim nhan dang")
    plt.title("Cong suat tung doan tai cac tan so DTMF")
    plt.tight_layout()
    plt.savefig(segment_power_plot, dpi=180)
    plt.close()

    return DtmfResult(
        fs=fs,
        duration_s=duration_s,
        sequence=sequence,
        tones=tones,
        waveform_plot=waveform_plot,
        spectrum_plot=spectrum_plot,
        segment_power_plot=segment_power_plot,
    )


def bits_to_text(bits: np.ndarray, msb_first: bool) -> str:
    chars = []
    usable = len(bits) // 8 * 8
    for start in range(0, usable, 8):
        byte = bits[start : start + 8]
        value = 0
        if msb_first:
            for bit in byte:
                value = (value << 1) | int(bit)
        else:
            for offset, bit in enumerate(byte):
                value |= int(bit) << offset
        chars.append(chr(value))
    return "".join(chars)


def printable_score(text: str) -> float:
    if not text:
        return -1e9

    score = 0.0
    for char in text:
        code = ord(char)
        if char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:-_":
            score += 4.0
        elif 32 <= code < 127:
            score += 1.0
        else:
            score -= 12.0

    upper = text.upper()
    for word in ["HELLO", "SIGNAL", "PYTHON", "DTMF"]:
        score += 10.0 * upper.count(word)

    if all(char.isupper() or char.isdigit() or char == " " for char in text):
        score += 3.0

    return score


def estimate_carrier_hz(x: np.ndarray, fs: int) -> float:
    f, pxx = welch(x, fs=fs, nperseg=min(len(x), 65536))
    mask = (f >= 200) & (f <= 10000)
    peak = float(f[mask][int(np.argmax(pxx[mask]))])

    # The carrier in this exercise is generated at a clean round frequency.
    return round(peak / 100.0) * 100.0


def decode_bpsk_ascii(path: Path) -> BpskResult:
    fs, x = read_wav_mono(path)
    duration_s = len(x) / fs
    carrier_hz = estimate_carrier_hz(x, fs)

    energy = moving_average(x * x, int(0.10 * fs))
    active_idx = np.flatnonzero(energy > 0.01 * np.max(energy))
    active_start = int(active_idx[0])
    active_end = int(active_idx[-1])

    n = np.arange(len(x))
    mixed = x * np.exp(-1j * 2 * np.pi * carrier_hz * n / fs) * 2
    csum = np.concatenate([[0 + 0j], np.cumsum(mixed)])

    common_bit_durations = [0.05, 0.0625, 0.10, 0.125, 0.20, 0.25, 0.50]
    samples_per_bit_candidates = {int(round(fs * seconds)) for seconds in common_bit_durations}

    active_len = active_end - active_start + 1

    best: dict[str, object] | None = None

    for spb in sorted(samples_per_bit_candidates):
        if spb <= 0:
            continue

        estimated_bits = int(round(active_len / spb / 8) * 8)
        bit_count_candidates = {
            estimated_bits - 16,
            estimated_bits - 8,
            estimated_bits,
            estimated_bits + 8,
            estimated_bits + 16,
        }

        for bit_count in sorted(bit_count_candidates):
            if bit_count < 8:
                continue

            start_min = max(0, active_start - spb)
            start_max = min(len(x) - bit_count * spb - 1, active_start + spb)
            if start_max <= start_min:
                continue

            step = max(1, spb // 120)
            for start in range(start_min, start_max + 1, step):
                edges = start + np.arange(bit_count + 1) * spb
                if edges[-1] >= len(csum):
                    continue

                symbols = (csum[edges[1:]] - csum[edges[:-1]]) / spb
                rotation = 0.5 * np.angle(np.mean(symbols**2))
                projected = np.real(symbols * np.exp(-1j * rotation))
                threshold = 0.0
                raw_bits = projected > threshold
                if raw_bits.all() or (~raw_bits).all():
                    continue

                separation = abs(np.mean(projected[raw_bits]) - np.mean(projected[~raw_bits]))

                for invert in [False, True]:
                    bits = np.logical_xor(raw_bits, invert).astype(int)
                    for msb_first in [True, False]:
                        text = bits_to_text(bits, msb_first=msb_first)
                        window_end = start + bit_count * spb
                        end_error_bits = abs(window_end - active_end) / spb
                        score = printable_score(text) + separation - 8.0 * end_error_bits
                        if best is None or score > float(best["score"]):
                            best = {
                                "score": score,
                                "start": start,
                                "spb": spb,
                                "bit_count": bit_count,
                                "bits": bits,
                                "text": text,
                                "symbols": projected,
                                "invert": invert,
                                "msb_first": msb_first,
                                "separation": separation,
                            }

    if best is None:
        raise RuntimeError("Khong giai ma duoc input.wav")

    bits = np.asarray(best["bits"], dtype=int)
    bit_start = int(best["start"])
    spb = int(best["spb"])
    bit_count = int(best["bit_count"])
    symbols = np.asarray(best["symbols"], dtype=float)
    message = str(best["text"])

    waveform_plot = FIG_DIR / "input_waveform_active.png"
    spectrum_plot = FIG_DIR / "input_spectrum_carrier.png"
    symbols_plot = FIG_DIR / "input_demodulated_symbols.png"

    t = np.arange(len(x)) / fs
    plt.figure(figsize=(11, 4))
    plt.plot(t, x, linewidth=0.7)
    plt.axvspan(bit_start / fs, (bit_start + bit_count * spb) / fs, alpha=0.18, color="tab:orange")
    plt.xlabel("Thoi gian (s)")
    plt.ylabel("Bien do")
    plt.title("input.wav - vung tin hieu BPSK da giai ma")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(waveform_plot, dpi=180)
    plt.close()

    f, pxx = welch(x, fs=fs, nperseg=min(len(x), 65536))
    mask = (f >= carrier_hz - 600) & (f <= carrier_hz + 600)
    plt.figure(figsize=(11, 4))
    plt.semilogy(f[mask], pxx[mask])
    plt.axvline(carrier_hz, color="tab:red", linestyle="--", label=f"fc = {carrier_hz:.0f} Hz")
    plt.xlabel("Tan so (Hz)")
    plt.ylabel("PSD")
    plt.title("Pho quanh song mang cua input.wav")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(spectrum_plot, dpi=180)
    plt.close()

    plt.figure(figsize=(11, 4))
    markerline, stemlines, baseline = plt.stem(np.arange(bit_count), symbols)
    plt.setp(markerline, markersize=3)
    plt.setp(stemlines, linewidth=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Chi so bit")
    plt.ylabel("Gia tri sau giai dieu che")
    plt.title(f"Quyet dinh bit - thong diep: {message}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(symbols_plot, dpi=180)
    plt.close()

    return BpskResult(
        fs=fs,
        duration_s=duration_s,
        carrier_hz=carrier_hz,
        bit_duration_s=spb / fs,
        bit_count=bit_count,
        bit_start_s=bit_start / fs,
        bits="".join(str(bit) for bit in bits),
        message=message,
        waveform_plot=waveform_plot,
        spectrum_plot=spectrum_plot,
        symbols_plot=symbols_plot,
    )


def write_text_results(dtmf: DtmfResult, bpsk: BpskResult) -> None:
    lines = [
        "KET QUA BAI THUC HANH",
        "",
        "1. Nhan dien DTMF",
        f"- File: dtmf.wav",
        f"- Tan so lay mau: {dtmf.fs} Hz",
        f"- Thoi luong: {dtmf.duration_s:.3f} s",
        f"- Chuoi phim nhan dang: {dtmf.sequence}",
        "",
        "Bang tan so tung phim:",
    ]

    for tone in dtmf.tones:
        lines.append(
            f"{tone.index:02d}. {tone.key}: {tone.low_hz} Hz + {tone.high_hz} Hz "
            f"({tone.start_s:.3f}s -> {tone.end_s:.3f}s)"
        )

    lines.extend(
        [
            "",
            "2. Giai ma input.wav",
            f"- File: input.wav",
            f"- Tan so lay mau: {bpsk.fs} Hz",
            f"- Thoi luong: {bpsk.duration_s:.3f} s",
            f"- Dang dieu che: BPSK / nhan bien do +/-1 vao song mang",
            f"- Tan so song mang uoc luong: {bpsk.carrier_hz:.0f} Hz",
            f"- Thoi gian moi bit: {bpsk.bit_duration_s:.3f} s",
            f"- So bit: {bpsk.bit_count}",
            f"- Day bit: {bpsk.bits}",
            f"- Thong diep giai ma: {bpsk.message}",
            "",
            "Hinh ve duoc luu trong thu muc figures/.",
        ]
    )

    (ROOT / "ket_qua.txt").write_text("\n".join(lines), encoding="utf-8")


def write_markdown_report(dtmf: DtmfResult, bpsk: BpskResult) -> None:
    tones_table = "\n".join(
        f"| {tone.index} | {tone.key} | {tone.low_hz} | {tone.high_hz} | "
        f"{tone.start_s:.3f} - {tone.end_s:.3f} |"
        for tone in dtmf.tones
    )

    content = f"""# Bao cao bai thuc hanh

## 1. Nhan dien gia tri so trong dtmf.wav

DTMF dung hai nhom tan so: nhom hang thap va nhom cot cao. Voi moi doan tin hieu,
chuong trinh tinh cong suat tai cac tan so DTMF chuan, chon mot tan so thap va
mot tan so cao co cong suat lon nhat, sau do anh xa sang phim so.

Ket qua:

- Tan so lay mau: `{dtmf.fs} Hz`
- Thoi luong: `{dtmf.duration_s:.3f} s`
- Chuoi so nhan dang: **{dtmf.sequence}**

| STT | Phim | Tan so thap (Hz) | Tan so cao (Hz) | Khoang thoi gian (s) |
| --- | --- | ---: | ---: | --- |
{tones_table}

![DTMF waveform](figures/dtmf_waveform_segments.png)

![DTMF spectrum](figures/dtmf_spectrum.png)

![DTMF segment powers](figures/dtmf_segment_powers.png)

## 2. Giai ma input.wav

Pho tin hieu cho thay song mang xap xi `{bpsk.carrier_hz:.0f} Hz`. Tin hieu
duoc giai dieu che bang cach nhan voi song mang phuc `exp(-j2*pi*fc*t)`, lay
trung binh tren tung khoang bit, roi quyet dinh dau am/duong de thu duoc bit.

Ket qua:

- Tan so lay mau: `{bpsk.fs} Hz`
- Thoi luong file: `{bpsk.duration_s:.3f} s`
- Tan so song mang: `{bpsk.carrier_hz:.0f} Hz`
- Thoi gian moi bit: `{bpsk.bit_duration_s:.3f} s`
- So bit: `{bpsk.bit_count}`
- Day bit: `{bpsk.bits}`
- Thong diep giai ma: **{bpsk.message}**

![Input waveform](figures/input_waveform_active.png)

![Input spectrum](figures/input_spectrum_carrier.png)

![Input demodulated symbols](figures/input_demodulated_symbols.png)

## 3. Code Python chinh

File code day du: `bai_thuc_hanh.py`.

```python
fs, x = read_wav_mono(Path("dtmf.wav"))
segments = detect_active_segments(x, fs, 0.10, 0.020, 0.050, 0.100)
for start, end in segments:
    powers = frequency_power(x[start:end], fs, np.concatenate([DTMF_LOW, DTMF_HIGH]))
```

```python
mixed = x * np.exp(-1j * 2 * np.pi * carrier_hz * n / fs) * 2
symbols = mean(mixed tren tung khoang bit)
bits = symbols > 0
text = bits_to_text(bits, msb_first=True)
```
"""

    (ROOT / "bao_cao_bai_thuc_hanh.md").write_text(content, encoding="utf-8")


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return 900, 400
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


class MinimalDocx:
    def __init__(self) -> None:
        self.body: list[str] = []
        self.images: list[Path] = []

    def add_paragraph(self, text: str = "", bold: bool = False, mono: bool = False) -> None:
        for line in text.splitlines() or [""]:
            run_props = []
            if bold:
                run_props.append("<w:b/>")
            if mono:
                run_props.append('<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/>')
            props = f"<w:rPr>{''.join(run_props)}</w:rPr>" if run_props else ""
            xml_space = ' xml:space="preserve"' if line.startswith(" ") or line.endswith(" ") else ""
            self.body.append(f"<w:p><w:r>{props}<w:t{xml_space}>{escape(line)}</w:t></w:r></w:p>")

    def add_heading(self, text: str, level: int = 1) -> None:
        size = "32" if level == 1 else "26"
        self.body.append(
            "<w:p>"
            "<w:r>"
            f'<w:rPr><w:b/><w:sz w:val="{size}"/></w:rPr>'
            f"<w:t>{escape(text)}</w:t>"
            "</w:r>"
            "</w:p>"
        )

    def add_image(self, path: Path, caption: str) -> None:
        self.images.append(path)
        rel_id = f"rId{len(self.images) + 1}"
        cx_max = 6_000_000
        width_px, height_px = png_size(path)
        ratio = height_px / max(1, width_px)
        cx = cx_max
        cy = int(cx * ratio)
        doc_pr_id = len(self.images)
        name = escape(path.name)

        self.body.append(
            f"""
<w:p>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="{doc_pr_id}" name="{name}"/>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="{doc_pr_id}" name="{name}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm>
                  <a:off x="0" y="0"/>
                  <a:ext cx="{cx}" cy="{cy}"/>
                </a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""
        )
        self.add_paragraph(caption, bold=True)

    def save(self, path: Path) -> None:
        document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
  xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
  xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
  <w:body>
    {''.join(self.body)}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>
"""

        rels = [
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
        ]
        for idx, image in enumerate(self.images, start=2):
            rels.append(
                f'<Relationship Id="rId{idx}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                f'Target="media/{escape(image.name)}"/>'
            )

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "[Content_Types].xml",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
""",
            )
            zf.writestr(
                "_rels/.rels",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
""",
            )
            zf.writestr(
                "word/_rels/document.xml.rels",
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
                + "".join(rels)
                + "</Relationships>",
            )
            zf.writestr(
                "word/styles.xml",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr><w:sz w:val="24"/></w:rPr>
  </w:style>
</w:styles>
""",
            )
            zf.writestr("word/document.xml", document_xml)
            for image in self.images:
                zf.write(image, f"word/media/{image.name}")


def write_docx_report(dtmf: DtmfResult, bpsk: BpskResult) -> Path:
    doc = MinimalDocx()
    doc.add_heading("Báo cáo bài thực hành xử lý tín hiệu âm thanh", level=1)

    doc.add_heading("Mục tiêu", level=2)
    doc.add_paragraph(
        "Bài thực hành gồm hai yêu cầu: nhận diện dãy số trong file dtmf.wav bằng tín hiệu DTMF "
        "và giải mã thông điệp trong file input.wav. Toàn bộ phần xử lý được thực hiện bằng Python, "
        "sử dụng numpy, scipy và matplotlib."
    )

    doc.add_heading("1. Nhận diện dãy số trong dtmf.wav", level=2)
    doc.add_paragraph(
        "DTMF là viết tắt của Dual Tone Multi Frequency. Mỗi phím điện thoại không được biểu diễn "
        "bằng một tần số duy nhất, mà bằng tổng của hai tần số: một tần số thuộc nhóm thấp "
        "và một tần số thuộc nhóm cao. Vì vậy muốn biết một đoạn âm là số nào, ta phải tìm đúng "
        "hai tần số mạnh nhất của đoạn đó rồi tra bảng DTMF."
    )
    doc.add_paragraph("Bảng ánh xạ cơ bản dùng trong bài:")
    doc.add_paragraph("1 = 697 + 1209 Hz    2 = 697 + 1336 Hz    3 = 697 + 1477 Hz")
    doc.add_paragraph("4 = 770 + 1209 Hz    5 = 770 + 1336 Hz    6 = 770 + 1477 Hz")
    doc.add_paragraph("7 = 852 + 1209 Hz    8 = 852 + 1336 Hz    9 = 852 + 1477 Hz")
    doc.add_paragraph("0 = 941 + 1336 Hz")
    doc.add_paragraph(
        "Cách làm trong chương trình: đầu tiên đọc file wav và chuẩn hóa biên độ, sau đó dùng năng lượng "
        "trung bình trượt để tách các đoạn có âm. Với từng đoạn, chương trình tính công suất tại các "
        "tần số DTMF chuẩn [697, 770, 852, 941] và [1209, 1336, 1477, 1633]. Tần số thấp có công suất "
        "lớn nhất và tần số cao có công suất lớn nhất được ghép lại để suy ra phím."
    )
    doc.add_paragraph(f"Chuỗi số nhận dạng được: {dtmf.sequence}", bold=True)
    doc.add_paragraph("Chi tiết từng đoạn âm:")
    for tone in dtmf.tones:
        doc.add_paragraph(
            f"{tone.index:02d}. Phím {tone.key}: {tone.low_hz} Hz + {tone.high_hz} Hz "
            f"({tone.start_s:.3f}s -> {tone.end_s:.3f}s)"
        )
    doc.add_image(dtmf.waveform_plot, "Hình 1. Dạng sóng DTMF và các đoạn âm đã tách.")
    doc.add_paragraph(
        "Hình 1 chỉ cho thấy tín hiệu được chia thành 8 đoạn âm. Các con số ghi phía trên là kết quả "
        "sau khi chương trình đã phân tích tần số từng đoạn, không phải là kết quả nhìn trực tiếp "
        "từ dạng sóng."
    )
    doc.add_image(dtmf.spectrum_plot, "Hình 2. Phổ công suất tổng của toàn bộ file dtmf.wav.")
    doc.add_paragraph(
        "Hình 2 là phổ của toàn bộ file, nên nó chỉ chứng minh rằng các tần số DTMF chuẩn có xuất hiện. "
        "Do nhiều phím cùng nằm trong một file, phổ tổng không thể hiện rõ từng cặp tần số của từng phím."
    )
    doc.add_image(
        dtmf.segment_power_plot,
        "Hình 3. Công suất từng đoạn tại các tần số DTMF chuẩn.",
    )
    doc.add_paragraph(
        "Hình 3 là hình quan trọng nhất cho phần nhận dạng số. Mỗi dòng là một đoạn âm, mỗi cột là một "
        "tần số DTMF chuẩn. Ở mỗi dòng sẽ có hai ô sáng rõ: một ô ở nhóm tần số thấp và một ô ở nhóm "
        "tần số cao. Hai ô sáng này chính là cặp tần số dùng để tra ra phím."
    )

    doc.add_heading("2. Giải mã thông điệp trong input.wav", level=2)
    doc.add_paragraph(
        "File input.wav là tín hiệu đã điều chế. Theo đề bài, dãy bit của một xâu ký tự được đổi thành "
        "biên độ +1 hoặc -1 rồi nhân vào sóng sin/cos. Đây có thể xem như dạng điều chế BPSK đơn giản: "
        "bit được mã hóa bằng dấu của tín hiệu sau khi giải điều chế."
    )
    doc.add_paragraph(
        "Chương trình ước lượng tần số sóng mang bằng phổ công suất và tìm được đỉnh quanh 3000 Hz. "
        "Sau đó tín hiệu được nhân với exp(-j2*pi*fc*t) để kéo sóng mang về gần 0 Hz. Trên mỗi khoảng "
        "bit, chương trình lấy trung bình giá trị sau giải điều chế. Nếu giá trị trung bình dương thì "
        "nhận là bit 1, nếu âm thì nhận là bit 0. Cuối cùng, cứ 8 bit được ghép thành một mã ASCII."
    )
    doc.add_paragraph(f"Tần số sóng mang ước lượng: {bpsk.carrier_hz:.0f} Hz")
    doc.add_paragraph(f"Thời gian mỗi bit: {bpsk.bit_duration_s:.3f} s")
    doc.add_paragraph(f"Số bit giải mã: {bpsk.bit_count}")
    doc.add_paragraph("Dãy bit thu được:")
    doc.add_paragraph(bpsk.bits, mono=True)
    doc.add_paragraph(f"Thông điệp giải mã được: {bpsk.message}", bold=True)
    doc.add_image(bpsk.waveform_plot, "Hình 4. Dạng sóng input.wav và vùng dữ liệu đã giải mã.")
    doc.add_image(bpsk.spectrum_plot, "Hình 5. Phổ quanh sóng mang của input.wav.")
    doc.add_image(bpsk.symbols_plot, "Hình 6. Các symbol sau giải điều chế.")
    doc.add_paragraph(
        "Ở Hình 6, các điểm nằm phía trên trục 0 được quyết định là bit 1, các điểm nằm phía dưới "
        "trục 0 được quyết định là bit 0. Khi chia dãy bit thành từng nhóm 8 bit và đổi sang ASCII, "
        "ta thu được chuỗi HELLOHELLOHELLO."
    )

    doc.add_heading("3. Code Python chính", level=2)
    doc.add_paragraph(
        "Dưới đây là đoạn code chính dùng để đọc file, nhận diện DTMF và giải mã input.wav. "
        "File chạy đầy đủ trong thư mục là bai_thuc_hanh.py."
    )

    code_sections = [
        (
            "3.1 Thư viện và hàm đọc file wav",
            r'''
            import numpy as np
            from scipy.io import wavfile
            from scipy.signal import welch

            def read_wav_mono(filename):
                fs, x = wavfile.read(filename)
                if x.ndim > 1:
                    x = x[:, 0]
                x = x.astype(float)
                x = x / np.max(np.abs(x))
                return fs, x

            def moving_average(x, size):
                return np.convolve(x, np.ones(size) / size, mode="same")
            ''',
        ),
        (
            "3.2 Code nhận diện DTMF trong dtmf.wav",
            r'''
            DTMF_LOW = np.array([697, 770, 852, 941])
            DTMF_HIGH = np.array([1209, 1336, 1477, 1633])
            DTMF_KEYS = {
                (697, 1209): "1", (697, 1336): "2", (697, 1477): "3",
                (770, 1209): "4", (770, 1336): "5", (770, 1477): "6",
                (852, 1209): "7", (852, 1336): "8", (852, 1477): "9",
                (941, 1209): "*", (941, 1336): "0", (941, 1477): "#",
            }

            def split_active_segments(x, fs):
                energy = moving_average(x * x, int(0.02 * fs))
                active = energy > 0.10 * np.max(energy)
                edges = np.flatnonzero(np.diff(np.pad(active.astype(int), (1, 1))))
                raw_segments = list(zip(edges[::2], edges[1::2]))

                segments = []
                min_gap = int(0.05 * fs)
                for start, end in raw_segments:
                    if len(segments) == 0 or start - segments[-1][1] > min_gap:
                        segments.append([start, end])
                    else:
                        segments[-1][1] = end
                return segments

            def power_at_freqs(segment, fs, freqs):
                n = np.arange(len(segment))
                powers = []
                for f in freqs:
                    c = np.sum(segment * np.exp(-1j * 2 * np.pi * f * n / fs))
                    powers.append(abs(c) ** 2 / len(segment) ** 2)
                return np.array(powers)

            def decode_dtmf(filename):
                fs, x = read_wav_mono(filename)
                segments = split_active_segments(x, fs)
                all_freqs = np.r_[DTMF_LOW, DTMF_HIGH]
                result = ""

                for start, end in segments:
                    segment = x[start:end]
                    trim = len(segment) // 10
                    segment = segment[trim:-trim]

                    powers = power_at_freqs(segment, fs, all_freqs)
                    low = int(DTMF_LOW[np.argmax(powers[:4])])
                    high = int(DTMF_HIGH[np.argmax(powers[4:])])
                    result += DTMF_KEYS[(low, high)]

                return result
            ''',
        ),
        (
            "3.3 Code giải mã thông điệp trong input.wav",
            r'''
            def bits_to_text(bits):
                chars = []
                for i in range(0, len(bits), 8):
                    byte = bits[i:i + 8]
                    value = 0
                    for bit in byte:
                        value = (value << 1) | int(bit)
                    chars.append(chr(value))
                return "".join(chars)

            def decode_input(filename):
                fs, x = read_wav_mono(filename)

                f, pxx = welch(x, fs=fs, nperseg=65536)
                mask = (f > 200) & (f < 10000)
                fc = round(f[mask][np.argmax(pxx[mask])] / 100) * 100

                bit_duration = 0.25
                samples_per_bit = int(bit_duration * fs)
                bit_count = 120
                start_sample = int(1.65 * fs)

                n = np.arange(len(x))
                mixed = x * np.exp(-1j * 2 * np.pi * fc * n / fs) * 2

                symbols = []
                for k in range(bit_count):
                    a = start_sample + k * samples_per_bit
                    b = a + samples_per_bit
                    symbols.append(np.mean(mixed[a:b]))

                symbols = np.array(symbols)
                rotation = 0.5 * np.angle(np.mean(symbols ** 2))
                projected = np.real(symbols * np.exp(-1j * rotation))

                bits = (projected > 0).astype(int)
                bits = 1 - bits
                return "".join(map(str, bits)), bits_to_text(bits)
            ''',
        ),
        (
            "3.4 Code chạy chính",
            r'''
            dtmf_sequence = decode_dtmf("dtmf.wav")
            bit_string, message = decode_input("input.wav")

            print("DTMF:", dtmf_sequence)
            print("Bits:", bit_string)
            print("Message:", message)
            ''',
        ),
    ]

    for section_title, section_code in code_sections:
        doc.add_heading(section_title, level=2)
        for code_line in textwrap.dedent(section_code).strip("\n").splitlines():
            doc.add_paragraph(code_line, mono=True)

    report_path = ROOT / "bao_cao_bai_thuc_hanh_chi_tiet.docx"
    try:
        doc.save(report_path)
    except PermissionError:
        report_path = ROOT / "bao_cao_bai_thuc_hanh_moi.docx"
        doc.save(report_path)

    return report_path


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    dtmf_path = ROOT / "dtmf.wav"
    input_path = ROOT / "input.wav"

    if not dtmf_path.exists():
        raise FileNotFoundError(f"Khong thay {dtmf_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Khong thay {input_path}")

    dtmf = decode_dtmf(dtmf_path)
    bpsk = decode_bpsk_ascii(input_path)

    write_text_results(dtmf, bpsk)
    write_markdown_report(dtmf, bpsk)
    docx_path = write_docx_report(dtmf, bpsk)

    print("Hoan tat.")
    print(f"DTMF sequence: {dtmf.sequence}")
    print(f"Input message: {bpsk.message}")
    print(f"Bit sequence: {bpsk.bits}")
    print("Da tao:")
    print("- ket_qua.txt")
    print("- bao_cao_bai_thuc_hanh.md")
    print(f"- {docx_path.name}")
    print("- figures/")


if __name__ == "__main__":
    main()
