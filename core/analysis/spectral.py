import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import welch


def compute_fft_spectrum(
    signal: np.ndarray, sampling_rate: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT spectrum of a signal.

    Args:
        signal: Input signal
        sampling_rate: Sampling rate (bars per unit time)

    Returns:
        Tuple of (frequencies, power_spectrum)
    """
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([])

    mean_centered = signal - np.mean(signal)

    fft_coeffs = np.fft.fft(mean_centered)
    power_spectrum = np.abs(fft_coeffs) ** 2

    frequencies = np.fft.fftfreq(n, d=1.0 / sampling_rate)

    positive_mask = frequencies > 0
    frequencies = frequencies[positive_mask]
    power_spectrum = power_spectrum[positive_mask]

    return frequencies, power_spectrum


def find_dominant_peaks(
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    n_peaks: int = 5,
    min_power_ratio: float = 0.01,
) -> list[dict[str, float]]:
    """
    Find dominant frequency peaks in the power spectrum.

    Args:
        frequencies: Frequency array
        power_spectrum: Power spectrum array
        n_peaks: Number of top peaks to return
        min_power_ratio: Minimum power as fraction of max power

    Returns:
        List of peak dictionaries with 'frequency', 'power', and 'period'
    """
    if len(power_spectrum) == 0:
        return []

    max_power = np.max(power_spectrum)
    threshold = max_power * min_power_ratio

    peak_indices = []
    for i in range(1, len(power_spectrum) - 1):
        if (
            power_spectrum[i] > power_spectrum[i - 1]
            and power_spectrum[i] > power_spectrum[i + 1]
            and power_spectrum[i] >= threshold
        ):
            peak_indices.append(i)

    peak_indices = sorted(peak_indices, key=lambda i: power_spectrum[i], reverse=True)[:n_peaks]

    peaks = []
    for idx in peak_indices:
        freq = frequencies[idx]
        power = power_spectrum[idx]
        period = 1.0 / freq if freq > 0 else np.inf
        peaks.append({"frequency": freq, "power": power, "period": period})

    return peaks


def plot_fft_spectrum(
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    peaks: list[dict[str, float]],
    interval: str = "1h",
) -> go.Figure:
    """
    Create Plotly figure for FFT spectrum with labeled peaks.

    Args:
        frequencies: Frequency array
        power_spectrum: Power spectrum array
        peaks: List of dominant peaks
        interval: Time interval for period conversion

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=power_spectrum,
            mode="lines",
            name="Power Spectrum",
            line=dict(color="blue", width=1),
        )
    )

    for i, peak in enumerate(peaks):
        freq = peak["frequency"]
        power = peak["power"]
        period_bars = peak["period"]

        period_str = format_period(period_bars, interval)

        fig.add_trace(
            go.Scatter(
                x=[freq],
                y=[power],
                mode="markers+text",
                name=f"Peak {i + 1}",
                marker=dict(size=10, color="red"),
                text=[period_str],
                textposition="top center",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="FFT Power Spectrum",
        xaxis_title="Frequency (cycles/bar)",
        yaxis_title="Power",
        height=400,
        hovermode="x unified",
    )

    return fig


def compute_welch_psd(
    signal: np.ndarray,
    window_length: int = 256,
    overlap_ratio: float = 0.5,
    sampling_rate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.

    Args:
        signal: Input signal
        window_length: Length of each window segment
        overlap_ratio: Overlap between segments (0.0 to 1.0)
        sampling_rate: Sampling rate

    Returns:
        Tuple of (frequencies, power_spectral_density)
    """
    if len(signal) < window_length:
        window_length = len(signal)

    noverlap = int(window_length * overlap_ratio)

    frequencies, psd = welch(
        signal,
        fs=sampling_rate,
        window="hann",
        nperseg=window_length,
        noverlap=noverlap,
        detrend="constant",
    )

    return frequencies, psd


def compute_sliding_dominant_period(
    signal: np.ndarray,
    window_length: int = 256,
    overlap_ratio: float = 0.5,
    sampling_rate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute dominant period over time using sliding windows.

    Args:
        signal: Input signal
        window_length: Window length for each PSD computation
        overlap_ratio: Overlap between windows
        sampling_rate: Sampling rate

    Returns:
        Tuple of (time_indices, dominant_periods)
    """
    if len(signal) < window_length:
        window_length = len(signal)

    step = max(1, int(window_length * (1 - overlap_ratio)))
    n_windows = (len(signal) - window_length) // step + 1

    time_indices = []
    dominant_periods = []

    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + window_length
        window_data = signal[start_idx:end_idx]

        freqs, psd = compute_welch_psd(
            window_data,
            window_length=min(window_length, len(window_data)),
            overlap_ratio=overlap_ratio,
            sampling_rate=sampling_rate,
        )

        if len(freqs) > 1 and len(psd) > 1:
            peak_idx = np.argmax(psd[1:]) + 1
            dominant_freq = freqs[peak_idx]
            if dominant_freq > 0:
                dominant_period = 1.0 / dominant_freq
            else:
                dominant_period = np.nan
        else:
            dominant_period = np.nan

        center_idx = start_idx + window_length // 2
        time_indices.append(center_idx)
        dominant_periods.append(dominant_period)

    return np.array(time_indices), np.array(dominant_periods)


def plot_sliding_dominant_period(
    timestamps: pd.DatetimeIndex,
    time_indices: np.ndarray,
    dominant_periods: np.ndarray,
    interval: str = "1h",
) -> go.Figure:
    """
    Create Plotly figure for sliding window dominant period.

    Args:
        timestamps: Original timestamps
        time_indices: Indices corresponding to window centers
        dominant_periods: Dominant periods at each window
        interval: Time interval for conversion

    Returns:
        Plotly figure
    """
    valid_mask = ~np.isnan(dominant_periods)
    valid_indices = time_indices[valid_mask]
    valid_periods = dominant_periods[valid_mask]

    if len(valid_indices) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid periods found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    valid_indices = valid_indices.astype(int)
    valid_indices = np.clip(valid_indices, 0, len(timestamps) - 1)
    valid_times = timestamps[valid_indices]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=valid_times,
            y=valid_periods,
            mode="lines+markers",
            name="Dominant Period",
            line=dict(color="purple", width=2),
            marker=dict(size=4),
        )
    )

    fig.update_layout(
        title="Sliding Window Dominant Period",
        xaxis_title="Time",
        yaxis_title="Period (bars)",
        height=400,
        hovermode="x unified",
    )

    return fig


def format_period(period_bars: float, interval: str) -> str:
    """
    Format period in bars to human-readable string with hours.

    Args:
        period_bars: Period in number of bars
        interval: Time interval (e.g., "30m", "1h", "4h")

    Returns:
        Formatted string (e.g., "24.5 bars (24.5h)")
    """
    interval_hours = {
        "30m": 0.5,
        "1h": 1.0,
        "4h": 4.0,
    }

    hours_per_bar = interval_hours.get(interval, 1.0)
    period_hours = period_bars * hours_per_bar

    if period_hours < 24:
        return f"{period_bars:.1f} bars ({period_hours:.1f}h)"
    else:
        period_days = period_hours / 24
        return f"{period_bars:.1f} bars ({period_days:.1f}d)"


def create_welch_heatmap(
    signal: np.ndarray,
    timestamps: pd.DatetimeIndex,
    window_length: int = 256,
    overlap_ratio: float = 0.5,
    max_period_bars: int = 100,
) -> go.Figure:
    """
    Create a heatmap of power spectral density over time using Welch's method.

    Args:
        signal: Input signal
        timestamps: Timestamps for the signal
        window_length: Window length for Welch PSD
        overlap_ratio: Overlap between windows
        max_period_bars: Maximum period to display in bars

    Returns:
        Plotly heatmap figure
    """
    if len(signal) < window_length:
        window_length = len(signal)

    step = max(1, int(window_length * (1 - overlap_ratio)))
    n_windows = (len(signal) - window_length) // step + 1

    all_freqs = None
    all_psds = []
    time_centers = []

    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + window_length
        window_data = signal[start_idx:end_idx]

        freqs, psd = compute_welch_psd(
            window_data,
            window_length=min(window_length, len(window_data)),
            overlap_ratio=overlap_ratio,
        )

        if all_freqs is None:
            all_freqs = freqs

        all_psds.append(psd)
        center_idx = start_idx + window_length // 2
        center_idx = min(center_idx, len(timestamps) - 1)
        time_centers.append(timestamps[center_idx])

    if all_freqs is None or len(all_psds) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for heatmap",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    psd_matrix = np.array(all_psds).T

    periods = np.array([1.0 / f if f > 0 else np.inf for f in all_freqs])
    valid_period_mask = (periods > 0) & (periods <= max_period_bars)

    filtered_periods = periods[valid_period_mask]
    filtered_psd = psd_matrix[valid_period_mask, :]

    fig = go.Figure(
        data=go.Heatmap(
            z=np.log10(filtered_psd + 1e-10),
            x=time_centers,
            y=filtered_periods,
            colorscale="Viridis",
            colorbar=dict(title="log10(PSD)"),
        )
    )

    fig.update_layout(
        title="Welch PSD Heatmap (Time vs Period)",
        xaxis_title="Time",
        yaxis_title="Period (bars)",
        height=500,
        yaxis=dict(autorange="reversed"),
    )

    return fig
