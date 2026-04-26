import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# ============================================================
# Output directory
# ============================================================

os.makedirs("out", exist_ok=True)


# ============================================================
# Data from Tables 5.6 and 5.7
# ============================================================

data = {
    "Backend": ["TVM", "ONNX Runtime", "TorchCompile", "LiteRT", "PyTorch eager"],
    "Latency_ms": [192.0, 189.2, 202.9, 154.1, 290.1],
    "Latency_CI95_halfwidth": [7.5, 7.0, 8.5, 2.5, 2.3],
    "FPS": [5.21, 5.28, 4.93, 6.49, 3.45],
    "Median_ms": [188.8, 187.2, 202.2, 155.2, 289.8],
    "Mean_power_mW": [4096.2, 4043.2, 2291.8, 2292.8, 2316.4],
    "Max_power_mW": [4783.1, 5415.1, 2742.5, 2591.5, 2759.8],
    "Power_std_mW": [46.1, 172.5, 63.2, 14.2, 30.3],
}

df = pd.DataFrame(data)

df["Mean_power_W"] = df["Mean_power_mW"] / 1000.0
df["Power_std_W"] = df["Power_std_mW"] / 1000.0
df["Max_power_W"] = df["Max_power_mW"] / 1000.0
df["FPS_per_W"] = df["FPS"] / df["Mean_power_W"]
df["Energy_per_inf_J"] = df["Mean_power_W"] * (df["Latency_ms"] / 1000.0)


# ============================================================
# Dark Sunset thesis style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,

    # Clean frame
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": "#4A403A",
    "axes.linewidth": 0.7,

    # Warm, soft grid
    "grid.color": "#E3DED5",
    "grid.linestyle": "-",
    "grid.linewidth": 0.55,
    "grid.alpha": 0.78,
})


# ============================================================
# Dark Sunset palette
# ============================================================
# From screenshot:
# dark teal, pale yellow, amber, muted red, dark burgundy

backend_colors = {
    "LiteRT": "#F6E8A6",          # pale yellow, best-performing point
    "ONNX Runtime": "#38636B",    # dark teal
    "TVM": "#E0A23A",             # amber
    "TorchCompile": "#A62B2F",    # muted red
    "PyTorch eager": "#5E0B10",   # dark burgundy
}

text_color = "#1F1F1F"
muted_text = "#6B625C"
edge_color = "#3F332E"
background_color = "#FFFFFF"


def style_axis(ax, grid_axis="y"):
    ax.grid(axis=grid_axis)
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both",
        which="both",
        length=3.0,
        width=0.7,
        color=edge_color,
        labelcolor=text_color,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(f"out/{name}.png", bbox_inches="tight", facecolor=background_color)
    fig.savefig(f"out/{name}.pdf", bbox_inches="tight", facecolor=background_color)


def add_bar_labels(ax, bars, values, fmt="{:.1f}", offset=0.02, fontsize=8.5):
    ymax = ax.get_ylim()[1]
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset * ymax,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=text_color,
        )


# ============================================================
# Sorting
# ============================================================

backend_order_latency = df.sort_values("Latency_ms")["Backend"].tolist()
backend_order_power = df.sort_values("Mean_power_W")["Backend"].tolist()
backend_order_efficiency = df.sort_values("FPS_per_W", ascending=False)["Backend"].tolist()

df_latency = df.set_index("Backend").loc[backend_order_latency].reset_index()
df_power = df.set_index("Backend").loc[backend_order_power].reset_index()
df_eff = df.set_index("Backend").loc[backend_order_efficiency].reset_index()


# ============================================================
# Figure 1: Mean latency with 95% confidence intervals
# ============================================================

fig, ax = plt.subplots(figsize=(6.6, 3.9), facecolor=background_color)

x = np.arange(len(df_latency))
colors = [backend_colors[b] for b in df_latency["Backend"]]

bars = ax.bar(
    x,
    df_latency["Latency_ms"],
    yerr=df_latency["Latency_CI95_halfwidth"],
    capsize=3,
    color=colors,
    edgecolor=edge_color,
    linewidth=0.55,
    width=0.62,
    error_kw={
        "elinewidth": 0.8,
        "capthick": 0.8,
        "ecolor": edge_color,
    },
)

ax.set_xticks(x)
ax.set_xticklabels(df_latency["Backend"], rotation=25, ha="right")
ax.set_ylabel("Mean latency (ms)")
ax.set_title("YOLOv8n inference latency on Raspberry Pi 5 CPU")
ax.set_ylim(0, df_latency["Latency_ms"].max() + 48)

style_axis(ax)
add_bar_labels(
    ax,
    bars,
    df_latency["Latency_ms"],
    fmt="{:.1f}",
    offset=0.025,
)

save_figure(fig, "pi5_latency_ci_dark_sunset")
plt.show()


# ============================================================
# Figure 2: Mean power with standard deviation
# No max-power marker
# ============================================================

fig, ax = plt.subplots(figsize=(6.6, 3.9), facecolor=background_color)

x = np.arange(len(df_power))
colors = [backend_colors[b] for b in df_power["Backend"]]

bars = ax.bar(
    x,
    df_power["Mean_power_W"],
    yerr=df_power["Power_std_W"],
    capsize=3,
    color=colors,
    edgecolor=edge_color,
    linewidth=0.55,
    width=0.62,
    error_kw={
        "elinewidth": 0.8,
        "capthick": 0.8,
        "ecolor": edge_color,
    },
)

ax.set_xticks(x)
ax.set_xticklabels(df_power["Backend"], rotation=25, ha="right")
ax.set_ylabel("Mean power (W)")
ax.set_title("Power consumption during YOLOv8n inference")
ax.set_ylim(0, df_power["Mean_power_W"].max() + 0.75)

style_axis(ax)
add_bar_labels(
    ax,
    bars,
    df_power["Mean_power_W"],
    fmt="{:.2f}",
    offset=0.025,
)

ax.text(
    0.01,
    0.96,
    "Error bars show standard deviation",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.5,
    color=muted_text,
)

save_figure(fig, "pi5_power_dark_sunset")
plt.show()


# ============================================================
# Figure 3: Latency-power trade-off
# ============================================================

fig, ax = plt.subplots(figsize=(6.4, 4.1), facecolor=background_color)

for _, row in df.iterrows():
    ax.scatter(
        row["Latency_ms"],
        row["Mean_power_W"],
        s=105 if row["Backend"] == "LiteRT" else 82,
        color=backend_colors[row["Backend"]],
        edgecolor=edge_color,
        linewidth=0.75,
        zorder=3,
    )

label_offsets = {
    "LiteRT": (7, -2),
    "TorchCompile": (8, 8),
    "PyTorch eager": (8, 8),
    "ONNX Runtime": (8, -5),
    "TVM": (8, 10),
}

for _, row in df.iterrows():
    dx, dy = label_offsets[row["Backend"]]
    ax.annotate(
        row["Backend"],
        xy=(row["Latency_ms"], row["Mean_power_W"]),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        color=text_color,
    )

ax.set_xlabel("Mean latency (ms)")
ax.set_ylabel("Mean power (W)")
ax.set_title("Latency–power trade-off on Raspberry Pi 5 CPU")
ax.set_xlim(145, 305)
ax.set_ylim(2.15, 4.25)

ax.grid(True)
ax.set_axisbelow(True)
ax.tick_params(
    axis="both",
    which="both",
    length=3.0,
    width=0.7,
    color=edge_color,
    labelcolor=text_color,
)
ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

save_figure(fig, "pi5_latency_power_tradeoff_dark_sunset")
plt.show()


# ============================================================
# Figure 4: Energy efficiency, FPS/W
# ============================================================

fig, ax = plt.subplots(figsize=(6.6, 3.9), facecolor=background_color)

x = np.arange(len(df_eff))
colors = [backend_colors[b] for b in df_eff["Backend"]]

bars = ax.bar(
    x,
    df_eff["FPS_per_W"],
    color=colors,
    edgecolor=edge_color,
    linewidth=0.55,
    width=0.62,
)

ax.set_xticks(x)
ax.set_xticklabels(df_eff["Backend"], rotation=25, ha="right")
ax.set_ylabel("Throughput per watt (FPS/W)")
ax.set_title("Energy efficiency of YOLOv8n inference")
ax.set_ylim(0, df_eff["FPS_per_W"].max() + 0.45)

style_axis(ax)
add_bar_labels(
    ax,
    bars,
    df_eff["FPS_per_W"],
    fmt="{:.2f}",
    offset=0.025,
)

save_figure(fig, "pi5_fps_per_watt_dark_sunset")
plt.show()


# ============================================================
# Combined 2x2 figure for thesis
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.3), facecolor=background_color)

# ------------------------------------------------------------
# (a) Latency
# ------------------------------------------------------------

ax = axes[0, 0]
df_plot = df_latency
x = np.arange(len(df_plot))
colors = [backend_colors[b] for b in df_plot["Backend"]]

bars = ax.bar(
    x,
    df_plot["Latency_ms"],
    yerr=df_plot["Latency_CI95_halfwidth"],
    capsize=3,
    color=colors,
    edgecolor=edge_color,
    linewidth=0.55,
    width=0.62,
    error_kw={
        "elinewidth": 0.8,
        "capthick": 0.8,
        "ecolor": edge_color,
    },
)

ax.set_xticks(x)
ax.set_xticklabels(df_plot["Backend"], rotation=25, ha="right")
ax.set_ylabel("Mean latency (ms)")
ax.set_title("(a) Latency")
ax.set_ylim(0, df_plot["Latency_ms"].max() + 48)
style_axis(ax)
add_bar_labels(ax, bars, df_plot["Latency_ms"], fmt="{:.1f}", offset=0.025, fontsize=8)


# ------------------------------------------------------------
# (b) Power
# ------------------------------------------------------------

ax = axes[0, 1]
df_plot = df_power
x = np.arange(len(df_plot))
colors = [backend_colors[b] for b in df_plot["Backend"]]

bars = ax.bar(
    x,
    df_plot["Mean_power_W"],
    yerr=df_plot["Power_std_W"],
    capsize=3,
    color=colors,
    edgecolor=edge_color,
    linewidth=0.55,
    width=0.62,
    error_kw={
        "elinewidth": 0.8,
        "capthick": 0.8,
        "ecolor": edge_color,
    },
)

ax.set_xticks(x)
ax.set_xticklabels(df_plot["Backend"], rotation=25, ha="right")
ax.set_ylabel("Mean power (W)")
ax.set_title("(b) Power")
ax.set_ylim(0, df_plot["Mean_power_W"].max() + 0.75)
style_axis(ax)
add_bar_labels(ax, bars, df_plot["Mean_power_W"], fmt="{:.2f}", offset=0.025, fontsize=8)

ax.text(
    0.01,
    0.96,
    "Error bars: std.",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8,
    color=muted_text,
)


# ------------------------------------------------------------
# (c) Latency-power scatter
# ------------------------------------------------------------

ax = axes[1, 0]

for _, row in df.iterrows():
    ax.scatter(
        row["Latency_ms"],
        row["Mean_power_W"],
        s=95 if row["Backend"] == "LiteRT" else 76,
        color=backend_colors[row["Backend"]],
        edgecolor=edge_color,
        linewidth=0.75,
        zorder=3,
    )

for _, row in df.iterrows():
    dx, dy = label_offsets[row["Backend"]]
    ax.annotate(
        row["Backend"],
        xy=(row["Latency_ms"], row["Mean_power_W"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.5,
        ha="left",
        va="center",
        color=text_color,
    )

ax.set_xlabel("Mean latency (ms)")
ax.set_ylabel("Mean power (W)")
ax.set_title("(c) Latency–power trade-off")
ax.set_xlim(145, 305)
ax.set_ylim(2.15, 4.25)
ax.grid(True)
ax.set_axisbelow(True)
ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax.tick_params(
    axis="both",
    which="both",
    length=3.0,
    width=0.7,
    color=edge_color,
    labelcolor=text_color,
)


# ------------------------------------------------------------
# (d) Energy efficiency
# ------------------------------------------------------------

ax = axes[1, 1]
df_plot = df_eff
x = np.arange(len(df_plot))
colors = [backend_colors[b] for b in df_plot["Backend"]]

bars = ax.bar(
    x,
    df_plot["FPS_per_W"],
    color=colors,
    edgecolor=edge_color,
    linewidth=0.55,
    width=0.62,
)

ax.set_xticks(x)
ax.set_xticklabels(df_plot["Backend"], rotation=25, ha="right")
ax.set_ylabel("Throughput per watt (FPS/W)")
ax.set_title("(d) Energy efficiency")
ax.set_ylim(0, df_plot["FPS_per_W"].max() + 0.45)
style_axis(ax)
add_bar_labels(ax, bars, df_plot["FPS_per_W"], fmt="{:.2f}", offset=0.025, fontsize=8)


fig.suptitle(
    "YOLOv8n inference performance and power on Raspberry Pi 5 CPU",
    fontsize=13,
    y=1.01,
    color=text_color,
)

fig.tight_layout()
fig.savefig(
    "out/pi5_yolov8n_cpu_results_dark_sunset.png",
    bbox_inches="tight",
    facecolor=background_color,
)
fig.savefig(
    "out/pi5_yolov8n_cpu_results_dark_sunset.pdf",
    bbox_inches="tight",
    facecolor=background_color,
)
plt.show()


# ============================================================
# Optional: print derived metrics
# ============================================================

summary = df[[
    "Backend",
    "Latency_ms",
    "FPS",
    "Mean_power_W",
    "FPS_per_W",
    "Energy_per_inf_J",
]].copy()

summary = summary.sort_values("FPS_per_W", ascending=False)

print(summary.to_string(index=False))