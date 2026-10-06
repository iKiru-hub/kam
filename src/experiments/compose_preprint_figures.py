"""Compose manuscript Figures 3 and 4 from the saved experiment panels.

No simulations are run. Figure 3 retains the existing A--D layout; Figure 4
embeds the current sensitivity and load SVGs, avoiding stale embedded copies.
Run from the repository root after regenerating individual experiment panels.
Requires rsvg-convert on PATH.
"""

from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/preprint"
OUTPUT = ROOT / "article/figures/preprint"
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def tag(name):
    return f"{{{SVG}}}{name}"


def embed(parent, path, prefix, x, y, width):
    source = ET.parse(path).getroot()
    _, _, source_width, source_height = map(float, source.attrib["viewBox"].split())
    ids = {node.attrib["id"]: prefix + node.attrib["id"]
           for node in source.iter() if "id" in node.attrib}
    for node in source.iter():
        for key, value in list(node.attrib.items()):
            if key == "id":
                node.set(key, ids[value])
            elif value.startswith("#") and value[1:] in ids:
                node.set(key, "#" + ids[value[1:]])
            else:
                node.set(key, re.sub(r"url\(#([^)]+)\)",
                                    lambda m: f"url(#{ids.get(m[1], m[1])})", value))
    group = ET.SubElement(parent, tag("g"), {
        "transform": f"translate({x},{y}) scale({width/source_width})"})
    for child in source:
        group.append(child)
    return width * source_height / source_width


def label(parent, text, x, y):
    ET.SubElement(parent, tag("text"), {
        "x": str(x), "y": str(y), "font-family": "sans-serif",
        "font-size": "14", "font-weight": "bold"}).text = text


def save(root, name):
    svg_path = OUTPUT / f"{name}.svg"
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o",
                    str(OUTPUT / f"{name}.pdf"), str(svg_path)], check=True)


def sensitivity_panel():
    """Use stacked heatmaps so cell annotations remain legible at page width."""
    data = np.load(RESULTS / "parameter_sensitivity/arrays.npz")
    means, deviations = data["paired_delta_mean"], data["paired_delta_std"]
    assert means.shape == deviations.shape == (2, 5, 10)
    assert data["rules"].tolist() == ["base", "err2"]
    limit = float(np.abs(means).max())
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 3.35))
    fig.subplots_adjust(left=0.12, right=0.90, bottom=0.12, top=0.93, hspace=0.52)
    labels = ["CA3 fan-in", r"$K_{\mathrm{CA3}}$", r"$\beta_{\mathrm{CA3}}$",
              r"$\beta_{\mathrm{CA1}}$", r"$\alpha$"]
    for index, (axis, title) in enumerate(zip(axes, ["Instructive-driven", "Error-driven"])):
        im = axis.imshow(means[index], cmap="RdBu_r", vmin=-limit,
                         vmax=limit, aspect="auto")
        axis.set_title(title, fontsize=9, pad=5)
        axis.set_yticks(range(5), labels, fontsize=7)
        axis.set_xticks(range(10), [f"{m:g}x" for m in data["multipliers"]], fontsize=7)
        axis.tick_params(length=2)
        for row in range(5):
            for col in range(10):
                value = means[index, row, col]
                axis.text(col, row, f"{value:+.3f}\n±{deviations[index,row,col]:.3f}",
                          ha="center", va="center", fontsize=6,
                          color="white" if abs(value) > 0.6 * limit else "black",
                          linespacing=0.95)
    axes[-1].set_xlabel("Multiplier of selected value", fontsize=8, labelpad=3)
    bar = fig.colorbar(im, cax=fig.add_axes([0.93, 0.20, 0.018, 0.60]))
    bar.ax.tick_params(labelsize=7, length=2)
    bar.ax.set_title(r"$\Delta J$", fontsize=8, pad=5)
    path = OUTPUT / "sensitivity_stacked.svg"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    # The legacy layout contains the unchanged degradation panels A--D.
    # Remove both obsolete optimization groups and their labels, then crop.
    degradation = ET.parse(RESULTS / "Figure3.svg").getroot()
    layer = next(n for n in degradation if n.attrib.get("id") == "layer1")
    for child in list(layer):
        if child.attrib.get("id") in {"figure_1", "figure_1-9"} or (
                child.tag == tag("text") and "".join(child.itertext()).strip() in {"E", "F"}):
            layer.remove(child)
    degradation.set("height", "81mm")
    degradation.set("viewBox", "0 0 179.39903 81")
    save(degradation, "figure_3")

    optimization = ET.Element(tag("svg"), {
        "width": "468pt", "height": "513pt", "viewBox": "0 0 468 513"})
    label(optimization, "A", 0, 14)
    embed(optimization, sensitivity_panel(), "sensitivity_", 0, 18, 468)
    label(optimization, "B", 0, 274)
    embed(optimization, RESULTS / "multiple_cues/plot_multiple_cues_preprint_capacity.svg",
          "load_", 61, 270, 346)
    save(optimization, "figure_4")


if __name__ == "__main__":
    main()
