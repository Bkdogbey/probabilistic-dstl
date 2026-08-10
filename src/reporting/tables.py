"""LaTeX booktabs table generation from CSV/in-memory results (stdlib only, no pandas)."""

import csv
import os

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape_latex(text):
    text = str(text)
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in text)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_cell(value, num, *, is_percent, is_int, float_format):
    if num is None:
        return "--" if value in (None, "") else _escape_latex(value)
    if is_percent:
        return f"{100 * num:.1f}\\%"
    if is_int:
        return str(int(round(num)))
    return float_format % num


def rows_to_booktabs(
    rows,
    fieldnames,
    *,
    column_labels=None,
    percent_columns=(),
    int_columns=(),
    float_format="%.3f",
    bold_best=None,
):
    """Build a booktabs tabular body from a list of dict rows.

    rows: list[dict] with keys covering `fieldnames`.
    column_labels: optional {field: header_text}; header_text is raw LaTeX and
        is NOT escaped, so math like "$P_\\downarrow(\\varphi)$" renders as-is.
        Fields without an override fall back to an escaped, title-cased name.
    percent_columns: fields rendered as "NN.N\\%" (value assumed in [0, 1]).
    bold_best: optional {field: "max"|"min"}; bolds the winning row's cell.
    """
    column_labels = column_labels or {}
    bold_best = bold_best or {}

    best_values = {}
    for field, mode in bold_best.items():
        nums = [n for row in rows if (n := _to_float(row.get(field))) is not None]
        if nums:
            best_values[field] = max(nums) if mode == "max" else min(nums)

    align = "l" + "r" * (len(fieldnames) - 1)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]

    header = " & ".join(
        column_labels.get(f, _escape_latex(f.replace("_", " ").title())) for f in fieldnames
    )
    lines.append(header + r" \\")
    lines.append("\\midrule")

    for row in rows:
        cells = []
        for field in fieldnames:
            value = row.get(field)
            num = _to_float(value)
            cell = _format_cell(
                value, num,
                is_percent=field in percent_columns,
                is_int=field in int_columns,
                float_format=float_format,
            )
            if num is not None and num == best_values.get(field):
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def render_latex_table(rows, fieldnames, out_path, *, caption=None, label=None, **booktabs_kwargs):
    """Wrap `rows_to_booktabs(rows, fieldnames, **booktabs_kwargs)` in a `table`
    environment and write it to `out_path`. Ready to `\\input{}` into a paper
    (requires `\\usepackage{booktabs}`).
    """
    body = rows_to_booktabs(rows, fieldnames, **booktabs_kwargs)

    lines = ["\\begin{table}[t]", "\\centering"]
    if caption is not None:
        lines.append(f"\\caption{{{caption}}}")
    if label is not None:
        lines.append(f"\\label{{{label}}}")
    lines.append(body)
    lines.append("\\end{table}")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return out_path


def write_latex_table(csv_path, out_path, **kwargs):
    """Read `csv_path` and render it via `render_latex_table` to `out_path`."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    return render_latex_table(rows, fieldnames, out_path, **kwargs)
