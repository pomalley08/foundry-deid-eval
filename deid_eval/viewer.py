"""Offline note viewer: per-note metrics and highlighted input-versus-output comparison.

Writes one self-contained HTML file. Every highlight is computed here from the grader's own
alignment and span rules, so the page shows exactly what was scored; the browser only renders,
sorts and filters.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import report
from .phi_grader import _PLACEHOLDER_PATTERN, align, occurrences, predicted_spans, score_note

_REASON_TEXT = {
    "missed": "not replaced",
    "partial": "only partly replaced",
    "repeat": "replaced here, but the same text appears elsewhere in the output",
    "failed": "the target returned no output",
}


def _runs(length: int, key_at):
    """Yield (start, end, key) for maximal runs of equal keys."""
    start = 0
    for i in range(1, length + 1):
        if i == length or key_at(i) != key_at(start):
            yield start, i, key_at(start)
            start = i


def note_view(note: dict, row: dict | None, price: dict | None = None) -> dict:
    """Metrics, issue marks and pane segments for one note and one target."""
    row = row or {}
    deid = row.get("deid_text") or ""
    error = row.get("error") or (None if row else "no result for this target")
    score = score_note({**note, "deid_text": deid})
    source = note["source_text"]
    failed = score["failed"]

    marks: list[dict] = []
    gold_at: list[int | None] = [None] * len(source)
    over_at: list[int | None] = [None] * len(source)
    edit_at = [False] * len(source)
    prot_at: list[str | None] = [None] * len(source)

    leaked = {(s["start"], s["end"]): s["reason"] for s in score["leaked"]}
    for span in note["phi_spans"]:
        reason = leaked.get((span["start"], span["end"]))
        mark_id = len(marks)
        marks.append({
            "id": mark_id,
            "kind": "leaked" if reason else "caught",
            "category": span["category"],
            "text": span["text"],
            "reason": reason,
        })
        for i in range(span["start"], span["end"]):
            gold_at[i] = mark_id

    over_ranges = {(p["start"], p["end"]) for p in score["false_positives"]}
    over_by_range: dict[tuple[int, int], int] = {}
    for pred in [] if failed else predicted_spans(source, deid):
        key = (pred["start"], pred["end"])
        if key in over_ranges:
            mark_id = len(marks)
            over_by_range[key] = mark_id
            marks.append({"id": mark_id, "kind": "over", "text": pred["text"].strip(), "label": pred["label"]})
            for i in range(*key):
                over_at[i] = mark_id
        else:
            for i in range(*key):
                edit_at[i] = True

    for value in note["protected_values"]:
        kept = not failed and bool(occurrences(value, deid))
        if not kept:
            marks.append({"id": len(marks), "kind": "lost", "text": value})
        for start, end in occurrences(value, source):
            for i in range(start, end):
                prot_at[i] = "kept" if kept else "lost"

    def source_key(i):
        gold, over = gold_at[i], over_at[i]
        return (gold, over, edit_at[i] and gold is None and over is None, prot_at[i])

    src_segments = []
    for start, end, (gold, over, edit, prot) in _runs(len(source), source_key):
        text = source[start:end]
        classes, titles, mark_id = [], [], -1
        if gold is not None:
            mark = marks[gold]
            mark_id = gold
            if mark["kind"] == "caught":
                classes.append("phi ok")
                titles.append("%s: caught" % mark["category"])
            else:
                classes.append("phi leak")
                titles.append("%s: leaked, %s" % (mark["category"], _REASON_TEXT[mark["reason"]]))
        if over is not None:
            mark = marks[over]
            mark_id = over
            classes.append("over")
            titles.append("Not PHI, but replaced" + (" with [%s]" % mark["label"] if mark["label"] else " or deleted"))
        if edit:
            classes.append("edit")
            titles.append("Not PHI; replaced together with neighboring PHI (not counted)")
        if prot:
            classes.append("prot" if prot == "kept" else "prot lost")
            titles.append("Protected clinical value %s" % prot)
        if not classes:
            src_segments.append([text])
        else:
            src_segments.append([text, " ".join(classes), "\n".join(titles), mark_id])

    out_segments = []
    if not failed:
        ph_at: list[str | None] = [None] * len(deid)
        for match in re.finditer(_PLACEHOLDER_PATTERN, deid):
            for j in range(match.start(), match.end()):
                ph_at[j] = match.group(1)
        leak_out: list[int | None] = [None] * len(deid)
        over_out: list[int | None] = [None] * len(deid)
        added = [False] * len(deid)
        for tag, i1, i2, j1, j2 in align(source, deid):
            if tag == "equal":
                for k in range(i2 - i1):
                    gold = gold_at[i1 + k]
                    if gold is not None and marks[gold]["kind"] == "leaked" and source[i1 + k].isalnum():
                        leak_out[j1 + k] = gold
            elif tag in ("replace", "insert"):
                over = over_by_range.get((i1, i2))
                for j in range(j1, j2):
                    if over is not None:
                        over_out[j] = over
                    if ph_at[j] is None and not deid[j].isspace():
                        added[j] = True
        for mark in marks:
            if mark["kind"] == "leaked":
                for start, end in occurrences(mark["text"], deid):
                    for j in range(start, end):
                        if leak_out[j] is None:
                            leak_out[j] = mark["id"]

        def out_key(j):
            return (ph_at[j], leak_out[j], over_out[j], added[j] and leak_out[j] is None)

        for start, end, (label, leak, over, add) in _runs(len(deid), out_key):
            text = deid[start:end]
            classes, titles, mark_id = [], [], -1
            if label is not None:
                classes.append("ph over" if over is not None else "ph")
                titles.append("[%s] placeholder" % label + (", replacing text that is not PHI" if over is not None else ""))
                mark_id = over if over is not None else -1
            if leak is not None:
                mark = marks[leak]
                classes.append("leak")
                titles.append("Leaked %s (%s)" % (mark["category"], mark["reason"]))
                mark_id = leak
            if add:
                classes.append("added")
                titles.append("Text that is not in the source (added or rewritten)")
            if not classes:
                out_segments.append([text])
            else:
                out_segments.append([text, " ".join(classes), "\n".join(titles), mark_id])

    cost = None
    if price and row.get("input_tokens") is not None and price.get("input_per_1m") is not None:
        cost = (row["input_tokens"] * price["input_per_1m"] + (row.get("output_tokens") or 0) * price["output_per_1m"]) / 1e6

    reasons: dict[str, int] = {}
    for span in score["leaked"]:
        reasons[span["reason"]] = reasons.get(span["reason"], 0) + 1
    return {
        "recall": score["recall"],
        "tp": score["tp"],
        "fn": score["fn"],
        "fp": score["fp"],
        "lost": sum(1 for m in marks if m["kind"] == "lost"),
        "preservation": score["clinical_preservation"],
        "reasons": reasons,
        "failed": failed,
        "error": error,
        "latency_ms": row.get("latency_ms"),
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "cost": cost,
        "marks": marks,
        "src": src_segments,
        "out": out_segments,
    }


def struggle_key(views: dict[str, dict], note_id: str):
    """Default order: most leaks across targets, then most over-redactions, then lowest preservation."""
    leaks = sum(v["fn"] for v in views.values())
    overs = sum(v["fp"] for v in views.values())
    preservation = sum(v["preservation"] for v in views.values()) / len(views) if views else 1.0
    return (-leaks, -overs, preservation, note_id)


def build_data(notes: list[dict], results_by_target: dict[str, list[dict]], pricing: dict,
               foundry_runs: dict[str, dict] | None = None) -> dict:
    foundry_runs = foundry_runs or {}
    summaries = report.rank([
        report.summarize(name, report.score_items(notes, rows), pricing) for name, rows in results_by_target.items()
    ])
    target_names = [s["target"] for s in summaries]
    rows_by_target = {name: {row["id"]: row for row in rows} for name, rows in results_by_target.items()}
    models = {s["target"]: s["model"] for s in summaries}

    note_records = []
    for note in notes:
        views = {}
        for name in target_names:
            price = report.find_price(pricing, name, models[name].split(", ")[0])
            views[name] = note_view(note, rows_by_target[name].get(note["id"]), price)
        note_records.append({
            "id": note["id"],
            "note_type": note.get("note_type", "note"),
            "gold": len(note["phi_spans"]),
            "leaks": sum(v["fn"] for v in views.values()),
            "overs": sum(v["fp"] for v in views.values()),
            "key": struggle_key(views, note["id"]),
            "results": views,
        })
    note_records.sort(key=lambda record: record["key"])
    for rank, record in enumerate(note_records):
        record["rank"] = rank
        del record["key"]

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "disclaimer": report.DISCLAIMER,
        "note_types": sorted({n["note_type"] for n in note_records}),
        "targets": [
            {
                "name": s["target"],
                "model": s["model"],
                "recall": s["recall"],
                "residual": s["residual_phi_rate"],
                "preservation": s["clinical_preservation"],
                "cost": s["cost"] if isinstance(s["cost"], str) else round(s["cost"], 4),
                "foundry_url": _https_or_none((foundry_runs.get(s["target"]) or {}).get("report_url")),
            }
            for s in summaries
        ],
        "notes": note_records,
    }


def _https_or_none(url) -> str | None:
    """Only plain https links are embedded; anything else (javascript:, file:) is dropped."""
    return url if isinstance(url, str) and url.startswith("https://") else None


def _embed_json(data: dict) -> str:
    """JSON that is safe inside a <script> block: no raw <, > or & can close or alter it."""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_html(data: dict) -> str:
    return _PAGE.replace("__DATA__", _embed_json(data))


def write_viewer(path: Path, notes: list[dict], results_by_target: dict[str, list[dict]], pricing: dict,
                 foundry_runs: dict[str, dict] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(build_data(notes, results_by_target, pricing, foundry_runs)), encoding="utf-8")
    return path


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>De-ID note viewer</title>
<style>
:root { --border:#d0d7de; --muted:#57606a; --bg:#f6f8fa; --ok:#d8f3dc; --okb:#2d6a4f; --leak:#ffd5d5; --leakb:#c1121f;
  --over:#ffe2b8; --overb:#c76b00; --prot:#2f6fdb; --lost:#8e24aa; --ph:#e7e9ee; --added:#fff3a3; }
* { box-sizing: border-box; }
body { margin:0; font:14px/1.45 "Segoe UI", system-ui, -apple-system, Arial, sans-serif; color:#1f2328;
  height:100vh; display:flex; flex-direction:column; }
header { padding:10px 16px; border-bottom:1px solid var(--border); background:#fff; }
header h1 { font-size:18px; margin:0 8px 0 0; display:inline-block; }
.disclaimer { display:inline-block; background:#fff4e5; border:1px solid #f0c27b; border-radius:4px; padding:2px 8px; font-size:12px; font-weight:600; }
.meta { color:var(--muted); font-size:12px; margin-left:8px; }
.summary { margin-top:8px; display:flex; gap:8px; flex-wrap:wrap; }
.tchip { border:1px solid var(--border); border-radius:6px; padding:4px 8px; font-size:12px; background:var(--bg); }
.tchip b { font-size:13px; }
.legend { margin-top:6px; font-size:12px; color:var(--muted); display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
main { flex:1; min-height:0; display:grid; grid-template-columns:minmax(480px, 45%) 1fr; }
#list { border-right:1px solid var(--border); display:flex; flex-direction:column; min-height:0; }
.tablewrap { flex:1; overflow:auto; }
.filters { padding:8px 10px; display:flex; gap:10px; flex-wrap:wrap; align-items:center; border-bottom:1px solid var(--border); background:var(--bg); font-size:12px; }
.filters select { font-size:12px; }
table { border-collapse:collapse; width:100%; }
th, td { border-bottom:1px solid #eaeef2; padding:4px 6px; text-align:left; font-size:12px; white-space:nowrap; }
th { position:sticky; top:0; background:#fff; cursor:pointer; user-select:none; z-index:1; }
th.sorted::after { content:" \25BE"; }
th.sorted.asc::after { content:" \25B4"; }
tr.row { cursor:pointer; }
tr.row:hover { background:#f3f6fa; }
tr.row.sel { background:#e1ecff; }
td.cell { text-align:center; min-width:64px; max-width:96px; white-space:normal; cursor:pointer; }
td.cell:hover .badge { outline:2px solid #0969da; }
td.cell.cur .badge { box-shadow:0 0 0 2px #0969da; }
.badge { display:inline-block; min-width:48px; padding:1px 6px; border-radius:10px; font-weight:600; }
.good { background:var(--ok); color:var(--okb); } .warn { background:#fff1c2; color:#8a6100; } .bad { background:var(--leak); color:var(--leakb); }
.err { background:#eee; color:#555; }
.sub { display:block; font-size:11px; color:var(--muted); }
#detail { overflow:auto; padding:12px 16px; }
.dhead { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
.dhead h2 { margin:0; font-size:18px; }
.dhead .nav { margin-left:auto; }
button { font:inherit; font-size:12px; border:1px solid var(--border); background:#fff; border-radius:6px; padding:3px 10px; cursor:pointer; }
button:hover { background:var(--bg); }
.tabs { margin:10px 0 8px; display:flex; gap:6px; flex-wrap:wrap; }
.tabs button.active { border-color:#0969da; box-shadow:0 0 0 1px #0969da inset; background:#f0f6ff; }
.flink { color:#0969da; font-size:12px; text-decoration:none; align-self:center; }
.flink:hover { text-decoration:underline; }
.tabs .flink { margin-left:6px; }
.metrics { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
.card { border:1px solid var(--border); border-radius:6px; padding:4px 9px; min-width:92px; background:#fff; }
.card .k { font-size:11px; color:var(--muted); } .card .v { font-size:16px; font-weight:600; } .card .s { font-size:11px; color:var(--muted); }
.panes { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.pane h3 { margin:0 0 4px; font-size:13px; color:var(--muted); font-weight:600; }
pre { margin:0; padding:10px; border:1px solid var(--border); border-radius:6px; background:#fff; white-space:pre-wrap; word-break:break-word;
  font:12.5px/1.6 Consolas, "Cascadia Mono", Menlo, monospace; max-height:62vh; overflow:auto; }
.errbox { padding:12px; border:1px dashed #999; border-radius:6px; background:#fafafa; color:#444; }
.phi.ok { background:var(--ok); border-bottom:2px solid var(--okb); }
.phi.leak, .leak { background:var(--leak); border-bottom:2px solid var(--leakb); font-weight:700; }
.over { background:var(--over); outline:1px dashed var(--overb); }
.edit { text-decoration:line-through; text-decoration-color:#999; color:#666; }
.prot { box-shadow:inset 0 -2px 0 var(--prot); }
.prot.lost { box-shadow:inset 0 -3px 0 var(--lost); }
.ph { background:var(--ph); border-radius:3px; padding:0 1px; color:#3d4451; }
.ph.over { background:var(--over); outline:1px dashed var(--overb); }
.added { background:var(--added); }
.flash { animation:flash 1.2s ease-out 2; }
@keyframes flash { 0% { box-shadow:0 0 0 3px #0969da; } 100% { box-shadow:0 0 0 0 transparent; } }
.issues { margin-top:12px; }
.issues h3 { font-size:13px; margin:0 0 4px; color:var(--muted); }
.issues ul { margin:0; padding-left:18px; }
.issues li { margin:2px 0; cursor:pointer; } .issues li:hover { text-decoration:underline; }
.empty { color:var(--muted); }
kbd { border:1px solid var(--border); border-bottom-width:2px; border-radius:3px; padding:0 4px; font-size:11px; background:var(--bg); }
</style>
</head>
<body>
<header>
  <h1>De-ID note viewer</h1><span class="disclaimer" id="disclaimer"></span><span class="meta" id="meta"></span>
  <div class="summary" id="summary"></div>
  <div class="legend">
    <span><span class="phi ok">caught PHI</span></span>
    <span><span class="phi leak">leaked PHI</span></span>
    <span><span class="over">over-redacted</span></span>
    <span><span class="prot">protected kept</span></span>
    <span><span class="prot lost">protected lost</span></span>
    <span><span class="ph">[PLACEHOLDER]</span></span>
    <span><span class="added">added text</span></span>
    <span><span class="edit">replaced with PHI</span></span>
    <span>Keys: <kbd>&uarr;</kbd><kbd>&darr;</kbd> note, <kbd>&larr;</kbd><kbd>&rarr;</kbd> target</span>
  </div>
</header>
<main>
  <section id="list">
    <div class="filters">
      <label>Type <select id="f-type"><option value="">all</option></select></label>
      <label>Issues for <select id="f-target"><option value="">any target</option></select></label>
      <label><input type="checkbox" id="f-issues"> issues only</label>
      <span id="count" class="meta"></span>
    </div>
    <div class="tablewrap"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>
  </section>
  <section id="detail"></section>
</main>
<script type="application/json" id="viewer-data">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("viewer-data").textContent);
const TARGETS = D.targets.map(t => t.name);
const NOTES = new Map(D.notes.map(n => [n.id, n]));
const state = { sort: "rank", asc: true, type: "", issuesFor: "", issuesOnly: false, note: null, target: TARGETS[0] };

function h(tag, props, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null) continue;
    if (k === "text") node.textContent = v;
    else if (k === "cls") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) node.append(kid);
  return node;
}
const pct = x => x == null ? "n/a" : (x * 100).toFixed(x === 1 || x === 0 ? 0 : 1) + "%";
const grade = r => r >= 1 ? "good" : r >= 0.9 ? "warn" : "bad";
const money = c => c == null ? "n/a" : typeof c === "string" ? c : "$" + c.toFixed(c < 0.01 ? 5 : 4);
const issueTargets = () => state.issuesFor ? [state.issuesFor] : TARGETS;
const hasIssue = r => r && (r.fn > 0 || r.fp > 0 || r.lost > 0 || r.error);
const shortType = t => t.split("_")[0];
const TARGET_INFO = new Map(D.targets.map(t => [t.name, t]));
const foundryLink = (name, text) => {
  const url = (TARGET_INFO.get(name) || {}).foundry_url;
  return url ? h("a", { cls: "flink", href: url, target: "_blank", rel: "noopener noreferrer",
    title: `Open the latest Foundry evaluation run for ${name}`, text, onclick: e => e.stopPropagation() }) : null;
};

function visible() {
  let list = D.notes.filter(n => !state.type || n.note_type === state.type);
  if (state.issuesOnly) list = list.filter(n => issueTargets().some(t => hasIssue(n.results[t])));
  const val = n => {
    if (state.sort === "rank") return n.rank;
    if (state.sort === "id") return n.id;
    if (state.sort === "type") return n.note_type;
    if (state.sort === "gold") return n.gold;
    const r = n.results[state.sort];
    return r ? r.recall * 1000 - r.fp - r.lost : -1;
  };
  list.sort((a, b) => { const x = val(a), y = val(b); const c = x < y ? -1 : x > y ? 1 : a.rank - b.rank; return state.asc ? c : -c; });
  return list;
}

function renderHeader() {
  document.getElementById("disclaimer").textContent = D.disclaimer;
  document.getElementById("meta").textContent = `Generated ${D.generated} · ${D.notes.length} notes`;
  const summary = document.getElementById("summary");
  for (const t of D.targets) {
    summary.append(h("span", { cls: "tchip", title: t.model },
      h("b", { text: t.name }), ` recall ${pct(t.recall)} · residual-PHI notes ${pct(t.residual)} · preservation ${pct(t.preservation)} · cost ${money(t.cost)}`,
      t.foundry_url ? " · " : null, foundryLink(t.name, "Foundry run \u2197")));
  }
  const typeSel = document.getElementById("f-type");
  for (const t of D.note_types) typeSel.append(h("option", { value: t, text: t }));
  const tgtSel = document.getElementById("f-target");
  for (const t of TARGETS) tgtSel.append(h("option", { value: t, text: t }));
  typeSel.onchange = e => { state.type = e.target.value; renderList(); };
  tgtSel.onchange = e => { state.issuesFor = e.target.value; renderList(); };
  document.getElementById("f-issues").onchange = e => { state.issuesOnly = e.target.checked; renderList(); };
}

function renderList() {
  const head = document.getElementById("thead");
  head.replaceChildren();
  const cols = [["rank", "#"], ["id", "Note"], ["type", "Type"], ["gold", "PHI"], ...TARGETS.map(t => [t, t])];
  const tr = h("tr");
  for (const [key, label] of cols) {
    const sorted = state.sort === key;
    tr.append(h("th", {
      text: label, cls: sorted ? "sorted" + (state.asc ? " asc" : "") : null,
      title: key === "rank" ? "Struggle rank: most leaks across targets first, then over-redactions, then lowest preservation" :
             TARGETS.includes(key) ? "Sort by this target's recall (worst first)" : null,
      onclick: () => { if (state.sort === key) state.asc = !state.asc; else { state.sort = key; state.asc = true; } renderList(); },
    }));
  }
  head.append(tr);
  const body = document.getElementById("tbody");
  body.replaceChildren();
  const list = visible();
  document.getElementById("count").textContent = `${list.length} of ${D.notes.length} notes`;
  for (const n of list) {
    const row = h("tr", { cls: "row" + (n.id === state.note ? " sel" : ""), onclick: () => select(n.id) },
      h("td", { text: String(n.rank + 1) }), h("td", { text: n.id }), h("td", { text: shortType(n.note_type) }), h("td", { text: String(n.gold) }));
    for (const t of TARGETS) {
      const r = n.results[t];
      const current = n.id === state.note && t === state.target;
      const td = h("td", { cls: "cell" + (current ? " cur" : ""), title: `Open ${n.id} for ${t}`,
        onclick: e => { e.stopPropagation(); state.target = t; select(n.id); } });
      if (!r || r.error) td.append(h("span", { cls: "badge err", text: "error", title: r ? r.error : "no result" }));
      else {
        td.append(h("span", { cls: "badge " + grade(r.recall), text: pct(r.recall) }));
        const bits = [];
        if (r.fn) bits.push(`${r.fn} leaked`);
        if (r.fp) bits.push(`${r.fp} over`);
        if (r.lost) bits.push(`${r.lost} lost`);
        if (bits.length) td.append(h("span", { cls: "sub", text: bits.join(" · ") }));
      }
      row.append(td);
    }
    body.append(row);
  }
}

function segs(target, segments) {
  target.replaceChildren();
  for (const s of segments) {
    if (s.length === 1) target.append(document.createTextNode(s[0]));
    else target.append(h("span", { cls: s[1], title: s[2], "data-mark": s[3] >= 0 ? String(s[3]) : null, text: s[0] }));
  }
}

function syncScroll(a, b) {
  let lock = false;
  const link = (x, y) => x.addEventListener("scroll", () => {
    if (lock) return; lock = true;
    const max = x.scrollHeight - x.clientHeight;
    y.scrollTop = max > 0 ? x.scrollTop / max * (y.scrollHeight - y.clientHeight) : 0;
    requestAnimationFrame(() => { lock = false; });
  });
  link(a, b); link(b, a);
}

function card(k, v, s, cls) {
  return h("div", { cls: "card" }, h("div", { cls: "k", text: k }), h("div", { cls: "v " + (cls || ""), text: v }), s ? h("div", { cls: "s", text: s }) : null);
}

function renderDetail() {
  const box = document.getElementById("detail");
  box.replaceChildren();
  const n = NOTES.get(state.note);
  if (!n) { box.append(h("p", { cls: "empty", text: "Select a note." })); return; }
  const list = visible();
  const idx = list.findIndex(x => x.id === n.id);
  box.append(h("div", { cls: "dhead" },
    h("h2", { text: n.id }),
    h("span", { cls: "meta", text: `${n.note_type.replace("_", " ")} · ${n.gold} gold PHI spans · struggle rank ${n.rank + 1}` }),
    h("span", { cls: "nav" },
      h("button", { text: "\u2190 Prev", onclick: () => idx > 0 && select(list[idx - 1].id) }), " ",
      h("button", { text: "Next \u2192", onclick: () => idx >= 0 && idx < list.length - 1 && select(list[idx + 1].id) }))));

  const tabs = h("div", { cls: "tabs" });
  for (const t of TARGETS) {
    const r = n.results[t];
    tabs.append(h("button", { cls: t === state.target ? "active" : null, onclick: () => setTarget(t) },
      t + " ", h("span", { cls: "badge " + (r && !r.error ? grade(r.recall) : "err"), text: r && !r.error ? pct(r.recall) : "error" })));
  }
  const link = foundryLink(state.target, `Open ${state.target} run in Foundry \u2197`);
  if (link) tabs.append(link);
  box.append(tabs);

  const r = n.results[state.target];
  const reasons = Object.entries(r.reasons).map(([k, v]) => `${v} ${k}`).join(", ");
  box.append(h("div", { cls: "metrics" },
    card("PHI recall", pct(r.recall), `${r.tp} of ${r.tp + r.fn} spans caught`, grade(r.recall)),
    card("Leaked", String(r.fn), reasons || "none"),
    card("Over-redacted", String(r.fp), "non-PHI replaced"),
    card("Clinical preservation", pct(r.preservation), r.lost ? `${r.lost} value(s) lost` : "all kept"),
    card("Latency", r.latency_ms == null ? "n/a" : Math.round(r.latency_ms) + " ms"),
    card("Tokens", r.input_tokens == null ? "n/a" : `${r.input_tokens} / ${r.output_tokens}`, "input / output"),
    card("Est. cost", money(r.cost))));

  const src = h("pre"), out = h("pre");
  segs(src, r.src);
  const panes = h("div", { cls: "panes" },
    h("div", { cls: "pane" }, h("h3", { text: "Input: source note with gold PHI" }), src),
    h("div", { cls: "pane" }, h("h3", { text: `Output: ${state.target}` }), r.error ? h("div", { cls: "errbox", text: "Error: " + r.error }) : out));
  if (!r.error) { segs(out, r.out); syncScroll(src, out); }
  box.append(panes);

  const issues = r.marks.filter(m => m.kind !== "caught");
  const ul = h("ul");
  const describe = m => m.kind === "leaked" ? `Leaked ${m.category} "${m.text}": ${m.reason}` :
    m.kind === "over" ? `Over-redacted "${m.text}"` + (m.label ? ` as [${m.label}]` : " (deleted)") : `Lost protected value "${m.text}"`;
  for (const m of issues) ul.append(h("li", { text: describe(m), onclick: () => flash(src, out, m) }));
  box.append(h("div", { cls: "issues" }, h("h3", { text: issues.length ? `Issues (${issues.length}); click to locate` : "Issues" }),
    issues.length ? ul : h("p", { cls: "empty", text: "No leaks, over-redactions or lost values for this target." })));
}

function flash(src, out, m) {
  let nodes = [...src.querySelectorAll(`[data-mark="${m.id}"]`), ...out.querySelectorAll(`[data-mark="${m.id}"]`)];
  if (m.kind === "lost") nodes = [...src.querySelectorAll(".prot.lost")].filter(x => x.textContent.includes(m.text));
  if (!nodes.length) return;
  nodes[0].scrollIntoView({ block: "center", behavior: "smooth" });
  for (const node of nodes) { node.classList.remove("flash"); void node.offsetWidth; node.classList.add("flash"); }
}

function select(id) {
  state.note = id;
  renderList();
  renderDetail();
  writeHash();
  const row = document.querySelector("tr.row.sel");
  if (row) row.scrollIntoView({ block: "nearest" });
}
function setTarget(t) { state.target = t; renderList(); renderDetail(); writeHash(); }
function writeHash() { history.replaceState(null, "", `#${state.note}/${encodeURIComponent(state.target)}`); }

document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  const list = visible();
  const idx = list.findIndex(x => x.id === state.note);
  if (e.key === "ArrowDown" || e.key === "j") { if (idx < list.length - 1) select(list[idx + 1].id); e.preventDefault(); }
  else if (e.key === "ArrowUp" || e.key === "k") { if (idx > 0) select(list[idx - 1].id); e.preventDefault(); }
  else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const t = TARGETS.indexOf(state.target) + (e.key === "ArrowRight" ? 1 : -1);
    if (t >= 0 && t < TARGETS.length) setTarget(TARGETS[t]);
    e.preventDefault();
  }
});

renderHeader();
const [hashNote, hashTarget] = decodeURIComponent(location.hash.slice(1)).split("/");
const first = NOTES.get(hashNote) || visible()[0];
if (hashTarget && TARGETS.includes(hashTarget)) state.target = hashTarget;
else {
  // Open on the target that struggled most with the first note, so the first view shows an issue.
  const badness = t => { const r = first.results[t]; return r ? r.fn * 100 + r.fp * 10 + r.lost : 0; };
  state.target = TARGETS.reduce((a, b) => badness(b) > badness(a) ? b : a, TARGETS[0]);
}
select(first.id);
</script>
</body>
</html>
"""
