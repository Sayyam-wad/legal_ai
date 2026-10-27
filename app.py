"""Web interface for the legal assistant.

Run:  python app.py            then open http://127.0.0.1:5000
API:  POST /api/ask  {"query": "...", "top_k": 5}  -> JSON response
      GET  /api/health
"""

import os
import threading

from flask import Flask, jsonify, render_template_string, request

import config

app = Flask(__name__)
_assistant = None
_lock = threading.Lock()


def get_assistant():
    """Load models once, on first request (keeps `flask run` startup fast)."""
    global _assistant
    with _lock:
        if _assistant is None:
            from assistant import LegalAssistant
            use_rerank = os.getenv("LEGAL_AI_NO_RERANK") != "1"
            _assistant = LegalAssistant(cross_encoder=config.CROSS_ENCODER if use_rerank else None)
    return _assistant


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "loaded": _assistant is not None})


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    if len(query) < 3:
        return jsonify({"error": "Please enter a query of at least 3 characters."}), 400
    if len(query) > 5000:
        return jsonify({"error": "Query is too long (max 5000 characters)."}), 400
    try:
        top_k = max(1, min(int(data.get("top_k", 5)), 20))
    except (TypeError, ValueError):
        top_k = 5
    try:
        want_answer = bool(data.get("answer", os.getenv("LEGAL_AI_ANSWER", "1") != "0"))
        resp = get_assistant().ask(query, top_k=top_k, answer=want_answer)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    payload = resp.to_dict()
    for c in payload["cases"]:
        c["passage"] = c["passage"][:1500]
    return jsonify(payload)


@app.get("/")
def index():
    return render_template_string(PAGE)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Legal AI · Indian Case Law</title>
<style>
  :root { --bg:#f7f5f0; --card:#fff; --ink:#1d1d1f; --muted:#6b6b70; --line:#e4e0d6;
          --accent:#7a2e1f; --ok:#2f6b3a; --bad:#9b2c2c; --mid:#8a6d1d; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16161a; --card:#202026; --ink:#ececf0; --muted:#a0a0aa; --line:#33333b;
            --accent:#e0937f; --ok:#7cc48a; --bad:#ec8a8a; --mid:#e0c26d; } }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.55 Georgia, "Iowan Old Style", serif; }
  main { max-width:860px; margin:0 auto; padding:32px 16px 64px }
  h1 { font-size:28px; margin:0 0 4px } .sub { color:var(--muted); margin:0 0 24px; font-family:system-ui,sans-serif; font-size:14px }
  form { display:flex; flex-direction:column; gap:10px }
  textarea { width:100%; min-height:96px; padding:12px; font:inherit; border:1px solid var(--line);
             border-radius:8px; background:var(--card); color:var(--ink); resize:vertical }
  .row { display:flex; gap:12px; align-items:center; font-family:system-ui,sans-serif; font-size:14px; flex-wrap:wrap }
  button { background:var(--accent); color:#fff; border:0; padding:10px 20px; border-radius:8px;
           font:600 14px system-ui,sans-serif; cursor:pointer } button:disabled { opacity:.6; cursor:wait }
  .examples button { background:transparent; color:var(--accent); border:1px solid var(--line); padding:6px 10px; font-weight:500 }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin:14px 0 }
  .card h3 { margin:0 0 6px; font-size:18px }
  .meta { font:13px system-ui,sans-serif; color:var(--muted); display:flex; gap:10px; flex-wrap:wrap; margin-bottom:8px }
  .pill { padding:1px 8px; border-radius:99px; border:1px solid var(--line) }
  .Allowed,.Partly { color:var(--ok) } .Dismissed { color:var(--bad) } .Disposed,.Unclear { color:var(--mid) }
  ul { padding-left:20px; margin:6px 0 } li { margin:4px 0 }
  .tag { font:11px system-ui,sans-serif; color:var(--muted) }
  .bar { height:4px; background:var(--line); border-radius:4px; overflow:hidden; width:90px; display:inline-block; vertical-align:middle }
  .bar span { display:block; height:100%; background:var(--accent) }
  details summary { cursor:pointer; font:13px system-ui,sans-serif; color:var(--accent) }
  details p { font-size:14px; color:var(--muted) }
  .section { font:600 12px system-ui,sans-serif; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin:28px 0 4px }
  .note { font:13px system-ui,sans-serif; color:var(--muted); border-left:3px solid var(--line); padding-left:10px }
  .err { color:var(--bad); font-family:system-ui,sans-serif }
  .answer { border-left:4px solid var(--accent) } .answer a { color:var(--accent); text-decoration:none; font:600 13px system-ui,sans-serif }
</style>
</head>
<body>
<main>
  <h1>Legal AI</h1>
  <p class="sub">Search 2,914 Indian Supreme Court judgments by situation or legal question.</p>
  <form id="f">
    <textarea id="q" placeholder="Describe the facts or ask a legal question, e.g. 'Is a dying declaration alone enough to convict for murder?'"></textarea>
    <div class="row">
      <button id="go" type="submit">Search</button>
      <label>Results <select id="k"><option>3</option><option selected>5</option><option>10</option></select></label>
    </div>
    <div class="row examples">
      <button type="button">anticipatory bail in a dowry harassment case</button>
      <button type="button">dying declaration as sole basis for conviction</button>
      <button type="button">dismissal from service without departmental inquiry</button>
    </div>
  </form>
  <div id="out"></div>
</main>
<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
document.querySelectorAll(".examples button").forEach(b => b.onclick = () => { $("#q").value = b.textContent; $("#f").requestSubmit(); });
$("#f").onsubmit = async e => {
  e.preventDefault();
  const query = $("#q").value.trim(); if (!query) return;
  $("#go").disabled = true; $("#go").textContent = "Searching…";
  $("#out").innerHTML = '<p class="note">First search loads the models and can take a minute.</p>';
  try {
    const r = await fetch("/api/ask", {method:"POST", headers:{"Content-Type":"application/json"},
                                       body: JSON.stringify({query, top_k: +$("#k").value})});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.statusText);
    render(d);
  } catch (err) { $("#out").innerHTML = `<p class="err">${esc(err.message)}</p>`; }
  finally { $("#go").disabled = false; $("#go").textContent = "Search"; }
};
function render(d) {
  let h = "";
  if (d.answer) {
    const linked = esc(d.answer.text).replace(/\[((?:C|S)\d+)\]/g, '<a href="#$1">[$1]</a>');
    h += `<p class="section">Answer · ${esc(d.answer.provider)}</p><div class="card answer"><p>${linked}</p>
      ${d.answer.removed_citations.length ? `<p class="note">Removed citations to sources that were not retrieved: ${esc(d.answer.removed_citations.join(", "))}</p>` : ""}</div>`;
  }
  h += `<p class="section">Judgments · intent: ${esc(d.intent)}</p>`;
  if (!d.cases.length) h += '<p class="note">No matching judgments.</p>';
  d.cases.forEach((c, i) => {
    h += `<div class="card" id="${esc(c.case_id)}"><h3>${i+1}. ${esc(c.title)}${c.year ? ` (${esc(c.year)})` : ""}</h3>
      <div class="meta"><span>${esc(c.case_id)}</span>
        <span>relevance <span class="bar"><span style="width:${Math.round(c.relevance*100)}%"></span></span> ${Math.round(c.relevance*100)}%</span>
        <span class="pill ${esc(c.outcome.split(" ")[0])}">${esc(c.outcome)}</span>
        <span>appellant: ${esc(c.appellant)}</span></div>
      ${c.citations.length ? `<div class="meta">${c.citations.map(x => `<span class="pill">${esc(x)}</span>`).join("")}</div>` : ""}
      <ul>${c.key_passages.map(p => `<li>${esc(p.text)} ${p.roles.length ? `<span class="tag">[${esc(p.roles.join(", "))}]</span>` : ""}</li>`).join("")}</ul>
      <details><summary>Matched passage</summary><p>${esc(c.passage)}</p></details></div>`;
  });
  if (d.statutes.length) {
    h += '<p class="section">Statutes</p>';
    d.statutes.forEach(s => h += `<div class="card" id="${esc(s.statute_id)}"><h3>${esc(s.title)}</h3><div class="meta"><span>${esc(s.statute_id)}</span><span>relevance ${Math.round(s.relevance*100)}%</span></div><p>${esc(s.text)}</p></div>`);
  }
  const o = d.outcome_summary;
  h += `<p class="section">Outcomes in similar cases</p><div class="card">
    <div class="meta">${Object.entries(o.counts).map(([k,v]) => `<span class="pill ${esc(k.split(" ")[0])}">${esc(k)}: ${v}</span>`).join("")}</div>
    <p>${esc(o.trend)}</p><p class="note">${esc(o.caveat)}</p></div>`;
  d.notes.forEach(n => h += `<p class="note">${esc(n)}</p>`);
  h += `<p class="note">${esc(d.disclaimer)}</p>`;
  $("#out").innerHTML = h;
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)
