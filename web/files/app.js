/* MenuMind front-end: instant hybrid search + streamed grounded answers. */
(() => {
  const $ = (id) => document.getElementById(id);
  const q = $("q"), searchBtn = $("search-btn"), askBtn = $("ask-btn"),
        results = $("results"), resultsHead = $("results-head"), emptyNote = $("empty-note"),
        answerPanel = $("answer-panel"), answerBody = $("answer-body"),
        answerCites = $("answer-cites"), answerMode = $("answer-mode");

  function filtersQS() {
    const p = new URLSearchParams();
    if ($("f-veg").checked) p.set("veg", "1");
    const price = parseInt($("f-price").value, 10);
    if (price > 0) p.set("max_price", String(price));
    const rating = parseFloat($("f-rating").value);
    if (rating > 0) p.set("min_rating", String(rating));
    return p;
  }

  function scoreChips(h) {
    const chips = [];
    if (h.bm25_rank > 0) chips.push(`<span class="score kw">BM25 #${h.bm25_rank}</span>`);
    if (h.vec_rank > 0) chips.push(`<span class="score vec">semantic #${h.vec_rank}${h.cosine ? " · " + h.cosine.toFixed(2) : ""}</span>`);
    if (chips.length === 0) chips.push(`<span class="score">fused</span>`);
    return chips.join("");
  }

  function renderHits(data, query) {
    answerPanel.classList.add("hidden");
    results.innerHTML = "";
    emptyNote.classList.add("hidden");
    const mode = data.semantic ? "hybrid: BM25 + semantic" : "lexical only (BM25)";
    resultsHead.innerHTML = `<span><b>${data.count}</b> dishes for “${escapeHTML(query)}”</span><span>${mode} · fused with RRF</span>`;
    if (!data.results || data.results.length === 0) {
      emptyNote.classList.remove("hidden");
      return;
    }
    for (const h of data.results) {
      const it = h.item;
      const card = document.createElement("div");
      card.className = "dish card";
      card.innerHTML = `
        <div class="dish-top">
          <span class="dish-name"><span class="${it.veg ? "veg-dot" : "nonveg-dot"}"></span>${escapeHTML(it.name)}</span>
          <span class="price">₹${it.price}</span>
        </div>
        <div class="dish-desc">${escapeHTML(it.description)}</div>
        <div class="dish-meta">
          <span>${escapeHTML(it.restaurant_name)}</span><span>·</span><span>${escapeHTML(it.area)}</span>
          <span>·</span><span>⭐ ${it.rating}</span>${it.bestseller ? "<span>·</span><span>🔥 bestseller</span>" : ""}
          <span>·</span><span>${it.spice >= 3 ? "🌶️🌶️🌶️" : it.spice === 2 ? "🌶️🌶️" : it.spice === 1 ? "🌶️" : ""}</span>
        </div>
        <div class="score-row">${scoreChips(h)}</div>`;
      results.appendChild(card);
    }
  }

  function escapeHTML(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  async function doSearch() {
    const query = q.value.trim();
    if (!query) return;
    searchBtn.disabled = true;
    resultsHead.innerHTML = "Searching…";
    try {
      const p = filtersQS(); p.set("q", query);
      const r = await fetch("/api/search?" + p.toString());
      if (!r.ok) throw new Error(await r.text());
      renderHits(await r.json(), query);
    } catch (e) {
      resultsHead.innerHTML = `<span style="color:#c23b3b">Error: ${escapeHTML(e.message).slice(0, 160)}</span>`;
    } finally { searchBtn.disabled = false; }
  }

  async function doAsk() {
    const query = q.value.trim();
    if (!query) return;
    askBtn.disabled = true; searchBtn.disabled = true;
    answerPanel.classList.remove("hidden");
    answerBody.innerHTML = '<span class="cursor"></span>';
    answerCites.innerHTML = "";
    answerMode.textContent = "· retrieving + generating…";
    try {
      const body = Object.fromEntries(filtersQS());
      body.veg_only = body.veg === "1"; delete body.veg;
      body.question = query;
      const r = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!r.ok) throw new Error(await r.text());
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "", text = "", sources = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, i); buf = buf.slice(i + 2);
          if (!frame.startsWith("data: ")) continue;
          const ev = JSON.parse(frame.slice(6));
          if (ev.type === "sources") {
            sources = ev.results || [];
            renderHits({ results: sources, semantic: true, count: sources.length }, query);
          } else if (ev.type === "delta") {
            text += ev.text;
            answerBody.innerHTML = escapeHTML(text) + '<span class="cursor"></span>';
          } else if (ev.type === "error") {
            text += "\n\n⚠ " + ev.text;
            answerBody.textContent = text;
          } else if (ev.type === "done") {
            answerBody.textContent = text;
          }
        }
      }
      answerMode.textContent = "";
      for (let i = 0; i < sources.length && i < 6; i++) {
        const s = sources[i].item;
        const c = document.createElement("span");
        c.className = "cite";
        c.textContent = `[D${i + 1}] ${s.name} · ₹${s.price} · ${s.restaurant_name}`;
        answerCites.appendChild(c);
      }
    } catch (e) {
      answerBody.textContent = "⚠ " + e.message;
    } finally { askBtn.disabled = false; searchBtn.disabled = false; }
  }

  searchBtn.onclick = doSearch;
  askBtn.onclick = doAsk;
  q.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) doSearch(); });
  document.querySelectorAll(".chip").forEach(c => c.onclick = () => { q.value = c.dataset.q; doSearch(); });

  fetch("/stats").then(r => r.json()).then(s => {
    $("index-stats").textContent = `${s.dishes} dishes · ${s.restaurants} restaurants · ${s.vectors || 0} vectors`;
  }).catch(() => {});
  fetch("/live").then(r => r.json()).then(h => {
    if (h.semantic && h.generation) $("live-badge").textContent = "● LIVE · SEMANTIC + RAG";
    else if (h.semantic) $("live-badge").textContent = "● LIVE · SEMANTIC";
    else $("live-badge").textContent = "● LIVE · LEXICAL";
  }).catch(() => {});
})();
