// Ground Control dashboard. Renders with zero events (empty states) and
// rebuilds the current job from get_state after any (re)connect.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const socket = io();

  const S = { jobId: null, tokens: [], text: "", prompt: null, plan: null, steps: {}, history: [],
              status: {}, frame: new Array(104).fill(0), repair: null };

  // ------------------------------------------------------------ themes
  const THEMES = { mission: "Mission (dark)", hud: "HUD", geocities: "Web 1.0", analog: "Chart recorder", brutalist: "Brutalist" };
  const themeSels = [$("theme"), $("theme-m")];
  themeSels.forEach((sel) => {
    sel.innerHTML = Object.entries(THEMES).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
  });
  function setTheme(t) {
    if (!THEMES[t]) t = "mission";
    document.documentElement.dataset.theme = t;
    themeSels.forEach((sel) => { sel.value = t; });
    try { localStorage.setItem("gc-theme", t); } catch (e) { /* private mode */ }
    setTimeout(() => { Draw.gantt($("gantt"), S.history); Draw.entropy($("entropy"), S.tokens); }, 50);
  }
  let saved = null;
  try { saved = localStorage.getItem("gc-theme"); } catch (e) { /* ignore */ }
  setTheme(new URLSearchParams(location.search).get("theme") || saved || "mission");
  themeSels.forEach((sel) => sel.addEventListener("change", (e) => setTheme(e.target.value)));

  // ------------------------------------------------------------ layout: desktop (fit to screen) / mobile
  const LAYOUTS = ["desktop", "mobile"];
  function setLayout(l, remember = true) {
    if (!LAYOUTS.includes(l)) l = "desktop";
    const root = document.documentElement;
    LAYOUTS.forEach((x) => root.classList.toggle(`layout-${x}`, x === l));
    document.querySelectorAll("#layout-bar [data-layout]").forEach((b) => b.classList.toggle("on", b.dataset.layout === l));
    if (remember) { try { localStorage.setItem("gc-layout", l); } catch (e) { /* ignore */ } }
    setTimeout(() => { Draw.gantt($("gantt"), S.history); Draw.entropy($("entropy"), S.tokens); }, 60);
  }
  let savedLayout = null;
  try { savedLayout = localStorage.getItem("gc-layout"); } catch (e) { /* ignore */ }
  const urlLayout = new URLSearchParams(location.search).get("layout");
  setLayout(urlLayout || savedLayout || (window.innerWidth < 760 ? "mobile" : "desktop"), Boolean(urlLayout || savedLayout));
  document.querySelectorAll("#layout-bar [data-layout]").forEach((b) =>
    b.addEventListener("click", () => setLayout(b.dataset.layout)));

  const SEGS = ["system", "catalog", "rules", "examples", "command", "template"];
  $("seg-legend").innerHTML = SEGS.map((k) => `<span><i style="background:var(--seg-${k})"></i>${k}</span>`).join("") +
    `<span><i style="background:#1a2336;opacity:.4"></i>dimmed = reused from KV cache</span>`;
  $("mode-legend").innerHTML = Duet.MODES.map((m) => `<span><i style="background:var(--m-${m})"></i>${m}</span>`).join("");

  const QUICK = [
    ["Assignment example", "Type Hi on the LED matrix and then send back on MQTT on topic luca message 'I am done'"],
    ["Info request", "what is your yaw? reply on luca"],
    ["Duet", "do a 4 bar dance at 100 bpm, the lego bot mirrors you, then tell me 'bow complete' on luca"],
    ["Call & response", "call and response dance at 90 bpm, the car spins and the lego claps, then tell me how in sync you were on luca"],
    ["Question", "what is 2+2?"],
    ["Trap", "don't turn left, turn right 90 degrees"],
    ["/status", "/status"],
  ];
  $("quick").innerHTML = QUICK.map(([l], i) => `<button type="button" data-i="${i}">${l}</button>`).join("");
  $("quick").addEventListener("click", (e) => {
    const i = e.target.dataset.i;
    if (i !== undefined) send(QUICK[i][1]);
  });

  function send(text, source = "ui") {
    socket.emit("send_command", { text, source });
  }

  // ------------------------------------------------------------ microphone (browser speech recognition)
  // Needs a secure page (https or localhost); the board serves https for this reason.
  (() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const btn = $("mic"), note = $("mic-note"), input = $("send-text");
    if (!SR) { btn.disabled = true; btn.title = "Speech recognition isn't available in this browser (use Chrome, Edge or Safari)"; return; }
    if (!window.isSecureContext) {
      btn.disabled = true;
      note.textContent = `The mic needs a secure page: open https://${location.hostname}:${location.port || 7000} (accept the certificate warning once).`;
      return;
    }
    const rec = new SR();
    rec.lang = "en-US"; rec.interimResults = true; rec.continuous = false;
    let listening = false, finalText = "";
    rec.onstart = () => { listening = true; finalText = ""; btn.classList.add("on"); note.textContent = "Listening… speak your command"; };
    rec.onresult = (e) => {
      let interim = ""; finalText = "";
      for (const r of e.results) (r.isFinal ? (finalText += r[0].transcript) : (interim += r[0].transcript));
      input.value = (finalText + interim).trim();
    };
    rec.onerror = (e) => { note.textContent = e.error === "not-allowed" ? "Microphone permission was denied." : `Speech error: ${e.error}`; };
    rec.onend = () => {
      listening = false; btn.classList.remove("on");
      const t = finalText.trim();
      if (t) { send(t, "voice"); input.value = ""; note.textContent = `Sent: “${t}”`; }
      else if (!note.textContent.startsWith("Speech error") && !note.textContent.startsWith("Microphone")) note.textContent = "";
    };
    btn.addEventListener("click", () => { if (listening) rec.stop(); else { note.textContent = ""; rec.start(); } });
  })();
  $("send-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const t = $("send-text").value.trim();
    if (t) { send(t); $("send-text").value = ""; }
  });
  $("estop").addEventListener("click", () => socket.emit("estop", {}));
  $("model").addEventListener("change", (e) => socket.emit("set_model", { model: e.target.value }));
  $("think").addEventListener("change", (e) => socket.emit("set_think", { chars: +e.target.value }));
  $("explain-btn").addEventListener("click", () => $("explain").classList.remove("hidden"));
  $("explain-close").addEventListener("click", () => $("explain").classList.add("hidden"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("explain").classList.add("hidden");
    if (e.key === " " && e.target.tagName !== "INPUT") { e.preventDefault(); socket.emit("estop", {}); }
  });

  // ------------------------------------------------------------ status
  function renderStatus(s) {
    S.status = s;
    const st = s.state || "IDLE";
    $("state").className = `pill ${st}`;
    $("state").textContent = st;
    $("doing").textContent = s.doing || "";
    $("queue").textContent = `queue ${s.queue ?? 0}`;
    $("toks").textContent = (s.llm_tok_s ? `${s.llm_tok_s} tok/s` : "– tok/s") + (s.llm_cache ? ` · cache: ${s.llm_cache}` : "");
    $("dot-mqtt").className = "dot" + (s.mqtt ? " on" : "");
    $("dot-llm").className = "dot" + (s.llm_ready ? " on" : "");
    const lg = s.lego || {};
    $("dot-lego").className = "dot" + (lg.connected ? " on" : " sim");
    $("dot-lego").title = lg.connected ? `LEGO via BLE, rtt ${lg.rtt_ms} ms` : `LEGO simulated (${lg.error || "not connected"})`;
    $("perf-state").textContent = lg.connected ? `LEGO: BLE ${lg.rtt_ms} ms rtt` : "LEGO: simulated";
    if (s.model && $("model").value !== s.model) $("model").value = s.model;
  }

  // ------------------------------------------------------------ input lane
  function newJob(j) {
    S.jobId = j.id; S.tokens = []; S.text = ""; S.prompt = null; S.plan = null; S.steps = {}; S.repair = null; S.guard = [];
    $("cmd-src").className = `badge ${j.source}`;
    $("cmd-src").textContent = (j.source || "").toUpperCase();
    $("cmd-id").textContent = j.id + (j.reply_to ? ` → ${j.reply_to}` : "");
    $("cmd-text").classList.remove("muted");
    $("cmd-text").textContent = j.text;
    $("cmd-reply").classList.add("hidden");
    $("chips").innerHTML = `<div class="empty">tokenizing…</div>`;
    $("prompt-stats").textContent = "";
    $("stream").innerHTML = "";
    $("think-bubble").innerHTML = `<span class="muted">reading the prompt (prefill)…</span>`;
    $("decisions").innerHTML = `<div class="empty">waiting for the first decision…</div>`;
    $("plan").innerHTML = `<div class="empty">planning…</div>`;
    $("plan-note").textContent = "";
    $("gen-stats").textContent = "";
    Draw.entropy($("entropy"), []);
  }

  function renderPrompt(p) {
    S.prompt = p;
    const nc = p.n_cached_predicted || 0;
    $("prompt-stats").textContent = `${p.n_tokens} tokens · ${nc} reused from KV cache · ${p.n_tokens - nc} to read`;
    const html = p.tokens.map((t, i) => {
      const piece = t.piece.replace(/\n/g, "↵");
      return `<span class="chip k-${t.kind}${i < nc ? " cached" : ""}" title="id ${t.id} · ${t.kind}">${esc(piece)}</span>`;
    }).join("");
    $("chips").innerHTML = html;
    $("chips").scrollTop = $("chips").scrollHeight;
  }

  // ------------------------------------------------------------ thinking lane
  // Confidence: blend the theme's --conf-lo -> --conf-hi (no fixed red/green).
  const hex = (c) => { c = c.replace("#", ""); if (c.length === 3) c = [...c].map((x) => x + x).join("");
    return [0, 2, 4].map((i) => parseInt(c.slice(i, i + 2), 16)); };
  function confColor(p) {
    if (p == null) return Draw.css("--conf-none");
    const a = hex(Draw.css("--conf-lo")), b = hex(Draw.css("--conf-hi"));
    const m = a.map((v, i) => Math.round(v + (b[i] - v) * p));
    return `rgba(${m.join(",")}, ${Draw.css("--conf-alpha")})`;
  }
  const BAR = [1, 2, 3, 4, 5].map((i) => `var(--bar${i})`);

  function addToken(t) {
    if (t.id !== S.jobId) return;
    if (!S.tokens.length) $("stream").innerHTML = "";
    S.tokens.push(t);
    S.text += t.piece;
    const el = document.createElement("span");
    el.className = "tok new" + (t.decision ? " decision" : "") + (t.forced ? " forced" : "");
    const bar = (t.top || []).map((a, k) => `<span style="width:${(a.p * 100).toFixed(1)}%;background:${BAR[k]}"></span>`).join("");
    el.innerHTML = `<span class="t" style="background:${confColor(t.p)}">${esc(t.piece)}</span><span class="bar">${bar}</span>`;
    el.dataset.i = S.tokens.length - 1;
    $("stream").appendChild(el);
    setTimeout(() => el.classList.remove("new"), 400);
    $("stream").scrollTop = $("stream").scrollHeight;
    const tm = S.text.match(/"think"\s*:\s*"((?:[^"\\]|\\.)*)/);
    if (tm) $("think-bubble").textContent = tm[1] + (S.text.includes(`${tm[1]}"`) ? "" : "…");
    const pm = S.text.match(/"parts"\s*:\s*\[([^\]]*)/);
    if (pm) {
      const parts = [...pm[1].matchAll(/"((?:[^"\\]|\\.)*)("?)/g)].map((x) => x[1] + (x[2] ? "" : "…"));
      $("think-bubble").innerHTML = parts.length
        ? parts.map((p, k) => `<span class="part"><b>${k + 1}</b> ${esc(p)}</span>`).join("")
        : `<span class="muted">splitting the command into parts…</span>`;
    }
    if (t.decision) renderDecisions();
    Draw.entropy($("entropy"), S.tokens);
    $("gen-stats").textContent = `${S.tokens.length} tokens · ${(t.t_ms / 1000).toFixed(1)} s`;
  }

  function renderDecisions() {
    const decs = S.tokens.filter((t) => t.decision && t.choices && t.choices.length);
    if (!decs.length) return;
    $("decisions").innerHTML = decs.slice(-6).map((d) => {
      const chosenFrag = d.piece.replace(/^[\s"]+/, "");
      const opts = d.choices.slice(0, 4).map((c) => {
        const chosen = c.choice.split("|").some((o) => o.startsWith(chosenFrag)) || c.choice === `(${chosenFrag})`;
        return `<div class="opt${chosen ? " chosen" : ""}"><div class="lbl"><div class="fill" style="width:${(c.p * 100).toFixed(1)}%"></div>` +
          `<span>${esc(c.choice.length > 46 ? c.choice.slice(0, 44) + "…" : c.choice)}</span></div><div class="p">${(c.p * 100).toFixed(1)}%</div></div>`;
      }).join("");
      return `<div class="dec"><div class="key">${esc(d.key)}</div><div class="opts">${opts}</div></div>`;
    }).join("");
  }

  // popover with top-5
  $("stream").addEventListener("mousemove", (e) => {
    const el = e.target.closest(".tok");
    const pop = $("popover");
    if (!el) { pop.classList.add("hidden"); return; }
    const t = S.tokens[+el.dataset.i];
    if (!t || !t.top) return;
    pop.innerHTML = `<div class="muted">token #${t.i}${t.key ? ` · ${esc(t.key)}` : ""}${t.forced ? " · ⚑ grammar override" : ""}</div>` +
      t.top.map((a) => `<div class="r${a.piece === t.piece ? " chosen" : ""}"><span>${esc(JSON.stringify(a.piece))}</span><span>${(a.p * 100).toFixed(1)}%</span></div>`).join("") +
      (t.entropy != null ? `<div class="muted">entropy ${t.entropy} bits</div>` : "");
    pop.classList.remove("hidden");
    // keep it on screen (an expanded panel fills the window, so the cursor is often near an edge)
    const w = pop.offsetWidth, h = pop.offsetHeight;
    pop.style.left = `${Math.min(e.clientX + 14, window.innerWidth - w - 8)}px`;
    pop.style.top = `${e.clientY + 14 + h > window.innerHeight ? e.clientY - h - 10 : e.clientY + 14}px`;
  });
  $("stream").addEventListener("mouseleave", () => $("popover").classList.add("hidden"));

  function llmDone(d) {
    if (d.id !== S.jobId) return;
    if (!d.plan) {
      $("plan").innerHTML = `<div class="step fail"><div class="i">!</div><div class="name">no plan</div><div class="st">FAIL</div><div class="res">${esc(d.error || "")}</div></div>`;
      return;
    }
    S.plan = d.plan;
    const t = d.timings || {};
    if (String(d.raw_text || "").startsWith("(fast path")) { S.plan = d.plan; renderPlan(); return; }
    $("gen-stats").textContent = `${t.predicted_n} tokens · ${t.tok_s} tok/s · prefill ${(t.prefill_ms / 1000).toFixed(1)} s (${t.prompt_n} new, ${t.cache_n} cached) · decode ${(t.decode_ms / 1000).toFixed(1)} s`;
    const parts = d.plan.parts || [];
    if (d.plan.think) $("think-bubble").textContent = d.plan.think;
    else $("think-bubble").innerHTML = parts.length
      ? parts.map((p, k) => `<span class="part"><b>${k + 1}</b> ${esc(p)}</span>`).join("")
      : `<span class="muted">${d.raw_text === "(hand-written plan)" ? "hand-written plan (no LLM)" : "no parts: the model found nothing it can do"}</span>`;
    $("plan-note").textContent = d.plan.repaired ? "repaired: one step per part was enforced" : "";
    renderPlan();
  }

  function renderPlan() {
    const steps = (S.plan && S.plan.steps) || [];
    if (!steps.length) {
      $("plan").innerHTML = `<div class="empty">No steps: the model declined. Reply: "Sorry, I can't do that".</div>`;
      return;
    }
    $("plan").innerHTML = steps.map((s, i) => {
      const st = S.steps[i] || {};
      const state = st.state || "pending";
      const args = st.args || s.args || {};
      let res = "";
      if (state === "ok" && st.result) res = `→ ${esc(JSON.stringify(st.result)).slice(0, 160)}`;
      if (state === "fail") res = `✗ ${esc(st.error || "")}`;
      return `<div class="step ${state}"><div class="i">${i + 1}</div><div><div class="name">${esc(s.skill)}</div>` +
        `<div class="args">${esc(JSON.stringify(args))}</div></div><div class="st">${state.toUpperCase()}${st.t_ms != null ? ` ${st.t_ms}ms` : ""}</div>` +
        (res ? `<div class="res">${res}</div>` : "") + `</div>`;
    }).join("");
  }

  // ------------------------------------------------------------ action lane
  function addMqtt(m) {
    const log = $("mqtt-log");
    if (log.querySelector(".empty")) log.innerHTML = "";
    const div = document.createElement("div");
    div.className = "m";
    const time = new Date((m.t || Date.now() / 1000) * 1000).toLocaleTimeString();
    div.innerHTML = `<span class="muted">${time}</span> <b>${esc(m.topic)}</b> ${esc(m.payload).slice(0, 200)}`;
    log.prepend(div);
    while (log.children.length > 30) log.lastChild.remove();
  }

  // ------------------------------------------------------------ socket
  socket.on("connect", () => socket.emit("get_state", {}));
  socket.on("disconnect", () => { $("state").className = "pill ERROR"; $("state").textContent = "OFFLINE"; });
  socket.on("get_state_response", (st) => {
    $("model").innerHTML = (st.models || []).map((m) => `<option>${esc(m)}</option>`).join("");
    if (st.think_chars != null) $("think").value = String(st.think_chars);
    renderStatus(st.status || {});
    S.frame = st.matrix || S.frame;
    S.history = st.history || [];
    (st.mqtt_log || []).forEach(addMqtt);
    const j = st.job || {};
    if (j.job) {
      newJob(j.job);
      if (j.prompt) renderPrompt(j.prompt);
      (j.tokens || []).forEach(addToken);
      if (j.done) llmDone(j.done);
      if (j.result && j.result.reply) {
        $("cmd-reply").innerHTML = `<b>REPLY</b>${esc(j.result.reply)}`;
        $("cmd-reply").classList.remove("hidden");
      }
      S.steps = Object.fromEntries(Object.entries(j.steps || {}).map(([k, v]) => [+k, v]));
      if (S.plan) renderPlan();
      if (j.dance) { Duet.start({ ...j.dance, server_now: st.server_now }); (j.dance_kf || []).forEach(Duet.addKf); }
      if (j.dance_done) Duet.finish(j.dance_done);
    }
  });
  socket.on("status", renderStatus);
  socket.on("job.queued", (j) => { if (S.status.state !== "IDLE") $("doing").textContent = `queued: ${j.text}`; });
  socket.on("job.new", newJob);
  socket.on("llm.prompt", (p) => { if (p.id === S.jobId) renderPrompt(p); });
  socket.on("llm.token", addToken);
  socket.on("llm.repair", (r) => {
    if (r.id !== S.jobId) return;
    const div = document.createElement("div");
    div.className = "muted";
    div.style.width = "100%";
    div.textContent = r.style === "parts"
      ? `↻ ${r.declared.length} parts but ${r.got.length} steps [${r.got.join(", ")}]: second pass forces one step per part`
      : `↻ think named [${r.declared.join(", ")}] but steps were [${r.got.join(", ")}]: second pass forces the steps to match`;
    $("stream").appendChild(div);
  });
  socket.on("llm.done", llmDone);
  // Fast path (Part 2, A3): the router's softmax over intents, its threshold, and its decision.
  socket.on("router", (d) => {
    if (d.id !== S.jobId) return;
    const tau = d.tau || 0;
    const rows = Object.entries(d.probs || {}).map(([k, p]) => {
      const chosen = k === d.intent;
      return `<div class="opt${chosen ? " chosen" : ""}"><div class="lbl"><div class="fill" style="width:${(p * 100).toFixed(1)}%"></div>` +
        `<div class="tau" style="left:${(tau * 100).toFixed(1)}%" title="threshold τ = ${tau}"></div>` +
        `<span>${esc(k)}</span></div><div class="p">${(p * 100).toFixed(1)}%</div></div>`;
    }).join("");
    const step = d.fast && d.plan ? d.plan.steps[0] : null;
    const verdict = d.fast
      ? `FAST PATH ${d.p.toFixed(3)} → ${esc(step.skill)}(${esc(Object.values(step.args).join(", "))}) · ${d.ms} ms`
      : `→ LLM (${esc(d.reason || "")})`;
    $("decisions").innerHTML = `<div class="dec router"><div class="key">router</div><div class="opts">${rows}` +
      `<div class="verdict ${d.fast ? "fast" : "llm"}">${verdict}</div></div></div>`;
    if (d.fast) {
      $("think-bubble").innerHTML = `<span class="muted">fast path: no tokens generated</span>`;
      $("stream").innerHTML = `<div class="empty">no tokens generated: a 134k-parameter network routed this in ${d.ms} ms</div>`;
      $("gen-stats").textContent = `fast path · ${d.ms} ms · 0 tokens`;
    }
  });
  socket.on("llm.guard", (g) => {
    if (g.id !== S.jobId) return;
    S.guard = g.removed;
    for (const r of g.removed) {
      const div = document.createElement("div");
      div.className = "guard-note";
      div.innerHTML = `✂ removed from the reply: <s>${esc(r.sentence)}</s> <span class="muted">(${esc(r.numbers.join(", "))} isn't in the source facts)</span>`;
      $("stream").appendChild(div);
    }
  });
  socket.on("llm.compose", (c) => {
    if (c.id !== S.jobId) return;
    const div = document.createElement("div");
    div.className = "muted compose-note";
    div.style.width = "100%";
    div.textContent = `↳ ${c.label || "composing from facts"}: ${c.facts.slice(0, 160)}${c.facts.length > 160 ? "…" : ""}`;
    $("stream").appendChild(div);
  });
  socket.on("plan.grounded", (g) => {
    if (g.id !== S.jobId) return;
    S.plan = g.plan;
    $("plan-note").textContent = `grounded: ${g.note}`;
    renderPlan();
  });
  socket.on("plan.step", (s) => { if (s.id !== S.jobId) return; S.steps[s.i] = { ...(S.steps[s.i] || {}), ...s }; renderPlan(); });
  socket.on("job.done", (j) => {
    if (j.id === S.jobId && j.reply) {
      const note = (S.guard || []).length
        ? `<div class="guard-reply">✂ ${S.guard.length} sentence${S.guard.length > 1 ? "s" : ""} with a number not found in the source ${S.guard.length > 1 ? "were" : "was"} removed (shown in Thinking)</div>` : "";
      $("cmd-reply").innerHTML = `<b>REPLY</b>${esc(j.reply)}${note}`;
      $("cmd-reply").classList.remove("hidden");
    }
    S.history.push(j);
    if (S.history.length > 8) S.history.shift();
    Draw.gantt($("gantt"), S.history);
  });
  socket.on("matrix", (m) => { S.frame = m.frame; });
  socket.on("mqtt.out", addMqtt);
  socket.on("dance.timeline", Duet.start);
  socket.on("dance.kf", Duet.addKf);
  socket.on("dance.done", Duet.finish);
  socket.on("drive", Duet.setLive);
  socket.on("llm.warm", (w) => {
    $("doing").textContent = w.restored ? `model ready: cache ${w.cache_note}`
      : `model warm: prefix prefilled in ${(w.ms / 1000).toFixed(1)} s (cache saved for next boot)`;
  });
  socket.on("reflex.action", (r) => {
    $("cmd-src").className = "badge reflex"; $("cmd-src").textContent = "STOP";
    $("doing").textContent = `STOP from ${r.source}: everything halted in ${r.latency_ms} ms (no LLM involved)`;
  });

  // ------------------------------------------------------------ expandable panels
  // Every panel gets a ⤢ button (and double-click on its title). The expanded panel
  // floats over a dimmed page and keeps updating live; Esc, ✕ or the backdrop close it.
  const backdrop = document.createElement("div");
  backdrop.id = "expand-backdrop";
  document.body.appendChild(backdrop);
  let expanded = null;
  function redrawCharts() {
    setTimeout(() => { Draw.gantt($("gantt"), S.history); Draw.entropy($("entropy"), S.tokens); }, 30);
  }
  function collapse() {
    if (!expanded) return;
    expanded.classList.remove("expanded");
    expanded.querySelector(".expand-btn").textContent = "⤢";
    expanded.querySelector(".expand-btn").title = "Expand";
    expanded = null;
    document.body.classList.remove("has-expanded");
    redrawCharts();
  }
  function expand(card) {
    if (expanded === card) return collapse();
    collapse();
    expanded = card;
    card.classList.add("expanded");
    card.querySelector(".expand-btn").textContent = "✕";
    card.querySelector(".expand-btn").title = "Close (Esc)";
    document.body.classList.add("has-expanded");
    redrawCharts();
  }
  document.querySelectorAll("main .card, #duet .card").forEach((card) => {
    if (card.id === "send-form") return;                  // the input stays put
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "expand-btn";
    btn.textContent = "⤢";
    btn.title = "Expand";
    btn.addEventListener("click", (e) => { e.stopPropagation(); expand(card); });
    card.appendChild(btn);
    const head = card.querySelector(".card-head");
    if (head) head.addEventListener("dblclick", () => expand(card));
  });
  backdrop.addEventListener("click", collapse);
  // ?expand=matrix (or timeline, stage, stream, decisions, plan, chips, gantt ...) opens that panel directly
  const want = new URLSearchParams(location.search).get("expand");
  if (want && $(want)) { const card = $(want).closest(".card"); if (card) setTimeout(() => expand(card), 300); }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") collapse(); });

  // ------------------------------------------------------------ render loop
  function frame() {
    // One broken panel must not freeze the others (or stop the loop).
    for (const f of [() => Draw.matrix($("matrix"), S.frame), () => Duet.drawTimeline($("timeline")),
                     () => Duet.drawStage($("stage")), () => { $("sync-readout").textContent = Duet.readout(); }]) {
      try { f(); } catch (e) { console.error(e); }
    }
    requestAnimationFrame(frame);
  }
  // Web 1.0 marquee: constant scroll speed (~90 px/s) whatever the message length.
  // Only re-time when the text actually changes; status ticks rewrite the same text.
  let marqueeText = null;
  new MutationObserver(() => {
    const el = $("doing");
    if (el.textContent === marqueeText) return;
    marqueeText = el.textContent;
    const travel = el.offsetWidth;            // the span's width = box width (its padding) + text width
    el.style.setProperty("--marquee-s", `${Math.max(6, travel / 90).toFixed(1)}s`);
  }).observe($("doing"), { childList: true, characterData: true, subtree: true });

  Draw.gantt($("gantt"), S.history);
  Draw.entropy($("entropy"), []);
  requestAnimationFrame(frame);
  window.addEventListener("resize", () => { Draw.gantt($("gantt"), S.history); Draw.entropy($("entropy"), S.tokens); });
})();
