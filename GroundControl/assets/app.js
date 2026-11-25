// Ground Control dashboard. Renders with zero events (empty states) and
// rebuilds the current job from get_state after any (re)connect.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const socket = io();

  const S = { jobId: null, tokens: [], text: "", prompt: null, plan: null, steps: {}, history: [],
              status: {}, frame: new Array(104).fill(0), repair: null };

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
    $("toks").textContent = s.llm_tok_s ? `${s.llm_tok_s} tok/s` : "– tok/s";
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
    S.jobId = j.id; S.tokens = []; S.text = ""; S.prompt = null; S.plan = null; S.steps = {}; S.repair = null;
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
  function confColor(p) {
    if (p == null) return "#2a3550";
    const h = Math.round(p * 130);            // 0 = red, 130 = green
    return `hsla(${h}, 70%, 45%, 0.55)`;
  }
  const BAR = ["#3ddc97", "#5fd4ff", "#b08cff", "#ffb020", "#ff5c7a"];

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
    pop.style.left = `${e.clientX + 14}px`;
    pop.style.top = `${e.clientY + 14}px`;
    pop.classList.remove("hidden");
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
  socket.on("plan.grounded", (g) => {
    if (g.id !== S.jobId) return;
    S.plan = g.plan;
    $("plan-note").textContent = `grounded: ${g.note}`;
    renderPlan();
  });
  socket.on("plan.step", (s) => { if (s.id !== S.jobId) return; S.steps[s.i] = { ...(S.steps[s.i] || {}), ...s }; renderPlan(); });
  socket.on("job.done", (j) => {
    if (j.id === S.jobId && j.reply) {
      $("cmd-reply").innerHTML = `<b>REPLY</b>${esc(j.reply)}`;
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
  socket.on("llm.warm", (w) => { $("doing").textContent = `model warm: prefix prefilled in ${(w.ms / 1000).toFixed(1)} s`; });
  socket.on("reflex.action", (r) => {
    $("cmd-src").className = "badge reflex"; $("cmd-src").textContent = "STOP";
    $("doing").textContent = `STOP from ${r.source}: everything halted in ${r.latency_ms} ms (no LLM involved)`;
  });

  // ------------------------------------------------------------ render loop
  function frame() {
    Draw.matrix($("matrix"), S.frame);
    Duet.drawTimeline($("timeline"));
    Duet.drawStage($("stage"));
    $("sync-readout").textContent = Duet.readout();
    requestAnimationFrame(frame);
  }
  Draw.gantt($("gantt"), S.history);
  Draw.entropy($("entropy"), []);
  requestAnimationFrame(frame);
  window.addEventListener("resize", () => Draw.gantt($("gantt"), S.history));
})();
