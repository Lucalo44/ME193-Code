// Choreography timeline + top-down stage. Plays the compiled timeline locally
// against the board's clock (offset from server_now), so no per-frame events
// are needed; measured keyframe errors arrive as dance.kf and are drawn as ticks.
const Duet = (() => {
  const MODES = ["unison", "mirror", "call_response", "canon", "solo_car", "solo_lego"];
  let D = null;          // dance.timeline payload
  let kfs = [];          // dance.kf reports
  let offsetMs = 0;      // board clock - browser clock
  let done = null;
  const stage = { car: { x: 0, y: 0, yaw: 0, l: 0, r: 0 }, arms: { a: 0, b: 0, ta: 0, tb: 0, va: 0, vb: 0 },
                  lastT: performance.now(), live: { l: 0, r: 0 } };

  function start(payload) {
    D = payload;
    kfs = [];
    done = null;
    offsetMs = payload.server_now * 1000 - Date.now();
    stage.car = { x: 0, y: 0, yaw: 0, l: 0, r: 0 };
    stage.arms = { a: 0, b: 0, ta: 0, tb: 0, va: 0, vb: 0 };
  }
  function addKf(k) { kfs.push(k); }
  function finish(r) { done = r; }
  function setLive(d) { stage.live = { l: d.l, r: d.r }; }
  function now() { return (Date.now() + offsetMs) / 1000; }
  function elapsed() { return D ? now() - D.t_start : null; }
  function playing() { const e = elapsed(); return D && e !== null && e >= 0 && e <= D.seconds + 0.5; }

  // Current wheel speeds / arm targets from the compiled timeline at time e.
  function stateAt(e) {
    let wheels = null, pose = null, run = null;
    for (const k of D.timelines.car) {
      if (k.t > e) break;
      if (k.action === "wheels") wheels = k.params;
    }
    for (const k of D.timelines.lego) {
      if (k.t > e) break;
      if (k.action === "pose") { pose = k.params; run = null; }
      else if (k.action === "run") run = k.params;
      else if (k.action === "stop") run = null;
      else if (k.action === "tank") run = { a: k.params.l, b: k.params.r };
    }
    return { wheels, pose, run };
  }

  function drawTimeline(canvas) {
    // Drawn on a 150-px-tall design; a bigger (expanded) canvas scales it up uniformly.
    const fit = Draw.fit(canvas), ctx = fit.ctx;
    const k = Math.max(1, Math.min(3, fit.h / 150, fit.w / 700));
    ctx.scale(k, k);
    const w = fit.w / k;
    const oy = Math.max(0, (fit.h / k - 150) / 2);         // centre the 150-px design vertically
    ctx.translate(0, oy);
    const h = 150;
    ctx.clearRect(0, 0, w, h);
    const F = Draw.css("--cv-font");
    ctx.font = `12px ${F}`;
    if (!D) {
      ctx.fillStyle = Draw.css("--muted");
      ctx.fillText("Ask for a dance: the model writes a score, the compiler turns it into beats for both robots.", 8, 24);
      return;
    }
    const b0 = -D.count_in, b1 = D.beats + 0.5;
    const left = 52, right = 10;
    const bx = (b) => left + ((b - b0) / (b1 - b0)) * (w - left - right);
    const lanes = { car: [18, 58], lego: [66, 106] };
    // count-in
    ctx.fillStyle = Draw.css("--cv-shade");
    ctx.fillRect(bx(b0), 12, bx(0) - bx(b0), 100);
    ctx.fillStyle = Draw.css("--muted");
    for (let i = 0; i < D.count_in; i++) ctx.fillText(String(D.count_in - i), bx(-D.count_in + i) + 4, 30);
    // beat grid
    const M = D.meter || 4;                 // beats per bar (3 for a waltz)
    for (let b = Math.ceil(b0); b <= D.beats; b++) {
      ctx.strokeStyle = b % M === 0 ? Draw.css("--cv-grid-strong") : Draw.css("--cv-grid");
      ctx.lineWidth = b % M === 0 ? 1.5 : 1;
      ctx.beginPath(); ctx.moveTo(bx(b), 12); ctx.lineTo(bx(b), 112); ctx.stroke();
      if (b >= 0 && b % M === 0) { ctx.fillStyle = Draw.css("--muted"); ctx.fillText(`bar ${b / M + 1}`, bx(b) + 3, 10); }
    }
    // lane labels
    ctx.fillStyle = Draw.css("--cv-ink");
    ctx.font = `bold 12px ${F}`;
    ctx.fillText(D.where === "matrix" ? "LEGS" : "CAR", 6, 42);       // on the matrix the figure's legs/arms
    ctx.fillText(D.where === "matrix" ? "ARMS" : "LEGO", 6, 90);       // follow the car/LEGO lanes
    ctx.font = `12px ${F}`;
    // blocks
    for (const bl of D.blocks) {
      const [y0, y1] = lanes[bl.lane];
      const x0 = bx(bl.start), x1 = bx(bl.start + bl.beats);
      ctx.fillStyle = Draw.css(`--m-${bl.mode}`) + (bl.move === "freeze" ? "44" : "cc");
      ctx.fillRect(x0 + 1, y0, x1 - x0 - 2, y1 - y0);
      ctx.fillStyle = Draw.css("--cv-on-block");
      if (x1 - x0 > 34) ctx.fillText(bl.move, x0 + 5, y0 + 24);
    }
    // measured ticks + error strip
    const spb = D.spb;
    for (const k of kfs) {
      if (!k.motion || k.err_ms == null) continue;
      const lane = lanes[k.performer] || lanes.car;
      const xc = bx(k.beat), xa = bx(k.beat + k.err_ms / 1000 / spb);
      const a = Math.abs(k.err_ms);
      const col = a < 40 ? Draw.css("--ok") : a < 80 ? Draw.css("--warn") : Draw.css("--bad");
      ctx.strokeStyle = Draw.css("--cv-tick"); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(xc, lane[1] - 6); ctx.lineTo(xc, lane[1]); ctx.stroke();
      ctx.strokeStyle = col; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(xa, lane[1] - 10); ctx.lineTo(xa, lane[1] + 2); ctx.stroke();
      // magnified strip: 0..150 ms -> 0..26 px
      const sh = Math.min(26, (a / 150) * 26);
      ctx.fillStyle = col;
      ctx.fillRect(xc - 1 + (k.performer === "lego" ? 2 : -2), 140 - sh, 2, sh);
    }
    ctx.fillStyle = Draw.css("--muted");
    ctx.font = `10px ${F}`;
    ctx.fillText("sync error", 2, 128);
    ctx.fillText("(0-150 ms)", 2, 140);
    // playhead
    const e = elapsed();
    if (e !== null) {
      const beat = e / spb - D.count_in;
      if (beat >= b0 && beat <= b1) {
        ctx.strokeStyle = Draw.css("--cv-playhead");
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(bx(beat), 8); ctx.lineTo(bx(beat), 114); ctx.stroke();
      }
    }
  }

  function step(dt) {
    let l = stage.live.l, r = stage.live.r;
    if (playing() && D.where !== "matrix") {
      const st = stateAt(elapsed());
      if (st.wheels) { l = st.wheels.l; r = st.wheels.r; } else { l = r = 0; }
      if (st.pose) { stage.arms.ta = st.pose.a; stage.arms.tb = st.pose.b; stage.arms.va = st.pose.speed * 6; stage.arms.vb = st.pose.speed * 6; }
      stage.arms.run = st.run;
    } else {
      stage.arms.run = null;
      stage.arms.ta = 0; stage.arms.tb = 0; stage.arms.va = stage.arms.vb = 240;
    }
    // differential drive, same constants as hal/__init__.py
    const V = 0.4, TRACK = 0.12;
    const vl = (l / 255) * V, vr = (r / 255) * V;
    const v = (vl + vr) / 2, wz = (vr - vl) / TRACK;
    const c = stage.car;
    c.yaw += wz * dt;
    c.x += v * Math.cos(c.yaw) * dt;
    c.y += v * Math.sin(c.yaw) * dt;
    c.x = Math.max(-0.4, Math.min(0.4, c.x));
    c.y = Math.max(-0.22, Math.min(0.22, c.y));
    c.l = l; c.r = r;
    const A = stage.arms;
    if (A.run) {
      A.a += A.run.a * 6 * dt; A.b += A.run.b * 6 * dt;
    } else {
      const mv = (cur, tgt, vel) => { const d = tgt - cur, m = vel * dt; return Math.abs(d) <= m ? tgt : cur + Math.sign(d) * m; };
      A.a = mv(A.a, A.ta, A.va || 240); A.b = mv(A.b, A.tb, A.vb || 240);
    }
  }

  function drawStage(canvas) {
    const t = performance.now();
    const dt = Math.min(0.05, (t - stage.lastT) / 1000);
    stage.lastT = t;
    step(dt);
    const fit = Draw.fit(canvas), ctx = fit.ctx;
    const k = Math.max(1, Math.min(fit.w / 420, fit.h / 190));    // scale up when expanded
    ctx.scale(k, k);
    const w = fit.w / k, h = fit.h / k;
    const F = Draw.css("--cv-font");
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = Draw.css("--stage-bg");
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = Draw.css("--stage-grid");
    for (let x = 0; x < w; x += 24) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = 0; y < h; y += 24) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    // car (left half), top-down
    const scale = 260;
    const c = stage.car;
    ctx.save();
    ctx.translate(w * 0.3 + c.x * scale, h / 2 - c.y * scale);
    ctx.rotate(-c.yaw);
    ctx.fillStyle = Draw.css("--car-body"); ctx.strokeStyle = Draw.css("--car-line"); ctx.lineWidth = 2;
    ctx.fillRect(-30, -22, 60, 44); ctx.strokeRect(-30, -22, 60, 44);
    ctx.fillStyle = Draw.css("--car-line");
    ctx.beginPath(); ctx.moveTo(30, 0); ctx.lineTo(20, -8); ctx.lineTo(20, 8); ctx.fill();
    const wheel = (y, s) => { ctx.fillStyle = s > 0 ? Draw.css("--wheel-fwd") : s < 0 ? Draw.css("--wheel-back") : Draw.css("--wheel-idle"); ctx.fillRect(-14, y - 4, 28, 8); };
    wheel(-26, c.l); wheel(26, c.r);
    ctx.restore();
    ctx.fillStyle = Draw.css("--cv-ink"); ctx.font = `12px ${F}`;
    ctx.fillText("UNO Q car", 8, h - 8);
    // LEGO bot (right half), front view: body + two arms
    const lx = w * 0.74, ly = h * 0.62;
    ctx.fillStyle = Draw.css("--lego-body"); ctx.strokeStyle = Draw.css("--lego-arm"); ctx.lineWidth = 2;
    ctx.fillRect(lx - 34, ly - 26, 68, 52); ctx.strokeRect(lx - 34, ly - 26, 68, 52);
    const arm = (sx, ang, dir) => {
      // 0 deg = arm hanging down (+y); positive raises it outward (A left, B right)
      ctx.save(); ctx.translate(lx + sx, ly - 18);
      ctx.rotate((-dir * ang * Math.PI) / 180);
      ctx.fillStyle = Draw.css("--lego-arm");
      ctx.fillRect(-5, 0, 10, 54);
      ctx.restore();
    };
    arm(-40, stage.arms.a, -1);
    arm(40, stage.arms.b, 1);
    ctx.fillStyle = Draw.css("--cv-ink");
    ctx.fillText("LEGO bot (A / B arms)", w * 0.56, h - 8);
    if (playing()) {
      const beat = elapsed() / D.spb - D.count_in;
      ctx.fillStyle = Draw.css("--cv-ink"); ctx.font = `bold 16px ${F}`;
      ctx.fillText(beat < 0 ? `count-in ${Math.ceil(-beat)}` : `beat ${Math.floor(beat) + 1} / ${D.beats}`, 8, 20);
      if (D.where === "matrix") { ctx.font = `13px ${F}`; ctx.fillText("dancing on the LED matrix: robots stay still", 8, 40); }
      else if (D.where === "all") { ctx.font = `13px ${F}`; ctx.fillText("robots + the stick figure on the LED matrix", 8, 40); }
    }
  }

  function readout() {
    if (!done && !D) return "–";
    if (!done) return playing() ? "playing…" : "–";
    const p = done.performers || {};
    const f = (x) => (x == null ? "–" : `${x} ms`);
    return `max err ${f(done.max_sync_error_ms)} · mean ${f(done.mean_sync_error_ms)} · car↔lego gap ${f(done.max_gap_ms)}` +
      (done.completed ? "" : " · STOPPED");
  }

  return { start, addKf, finish, setLive, drawTimeline, drawStage, readout, MODES, get: () => D };
})();
