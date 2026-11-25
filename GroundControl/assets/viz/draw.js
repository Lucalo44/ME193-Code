// Canvas helpers shared by the dashboard: LED matrix, entropy sparkline, latency Gantt.
const Draw = (() => {
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  function fit(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.width;
    const h = canvas.clientHeight || canvas.height;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function matrix(canvas, frame) {
    const { ctx, w, h } = fit(canvas);
    ctx.clearRect(0, 0, w, h);
    const cols = 13, rows = 8;
    const cell = Math.min(w / cols, h / rows);
    const ox = (w - cell * cols) / 2, oy = (h - cell * rows) / 2;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = (frame && frame[r * cols + c]) || 0;
        ctx.beginPath();
        ctx.arc(ox + c * cell + cell / 2, oy + r * cell + cell / 2, cell * 0.36, 0, Math.PI * 2);
        if (v > 0) {
          const a = 0.25 + 0.75 * (v / 7);
          ctx.fillStyle = `rgba(255, 70, 70, ${a})`;
          ctx.shadowColor = "rgba(255,60,60,0.9)";
          ctx.shadowBlur = 8 * (v / 7);
        } else {
          ctx.fillStyle = "#1a1f2b";
          ctx.shadowBlur = 0;
        }
        ctx.fill();
      }
    }
    ctx.shadowBlur = 0;
  }

  function entropy(canvas, tokens) {
    const { ctx, w, h } = fit(canvas);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0e1522";
    ctx.fillRect(0, 0, w, h);
    if (!tokens.length) return;
    const n = Math.max(tokens.length, 60);
    const bw = w / n;
    const maxH = 2.6;
    tokens.forEach((t, i) => {
      const e = t.entropy == null ? 0 : t.entropy;
      const bh = Math.max(1, (e / maxH) * (h - 6));
      ctx.fillStyle = t.decision ? css("--warn") : t.forced ? css("--bad") : e > 1 ? "#5fa8ff" : "#33507a";
      ctx.fillRect(i * bw + 0.5, h - bh, Math.max(1, bw - 1), bh);
    });
  }

  const SEG_COLORS = { queue: "#3a465e", prefill: "--seg-catalog", decode: "--accent", dance: "--m-canon" };

  function gantt(canvas, jobs) {
    const { ctx, w, h } = fit(canvas);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px ui-monospace, Menlo, monospace";
    if (!jobs.length) {
      ctx.fillStyle = css("--muted");
      ctx.fillText("Finished jobs appear here: where the time went, and the KV cache effect.", 6, 20);
      return;
    }
    const rows = jobs.slice(-6);
    const tmax = Math.max(...rows.map((j) => j.total_ms / 1000), 5);
    const left = 130, right = 52, rh = Math.min(22, (h - 18) / rows.length);
    const sx = (t) => left + (t / tmax) * (w - left - right);
    // axis
    ctx.strokeStyle = "#24314a";
    ctx.fillStyle = css("--muted");
    const span = w - left - right;
    const step = [1, 2, 5, 10, 20, 30, 60, 120, 300].find((s) => (s / tmax) * span >= 42) || 600;
    for (let t = 0; t <= tmax; t += step) {
      ctx.beginPath(); ctx.moveTo(sx(t), 0); ctx.lineTo(sx(t), h - 14); ctx.stroke();
      ctx.fillText(`${t}s`, sx(t) - 6, h - 2);
    }
    rows.forEach((j, r) => {
      const y = r * rh + 2;
      ctx.fillStyle = css("--text");
      const label = (j.text || "").slice(0, 11) + ((j.text || "").length > 11 ? "…" : "");
      ctx.fillText(label, 4, y + rh * 0.65);
      (j.segments || []).forEach((s) => {
        let col = SEG_COLORS[s.name] || (s.name === "dance" ? "--m-canon" : "--ok");
        if (col.startsWith("--")) col = css(col);
        ctx.fillStyle = col;
        const x0 = sx(s.t0), x1 = Math.max(sx(s.t1), x0 + 2);
        ctx.fillRect(x0, y + 2, x1 - x0, rh - 6);
        if (x1 - x0 > 46) {
          ctx.fillStyle = "#0a0e16";
          ctx.fillText(s.name, x0 + 3, y + rh * 0.62);
        }
      });
      ctx.fillStyle = css("--muted");
      ctx.fillText(`${(j.total_ms / 1000).toFixed(1)}s`, w - right + 4, y + rh * 0.65);
      if (j.cache_n) {
        ctx.fillStyle = css("--seg-catalog");
        ctx.fillText(`♻${j.cache_n}`, left - 40, y + rh * 0.65);
      }
    });
  }

  return { fit, matrix, entropy, gantt, css };
})();
