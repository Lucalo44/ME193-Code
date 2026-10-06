// hud.js -- DOM overlay: scorebug, status readouts, last-shot spin compass and power
// meter, messages, camera picture-in-picture, lobby / game-over / latency / pause.
import { THEME, OPPONENTS, opponentFor } from './scene/theme.js';

const $ = (id) => document.getElementById(id);
const SEGMENTS = 12;

const REASON_SUB = {
  'CPU MISSED': 'missed the return', 'OUT': 'hit it long', 'NET': 'into the net', 'OWN SIDE': 'own side',
};

export class Hud {
  constructor() {
    this.msgTimer = null;
    this.toastTimer = null;
    this.lastPhase = null;
    this.traceCtx = $('trace').getContext('2d');
    this.lastFrameAt = 0;
    this.cpuName = opponentFor('medium').name;
    // Lobby cards: who you face at each speed.
    document.querySelectorAll('.tag .who').forEach((el) => {
      const o = OPPONENTS[el.dataset.speed];
      el.innerHTML = `VS ${o.name}<i>${o.tagline}</i>`;
    });
    $('bar-speed').innerHTML = '<i></i>'.repeat(SEGMENTS);
  }

  // Pause / resume buttons call onToggle (main.js sends the same "p" key as the keyboard).
  bindPause(onToggle) {
    for (const id of ['pause-btn', 'resume-btn']) {
      const b = $(id);
      b.addEventListener('mousedown', (e) => e.preventDefault());   // never take focus (Space would click it)
      b.addEventListener('click', onToggle);
    }
  }

  // ---------- per-state updates ----------
  update(s) {
    if (!s) return;
    const sc = s.score;
    const opp = opponentFor(s.speed_setting);
    if (opp.name !== $('cpu-name').textContent) {
      this.cpuName = opp.name;
      $('cpu-name').textContent = opp.name;
    }
    $('pts-player').textContent = sc.player;
    $('pts-cpu').textContent = sc.cpu;
    this._games('games-player', sc.games_player, sc.games_needed);
    this._games('games-cpu', sc.games_cpu, sc.games_needed);
    const playing = !['LOBBY', 'GAME_OVER', 'LATENCY_CAL'].includes(s.phase);
    $('srv-player').classList.toggle('on', playing && sc.server === 'player');
    $('srv-cpu').classList.toggle('on', playing && sc.server === 'cpu');
    const chip = $('speed-chip');
    chip.textContent = s.speed_setting;
    chip.className = s.speed_setting;
    $('phase-label').textContent = s.paused ? 'PAUSED' : s.phase.replace('_', ' ');

    const st = s.status;
    this._stat('chip-paddle', 'PADDLE', st.paddle_kind === 'sim' ? 'keyboard (sim)' : st.paddle,
      st.paddle_kind === 'sim' ? 'warn' : st.paddle === 'connected' ? 'ok' : st.paddle === 'connecting' ? 'warn' : 'bad');
    this._stat('chip-camera', 'CAMERA', st.camera, st.camera === 'ok' ? 'ok' : st.camera === 'off' ? '' : 'bad');
    const poseText = st.camera === 'off' ? 'off (stroke check skipped)'
      : !st.stroke_check ? 'untrained (stroke check off)'
      : (st.pose ? (st.pose_label || 'detected') : 'not detected');
    this._stat('chip-pose', 'POSE', poseText, st.camera === 'off' ? '' : !st.stroke_check ? 'warn' : st.pose ? 'ok' : 'bad');
    this._stat('chip-cal', 'CALIBRATION', st.calibration, st.calibration === 'file' ? 'ok' : 'warn');

    // Stroke hint (arrow points to the side the ball is coming to).
    const hint = $('stroke-hint');
    const req = s.required_stroke;
    if (s.settings.hint && req && req !== 'either' && s.phase === 'RALLY') {
      const left = s.settings.handedness === 'left';
      const rightSide = (req === 'forehand') !== left;
      $('hint-text').textContent = rightSide ? `${req} ▸` : `◂ ${req}`;
      hint.className = 'show';
    } else {
      hint.className = '';
    }

    // Overlays.
    $('lobby').classList.toggle('hidden', s.phase !== 'LOBBY');
    $('gameover').classList.toggle('hidden', s.phase !== 'GAME_OVER');
    $('latency').classList.toggle('hidden', s.phase !== 'LATENCY_CAL');
    $('paused').classList.toggle('hidden', !s.paused);
    const pb = $('pause-btn');
    pb.classList.toggle('hidden', !playing && !s.paused);
    pb.classList.toggle('paused', s.paused);
    $('pause-label').textContent = s.paused ? 'RESUME' : 'PAUSE';
    if (s.phase === 'GAME_OVER' && this.lastPhase !== 'GAME_OVER') {
      const won = sc.games_player > sc.games_cpu;
      $('go-title').textContent = won ? 'MATCH WON' : 'MATCH LOST';
      $('go-score').textContent = `YOU ${sc.games_player} – ${sc.games_cpu} ${this.cpuName}`;
    }
    this.lastPhase = s.phase;

    // Tag progress rings (lobby).
    document.querySelectorAll('.tag').forEach((el) => {
      const id = Number(el.dataset.id);
      const p = s.tag && s.tag.id === id ? s.tag.progress : 0;
      el.querySelector('circle').style.strokeDashoffset = String(289 * (1 - p));
      el.classList.toggle('active', p > 0);
    });
    let howto = st.camera === 'off'
      ? 'Camera is off: press <b>1</b>, <b>2</b> or <b>3</b> to start.'
      : 'Hold a printed tag up to the webcam until its ring fills. Forehand when the ball comes to the ' +
        'right half of the screen, backhand on the left.';
    if (st.paddle_kind === 'sim') {
      howto += '<br>Keyboard paddle: <b>Space</b> swing, <b>Shift+Space</b> hard, hold <b>W/S</b> top/backspin, ' +
        '<b>A/D</b> sidespin' + (st.camera === 'off' ? ', <b>F/B</b> force forehand/backhand.' : '.');
    }
    if ($('howto').innerHTML !== howto) $('howto').innerHTML = howto;

    $('pip-label').textContent = st.camera === 'off' ? 'camera off'
      : (performance.now() - this.lastFrameAt > 1500 ? 'no frames yet' : (st.pose_label || ''));
  }

  _games(id, won, needed) {
    const el = $(id);
    if (el.childElementCount !== needed) el.innerHTML = '<i></i>'.repeat(needed);
    [...el.children].forEach((c, i) => c.classList.toggle('won', i < won));
  }

  _stat(id, key, value, cls) {
    const el = $(id);
    const html = `<span class="k">${key}</span>${value}`;
    if (el.innerHTML !== html) el.innerHTML = html;
    el.className = `stat ${cls}`;
  }

  // ---------- events ----------
  centerMessage(text, sub = '', cls = 'bad', ms = 1300) {
    const el = $('center-msg');
    $('msg-big').textContent = text;
    $('msg-small').textContent = sub;
    el.className = cls;
    void el.offsetWidth;          // restart the slide-in
    el.className = `show ${cls}`;
    clearTimeout(this.msgTimer);
    this.msgTimer = setTimeout(() => { el.className = cls; }, ms);
  }

  toast(text, ms = 2500) {
    const el = $('toast');
    el.textContent = text;
    el.classList.add('show');
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => el.classList.remove('show'), ms);
  }

  miss(ev) {
    let sub = '';
    if (ev.reason === 'WRONG STROKE') {
      sub = `needed ${ev.needed}` + (ev.judged ? ` · saw ${ev.judged}` : ' · no stroke seen');
    } else if (ev.reason === 'EARLY' || ev.reason === 'LATE') {
      sub = `by ${Math.round(ev.by_s * 1000)} ms`;
    }
    this.centerMessage(ev.reason, sub, 'bad');
  }

  point(ev) {
    if (ev.winner === 'player') this.centerMessage('POINT', `${this.cpuName} ${REASON_SUB[ev.reason] || ''}`, 'good', 1100);
  }

  shot(ev) {
    const strength = Math.max(0, Math.min(1, ev.strength ?? 0));
    const lit = Math.round(strength * SEGMENTS);
    [...$('bar-speed').children].forEach((c, i) => {
      c.classList.toggle('on', i < lit);
      c.classList.toggle('hot', i >= SEGMENTS - 3);
    });
    $('val-speed').textContent = `${ev.speed.toFixed(1)} m/s`;
    const top = Math.max(-1, Math.min(1, ev.topspin)), side = Math.max(-1, Math.min(1, ev.sidespin));
    $('val-top').textContent = top >= 0.05 ? `top ${top.toFixed(2)}` : top <= -0.05 ? `back ${(-top).toFixed(2)}` : 'flat';
    $('val-side').textContent = side >= 0.05 ? `right ${side.toFixed(2)}` : side <= -0.05 ? `left ${(-side).toFixed(2)}` : 'none';
    const stroke = $('val-stroke');
    stroke.textContent = ev.stroke ? `${ev.stroke} ${Math.round((ev.confidence ?? 1) * 100)}%` : '--';
    stroke.className = 'ok';

    // Spin compass: up = topspin, down = backspin, left/right = sidespin.
    const x = side * 30, y = -top * 30;
    $('spin-dot').setAttribute('cx', x); $('spin-dot').setAttribute('cy', y);
    $('spin-vec').setAttribute('x2', x); $('spin-vec').setAttribute('y2', y);
    const sp = THEME.spin;
    const color = Math.abs(top) < 0.05 && Math.abs(side) < 0.05 ? sp.none
      : Math.abs(top) >= Math.abs(side) ? (top >= 0 ? sp.topspin : sp.backspin) : (side >= 0 ? sp.right : sp.left);
    $('spin-dot').style.fill = color;
    $('spin-vec').style.stroke = color;
    const tri = (cls, on) => document.querySelector(`#spin-compass .tri.${cls}`).classList.toggle('on', on);
    tri('top', top >= 0.05); tri('back', top <= -0.05); tri('right', side >= 0.05); tri('left', side <= -0.05);
  }

  trace(rows) {
    const g = this.traceCtx, W = g.canvas.width, H = g.canvas.height;
    g.clearRect(0, 0, W, H);
    if (!rows || rows.length < 2) return;
    const t0 = rows[0][0], t1 = rows[rows.length - 1][0];
    const maxA = Math.max(2, ...rows.map((r) => r[1]));
    const maxG = Math.max(200, ...rows.map((r) => r[2]));
    const plot = (idx, max, color) => {
      g.strokeStyle = color; g.lineWidth = 1.6; g.beginPath();
      rows.forEach((r, i) => {
        const x = ((r[0] - t0) / Math.max(1e-3, t1 - t0)) * W;
        const y = H - 2 - (r[idx] / max) * (H - 6);
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.stroke();
    };
    plot(2, maxG, 'rgba(255,255,255,0.45)');   // gyro
    plot(1, maxA, THEME.spin.backspin);          // linear acceleration
    const xPeak = ((0 - t0) / Math.max(1e-3, t1 - t0)) * W;
    g.strokeStyle = 'rgba(255,255,255,0.3)'; g.beginPath(); g.moveTo(xPeak, 0); g.lineTo(xPeak, H); g.stroke();
  }

  cameraFrame(b64) {
    $('cam').src = `data:image/jpeg;base64,${b64}`;
    this.lastFrameAt = performance.now();
  }

  connected(ok) { $('disconnected').classList.toggle('hidden', ok); }

  latency(s, renderT, beatPulse) {
    $('beat').style.transform = `scale(${0.7 + 0.4 * beatPulse})`;
    $('beat').style.background = `rgba(255,255,255,${0.85 * beatPulse})`;
    if (s.latency) {
      const beats = s.latency.beats;
      const left = beats.filter((b) => b > renderT).length;
      $('beat-count').textContent = renderT < beats[0] ? 'GET READY'
        : `${s.latency.count} SWINGS MATCHED · ${left} BEATS LEFT`;
    }
  }

  debug(text) {
    const el = $('debug-panel');
    el.classList.toggle('show', !!text);
    if (text) el.textContent = text;
  }
}
