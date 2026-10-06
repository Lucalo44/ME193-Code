// main.js -- renders the game. Python owns all logic and physics; this file only
// draws what the WebSocket says, interpolating between ~60 Hz snapshots.
//
// ---------------------------------------------------------------------------
// Protocol (mirror of server/protocol.py -- keep in sync)
//
// Python -> browser
//   {type:"state", t, phase, paused,
//    ball:{pos:[x,y,z], vel:[...], spin:[...], visible},
//    opponent:{x, swing:"forehand"|"backhand"|null, swing_t},
//    paddle:{pitch, roll, yaw},             // degrees, relative to the zeroed ready pose
//    required_stroke, incoming:{x, t_to_arrival, required}|null,
//    score:{player, cpu, games_player, games_cpu, server, games_needed},
//    speed_setting, status:{paddle, paddle_kind, camera, pose, stroke_check, calibration, pose_label},
//    tag:{id, label, progress}|null, latency:{beats, count}|null, debug:{...}|null,
//    settings:{hint, handedness, debug, hit_window_s}}
//   {type:"event", name:"hit"|"bounce"|"net"|"miss"|"point"|"game_over"|"tag_progress"
//                       |"tag_confirmed"|"swing"|"serve"|"message", ...}
//   {type:"camera_frame", jpeg_b64}
// Browser -> Python
//   {type:"key", key, down, shift}
//
// Physics frame: meters, table top y=0, player at -z. Scene frame: table top at
// y=TABLE_H and the player at +z (so three.js's default camera looks down the
// table and world +x is screen-right). Conversion: (x, y + TABLE_H, -z).
// ---------------------------------------------------------------------------

import * as THREE from 'three';
import { THEME, TABLE_H, opponentFor } from './scene/theme.js';
import { buildArena, buildSideHint } from './scene/table.js';
import { BallView } from './scene/ball.js';
import { OpponentView } from './scene/opponent.js';
import { PlayerPaddleView } from './scene/paddle.js';
import { Effects, Sounds } from './scene/effects.js';
import { Hud } from './hud.js';

const INTERP_DELAY = 0.05;     // render this far behind the newest snapshot (s)
const LOCAL_KEYS = new Set(['v', 'o', 'm']);   // handled here, not sent to Python

const runtime = await fetch('runtime.json').then(r => r.json()).catch(() => ({}));
const PLAYER_PLANE_Z = runtime.player_hit_plane_z ?? -1.55;
const OPP_PLANE_Z = runtime.opponent_hit_plane_z ?? 1.55;
const SPIN_MAX = runtime.spin_max_rads ?? 150;
const SIDESPIN_MAX = runtime.sidespin_max_rads ?? 80;

const toScene = (p) => new THREE.Vector3(p[0], p[1] + TABLE_H, -p[2]);
// Angular velocity is a pseudo-vector: under the z-mirror it picks up an extra sign.
const spinToScene = (w) => new THREE.Vector3(-w[0], -w[1], w[2]);
function topspinOf(vel, spin) {
  const h = Math.hypot(vel[0], vel[2]);
  if (h < 1e-6) return 0;
  return (spin[0] * vel[2] / h - spin[2] * vel[0] / h) / SPIN_MAX;
}
// Positive = the ball curves toward screen right (+x). Magnus a_x = spin_y * v_z.
function sidespinOf(vel, spin) {
  return Math.abs(vel[2]) < 1e-6 ? 0 : (spin[1] * Math.sign(vel[2])) / SIDESPIN_MAX;
}

// ---------------------------------------------------------------- renderer
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
const cam = THEME.camera;
const camera = new THREE.PerspectiveCamera(cam.fov, 1, 0.05, 60);
camera.position.set(0, TABLE_H + cam.height, cam.distance);
camera.lookAt(0, TABLE_H, cam.lookAtZ);

const arena = buildArena(scene);
const sideHint = buildSideHint(scene);
const ballView = new BallView(scene);
const opponentView = new OpponentView(scene, OPP_PLANE_Z);
const paddleView = new PlayerPaddleView(scene, -PLAYER_PLANE_Z);
const effects = new Effects(scene, arena.net);
const sounds = new Sounds();
const hud = new Hud();

// Debug helpers (only visible with the debug overlay on).
const debugGroup = new THREE.Group();
scene.add(debugGroup);
const velArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.3, 0xffff00);
debugGroup.add(velArrow);
const markerGeo = new THREE.SphereGeometry(0.03, 12, 8);
const arrivalMarker = new THREE.Mesh(markerGeo, new THREE.MeshBasicMaterial({ color: 0x4dffb2 }));
const cpuMarker = new THREE.Mesh(markerGeo, new THREE.MeshBasicMaterial({ color: 0xff3d7f }));
debugGroup.add(arrivalMarker, cpuMarker);
const hitPlane = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 0.8),
  new THREE.MeshBasicMaterial({ color: 0x4dffb2, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }));
hitPlane.position.set(0, TABLE_H + 0.3, -PLAYER_PLANE_Z);
debugGroup.add(hitPlane);
debugGroup.visible = false;

// Optional bloom (toggle with V). Auto-disabled if it drags the frame rate down.
let composer = null, bloomOn = false;
try {
  const [{ EffectComposer }, { RenderPass }, { UnrealBloomPass }, { OutputPass }] = await Promise.all([
    import('three/addons/postprocessing/EffectComposer.js'),
    import('three/addons/postprocessing/RenderPass.js'),
    import('three/addons/postprocessing/UnrealBloomPass.js'),
    import('three/addons/postprocessing/OutputPass.js'),
  ]);
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const b = THEME.bloom;
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(512, 512), b.strength, b.radius, b.threshold));
  composer.addPass(new OutputPass());
  bloomOn = !!b.enabledByDefault;
} catch (e) {
  console.warn('Bloom unavailable:', e);
}

let orbit = null;
async function toggleOrbit() {
  if (orbit) { orbit.dispose(); orbit = null; resetCamera(); return; }
  const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
  orbit = new OrbitControls(camera, canvas);
  orbit.target.set(0, TABLE_H, 0);
  orbit.update();
}
function resetCamera() {
  camera.position.set(0, TABLE_H + cam.height, cam.distance);
  camera.lookAt(0, TABLE_H, cam.lookAtZ);
}

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (composer) composer.setSize(w, h);
}
window.addEventListener('resize', resize);
resize();

// ---------------------------------------------------------------- network
const snapshots = [];           // recent state messages, oldest first
let latest = null;
let clockOffset = null;         // performance.now()/1000 - state.t
let ws = null;

function onState(s) {
  const now = performance.now() / 1000;
  const off = now - s.t;
  if (clockOffset === null || off < clockOffset || off - clockOffset > 0.25) clockOffset = off;
  else clockOffset += (off - clockOffset) * 0.02;
  snapshots.push(s);
  while (snapshots.length > 30) snapshots.shift();
  latest = s;
  hud.update(s);
}

function onEvent(ev) {
  switch (ev.name) {
    case 'hit':
      effects.hit(toScene(ev.pos));
      sounds.hit(ev.strength ?? 0.5);
      if (ev.who === 'player') hud.shot(ev);
      break;
    case 'bounce': effects.bounce(toScene(ev.pos)); sounds.bounce(); break;
    case 'net': effects.netHit(); sounds.net(); break;
    case 'miss': hud.miss(ev); sounds.miss(); break;
    case 'point': hud.point(ev); sounds.point(ev.winner === 'player'); break;
    case 'game_over':
      if (!ev.match_over) hud.centerMessage(ev.winner === 'player' ? 'GAME' : `GAME ${opponentFor(latest && latest.speed_setting).name}`,
        `games ${ev.score.games_player} – ${ev.score.games_cpu}`, ev.winner === 'player' ? 'good' : 'bad', 2600);
      break;
    case 'swing': paddleView.swing(); hud.trace(ev.trace); break;
    case 'tag_confirmed': sounds.point(true); break;
    case 'message': hud.toast(ev.text); break;
  }
}

function connect() {
  ws = new WebSocket(`ws://${location.hostname || 'localhost'}:${runtime.ws_port ?? 8765}`);
  ws.onopen = () => hud.connected(true);
  ws.onclose = () => { hud.connected(false); setTimeout(connect, 1000); };
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.type === 'state') onState(msg);
    else if (msg.type === 'event') onEvent(msg);
    else if (msg.type === 'camera_frame') hud.cameraFrame(msg.jpeg_b64);
  };
}
connect();

// ---------------------------------------------------------------- input
function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

window.addEventListener('keydown', (e) => {
  sounds.unlock();
  if (e.metaKey || e.ctrlKey) return;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (key === ' ') e.preventDefault();
  if (e.repeat) return;
  if (LOCAL_KEYS.has(key)) {
    if (key === 'v') { bloomOn = !!composer && !bloomOn; hud.toast(`Bloom ${bloomOn ? 'on' : 'off'}`); }
    if (key === 'o') toggleOrbit();
    if (key === 'm') { sounds.enabled = !sounds.enabled; hud.toast(`Sound ${sounds.enabled ? 'on' : 'off'}`); }
    return;
  }
  send({ type: 'key', key, down: true, shift: e.shiftKey });
});
window.addEventListener('keyup', (e) => {
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (!LOCAL_KEYS.has(key)) send({ type: 'key', key, down: false, shift: e.shiftKey });
});
window.addEventListener('pointerdown', () => sounds.unlock());
hud.bindPause(() => {
  send({ type: 'key', key: 'p', down: true, shift: false });
  send({ type: 'key', key: 'p', down: false, shift: false });
});

// ---------------------------------------------------------------- interpolation
function sample(renderT) {
  if (!snapshots.length) return null;
  let a = snapshots[0], b = snapshots[snapshots.length - 1];
  if (renderT >= b.t) return { ...b, _alpha: 1 };
  for (let i = snapshots.length - 1; i > 0; i--) {
    if (snapshots[i - 1].t <= renderT) { a = snapshots[i - 1]; b = snapshots[i]; break; }
  }
  const span = b.t - a.t;
  const u = span > 1e-6 ? Math.min(1, Math.max(0, (renderT - a.t) / span)) : 1;
  const lerp = (x, y) => x + (y - x) * u;
  const pa = a.ball.pos, pb = b.ball.pos;
  const jump = Math.hypot(pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]) > 0.6;   // teleport (new serve)
  return {
    ...b,
    ball: { ...b.ball, pos: jump ? pb : pa.map((v, i) => lerp(v, pb[i])) },
    opponent: { ...b.opponent, x: lerp(a.opponent.x, b.opponent.x),
                swing_t: b.opponent.swing_t - (b.t - renderT) },
    paddle: { pitch: lerp(a.paddle.pitch, b.paddle.pitch), roll: lerp(a.paddle.roll, b.paddle.roll),
              yaw: lerp(a.paddle.yaw, b.paddle.yaw) },
  };
}

// ---------------------------------------------------------------- frame loop
const clock = new THREE.Clock();
let fps = 60, slowFor = 0, lastBeat = -1;
const sideColorFH = new THREE.Color(THEME.effects.hintForehand);
const sideColorBH = new THREE.Color(THEME.effects.hintBackhand);

function frame() {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, clock.getDelta());
  fps += ((1 / Math.max(dt, 1e-3)) - fps) * 0.05;
  if (bloomOn && fps < 42) { slowFor += dt; if (slowFor > 3) { bloomOn = false; hud.toast('Bloom off (low frame rate)'); } }
  else slowFor = 0;

  const renderT = clockOffset === null ? 0 : performance.now() / 1000 - clockOffset - INTERP_DELAY;
  const s = latest && !latest.paused ? sample(renderT) : latest;

  if (s) {
    const ballPos = toScene(s.ball.pos);
    const spin = spinToScene(s.ball.spin);
    ballView.update(ballPos, spin, topspinOf(s.ball.vel, s.ball.spin), sidespinOf(s.ball.vel, s.ball.spin),
      s.ball.visible, s.paused ? 0 : dt);
    opponentView.setCharacter(latest.speed_setting);   // one athlete per speed setting
    opponentView.update(s.opponent, dt);
    const inc = s.incoming;
    paddleView.update(s.paddle, inc ? inc.x : null, dt);

    // Target ring + side highlight for the incoming ball.
    const hintOn = s.settings.hint && s.phase === 'RALLY';
    const tToArr = inc ? inc.t_to_arrival - (latest.t - s.t) : null;
    effects.target(inc ? new THREE.Vector3(inc.x, TABLE_H + 0.25, -PLAYER_PLANE_Z) : null, tToArr,
      s.settings.hit_window_s, hintOn);
    const req = s.required_stroke;
    if (hintOn && inc && req && req !== 'either') {
      sideHint.position.x = inc.x > 0 ? 1.525 / 4 : -1.525 / 4;
      sideHint.material.color.copy(req === 'forehand' ? sideColorFH : sideColorBH);
      sideHint.material.opacity += (0.06 - sideHint.material.opacity) * Math.min(1, dt * 10);
    } else {
      sideHint.material.opacity *= 0.85;
    }

    // Gentle camera sway toward the ball (centered at rest, so the screen's
    // vertical centerline stays on the table's center line).
    if (!orbit) {
      const sway = s.ball.visible ? ballPos.x * cam.sway : 0;
      camera.position.x += (sway - camera.position.x) * Math.min(1, dt * 3);
      camera.lookAt(camera.position.x, TABLE_H, cam.lookAtZ);
    }

    // Latency calibration beat.
    if (s.phase === 'LATENCY_CAL' && s.latency) {
      const beats = s.latency.beats;
      let nearest = beats[0];
      for (const b of beats) if (Math.abs(b - renderT) < Math.abs(nearest - renderT)) nearest = b;
      const pulse = Math.exp(-(((renderT - nearest) / 0.09) ** 2));
      const idx = beats.findIndex(b => b > renderT) - 1;
      if (idx !== lastBeat && idx >= 0 && renderT - beats[idx] < 0.1) { sounds.beat(); }
      lastBeat = idx;
      hud.latency(s, renderT, pulse);
    }

    // Debug overlay.
    const dbg = latest.debug;
    debugGroup.visible = !!dbg;
    if (dbg) {
      const v = new THREE.Vector3(s.ball.vel[0], s.ball.vel[1], -s.ball.vel[2]);
      velArrow.position.copy(ballPos);
      if (v.length() > 0.01) { velArrow.setDirection(v.clone().normalize()); velArrow.setLength(Math.min(1.2, v.length() * 0.1)); }
      arrivalMarker.visible = !!dbg.arrival;
      if (dbg.arrival) arrivalMarker.position.copy(toScene(dbg.arrival));
      cpuMarker.visible = !!dbg.cpu_plan;
      if (dbg.cpu_plan) cpuMarker.position.copy(toScene(dbg.cpu_plan));
      const f = (x, n = 2) => (x === null || x === undefined ? '--' : Number(x).toFixed(n));
      hud.debug(
        `fps          ${f(fps, 0)}  bloom ${bloomOn ? 'on' : 'off'}\n` +
        `phase        ${latest.phase}\n` +
        `ball speed   ${f(dbg.speed)} m/s   topspin ${f(dbg.topspin, 0)} rad/s\n` +
        `arrival      ${dbg.arrival ? `x ${f(dbg.arrival[0])}  y ${f(dbg.arrival[1])}` : '--'}\n` +
        `t to arrival ${f(dbg.t_to_arrival)} s   window ${dbg.window_open ? 'OPEN' : 'closed'} (±${dbg.hit_window_s}s)\n` +
        `required     ${latest.required_stroke ?? '--'}\n` +
        `paddle       pitch ${f(s.paddle.pitch, 1)}  roll ${f(s.paddle.roll, 1)}  yaw ${f(s.paddle.yaw, 1)}\n` +
        `latency off  ${f(dbg.latency_offset_s * 1000, 0)} ms`);
    } else {
      hud.debug(null);
    }
  }

  effects.update(dt, camera);
  if (orbit) orbit.update();
  if (bloomOn && composer) composer.render(dt);
  else renderer.render(scene, camera);
}
frame();
