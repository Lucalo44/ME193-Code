// effects.js -- understated, broadcast-style effects: soft bounce scuffs, a white
// contact flash, a thin target ring, net shake, and synthesized sounds
// (celluloid "tock"s, a wooden paddle "pock", crowd applause on points).
import * as THREE from 'three';
import { THEME, TABLE_H } from './theme.js';

function softDot() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

export class Effects {
  constructor(scene, netMesh) {
    this.scene = scene;
    this.net = netMesh;
    this.netBaseY = netMesh.position.y;
    this.marks = [];
    this.flashes = [];
    this.netShake = 0;
    this.dot = softDot();

    this.ring = new THREE.Mesh(new THREE.RingGeometry(0.1, 0.108, 48),
      new THREE.MeshBasicMaterial({ color: THEME.effects.targetRing, transparent: true, opacity: 0.6,
        side: THREE.DoubleSide, depthWrite: false }));
    this.ring.visible = false;
    scene.add(this.ring);
  }

  bounce(p) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(0.07, 0.07),
      new THREE.MeshBasicMaterial({ map: this.dot, color: THEME.effects.bounceMark, transparent: true,
        opacity: 0.35, depthWrite: false }));
    m.rotation.x = -Math.PI / 2;
    m.position.set(p.x, TABLE_H + 0.002, p.z);
    this.scene.add(m);
    this.marks.push({ mesh: m, age: 0 });
  }

  hit(p, color = THEME.effects.hitFlash) {
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.dot, color, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0.9 }));
    s.position.copy(p);
    s.scale.setScalar(0.12);
    this.scene.add(s);
    this.flashes.push({ s, age: 0 });
  }

  netHit() { this.netShake = 1; }

  // arrival: scene Vector3 or null; tToArrival seconds until the ball reaches it;
  // win = [early, late]: the hit window (seconds before / after arrival).
  target(arrival, tToArrival, win, visible) {
    const [early, late] = win || [0.2, 0.2];
    if (!arrival || !visible || tToArrival === null || tToArrival < -late) { this.ring.visible = false; return; }
    this.ring.visible = true;
    this.ring.position.copy(arrival);
    this.ring.scale.setScalar(0.5 + Math.max(0, tToArrival) * 1.2);
    const open = tToArrival <= early && tToArrival >= -late;
    this.ring.material.color.set(open ? THEME.effects.targetRingOpen : THEME.effects.targetRing);
    this.ring.material.opacity = open ? 0.9 : 0.3;
  }

  update(dt, camera) {
    for (const m of this.marks) {
      m.age += dt;
      m.mesh.material.opacity = Math.max(0, 0.35 * (1 - m.age / 1.2));
    }
    this.marks = this.marks.filter((m) => {
      if (m.age < 1.2) return true;
      this.scene.remove(m.mesh); m.mesh.geometry.dispose(); m.mesh.material.dispose();
      return false;
    });
    for (const f of this.flashes) {
      f.age += dt;
      f.s.scale.setScalar(0.12 + f.age * 0.9);
      f.s.material.opacity = Math.max(0, 0.9 * (1 - f.age / 0.18));
    }
    this.flashes = this.flashes.filter((f) => {
      if (f.age < 0.18) return true;
      this.scene.remove(f.s); f.s.material.dispose();
      return false;
    });
    this.netShake = Math.max(0, this.netShake - dt * 4);
    this.net.position.y = this.netBaseY + Math.sin(performance.now() * 0.06) * 0.004 * this.netShake;
    if (this.ring.visible && camera) this.ring.lookAt(camera.position);
  }
}

// --- Sound: small WebAudio synths, no audio files needed. ---
export class Sounds {
  constructor() { this.ctx = null; this.enabled = true; }

  unlock() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === 'suspended') this.ctx.resume();
  }

  _tone(freq, dur, type = 'sine', gain = 0.25, slideTo = null, delay = 0) {
    if (!this.ctx || !this.enabled) return;
    const t = this.ctx.currentTime + delay;
    const o = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, t);
    if (slideTo) o.frequency.exponentialRampToValueAtTime(slideTo, t + dur);
    g.gain.setValueAtTime(gain, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(this.ctx.destination);
    o.start(t); o.stop(t + dur + 0.02);
  }

  _noise(dur, freq, q, gain, type = 'bandpass', shape = 3, delay = 0) {
    if (!this.ctx || !this.enabled) return;
    const t = this.ctx.currentTime + delay;
    const n = Math.floor(this.ctx.sampleRate * dur);
    const buf = this.ctx.createBuffer(1, n, this.ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / n, shape);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    const f = this.ctx.createBiquadFilter();
    f.type = type; f.frequency.value = freq; f.Q.value = q;
    const g = this.ctx.createGain();
    g.gain.value = gain;
    src.connect(f).connect(g).connect(this.ctx.destination);
    src.start(t);
  }

  // Paddle: wooden "pock" -- louder and lower for harder hits.
  hit(strength = 0.5) {
    this._noise(0.03, 1400 - strength * 300, 2.5, 0.6 + strength * 0.5);
    this._tone(900 - strength * 150, 0.05, 'sine', 0.18);
  }
  // Table: bright celluloid "tock".
  bounce() { this._noise(0.02, 3600, 4, 0.45); this._tone(2300, 0.03, 'sine', 0.08); }
  net() { this._noise(0.12, 300, 1, 0.5, 'lowpass', 2); }
  miss() { this._noise(0.5, 500, 0.7, 0.12, 'lowpass', 1.5); }   // murmur
  point(good) { if (good) this.applause(0.9); else this._noise(0.6, 400, 0.6, 0.1, 'lowpass', 1.2); }
  applause(level = 1) {
    // Many short claps, randomly spaced, under a soft swell.
    this._noise(1.6, 2000, 0.4, 0.06 * level, 'bandpass', 1.2);
    for (let i = 0; i < 40; i++) this._noise(0.025, 1500 + Math.random() * 1500, 1.5, 0.12 * level * Math.random(), 'bandpass', 4, Math.random() * 1.4);
  }
  beat() { this._tone(1000, 0.06, 'sine', 0.12); }
  toss() { this._noise(0.05, 900, 1.2, 0.18, 'bandpass', 2); }
  tag() { this._tone(660, 0.08, 'triangle', 0.1); }
}
