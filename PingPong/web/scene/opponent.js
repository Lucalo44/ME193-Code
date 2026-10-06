// opponent.js -- the computer opponent: one athlete per speed setting (see OPPONENTS
// in theme.js), built from primitives -- lathed torso, tapered limbs with joints,
// polo collar, a face with eyes/brows/nose/mouth, three hairstyles, shorts or skirt.
//
// It faces the player (+z in scene coords). Its right hand is on scene -x, so its
// forehand side is -x, matching game/opponent.py.
import * as THREE from 'three';
import { THEME, opponentFor } from './theme.js';

const SWING_DURATION = 0.45;
const REF_HEIGHT = 1.75;                 // the rig is modelled at this height, then scaled
const THIGH = 0.43, SHIN = 0.43, FOOT_H = 0.08;

const lerp = (a, b, u) => a + (b - a) * u;
const smooth = (u) => u * u * (3 - 2 * u);

// Swing keyframes: [u, torsoYaw, shoulderYaw, shoulderPitch, elbow, wrist]
const FOREHAND = [
  [0.0, 0.0, -0.3, -1.0, -1.0, 0.0],
  [0.35, -0.8, -1.15, -0.7, -0.55, 0.35],   // wind-up: shoulders turn right, paddle back and low
  [0.6, 0.35, 0.4, -1.3, -1.2, -0.2],       // contact, brushing up
  [1.0, 0.55, 0.95, -1.75, -1.65, -0.4],    // follow-through over the left shoulder
];
const BACKHAND = [
  [0.0, 0.0, -0.3, -1.0, -1.0, 0.0],
  [0.35, 0.55, 0.95, -1.0, -1.75, 0.5],     // paddle tucked in front of the body
  [0.6, -0.2, -0.3, -1.25, -0.9, -0.3],     // flick out
  [1.0, -0.35, -0.8, -1.45, -0.6, -0.5],
];
function keyframe(frames, u) {
  for (let i = 1; i < frames.length; i++) {
    if (u <= frames[i][0]) {
      const a = frames[i - 1], b = frames[i];
      const k = smooth((u - a[0]) / (b[0] - a[0]));
      return a.map((v, j) => lerp(v, b[j], k)).slice(1);
    }
  }
  return frames[frames.length - 1].slice(1);
}

// ---------------------------------------------------------------- materials
function fabric(color) {
  const c = new THREE.Color(color);
  return new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.85, sheen: 0.6, sheenRoughness: 0.8,
    sheenColor: c.clone().lerp(new THREE.Color('#ffffff'), 0.35) });
}
const plain = (color, roughness = 0.6, metalness = 0) => new THREE.MeshStandardMaterial({ color, roughness, metalness });

// ---------------------------------------------------------------- helpers
function pivot(parent, x = 0, y = 0, z = 0) {
  const g = new THREE.Group();
  g.position.set(x, y, z);
  g.rotation.order = 'YXZ';
  parent.add(g);
  return g;
}
function add(parent, geo, material, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(geo, material);
  m.position.set(x, y, z);
  m.castShadow = true;
  m.receiveShadow = true;
  parent.add(m);
  return m;
}
// Tapered limb segment hanging down from its pivot.
function segment(parent, rTop, rBottom, length, material) {
  return add(parent, new THREE.CylinderGeometry(rBottom, rTop, length, 18, 1), material, 0, -length / 2, 0);
}
// Revolved body part from a (radius, y) profile, squashed front-to-back.
function lathe(parent, profile, material, depth = 0.7, width = 1.0) {
  const pts = profile.map(([r, y]) => new THREE.Vector2(r, y));
  const m = add(parent, new THREE.LatheGeometry(pts, 32), material);
  m.scale.set(width, 1, depth);
  return m;
}

// ---------------------------------------------------------------- the athlete
function buildAthlete(spec) {
  const o = THEME.opponent;
  const skin = new THREE.MeshPhysicalMaterial({ color: spec.skin, roughness: 0.5, sheen: 0.3,
    sheenColor: new THREE.Color('#ffd9c0') });
  const shirt = fabric(spec.shirt), trim = fabric(spec.trim), bottom = fabric(spec.bottomColor);
  const hairMat = plain(spec.hair, 0.75);
  const rig = { spec };

  rig.root = new THREE.Group();
  rig.root.scale.setScalar(spec.heightM / REF_HEIGHT);
  rig.hips = pivot(rig.root, 0, 0.9, 0);

  // ---- legs ----
  rig.legs = [];
  for (const s of [-1, 1]) {
    const thigh = pivot(rig.hips, s * 0.095, -0.02, 0);
    segment(thigh, 0.078, 0.054, THIGH, skin);
    const knee = pivot(thigh, 0, -THIGH, 0);
    add(knee, new THREE.SphereGeometry(0.054, 14, 10), skin);
    segment(knee, 0.052, 0.034, SHIN, skin);
    const calf = add(knee, new THREE.SphereGeometry(0.05, 14, 10), skin, 0, -0.13, -0.012);
    calf.scale.set(1.0, 2.3, 1.1);
    add(knee, new THREE.CylinderGeometry(0.039, 0.036, 0.1, 14), plain(spec.socks, 0.9), 0, -SHIN + 0.06, 0);
    const ankle = pivot(knee, 0, -SHIN, 0);
    add(ankle, new THREE.BoxGeometry(0.1, 0.022, 0.27), plain('#f2f2f2', 0.8), 0, -FOOT_H + 0.011, 0.045);  // sole
    const upper = add(ankle, new THREE.CapsuleGeometry(0.045, 0.16, 6, 14), plain(spec.shoes, 0.55), 0, -0.04, 0.045);
    upper.rotation.x = Math.PI / 2;
    upper.scale.set(1.05, 1, 0.75);
    add(ankle, new THREE.BoxGeometry(0.03, 0.012, 0.08), plain(spec.shoeAccent, 0.5), s * 0.044, -0.045, 0.05);   // side stripe
    rig.legs.push({ thigh, knee, ankle, side: s });
  }

  // ---- shorts / skirt (on the hips, so they follow the pelvis) ----
  if (spec.bottom === 'skirt') {
    lathe(rig.hips, [[0.0, 0.09], [0.155, 0.09], [0.16, 0.05], [0.195, -0.08], [0.225, -0.17]], bottom, 0.72, 1.08);
  } else {
    lathe(rig.hips, [[0.0, 0.09], [0.155, 0.09], [0.162, 0.0], [0.172, -0.08], [0.0, -0.08]], bottom, 0.72, 1.08);
    for (const s of [-1, 1]) {
      const leg = add(rig.hips, new THREE.CylinderGeometry(0.088, 0.096, 0.17, 18, 1, true), bottom, s * 0.095, -0.12, 0);
      leg.material = bottom.clone();
      leg.material.side = THREE.DoubleSide;
      rig.legs.find((l) => l.side === s).shortsLeg = leg;
    }
  }

  // ---- torso ----
  rig.torso = pivot(rig.hips, 0, 0.04, 0);
  lathe(rig.torso, [[0.0, 0.04], [0.14, 0.04], [0.136, 0.12], [0.155, 0.24], [0.176, 0.36],
    [0.172, 0.45], [0.13, 0.52], [0.07, 0.55], [0.0, 0.555]], shirt, 0.66, 1.08);
  // Hem trim, chest stripe, placket, collar.
  const band = add(rig.torso, new THREE.CylinderGeometry(0.179, 0.177, 0.03, 32, 1, true), trim, 0, 0.33, 0);
  band.scale.set(1.085, 1, 0.665);
  add(rig.torso, new THREE.BoxGeometry(0.03, 0.11, 0.006), trim, 0, 0.475, 0.098);                // placket
  for (const s of [-1, 1]) {
    const flap = add(rig.torso, new THREE.BoxGeometry(0.07, 0.006, 0.05), trim, s * 0.035, 0.548, 0.06);
    flap.rotation.set(0.5, s * 0.5, s * 0.35);
  }
  const crest = add(rig.torso, new THREE.CircleGeometry(0.022, 16), plain(spec.accent, 0.5), 0.07, 0.42, 0.12);
  crest.rotation.x = -0.25;

  // ---- neck + head ----
  add(rig.torso, new THREE.CylinderGeometry(0.044, 0.05, 0.11, 16), skin, 0, 0.6, 0.005);
  rig.head = pivot(rig.torso, 0, 0.64, 0.01);
  const skull = add(rig.head, new THREE.SphereGeometry(0.1, 28, 22), skin, 0, 0.1, 0);
  skull.scale.set(0.9, 1.08, 0.98);
  const jaw = add(rig.head, new THREE.SphereGeometry(0.075, 22, 16), skin, 0, 0.045, 0.02);
  jaw.scale.set(0.86, 0.72, 0.9);
  for (const s of [-1, 1]) {
    const ear = add(rig.head, new THREE.SphereGeometry(0.022, 10, 8), skin, s * 0.09, 0.095, -0.005);
    ear.scale.set(0.45, 1, 0.75);
    const white = add(rig.head, new THREE.SphereGeometry(0.014, 12, 10), plain('#f4f1ea', 0.3), s * 0.034, 0.108, 0.082);
    white.scale.set(1.3, 0.85, 0.6);
    add(rig.head, new THREE.SphereGeometry(0.0085, 10, 8), plain('#1b130e', 0.2), s * 0.034, 0.108, 0.0895);
    const brow = add(rig.head, new THREE.BoxGeometry(0.03, 0.006, 0.008), plain(spec.brows, 0.8), s * 0.035, 0.127, 0.087);
    brow.rotation.z = -s * 0.12;
  }
  const nose = add(rig.head, new THREE.ConeGeometry(0.012, 0.034, 10), skin, 0, 0.085, 0.098);
  nose.rotation.x = Math.PI / 2 + 0.35;
  add(rig.head, new THREE.BoxGeometry(0.03, 0.005, 0.006), plain('#8a4a3c', 0.6), 0, 0.048, 0.086);  // mouth

  // ---- hair ----
  const cap = add(rig.head, new THREE.SphereGeometry(0.106, 28, 18, 0, Math.PI * 2, 0, Math.PI * 0.5), hairMat, 0, 0.103, -0.006);
  cap.scale.set(0.94, 1.05, 1.0);
  cap.rotation.x = -0.55;               // hairline on the forehead, down to the nape at the back
  const nape = add(rig.head, new THREE.SphereGeometry(0.1, 20, 12, Math.PI * 0.6, Math.PI * 0.8, Math.PI * 0.35, Math.PI * 0.3), hairMat, 0, 0.1, -0.006);
  nape.scale.set(0.95, 1.05, 1.0);
  if (spec.hairStyle === 'short') {
    for (const s of [-1, 1]) add(rig.head, new THREE.BoxGeometry(0.008, 0.035, 0.018), hairMat, s * 0.088, 0.1, 0.02);  // sideburns
  } else if (spec.hairStyle === 'ponytail') {
    rig.ponytail = pivot(rig.head, 0, 0.155, -0.085);
    add(rig.ponytail, new THREE.TorusGeometry(0.018, 0.007, 6, 14), plain(spec.accent, 0.5), 0, 0, 0).rotation.x = 1.2;
    const tail = add(rig.ponytail, new THREE.CapsuleGeometry(0.028, 0.15, 6, 12), hairMat, 0, -0.09, -0.02);
    tail.scale.set(1, 1, 0.8);
    rig.ponytail.rotation.x = 0.45;
  } else if (spec.hairStyle === 'buns') {
    for (const s of [-1, 1]) {
      add(rig.head, new THREE.SphereGeometry(0.038, 16, 12), hairMat, s * 0.065, 0.19, -0.03);
      add(rig.head, new THREE.TorusGeometry(0.03, 0.006, 6, 14), plain(spec.accent, 0.5), s * 0.06, 0.17, -0.025)
        .rotation.set(Math.PI / 2, 0, s * 0.6);
    }
    const bangs = add(rig.head, new THREE.SphereGeometry(0.104, 24, 8, -Math.PI * 0.3, Math.PI * 0.6, Math.PI * 0.18, Math.PI * 0.16), hairMat, 0, 0.105, 0.004);
    bangs.scale.set(0.93, 1.06, 1.0);
  }

  // ---- arms ----
  const arm = (s) => {
    const shoulder = pivot(rig.torso, s * 0.195, 0.46, 0);
    add(shoulder, new THREE.SphereGeometry(0.056, 16, 12), shirt);
    const sleeve = segment(shoulder, 0.056, 0.05, 0.14, shirt);
    sleeve.position.y = -0.065;
    const cuff = add(shoulder, new THREE.TorusGeometry(0.05, 0.006, 6, 20), trim, 0, -0.135, 0);
    cuff.rotation.x = Math.PI / 2;
    segment(shoulder, 0.046, 0.037, 0.28, skin);
    const elbow = pivot(shoulder, 0, -0.28, 0);
    add(elbow, new THREE.SphereGeometry(0.038, 12, 10), skin);
    segment(elbow, 0.038, 0.029, 0.24, skin);
    if (spec.wristband) add(elbow, new THREE.CylinderGeometry(0.034, 0.034, 0.05, 14), fabric(spec.wristband), 0, -0.2, 0);
    const wrist = pivot(elbow, 0, -0.245, 0);
    const palm = add(wrist, new THREE.CapsuleGeometry(0.028, 0.035, 6, 10), skin, 0, -0.04, 0);
    palm.scale.set(1.0, 1, 0.65);
    add(wrist, new THREE.CapsuleGeometry(0.011, 0.025, 4, 8), skin, -s * 0.025, -0.035, 0.012).rotation.z = s * 0.6;  // thumb
    return { shoulder, elbow, wrist };
  };
  rig.rArm = arm(-1);
  rig.lArm = arm(1);

  // ---- paddle in the right hand: wood blade, two rubbers, flared handle ----
  rig.paddle = pivot(rig.rArm.wrist, 0, -0.05, 0.005);
  const blade = new THREE.Group();
  blade.position.y = -0.13;
  blade.scale.set(1, 1.05, 1);
  rig.paddle.add(blade);
  const disc = (r, h) => new THREE.CylinderGeometry(r, r, h, 36);
  add(blade, disc(0.077, 0.006), plain(o.paddleWood, 0.6)).rotation.z = Math.PI / 2;
  add(blade, disc(0.075, 0.0035), plain(o.paddleRubber, 0.85), -0.0045, 0, 0).rotation.z = Math.PI / 2;
  add(blade, disc(0.075, 0.0035), plain(o.paddleBack, 0.85), 0.0045, 0, 0).rotation.z = Math.PI / 2;
  add(rig.paddle, new THREE.CylinderGeometry(0.013, 0.016, 0.1, 12), plain(o.paddleHandle, 0.55), 0, -0.02, 0);

  rig.root.traverse((m) => { if (m.isMesh) m.castShadow = true; });
  return rig;
}

function disposeTree(obj) {
  obj.traverse((m) => {
    if (m.geometry) m.geometry.dispose();
    if (m.material) m.material.dispose();
  });
}

// ---------------------------------------------------------------- view
export class OpponentView {
  constructor(scene, hitPlaneZ) {
    this.anchor = new THREE.Group();
    this.anchor.position.set(0, 0, -(hitPlaneZ + THEME.opponent.distanceBehindEnd));
    scene.add(this.anchor);
    this.time = 0;
    this.stride = 0;
    this.key = null;
    this.setCharacter('medium');
  }

  // speedSetting: 'slow' | 'medium' | 'fast' -- swaps the athlete if it changed.
  setCharacter(speedSetting) {
    const spec = opponentFor(speedSetting);
    if (this.rig && this.rig.spec === spec) return;
    if (this.rig) {
      this.anchor.remove(this.rig.root);
      disposeTree(this.rig.root);
    }
    this.rig = buildAthlete(spec);
    this.anchor.add(this.rig.root);
  }

  // state: {x, swing, swing_t}
  update(state, dt) {
    const r = this.rig, st = r.spec.style;
    this.time += dt;
    const x = state ? state.x : 0;
    const vx = this._lastX === undefined ? 0 : (x - this._lastX) / Math.max(dt, 1e-3);
    this._lastX = x;
    this.anchor.position.x = x;

    // Ready stance: crouch depth, bounce and shuffle footwork are per character.
    const moving = Math.min(1, Math.abs(vx) / 1.5);
    this.stride += Math.abs(vx) * dt * 9 * st.footwork;
    const bounce = Math.abs(Math.sin(this.time * Math.PI * st.bounceHz)) * st.bounceAmp;
    const step = Math.sin(this.stride) * 0.22 * moving;
    let lowest = Infinity;
    for (const leg of r.legs) {
      const spread = leg.side * 0.13 + step * leg.side;
      const crouch = st.crouch - moving * 0.06;
      leg.thigh.rotation.set(crouch, 0, spread);
      leg.knee.rotation.set(st.knee + moving * 0.1, 0, -step * leg.side * 0.5);
      leg.ankle.rotation.set(-(crouch + st.knee + moving * 0.1), 0, -spread);    // keep the sole flat
      if (leg.shortsLeg) leg.shortsLeg.rotation.set(crouch * 0.8, 0, spread);
      const a = Math.abs(crouch), b = st.knee + moving * 0.1;
      const h = (THIGH * Math.cos(a) + SHIN * Math.cos(b - a)) * Math.cos(spread) + FOOT_H;
      lowest = Math.min(lowest, h);
    }
    r.hips.position.y = lowest + 0.02 + bounce + Math.abs(Math.sin(this.stride)) * 0.015 * moving;

    // Swing: torso, hips and right arm follow keyframes; otherwise hold the ready pose.
    const t = state && state.swing ? state.swing_t : 99;
    let torsoYaw = 0, sYaw = -0.3, sPitch = -1.0, elbow = -1.0, wrist = 0;
    if (t >= 0 && t < SWING_DURATION) {
      [torsoYaw, sYaw, sPitch, elbow, wrist] = keyframe(state.swing === 'forehand' ? FOREHAND : BACKHAND, t / SWING_DURATION);
    }
    const k = Math.min(1, dt * 30);
    r.hips.rotation.y += (torsoYaw * 0.35 - r.hips.rotation.y) * k;
    r.torso.rotation.x = st.lean;
    r.torso.rotation.y += (torsoYaw * 0.65 - r.torso.rotation.y) * k;
    r.torso.rotation.z += ((-vx * 0.05) - r.torso.rotation.z) * Math.min(1, dt * 6);
    const ra = r.rArm;
    ra.shoulder.rotation.y += (sYaw - ra.shoulder.rotation.y) * k;
    ra.shoulder.rotation.x += (sPitch - ra.shoulder.rotation.x) * k;
    ra.shoulder.rotation.z = 0.25;
    ra.elbow.rotation.x += (elbow - ra.elbow.rotation.x) * k;
    ra.wrist.rotation.x += (wrist - ra.wrist.rotation.x) * k;
    r.paddle.rotation.y = 0.4;
    // Free arm out front for balance, swinging slightly against the stroke.
    r.lArm.shoulder.rotation.set(-0.65, 0.25 - torsoYaw * 0.3, -0.28);
    r.lArm.elbow.rotation.x = -1.35;
    r.head.rotation.x = -st.lean * 0.6;
    r.head.rotation.y = -r.torso.rotation.y * 0.8;           // eyes stay on the ball
    if (r.ponytail) {
      r.ponytail.rotation.x = 0.45 + bounce * 8;
      r.ponytail.rotation.z += ((vx * 0.25 + r.torso.rotation.y * 0.6) - r.ponytail.rotation.z) * Math.min(1, dt * 5);
    }
  }
}
