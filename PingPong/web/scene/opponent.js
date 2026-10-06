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

// Swing keyframes, in torso space (x = the athlete's left, y up, z toward the player):
//   [u, torsoYaw, wrist target xyz, elbow pole xyz, paddle-face normal xyz]
// The arm is solved with two-bone IK, so only the hand's path is keyframed.
const READY = [0.0, 0.0, -0.17, 0.26, 0.34, -0.6, -1, -0.4, 0.15, 0.1, 1];
const FOREHAND = [
  READY,
  [0.35, -0.8, -0.40, 0.02, 0.02, -0.7, -1, -0.5, -0.2, 0.3, 1],    // wind-up: low and back, shoulders turned
  [0.6, 0.35, -0.14, 0.30, 0.42, -0.6, -1, -0.1, 0.1, 0.4, 1],      // contact out front, face closing (topspin)
  [1.0, 0.55, 0.12, 0.52, 0.22, -0.3, -0.6, -0.6, 0.6, 0.5, 0.6],   // follow-through up by the left shoulder
];
const BACKHAND = [
  READY,
  [0.35, 0.5, 0.02, 0.12, 0.22, -1, -0.4, 0.3, 0.2, 0.2, 1],        // paddle tucked in front, elbow out
  [0.6, -0.15, -0.14, 0.30, 0.42, -1, -0.3, 0.2, 0.0, 0.3, 1],      // flick through the ball
  [1.0, -0.35, -0.36, 0.40, 0.32, -1, -0.2, 0.0, -0.4, 0.4, 0.8],   // finish out to the right
];
const UPPER_ARM = 0.28, FOREARM = 0.245;
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

// ---------------------------------------------------------------- head + face
// The head is a lathe (egg shape, narrower jaw) whose UVs wrap like a globe with the
// face centered at u = 0.5, so features can be painted onto a canvas texture.
const HEAD_R = 0.1, HEAD_HH = 0.112, HEAD_CY = 0.1, HEAD_SX = 0.88, HEAD_ROWS = 40;
const FACE_W = 1024, FACE_H = 512;
const headTheta = (j) => Math.PI * (1 - j / HEAD_ROWS);           // polar angle from the top
const headRadius = (th) => {
  const low = Math.max(0, -Math.cos(th));                         // 0 above the equator, 1 at the chin
  return Math.sin(th) * HEAD_R * (1 - 0.3 * low * low);
};
function headGeometry() {
  const pts = [];
  for (let j = 0; j <= HEAD_ROWS; j++) {
    const th = headTheta(j);
    pts.push(new THREE.Vector2(Math.max(headRadius(th), 1e-4), Math.cos(th) * HEAD_HH));
  }
  return new THREE.LatheGeometry(pts, 64, Math.PI);                 // phiStart = PI puts u = 0.5 at the front
}
// Head-local point (x toward the athlete's left, y up from head center) -> canvas px.
function facePx(x, y) {
  const th = Math.acos(Math.max(-1, Math.min(1, y / HEAD_HH)));
  const r = headRadius(th) * HEAD_SX;
  const ang = Math.asin(Math.max(-1, Math.min(1, x / Math.max(r, 1e-4))));
  return [(0.5 + ang / (2 * Math.PI)) * FACE_W, (th / Math.PI) * FACE_H];
}
const PX_X = FACE_W / (2 * Math.PI * HEAD_R * HEAD_SX);           // canvas px per meter near the face
const PX_Y = FACE_H / (Math.PI * HEAD_HH);

function faceTexture(spec) {
  const c = document.createElement('canvas');
  c.width = FACE_W; c.height = FACE_H;
  const g = c.getContext('2d');
  const skin = new THREE.Color(spec.skin);
  const shade = (k) => `#${skin.clone().multiplyScalar(k).getHexString()}`;
  g.fillStyle = spec.skin;
  g.fillRect(0, 0, FACE_W, FACE_H);

  const soft = (x, y, rx, ry, color, alpha) => {           // soft radial blob
    const [cx, cy] = facePx(x, y);
    g.save();
    g.translate(cx, cy);
    g.scale(rx * PX_X, ry * PX_Y);
    const grad = g.createRadialGradient(0, 0, 0, 0, 0, 1);
    grad.addColorStop(0, color);
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    g.globalAlpha = alpha;
    g.fillStyle = grad;
    g.beginPath(); g.arc(0, 0, 1, 0, Math.PI * 2); g.fill();
    g.restore();
  };

  // Gentle modelling: eye sockets, cheeks, under the nose and lower lip, jawline.
  for (const s of [-1, 1]) {
    soft(s * 0.033, 0.012, 0.024, 0.016, shade(0.82), 0.55);  // eye socket
    soft(s * 0.042, -0.03, 0.022, 0.016, '#e8908a', 0.18);    // cheek warmth
  }
  soft(0, -0.03, 0.012, 0.006, shade(0.72), 0.5);              // under the nose
  soft(0, -0.066, 0.016, 0.007, shade(0.8), 0.45);             // under the lip
  soft(0, -0.1, 0.06, 0.02, shade(0.82), 0.35);                // jaw shadow

  // Brows.
  g.strokeStyle = spec.brows;
  g.lineCap = 'round';
  for (const s of [-1, 1]) {
    const a = facePx(s * 0.017, 0.03), b = facePx(s * 0.034, 0.035), e = facePx(s * 0.05, 0.028);
    g.lineWidth = 0.0045 * PX_Y;
    g.globalAlpha = 0.85;
    g.beginPath(); g.moveTo(a[0], a[1]); g.quadraticCurveTo(b[0], b[1] - 3, e[0], e[1]); g.stroke();
  }
  g.globalAlpha = 1;

  // Eyes: almond-shaped white, brown iris, pupil, catch-light, upper lid line, crease.
  for (const s of [-1, 1]) {
    const [cx, cy] = facePx(s * 0.033, 0.008);
    const w = 0.0125 * PX_X, h = 0.0052 * PX_Y;
    const almond = () => {
      g.beginPath();
      g.moveTo(cx - w, cy);
      g.quadraticCurveTo(cx, cy - h * 2.0, cx + w, cy);
      g.quadraticCurveTo(cx, cy + h * 1.5, cx - w, cy);
      g.closePath();
    };
    g.save();
    almond();
    g.fillStyle = '#efe7de';
    g.fill();
    g.clip();
    g.fillStyle = '#4a3020';
    g.beginPath(); g.arc(cx, cy, h * 1.35, 0, Math.PI * 2); g.fill();      // iris
    g.fillStyle = '#120b07';
    g.beginPath(); g.arc(cx, cy, h * 0.6, 0, Math.PI * 2); g.fill();       // pupil
    g.fillStyle = 'rgba(0,0,0,0.25)';
    g.fillRect(cx - w, cy - h * 2, w * 2, h * 0.9);                          // lid shadow on the eye
    g.restore();
    g.fillStyle = 'rgba(255,255,255,0.9)';
    g.beginPath(); g.arc(cx + h * 0.45, cy - h * 0.45, h * 0.3, 0, Math.PI * 2); g.fill();
    g.strokeStyle = '#2a1a12';
    g.lineWidth = 3;
    g.beginPath(); g.moveTo(cx - w * 1.05, cy + 1); g.quadraticCurveTo(cx, cy - h * 2.05, cx + w * 1.05, cy); g.stroke();
    g.strokeStyle = shade(0.7);
    g.lineWidth = 2;
    g.globalAlpha = 0.6;
    g.beginPath(); g.moveTo(cx - w * 0.8, cy - h * 1.6); g.quadraticCurveTo(cx, cy - h * 3.0, cx + w * 0.85, cy - h * 1.5); g.stroke();
    g.globalAlpha = 1;
  }

  // Nostrils.
  for (const s of [-1, 1]) soft(s * 0.007, -0.026, 0.004, 0.0025, shade(0.45), 0.7);

  // Lips: soft rosy shape with a darker parting line.
  const [mx, my] = facePx(0, -0.052);
  const lw = 0.019 * PX_X, lh = 0.0045 * PX_Y;
  const lip = skin.clone().lerp(new THREE.Color('#b0524a'), 0.45);
  g.fillStyle = `#${lip.getHexString()}`;
  g.globalAlpha = 0.85;
  g.beginPath();
  g.moveTo(mx - lw, my);
  g.quadraticCurveTo(mx - lw * 0.4, my - lh * 1.6, mx, my - lh * 0.9);
  g.quadraticCurveTo(mx + lw * 0.4, my - lh * 1.6, mx + lw, my);
  g.quadraticCurveTo(mx, my + lh * 2.2, mx - lw, my);
  g.fill();
  g.globalAlpha = 1;
  g.strokeStyle = `#${lip.clone().multiplyScalar(0.55).getHexString()}`;
  g.lineWidth = 2.5;
  g.beginPath(); g.moveTo(mx - lw * 0.95, my); g.quadraticCurveTo(mx, my + lh * 0.5, mx + lw * 0.95, my); g.stroke();

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

// ---------------------------------------------------------------- two-bone arm IK
const _v = () => new THREE.Vector3();
const IK = { dir: _v(), pole: _v(), upper: _v(), fore: _v(), x: _v(), y: _v(), z: _v(), m: new THREE.Matrix4(),
             q: new THREE.Quaternion(), qe: new THREE.Quaternion() };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// Point the arm's wrist at `target` (torso space) with the elbow bending toward
// `pole`. Returns the forearm direction (torso space) for orienting the paddle.
function solveArm(arm, target, pole) {
  const S = arm.shoulder.position;
  const d0 = IK.dir.copy(target).sub(S);
  const d = clamp(d0.length(), Math.abs(UPPER_ARM - FOREARM) + 1e-3, UPPER_ARM + FOREARM - 1e-3);
  const dir = d0.normalize();
  const a = Math.acos(clamp((UPPER_ARM ** 2 + d * d - FOREARM ** 2) / (2 * UPPER_ARM * d), -1, 1));
  const b = Math.acos(clamp((UPPER_ARM ** 2 + FOREARM ** 2 - d * d) / (2 * UPPER_ARM * FOREARM), -1, 1));
  const p = IK.pole.copy(pole).addScaledVector(dir, -pole.dot(dir));
  if (p.lengthSq() < 1e-6) p.set(0, -1, 0).addScaledVector(dir, dir.y);
  p.normalize();
  const upper = IK.upper.copy(dir).multiplyScalar(Math.cos(a)).addScaledVector(p, Math.sin(a));
  // Forearm runs from the elbow to the target.
  const fore = IK.fore.copy(dir).multiplyScalar(d).addScaledVector(upper, -UPPER_ARM).normalize();
  const y = IK.y.copy(upper).negate();                                   // local +y runs up the arm
  const z = IK.z.copy(fore).addScaledVector(upper, -fore.dot(upper));    // bend direction
  if (z.lengthSq() < 1e-6) z.copy(p).negate();
  z.normalize();
  const x = IK.x.crossVectors(y, z).normalize();
  arm.shoulder.quaternion.setFromRotationMatrix(IK.m.makeBasis(x, y, z));
  arm.elbow.rotation.set(-(Math.PI - b), 0, 0);
  return fore;
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
  const head = add(rig.head, headGeometry(), new THREE.MeshStandardMaterial({ map: faceTexture(spec), roughness: 0.6 }), 0, HEAD_CY, 0);
  head.scale.set(HEAD_SX, 1, 1);
  for (const sx of [-1, 1]) {
    const ear = add(rig.head, new THREE.SphereGeometry(0.022, 12, 10), skin, sx * 0.087, 0.098, -0.005);
    ear.scale.set(0.4, 1, 0.72);
  }
  // A soft nose bump so the profile isn't flat; the painted shading does the rest.
  const nose = add(rig.head, new THREE.SphereGeometry(0.013, 14, 10), skin, 0, HEAD_CY - 0.012, 0.096);
  nose.scale.set(0.85, 1.5, 1.0);

  // ---- hair ----
  const cap = add(rig.head, new THREE.SphereGeometry(0.109, 32, 20, 0, Math.PI * 2, 0, Math.PI * 0.5), hairMat, 0, 0.104, -0.006);
  cap.scale.set(0.93, 1.12, 1.0);
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
  } else if (spec.hairStyle === 'bob') {
    // Jaw-length bob: a shell around the sides and back, open over the face.
    const bob = add(rig.head, new THREE.SphereGeometry(0.112, 32, 20, Math.PI / 2 + 0.95, Math.PI * 2 - 1.9, 0, Math.PI * 0.66), hairMat, 0, 0.104, -0.004);
    bob.scale.set(0.95, 1.1, 1.0);
    bob.material = hairMat.clone();
    bob.material.side = THREE.DoubleSide;
    const fringe = add(rig.head, new THREE.SphereGeometry(0.111, 24, 8, -Math.PI * 0.32, Math.PI * 0.64, Math.PI * 0.17, Math.PI * 0.13), hairMat, 0, 0.104, 0.002);
    fringe.scale.set(0.94, 1.1, 1.0);
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
    const shoulder = pivot(rig.torso, s * 0.185, 0.45, 0);
    add(shoulder, new THREE.SphereGeometry(0.05, 16, 12), shirt);
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
  rig.paddle = pivot(rig.rArm.wrist, 0, -0.045, 0);
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
    this._target = new THREE.Vector3();
    this._pole = new THREE.Vector3();
    this._px = new THREE.Vector3();
    this._py = new THREE.Vector3();
    this._pz = new THREE.Vector3();
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

    // Swing: keyframed hand path + torso turn; arms solved with IK every frame.
    const t = state && state.swing ? state.swing_t : 99;
    const swinging = t >= 0 && t < SWING_DURATION;
    const kf = swinging ? keyframe(state.swing === 'forehand' ? FOREHAND : BACKHAND, t / SWING_DURATION) : READY.slice(1);
    const [torsoYaw, hx, hy, hz, px, py, pz, nx, ny, nz] = kf;
    const k = Math.min(1, dt * 30);
    r.hips.rotation.y += (torsoYaw * 0.35 - r.hips.rotation.y) * k;
    r.torso.rotation.x = st.lean;
    r.torso.rotation.y += (torsoYaw * 0.65 - r.torso.rotation.y) * k;
    r.torso.rotation.z += ((-vx * 0.05) - r.torso.rotation.z) * Math.min(1, dt * 6);

    // Paddle hand. A little idle sway keeps the ready position alive.
    const sway = swinging ? 0 : Math.sin(this.time * 1.7) * 0.012;
    this._target.set(hx + sway, hy + bounce * 2 + Math.sin(this.time * 2.3) * 0.008, hz);
    this._pole.set(px, py, pz);
    const fore = solveArm(r.rArm, this._target, this._pole);

    // Paddle: handle continues the forearm; forehand rubber (red) faces the ball on
    // forehands, the backhand rubber (black) on backhands.
    const side = state && state.swing === 'backhand' && swinging ? 1 : -1;
    const py_ = this._py.copy(fore).negate();
    const px_ = this._px.set(nx, ny, nz).multiplyScalar(side);
    px_.addScaledVector(py_, -px_.dot(py_)).normalize();
    const pz_ = this._pz.crossVectors(px_, py_);
    const want = IK.q.setFromRotationMatrix(IK.m.makeBasis(px_, py_, pz_));
    const parent = IK.qe.copy(r.rArm.shoulder.quaternion).multiply(r.rArm.elbow.quaternion);
    r.paddle.quaternion.copy(parent.invert().multiply(want));

    // Free arm: out front for balance, drawn back as the body turns into the stroke.
    this._target.set(0.17 + torsoYaw * 0.06 - sway, 0.25 + bounce * 2, 0.3 - Math.abs(torsoYaw) * 0.12);
    this._pole.set(0.6, -1, -0.4);
    solveArm(r.lArm, this._target, this._pole);
    r.head.rotation.x = -st.lean * 0.6;
    r.head.rotation.y = -r.torso.rotation.y * 0.8;           // eyes stay on the ball
    if (r.ponytail) {
      r.ponytail.rotation.x = 0.45 + bounce * 8;
      r.ponytail.rotation.z += ((vx * 0.25 + r.torso.rotation.y * 0.6) - r.ponytail.rotation.z) * Math.min(1, dt * 5);
    }
  }
}
