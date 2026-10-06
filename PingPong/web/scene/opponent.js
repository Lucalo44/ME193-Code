// opponent.js -- the computer opponent as a realistic athlete built from primitives:
// polo shirt, shorts, knee-bent ready stance, shuffle steps, and full-body
// forehand / backhand swings. It faces the player (+z in scene coords). Its right
// hand is on scene -x, so its forehand side is -x, matching game/opponent.py.
import * as THREE from 'three';
import { THEME, TABLE_H } from './theme.js';

const SWING_DURATION = 0.45;

const mat = (color, rough = 0.7) => new THREE.MeshStandardMaterial({ color, roughness: rough });
function pivot(parent, x, y, z) {
  const g = new THREE.Group();
  g.position.set(x, y, z);
  g.rotation.order = 'YXZ';
  parent.add(g);
  return g;
}
function limb(parent, radius, length, material) {
  const m = new THREE.Mesh(new THREE.CapsuleGeometry(radius, length, 4, 10), material);
  m.position.y = -length / 2;
  m.castShadow = true;
  parent.add(m);
  return m;
}
const lerp = (a, b, u) => a + (b - a) * u;
const smooth = (u) => u * u * (3 - 2 * u);

// Swing keyframes: [u, torsoYaw, shoulderYaw, shoulderPitch, elbow]
const FOREHAND = [
  [0.0, 0.0, -0.3, -1.0, -1.0],
  [0.35, -0.75, -1.1, -0.7, -0.6],   // wind-up: shoulders turn right, paddle back
  [0.6, 0.35, 0.4, -1.3, -1.2],      // contact / brush up
  [1.0, 0.55, 0.9, -1.7, -1.6],      // follow-through over the left shoulder
];
const BACKHAND = [
  [0.0, 0.0, -0.3, -1.0, -1.0],
  [0.35, 0.55, 0.9, -1.0, -1.7],     // paddle tucked in front of the body
  [0.6, -0.2, -0.3, -1.25, -0.9],    // flick out
  [1.0, -0.35, -0.8, -1.4, -0.6],
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

export class OpponentView {
  constructor(scene, hitPlaneZ) {
    const o = THEME.opponent;
    const skin = mat(o.skin, 0.6), shirt = mat(o.shirt, 0.8), shorts = mat(o.shorts, 0.8);
    this.root = new THREE.Group();
    this.root.position.set(0, 0, -(hitPlaneZ + o.distanceBehindEnd));
    scene.add(this.root);

    // Legs.
    this.hips = pivot(this.root, 0, 0.86, 0);
    this.legs = [];
    for (const s of [-1, 1]) {
      const thigh = pivot(this.hips, s * 0.1, 0, 0);
      limb(thigh, 0.075, 0.3, shorts).scale.set(1.05, 1, 1.05);
      const knee = pivot(thigh, 0, -0.44, 0);
      limb(knee, 0.055, 0.32, skin);
      const sock = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.048, 0.1, 10), mat(o.socks));
      sock.position.y = -0.36;
      knee.add(sock);
      const shoe = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.07, 0.27), mat(o.shoes, 0.5));
      shoe.position.set(0, -0.43, 0.05);
      shoe.castShadow = true;
      knee.add(shoe);
      this.legs.push({ thigh, knee, side: s });
    }

    // Torso, head.
    this.torso = pivot(this.hips, 0, 0.02, 0);
    const chest = new THREE.Mesh(new THREE.CapsuleGeometry(0.15, 0.3, 6, 14), shirt);
    chest.scale.set(1.15, 1, 0.75);
    chest.position.y = 0.28;
    chest.castShadow = true;
    this.torso.add(chest);
    const waist = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.16, 0.12, 14), shorts);
    waist.scale.set(1.1, 1, 0.78);
    waist.position.y = 0.02;
    this.torso.add(waist);
    const collar = new THREE.Mesh(new THREE.TorusGeometry(0.065, 0.014, 6, 16), mat(o.shirtTrim));
    collar.rotation.x = Math.PI / 2;
    collar.position.y = 0.55;
    this.torso.add(collar);
    const stripe = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.035, 0.24), mat(o.shirtTrim));
    stripe.position.y = 0.36;
    this.torso.add(stripe);
    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.05, 0.1, 10), skin);
    neck.position.y = 0.6;
    this.torso.add(neck);
    this.head = pivot(this.torso, 0, 0.75, 0);
    const skull = new THREE.Mesh(new THREE.SphereGeometry(0.105, 20, 16), skin);
    skull.scale.set(0.92, 1.08, 1);
    skull.castShadow = true;
    this.head.add(skull);
    const hair = new THREE.Mesh(new THREE.SphereGeometry(0.11, 20, 12, 0, Math.PI * 2, 0, Math.PI * 0.55), mat(o.hair, 0.9));
    hair.scale.set(0.95, 1.05, 1.03);
    hair.position.set(0, 0.012, -0.008);
    hair.rotation.x = -0.25;
    this.head.add(hair);
    for (const s of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.011, 8, 6), mat('#111', 0.3));
      eye.position.set(s * 0.035, 0.01, 0.094);
      this.head.add(eye);
      const ear = new THREE.Mesh(new THREE.SphereGeometry(0.022, 8, 6), skin);
      ear.scale.set(0.5, 1, 0.8);
      ear.position.set(s * 0.098, 0, 0);
      this.head.add(ear);
    }
    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.014, 0.035, 6), skin);
    nose.rotation.x = Math.PI / 2;
    nose.position.set(0, -0.01, 0.105);
    this.head.add(nose);

    // Arms. Right (paddle) arm on scene -x.
    const arm = (s) => {
      const shoulder = pivot(this.torso, s * 0.2, 0.5, 0);
      const sleeve = limb(shoulder, 0.052, 0.1, shirt);
      sleeve.position.y = -0.06;
      limb(shoulder, 0.042, 0.22, skin);
      const elbow = pivot(shoulder, 0, -0.29, 0);
      limb(elbow, 0.036, 0.2, skin);
      const hand = new THREE.Mesh(new THREE.SphereGeometry(0.04, 10, 8), skin);
      hand.position.y = -0.27;
      elbow.add(hand);
      return { shoulder, elbow, hand };
    };
    this.rArm = arm(-1);
    this.lArm = arm(1);

    // Paddle in the right hand.
    this.paddle = new THREE.Group();
    this.paddle.position.y = -0.29;
    this.rArm.elbow.add(this.paddle);
    const blade = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.008, 28), mat(o.paddleRubber, 0.6));
    blade.rotation.z = Math.PI / 2;
    blade.position.y = -0.12;
    blade.castShadow = true;
    this.paddle.add(blade);
    const back = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.004, 28), mat(o.paddleBack));
    back.rotation.z = Math.PI / 2;
    back.position.set(0.006, -0.12, 0);
    this.paddle.add(back);
    const handle = new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.09, 0.028), mat(o.paddleHandle, 0.6));
    handle.position.y = -0.03;
    this.paddle.add(handle);

    this.time = 0;
    this.stride = 0;
  }

  // state: {x, swing, swing_t}
  update(state, dt) {
    this.time += dt;
    const x = state ? state.x : 0;
    const vx = this._lastX === undefined ? 0 : (x - this._lastX) / Math.max(dt, 1e-3);
    this._lastX = x;
    this.root.position.x = x;

    // Ready stance with breathing, and side-shuffle steps while moving.
    const moving = Math.min(1, Math.abs(vx) / 1.5);
    this.stride += Math.abs(vx) * dt * 9;
    const bob = Math.sin(this.time * 2.4) * 0.006;
    this.hips.position.y = 0.86 + bob - moving * 0.02 + Math.abs(Math.sin(this.stride)) * 0.02 * moving;
    for (const leg of this.legs) {
      const step = Math.sin(this.stride) * 0.22 * moving * leg.side;
      leg.thigh.rotation.set(-0.45, 0, leg.side * 0.13 + step);
      leg.knee.rotation.set(0.8, 0, -step * 0.5);
    }

    // Swing: torso and right arm follow keyframes; otherwise hold the ready pose.
    const t = state && state.swing ? state.swing_t : 99;
    let torsoYaw = 0, sYaw = -0.3, sPitch = -1.0, elbow = -1.0;
    if (t >= 0 && t < SWING_DURATION) {
      [torsoYaw, sYaw, sPitch, elbow] = keyframe(state.swing === 'forehand' ? FOREHAND : BACKHAND, t / SWING_DURATION);
    }
    const k = Math.min(1, dt * 30);
    this.torso.rotation.x = 0.28;                       // lean over the table
    this.torso.rotation.y += (torsoYaw - this.torso.rotation.y) * k;
    this.torso.rotation.z += ((-vx * 0.05) - this.torso.rotation.z) * Math.min(1, dt * 6);
    const r = this.rArm;
    r.shoulder.rotation.y += (sYaw - r.shoulder.rotation.y) * k;
    r.shoulder.rotation.x += (sPitch - r.shoulder.rotation.x) * k;
    r.shoulder.rotation.z = 0.25;
    r.elbow.rotation.x += (elbow - r.elbow.rotation.x) * k;
    this.paddle.rotation.y = 0.4;
    // Free arm: bent forward for balance.
    this.lArm.shoulder.rotation.set(-0.6, 0.2, -0.25);
    this.lArm.elbow.rotation.x = -1.3;
    this.head.rotation.x = -0.18;
    this.head.rotation.y = -this.torso.rotation.y * 0.7;   // eyes stay on the ball
  }
}
