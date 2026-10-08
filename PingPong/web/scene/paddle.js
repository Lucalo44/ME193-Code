// paddle.js -- the player's paddle, Wii-sports style.
//
// The GAME moves the paddle to the ball: as a ball comes in, the paddle glides to
// the predicted contact point (state.contact). The PLAYER drives the stroke:
//
//  * Real paddle (state.stroke present): LIVE. Drawing the paddle back draws the
//    on-screen paddle back along the stroke arc, in step; the forward swing sweeps
//    it through the contact point into the follow-through at the player's speed.
//    (state.stroke.phase: +1 drawn back, 0 at the ball, -1 follow-through --
//    hardware/stroke.py.)
//  * Keyboard paddle: SCRIPTED. The paddle winds up automatically before the ball
//    arrives, and Space plays a stroke through the ball (or in the air on a miss).
//
// Orientation: with the real paddle the on-screen paddle mirrors its FULL
// orientation (state.paddle.q) -- tilt, turn, a full 360 -- lightly smoothed. With
// the keyboard paddle the face only tilts a little with W/S/A/D.
import * as THREE from 'three';
import { THEME, TABLE_H } from './theme.js';

const READY = { x: 0.3, y: 0.22, dz: 0.05 };     // ready pose: right of center, above table height
const TRACK_RATE = 11;                            // 1/s: how fast the paddle glides to its target
const WINDUP_LEAD = 0.38;                         // s before contact the backswing starts
const FACE_SMOOTH = 9;                            // 1/s: orientation smoothing (keyboard paddle)
const LIVE_FACE_SMOOTH = 25;                      // 1/s: orientation smoothing (real paddle: full, unlimited)
const MAX_TILT = THREE.MathUtils.degToRad(55);    // limit on face tilt from the ready pose
const YAW_SHARE = 0.3;                            // fraction of the real paddle's turn shown
const STROKE = { toContact: 0.06, follow: 0.24, settle: 0.22 };   // stroke timing (s)
const AIR_SWING_AFTER = 0.75;                     // s to wait for a hit before showing an air swing
const LIVE_RATE = 40;                             // 1/s: light position smoothing on top of the filtered phase
// One-euro filter on the live stroke phase: heavy smoothing when the paddle moves
// slowly (hides Bluetooth burst jitter), almost none during a fast swing (no lag).
const PHASE_MIN_CUTOFF = 2.0;                     // Hz at rest
const PHASE_BETA = 1.2;                           // extra Hz per phase-unit/s of motion
const PHASE_D_CUTOFF = 1.0;                       // Hz, for the speed estimate

class OneEuro {
  constructor(minCutoff, beta, dCutoff) { Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: 0 }); }
  static alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  filter(value, dt) {
    if (this.x === null || dt <= 0) { this.x = value; return value; }
    const d = (value - this.x) / dt;
    this.dx += OneEuro.alpha(this.dCutoff, dt) * (d - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += OneEuro.alpha(cutoff, dt) * (value - this.x);
    return this.x;
  }
  reset(value) { this.x = value; this.dx = 0; }
}

// Stroke geometry relative to the contact point (scene meters; +x screen-right,
// +z toward the camera). Right-handed forehand; backhands and left-handers mirror x.
// The camera sits high behind the player, so offsets toward it (+z) drop the paddle down
// the screen; the wind-up is kept mostly sideways so it stays visibly next to the ball.
const BACKSWING = new THREE.Vector3(0.24, -0.02, 0.1);
const FOLLOW = new THREE.Vector3(-0.28, 0.2, -0.22);
const READY_NEAR = new THREE.Vector3(0.07, 0.0, 0.04);   // waiting spot near the contact point
// Live (real paddle) arc: a fuller draw-back and follow-through, like Wii Sports.
const LIVE_BACK = new THREE.Vector3(0.36, -0.03, 0.16);
const LIVE_FOLLOW = new THREE.Vector3(-0.38, 0.24, -0.24);
const YAW_BACK = THREE.MathUtils.degToRad(40), YAW_FOLLOW = THREE.MathUtils.degToRad(-55);

const ease = (u) => u * u * (3 - 2 * u);
const easeOut = (u) => 1 - (1 - u) * (1 - u);

export class PlayerPaddleView {
  constructor(scene, hitPlaneZ, toScene) {
    const p = THEME.playerPaddle;
    this.toScene = toScene;
    this.ready = new THREE.Vector3(READY.x, TABLE_H + READY.y, hitPlaneZ + READY.dz);
    this.root = new THREE.Group();                 // position (game-driven)
    this.root.position.copy(this.ready);
    scene.add(this.root);
    this.stroke = new THREE.Group();               // stroke turn (animation)
    this.root.add(this.stroke);
    this.face = new THREE.Group();                 // face tilt (real paddle)
    this.stroke.add(this.face);

    const disc = (r, h) => new THREE.CylinderGeometry(r, r, h, 40);
    this.rubberMat = new THREE.MeshStandardMaterial({ color: p.rubber, transparent: true, opacity: p.opacity,
      emissive: p.flash, emissiveIntensity: 0, roughness: 0.85 });
    // The blade is centered on the root, so "paddle at the ball" means the ball hits the rubber.
    const rubber = new THREE.Mesh(disc(0.078, 0.004), this.rubberMat);
    rubber.rotation.x = Math.PI / 2;
    rubber.position.set(0, 0, 0.005);
    this.face.add(rubber);
    const blade = new THREE.Mesh(disc(0.081, 0.006),
      new THREE.MeshStandardMaterial({ color: p.blade, transparent: true, opacity: p.opacity, roughness: 0.6 }));
    blade.rotation.x = Math.PI / 2;
    this.face.add(blade);
    const back = new THREE.Mesh(disc(0.078, 0.004),
      new THREE.MeshStandardMaterial({ color: p.back, transparent: true, opacity: p.opacity, roughness: 0.85 }));
    back.rotation.x = Math.PI / 2;
    back.position.set(0, 0, -0.005);
    this.face.add(back);
    const handle = new THREE.Mesh(new THREE.BoxGeometry(0.028, 0.1, 0.024),
      new THREE.MeshStandardMaterial({ color: p.handle, roughness: 0.55 }));
    handle.position.y = -0.105;
    this.face.add(handle);
    this.face.traverse((o) => { if (o.isMesh) o.castShadow = true; });

    // Pose arm tracking (state.arm): hand offset moves the paddle; a forearm follows.
    this.armCfg = { weight: 0, scale: 0.45, max: 0.4 };
    this.armOffset = new THREE.Vector3();
    this.handFilterX = new OneEuro(1.5, 0.8, 1.0);
    this.handFilterY = new OneEuro(1.5, 0.8, 1.0);
    this.forearm = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.036, 1, 16, 1, true),
      new THREE.MeshStandardMaterial({ color: '#d9a77c', transparent: true, opacity: 0.4, roughness: 0.7,
        side: THREE.DoubleSide, depthWrite: false }));
    this.forearm.visible = false;
    scene.add(this.forearm);
    this._up = new THREE.Vector3(0, 1, 0);
    this.flash = 0;
    this.punch = 0;              // brief impact pulse on contact
    this.phaseFilter = new OneEuro(PHASE_MIN_CUTOFF, PHASE_BETA, PHASE_D_CUTOFF);
    this._liveSide = 1;
    this.anim = null;            // active stroke animation
    this.swingWait = null;       // a swing registered; waiting to see if it becomes a hit
    this.handed = 1;             // +1 right-handed, -1 left-handed
    this._qTarget = new THREE.Quaternion();
    this._qFace = new THREE.Quaternion();
    this._twist = new THREE.Quaternion();
    this._tmp = new THREE.Vector3();
    this._target = new THREE.Vector3();
    this._yawQ = new THREE.Quaternion();
    this._yAxis = new THREE.Vector3(0, 1, 0);
  }

  // --- events -------------------------------------------------------------
  // A swing registered on the real paddle (it may or may not become a hit).
  swing() {
    this.flash = 1;
    if (!this.anim && !this.live) this.swingWait = { age: 0 };
  }

  // The swing struck the ball at physics position `pos`; play the stroke through it.
  hit(pos, stroke) {
    this.swingWait = null;
    if (this.live) { this.flash = 1; this.punch = 1; return; }   // the live stroke already shows the swing
    const C = this.toScene(pos);
    const side = this._side(stroke, C.x);
    this.anim = { t: 0, from: this.root.position.clone(), C, side, contact: true };
  }

  // The swing missed (early / late / wrong stroke): show it as an air swing.
  miss() {
    if (this.swingWait) { this.swingWait = null; this._airSwing(); }
  }

  _airSwing() {
    const C = this.root.position.clone().add(new THREE.Vector3(-0.08 * this.handed, 0.04, -0.12));
    this.anim = { t: 0, from: this.root.position.clone(), C, side: this.handed, contact: false };
  }

  _side(stroke, x) {
    // +1 = forehand on the right (right-handed). Backhands and left-handers mirror.
    let fh = stroke === 'forehand' ? true : stroke === 'backhand' ? false : (x >= 0) === (this.handed > 0);
    return (fh ? 1 : -1) * this.handed;
  }

  // --- per frame ------------------------------------------------------------
  // s: interpolated game state (paddle, contact, required_stroke, settings, phase)
  update(s, dt) {
    this.handed = s && s.settings && s.settings.handedness === 'left' ? -1 : 1;
    this.live = !!(s && s.stroke);
    this._updateFace(s ? s.paddle : null, dt);
    this._updateArm(s ? s.arm : null, dt);
    if (this.live) {
      this._updateLive(s, dt);
      this.flash = Math.max(0, this.flash - dt * 5);
      this.rubberMat.emissiveIntensity = this.flash * 0.6;
      return;
    }

    // Swing that never turned into a hit -> air swing.
    if (this.swingWait) {
      this.swingWait.age += dt;
      const c = s && s.contact;
      const ballNear = c && c.t_to_contact < 0.7 && c.t_to_contact > -0.4;
      if (!ballNear || this.swingWait.age > AIR_SWING_AFTER) { this.swingWait = null; this._airSwing(); }
    }

    let strokeYaw = 0;
    if (this.anim) {
      strokeYaw = this._animate(dt);
    } else {
      // Track: glide toward the contact point, winding up as the ball arrives.
      const c = s && s.contact;
      if (c && c.t_to_contact > -0.25 && c.t_to_contact < 1.6 && s.phase === 'RALLY') {
        const C = this.toScene(c.pos);
        const side = this._side(c.serve ? 'forehand' : s.required_stroke, C.x);
        const wind = ease(Math.max(0, Math.min(1, 1 - c.t_to_contact / WINDUP_LEAD)));
        this._target.copy(READY_NEAR).multiply(new THREE.Vector3(side, 1, 1))
          .lerp(this._tmp.copy(BACKSWING).multiply(new THREE.Vector3(side, 1, 1)), wind).add(C);
        strokeYaw = YAW_BACK * side * wind;
      } else {
        this._target.copy(this.ready);
      }
      this._target.add(this.armOffset);
      const k = 1 - Math.exp(-TRACK_RATE * dt);
      this.root.position.lerp(this._target, k);
      this._lastYaw = (this._lastYaw ?? 0) + (strokeYaw - (this._lastYaw ?? 0)) * k;
      strokeYaw = this._lastYaw;
    }
    this._yawQ.setFromAxisAngle(this._yAxis, strokeYaw);
    this.stroke.quaternion.copy(this._yawQ);

    this.flash = Math.max(0, this.flash - dt * 5);
    this.rubberMat.emissiveIntensity = this.flash * 0.6;
  }

  // Real paddle: anchor at the ball (or ready), stroke offset from the live phase.
  _updateLive(s, dt) {
    this.anim = null;
    this.swingWait = null;
    const c = s.contact;
    // While the ball is held at the paddle (hit-stop), stay with the ball itself.
    const anchorTarget = s.hold ? this.toScene(s.ball.pos)
      : (c && c.t_to_contact > -0.25 && c.t_to_contact < 1.6 && s.phase === 'RALLY')
        ? this.toScene(c.pos) : this.ready;
    if (!this.anchor) this.anchor = this.root.position.clone();
    this.anchor.lerp(anchorTarget, 1 - Math.exp(-TRACK_RATE * dt));

    const st = s.stroke;
    const side = (st.side === 'backhand' ? -1 : 1) * this.handed;
    if (side !== this._liveSide) { this._liveSide = side; this.phaseFilter.reset(st.phase || 0); }
    let p = this.phaseFilter.filter(Math.max(-1.3, Math.min(1.3, st.phase || 0)), dt);
    // Hit-stop: while the ball waits at the paddle for the swing to register, the
    // paddle stops AT the ball instead of passing through it; on the hit both leave
    // together (the filter and smoothing carry the paddle on into the follow-through).
    if (s.hold && p < 0.02) p = 0.02;
    const sideV = new THREE.Vector3(side, 1, 1);
    const offset = p >= 0 ? LIVE_BACK.clone().multiply(sideV).multiplyScalar(p)
                          : LIVE_FOLLOW.clone().multiply(sideV).multiplyScalar(-p);
    this._target.copy(this.anchor).add(offset).add(this.armOffset);
    const k = 1 - Math.exp(-LIVE_RATE * dt);
    this.root.position.lerp(this._target, k);
    // No scripted turn here: the face shows the real paddle's full orientation,
    // which already includes the stroke's turn.
    this._lastYaw = 0;
    this.stroke.quaternion.identity();
    // Impact: a quick swell of the paddle as it strikes, easing back over ~0.15 s.
    this.punch = Math.max(0, this.punch - dt * 7);
    this.stroke.scale.setScalar(1 + 0.12 * Math.sin(Math.PI * Math.min(1, 1 - this.punch)) * (this.punch > 0 ? 1 : 0));
  }

  _animate(dt) {
    const a = this.anim;
    a.t += dt;
    const side = a.side;
    const F = FOLLOW.clone().multiply(new THREE.Vector3(side, 1, 1)).add(a.C);
    const { toContact, follow, settle } = STROKE;
    let yaw;
    if (a.t < toContact) {
      // Snap from wherever we are (usually the backswing) into the ball.
      const u = ease(a.t / toContact);
      this.root.position.copy(a.from).lerp(a.C, u);
      yaw = YAW_BACK * side * (1 - u);
      if (a.contact && a.t + dt >= toContact) this.flash = 1;
    } else if (a.t < toContact + follow) {
      const u = easeOut((a.t - toContact) / follow);
      this.root.position.copy(a.C).lerp(F, u);
      yaw = YAW_FOLLOW * side * u;
    } else if (a.t < toContact + follow + settle) {
      const u = ease((a.t - toContact - follow) / settle);
      this.root.position.copy(F).lerp(this.ready, u * 0.6);
      yaw = YAW_FOLLOW * side * (1 - u);
    } else {
      this.anim = null;
      yaw = 0;
    }
    this._lastYaw = yaw;
    return yaw;
  }

  // Pose arm tracking: the hand's offset from its ready position shifts the paddle
  // (smoothed -- the camera is ~30 fps and noisier than the paddle's IMU), and a
  // translucent forearm is drawn from the tracked elbow to the paddle's grip.
  _updateArm(arm, dt) {
    const cfg = this.armCfg;
    if (!arm || !arm.ok || cfg.weight <= 0) {
      this.armOffset.multiplyScalar(Math.exp(-6 * dt));      // ease back if tracking drops out
      this.forearm.visible = false;
      return;
    }
    if (arm.zeroed) {
      const hx = this.handFilterX.filter(arm.hand[0], dt), hy = this.handFilterY.filter(arm.hand[1], dt);
      this.armOffset.set(hx * cfg.scale * cfg.weight, hy * cfg.scale * cfg.weight, 0);
      if (this.armOffset.length() > cfg.max) this.armOffset.setLength(cfg.max);
    }
    // Forearm: elbow placed relative to the grip using the real elbow->wrist direction.
    const grip = this._tmp.copy(this.root.position).add(new THREE.Vector3(0, -0.1, 0.02));
    const ew = [arm.elbow[0] - arm.wrist[0], arm.elbow[1] - arm.wrist[1]];
    const len = Math.hypot(ew[0], ew[1]) || 1;
    const elbow = grip.clone().add(new THREE.Vector3(ew[0] / len * 0.24, ew[1] / len * 0.24, 0.14));
    const dir = elbow.clone().sub(grip);
    this.forearm.position.copy(grip).addScaledVector(dir, 0.5);
    this.forearm.scale.set(1, dir.length(), 1);
    this.forearm.quaternion.setFromUnitVectors(this._up, dir.normalize());
    this.forearm.visible = true;
  }

  _updateFace(paddle, dt) {
    const o = paddle || {};
    if (o.q && o.q.length === 4) this._qTarget.set(o.q[0], o.q[1], o.q[2], o.q[3]).normalize();
    else this._qTarget.identity();
    if (this.live) {
      // Real paddle: mirror its full orientation -- any tilt, a full 360 turn, anything.
      this._qFace.slerp(this._qTarget, 1 - Math.exp(-LIVE_FACE_SMOOTH * dt));
      this.face.quaternion.copy(this._qFace);
      return;
    }
    // Keep only part of the turn about vertical (the stroke animation shows the swing).
    const t = this._twist.set(0, this._qTarget.y, 0, this._qTarget.w);
    if (t.lengthSq() < 1e-9) t.identity(); else t.normalize();
    const tilt = this._qTarget.clone().multiply(t.clone().invert());
    const yaw = 2 * Math.atan2(t.y, t.w);
    // Limit the tilt so a wild wrist can't flip the paddle over.
    const ang = 2 * Math.acos(Math.min(1, Math.abs(tilt.w)));
    if (ang > MAX_TILT) tilt.slerp(new THREE.Quaternion(), 1 - MAX_TILT / ang);
    const shown = new THREE.Quaternion().setFromAxisAngle(this._yAxis, yaw * YAW_SHARE).multiply(tilt);
    this._qFace.slerp(shown, 1 - Math.exp(-FACE_SMOOTH * dt));
    this.face.quaternion.copy(this._qFace);
  }
}
