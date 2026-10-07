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
// The real paddle's orientation (state.paddle.q) only shapes the paddle FACE --
// open/closed tilt plus a little turn -- smoothed and limited, so it reads clearly
// without throwing the paddle around.
import * as THREE from 'three';
import { THEME, TABLE_H } from './theme.js';

const READY = { x: 0.3, y: 0.22, dz: 0.05 };     // ready pose: right of center, above table height
const TRACK_RATE = 11;                            // 1/s: how fast the paddle glides to its target
const WINDUP_LEAD = 0.38;                         // s before contact the backswing starts
const FACE_SMOOTH = 9;                            // 1/s: orientation smoothing
const MAX_TILT = THREE.MathUtils.degToRad(55);    // limit on face tilt from the ready pose
const YAW_SHARE = 0.3;                            // fraction of the real paddle's turn shown
const STROKE = { toContact: 0.06, follow: 0.24, settle: 0.22 };   // stroke timing (s)
const AIR_SWING_AFTER = 0.75;                     // s to wait for a hit before showing an air swing
const LIVE_RATE = 30;                             // 1/s: smoothing of the live stroke (Bluetooth jitter)

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

    this.flash = 0;
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
    if (this.live) { this.flash = 1; return; }     // the live stroke already shows the swing
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
    this._updateFace(s ? s.paddle : null, dt);
    this.live = !!(s && s.stroke);
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
    const anchorTarget = (c && c.t_to_contact > -0.25 && c.t_to_contact < 1.6 && s.phase === 'RALLY')
      ? this.toScene(c.pos) : this.ready;
    if (!this.anchor) this.anchor = this.root.position.clone();
    this.anchor.lerp(anchorTarget, 1 - Math.exp(-TRACK_RATE * dt));

    const st = s.stroke;
    const side = (st.side === 'backhand' ? -1 : 1) * this.handed;
    const p = Math.max(-1.3, Math.min(1.3, st.phase || 0));
    const sideV = new THREE.Vector3(side, 1, 1);
    const offset = p >= 0 ? LIVE_BACK.clone().multiply(sideV).multiplyScalar(p)
                          : LIVE_FOLLOW.clone().multiply(sideV).multiplyScalar(-p);
    this._target.copy(this.anchor).add(offset);
    const k = 1 - Math.exp(-LIVE_RATE * dt);
    this.root.position.lerp(this._target, k);
    const yaw = p >= 0 ? YAW_BACK * side * p : YAW_FOLLOW * side * -p;
    this._lastYaw = (this._lastYaw ?? 0) + (yaw - (this._lastYaw ?? 0)) * k;
    this._yawQ.setFromAxisAngle(this._yAxis, this._lastYaw);
    this.stroke.quaternion.copy(this._yawQ);
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

  _updateFace(paddle, dt) {
    const o = paddle || {};
    if (o.q && o.q.length === 4) this._qTarget.set(o.q[0], o.q[1], o.q[2], o.q[3]).normalize();
    else this._qTarget.identity();
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
