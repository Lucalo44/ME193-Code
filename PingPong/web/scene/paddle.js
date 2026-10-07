// paddle.js -- the player's paddle near the bottom of the screen.
//
// It mirrors the real paddle's 3D orientation (state.paddle.q, a scene-frame
// quaternion relative to the zeroed ready pose, from hardware/orientation.py).
// Orientation alone can't give position, so the paddle sits at the end of an
// invisible forearm that pivots at the elbow: turning the paddle back for a
// backswing swings it back and out, the follow-through carries it across.
import * as THREE from 'three';
import { THEME, TABLE_H } from './theme.js';

// Elbow -> paddle-center offset in the ready pose (scene meters): up and toward the table.
const FOREARM = new THREE.Vector3(0, 0.2, -0.36);
const READY_X = 0.3;

export class PlayerPaddleView {
  constructor(scene, hitPlaneZ) {
    const p = THEME.playerPaddle;
    // Elbow position: below and behind where the paddle rests.
    this.elbow = new THREE.Group();
    this.elbow.position.set(READY_X, TABLE_H + 0.0, hitPlaneZ + 0.1 - FOREARM.z);
    scene.add(this.elbow);
    this.arm = new THREE.Group();               // rotates with the real paddle
    this.elbow.add(this.arm);
    this.tilt = new THREE.Group();              // the paddle model, at the end of the forearm
    this.tilt.position.copy(FOREARM);
    this.arm.add(this.tilt);

    const disc = (r, h) => new THREE.CylinderGeometry(r, r, h, 40);
    this.rubberMat = new THREE.MeshStandardMaterial({ color: p.rubber, transparent: true, opacity: p.opacity,
      emissive: p.flash, emissiveIntensity: 0, roughness: 0.85 });
    const rubber = new THREE.Mesh(disc(0.078, 0.004), this.rubberMat);
    rubber.rotation.x = Math.PI / 2;
    rubber.position.set(0, 0.075, 0.005);
    this.tilt.add(rubber);
    const blade = new THREE.Mesh(disc(0.081, 0.006),
      new THREE.MeshStandardMaterial({ color: p.blade, transparent: true, opacity: p.opacity, roughness: 0.6 }));
    blade.rotation.x = Math.PI / 2;
    blade.position.set(0, 0.075, 0);
    this.tilt.add(blade);
    const back = new THREE.Mesh(disc(0.078, 0.004),
      new THREE.MeshStandardMaterial({ color: p.back, transparent: true, opacity: p.opacity, roughness: 0.85 }));
    back.rotation.x = Math.PI / 2;
    back.position.set(0, 0.075, -0.005);
    this.tilt.add(back);
    const handle = new THREE.Mesh(new THREE.BoxGeometry(0.028, 0.1, 0.024),
      new THREE.MeshStandardMaterial({ color: p.handle, roughness: 0.55 }));
    handle.position.y = -0.03;
    this.tilt.add(handle);
    this.tilt.traverse((o) => { if (o.isMesh) o.castShadow = true; });

    this.flash = 0;
    this._target = new THREE.Quaternion();
    this._euler = new THREE.Euler(0, 0, 0, 'YXZ');
  }

  swing() { this.flash = 1; }

  // orientation: {q: [x, y, z, w]} scene-frame quaternion (preferred), or Euler
  // {pitch, roll, yaw} degrees; targetX: scene x to drift toward (incoming ball), or null.
  update(orientation, targetX, dt) {
    const o = orientation || {};
    if (o.q && o.q.length === 4) {
      this._target.set(o.q[0], o.q[1], o.q[2], o.q[3]).normalize();
    } else {
      const d2r = Math.PI / 180;
      this._euler.set((o.pitch || 0) * d2r, (o.yaw || 0) * d2r, -(o.roll || 0) * d2r, 'YXZ');
      this._target.setFromEuler(this._euler);
    }
    // Light smoothing on top of the 60 Hz stream; fast enough to follow a swing.
    this.arm.quaternion.slerp(this._target, Math.min(1, dt * 30));

    const tx = targetX === null || targetX === undefined ? READY_X : Math.max(-0.75, Math.min(0.75, targetX));
    this.elbow.position.x += (tx - this.elbow.position.x) * Math.min(1, dt * 4);
    this.flash = Math.max(0, this.flash - dt * 5);
    this.rubberMat.emissiveIntensity = this.flash * 0.6;
  }
}
