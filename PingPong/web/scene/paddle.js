// paddle.js -- the player's paddle near the bottom of the screen. Its tilt follows
// the live IMU pitch/roll so the player sees the face angle they present.
import * as THREE from 'three';
import { THEME, TABLE_H } from './theme.js';

export class PlayerPaddleView {
  constructor(scene, hitPlaneZ) {
    const p = THEME.playerPaddle;
    this.group = new THREE.Group();
    this.group.position.set(0.3, TABLE_H + 0.2, hitPlaneZ + 0.1);
    scene.add(this.group);
    this.tilt = new THREE.Group();
    this.group.add(this.tilt);

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
  }

  swing() { this.flash = 1; }

  // orientation: {pitch, roll, yaw} degrees relative to the zeroed ready pose;
  // targetX: scene x the paddle drifts toward (incoming ball), or null.
  update(orientation, targetX, dt) {
    const d2r = Math.PI / 180;
    const o = orientation || { pitch: 0, roll: 0, yaw: 0 };
    const k = Math.min(1, dt * 20);
    this.tilt.rotation.x += (o.pitch * d2r - this.tilt.rotation.x) * k;
    this.tilt.rotation.z += (-o.roll * d2r - this.tilt.rotation.z) * k;
    this.tilt.rotation.y += (o.yaw * d2r * 0.5 - this.tilt.rotation.y) * k;
    const tx = targetX === null || targetX === undefined ? 0.3 : Math.max(-0.75, Math.min(0.75, targetX));
    this.group.position.x += (tx - this.group.position.x) * Math.min(1, dt * 6);
    this.flash = Math.max(0, this.flash - dt * 5);
    this.rubberMat.emissiveIntensity = this.flash * 0.6;
    // Small punch forward on a swing.
    this.tilt.position.z = -this.flash * 0.06;
  }
}
