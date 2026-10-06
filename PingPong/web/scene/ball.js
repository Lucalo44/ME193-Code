// ball.js -- the ball, its visible rotation, a halo that glows in the color of its
// spin (green topspin, yellow backspin, blue left, red right), and a short streak.
import * as THREE from 'three';
import { THEME, spinColor } from './theme.js';

function ballTexture() {
  const c = document.createElement('canvas');
  c.width = 128; c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = THEME.ball.color;
  g.fillRect(0, 0, 128, 64);
  g.fillStyle = THEME.ball.stripe;
  g.fillRect(0, 31, 128, 2);                 // seam
  g.font = 'bold 10px sans-serif';
  g.fillText('★★★', 22, 24);   // tiny maker's stamp, makes spin readable
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function glowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 4, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.25, 'rgba(255,255,255,0.55)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

export class BallView {
  constructor(scene) {
    const b = THEME.ball;
    this.mesh = new THREE.Mesh(
      new THREE.SphereGeometry(b.visualRadius, 24, 16),
      new THREE.MeshStandardMaterial({ map: ballTexture(), emissive: b.emissive,
        emissiveIntensity: b.emissiveIntensity, roughness: 0.35 }));
    this.mesh.castShadow = true;
    scene.add(this.mesh);

    this.glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0 }));
    this.glow.scale.setScalar(b.glowSize);
    scene.add(this.glow);

    // Streak: fading, shrinking instanced spheres in the spin color.
    this.trailN = b.trailLength;
    this.trail = new THREE.InstancedMesh(
      new THREE.SphereGeometry(b.visualRadius * 0.7, 8, 6),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.35, depthWrite: false,
        blending: THREE.AdditiveBlending }),
      this.trailN);
    this.trail.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.trail.frustumCulled = false;
    scene.add(this.trail);
    this.history = [];
    this.tmp = new THREE.Object3D();
    this.col = new THREE.Color('#ffffff');
    this.target = new THREE.Color('#ffffff');
    for (let i = 0; i < this.trailN; i++) this.trail.setColorAt(i, this.col);
  }

  // pos: scene Vector3; spinScene: angular velocity (scene coords, rad/s);
  // topspin / sidespin: signed amounts in -1..1 that pick the glow color.
  update(pos, spinScene, topspin, sidespin, visible, dt) {
    this.mesh.visible = this.glow.visible = this.trail.visible = visible;
    if (!visible) { this.history.length = 0; return; }
    this.mesh.position.copy(pos);
    this.glow.position.copy(pos);
    const w = spinScene.length();
    if (w > 1e-3) this.mesh.rotateOnWorldAxis(spinScene.clone().normalize(), (w * dt) / 6);

    spinColor(topspin, sidespin, this.target);
    this.col.lerp(this.target, Math.min(1, dt * 12));
    const amount = Math.min(1, Math.max(Math.abs(topspin), Math.abs(sidespin)) * 2);
    this.glow.material.color.copy(this.col);
    this.glow.material.opacity = 0.25 + 0.6 * amount;
    this.glow.scale.setScalar(THEME.ball.glowSize * (0.7 + 0.5 * amount));

    const last = this.history[0];
    if (!last || last.distanceToSquared(pos) > 1e-6) this.history.unshift(pos.clone());
    if (this.history.length > this.trailN) this.history.length = this.trailN;
    if (this.history.length > 1 && this.history[0].distanceTo(this.history[1]) > 0.5) this.history.length = 1;
    for (let i = 0; i < this.trailN; i++) {
      const p = this.history[Math.min(i, this.history.length - 1)];
      const s = i < this.history.length ? (1 - i / this.trailN) * (0.4 + amount) : 0;
      this.tmp.position.copy(p);
      this.tmp.scale.setScalar(s);
      this.tmp.updateMatrix();
      this.trail.setMatrixAt(i, this.tmp.matrix);
      this.trail.setColorAt(i, this.col);
    }
    this.trail.instanceMatrix.needsUpdate = true;
    this.trail.instanceColor.needsUpdate = true;
  }
}
