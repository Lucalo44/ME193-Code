// crowd.js -- the audience: paper cut-outs (web/crowd/, made by tools/make_cutouts.py)
// standing on the stand steps, one per seat. They sway a little while play goes on,
// and jump and shake after every point (cheer()).
//
// If web/crowd/index.json is missing or empty, the plain silhouette crowd built by
// table.js stays instead.
import * as THREE from 'three';
import { THEME } from './theme.js';

const CHEER_S = 2.4;                     // how long a cheer lasts

export class Crowd {
  // seats: [{x, y (top of the step), z, rotY (facing the table)}]; silhouettes: meshes to
  // remove once the cut-outs are in.
  constructor(scene, seats, silhouettes = []) {
    this.scene = scene;
    this.seats = seats;
    this.silhouettes = silhouettes;
    this.groups = [];                    // [{mesh, people: [{seat, h, w, phase, ...}]}]
    this.time = 0;
    this.cheerT = Infinity;
    this.cheerLevel = 0;
    this._m = new THREE.Matrix4();
    this._q = new THREE.Quaternion();
    this._e = new THREE.Euler(0, 0, 0, 'YXZ');
    this._p = new THREE.Vector3();
    this._s = new THREE.Vector3();
    this._load();
  }

  async _load() {
    let names = [];
    try {
      const res = await fetch('crowd/index.json', { cache: 'no-cache' });
      if (res.ok) names = (await res.json()).cutouts || [];
    } catch (e) { /* no cut-outs: keep the silhouettes */ }
    if (!names.length) return;
    const loader = new THREE.TextureLoader();
    const textures = await Promise.all(names.map((n) => loader.loadAsync(`crowd/${n}`).catch(() => null)));
    const ok = textures.filter(Boolean);
    if (!ok.length) return;

    // Scatter: every seat gets a random picture, nudged along its row and a little
    // forward/back so the crowd isn't lined up.
    const bySheet = ok.map(() => []);
    const j = THEME.venue.cutouts.scatter;
    for (const seat of this.seats) {
      const along = (Math.random() - 0.5) * 2 * j.along, depth = (Math.random() - 0.5) * 2 * j.depth;
      const c = Math.cos(seat.rotY), s = Math.sin(seat.rotY);
      bySheet[Math.floor(Math.random() * ok.length)].push(
        { ...seat, x: seat.x + along * c + depth * s, z: seat.z - along * s + depth * c });
    }
    const cfg = THEME.venue.cutouts;
    ok.forEach((tex, k) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = 4;
      const aspect = tex.image.width / tex.image.height;
      // Unlit, so they read in the dim stands; brightness keeps them from glaring.
      const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, alphaTest: 0.5, side: THREE.DoubleSide,
        color: new THREE.Color(cfg.brightness, cfg.brightness, cfg.brightness) });
      const mesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1), mat, bySheet[k].length);
      mesh.frustumCulled = false;
      const people = bySheet[k].map((seat) => {
        const h = cfg.height * (0.88 + Math.random() * 0.24);
        return {
          seat, h, w: h * aspect,
          yaw: seat.rotY + (Math.random() - 0.5) * 0.35,      // not perfectly lined up
          tilt: (Math.random() - 0.5) * 0.08,
          phase: Math.random() * Math.PI * 2,
          jumpHz: 2.6 + Math.random() * 1.4,
          shakeHz: 5 + Math.random() * 3,
          delay: Math.random() * 0.35,                          // a ripple, not one synchronized hop
          eager: 0.6 + Math.random() * 0.6,
        };
      });
      this.groups.push({ mesh, people });
      this.scene.add(mesh);
    });
    for (const s of this.silhouettes) this.scene.remove(s);
    this.update(0);
  }

  // After a point. level: 1 = full cheer (the player won it), less for a polite clap.
  cheer(level = 1) {
    this.cheerT = 0;
    this.cheerLevel = level;
  }

  update(dt) {
    this.time += dt;
    this.cheerT += dt;
    for (const { mesh, people } of this.groups) {
      people.forEach((p, i) => {
        const t = this.time;
        const c = this.cheerT - p.delay;
        // Envelope: quick rise, slow fade over CHEER_S.
        const env = c > 0 && c < CHEER_S ? Math.min(1, c / 0.15) * (1 - c / CHEER_S) ** 1.5 : 0;
        const amp = env * this.cheerLevel * p.eager;
        const hop = Math.abs(Math.sin(p.phase + t * p.jumpHz * Math.PI)) * 0.16 * amp;
        const idle = Math.sin(p.phase + t * 0.9) * 0.004;
        const shake = Math.sin(p.phase * 2 + t * p.shakeHz * Math.PI * 2) * 0.22 * amp
          + Math.sin(p.phase + t * 0.7) * 0.015;
        this._p.set(p.seat.x, p.seat.y + p.h / 2 + hop + idle, p.seat.z);
        this._e.set(p.tilt, p.yaw, shake);
        this._q.setFromEuler(this._e);
        this._s.set(p.w, p.h, 1);
        mesh.setMatrixAt(i, this._m.compose(this._p, this._q, this._s));
      });
      mesh.instanceMatrix.needsUpdate = true;
    }
  }
}
