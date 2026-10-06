// table.js -- the venue (floor, court barriers, stands, overhead lamps) and the
// regulation-proportioned table with lines and net.
import * as THREE from 'three';
import { THEME, TABLE_H } from './theme.js';

const L = 2.74, W = 1.525, NET_H = 0.1525, OVERHANG = 0.1525, THICK = 0.03;

function barrierTexture(word) {
  const v = THEME.venue;
  const c = document.createElement('canvas');
  c.width = 1024; c.height = 128;
  const g = c.getContext('2d');
  g.fillStyle = v.barrier;
  g.fillRect(0, 0, 1024, 128);
  g.fillStyle = 'rgba(255,255,255,0.06)';
  g.fillRect(0, 0, 1024, 6);
  g.fillStyle = v.barrierText;
  g.font = '700 64px "Futura", "Avenir Next Condensed", "Arial Narrow", sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(word, 512, 68);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

function buildVenue(scene) {
  const v = THEME.venue;
  scene.background = new THREE.Color(THEME.background.color);
  scene.fog = new THREE.Fog(THEME.background.fog, THEME.background.fogNear, THEME.background.fogFar);

  // Concrete around, red court mat inside the barriers.
  const concrete = new THREE.Mesh(new THREE.PlaneGeometry(60, 60),
    new THREE.MeshStandardMaterial({ color: v.concrete, roughness: 0.95 }));
  concrete.rotation.x = -Math.PI / 2;
  concrete.receiveShadow = true;
  scene.add(concrete);
  const court = new THREE.Mesh(new THREE.PlaneGeometry(v.courtHalfWidth * 2, v.courtHalfLength * 2),
    new THREE.MeshStandardMaterial({ color: v.floor, roughness: 0.55, metalness: 0.0 }));
  court.rotation.x = -Math.PI / 2;
  court.position.y = 0.002;
  court.receiveShadow = true;
  scene.add(court);

  // Court barriers: low dark-blue panels with printed words.
  const panelW = 2.0, panelH = 0.7;
  const words = v.barrierWords;
  let n = 0;
  const addPanel = (x, z, rotY) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(panelW - 0.04, panelH, 0.05),
      new THREE.MeshStandardMaterial({ map: barrierTexture(words[n++ % words.length]), roughness: 0.6 }));
    m.position.set(x, panelH / 2, z);
    m.rotation.y = rotY;
    m.castShadow = true;
    m.receiveShadow = true;
    scene.add(m);
  };
  const hw = v.courtHalfWidth, hl = v.courtHalfLength;
  // Whole panels only, centered on each side so corners don't overlap.
  const run = (half) => {
    const n = Math.floor((2 * half - 0.2) / panelW);
    return Array.from({ length: n }, (_, i) => (i - (n - 1) / 2) * panelW);
  };
  for (const x of run(hw)) addPanel(x, -hl, 0);                                  // far end
  for (const z of run(hl)) {
    addPanel(-hw, z, Math.PI / 2);
    addPanel(hw, z, -Math.PI / 2);
  }

  // Dark walls and stepped stands with a silhouette crowd (barely lit, mood only).
  const wallMat = new THREE.MeshStandardMaterial({ color: v.wall, roughness: 1 });
  const back = new THREE.Mesh(new THREE.PlaneGeometry(40, 14), wallMat);
  back.position.set(0, 7, -hl - 7);
  scene.add(back);
  const stepMat = new THREE.MeshStandardMaterial({ color: '#0d0d11', roughness: 1 });
  const crowdMat = new THREE.MeshStandardMaterial({ color: v.crowd, roughness: 1 });
  const person = new THREE.CapsuleGeometry(0.17, 0.2, 3, 8);
  const spots = [];
  const stand = (cx, cz, rotY, len) => {
    for (let r = 0; r < v.standsRows; r++) {
      const step = new THREE.Mesh(new THREE.BoxGeometry(len, 0.45, 0.9), stepMat);
      const d = 1.2 + r * 0.9, h = 0.25 + r * 0.45;
      const off = new THREE.Vector3(0, h, -d).applyAxisAngle(new THREE.Vector3(0, 1, 0), rotY);
      step.position.set(cx + off.x, h, cz + off.z);
      step.rotation.y = rotY;
      scene.add(step);
      for (let s = -len / 2 + 0.3; s < len / 2; s += 0.55) {
        if (Math.random() < 0.35) continue;  // empty seats
        const p = new THREE.Vector3(s, h + 0.62 + Math.random() * 0.06, -d)
          .applyAxisAngle(new THREE.Vector3(0, 1, 0), rotY);
        spots.push([cx + p.x, p.y, cz + p.z]);
      }
    }
  };
  stand(0, -hl - 0.4, 0, 9);                       // behind the opponent
  stand(-hw - 0.4, -1.5, Math.PI / 2, 9);          // left side
  stand(hw + 0.4, -1.5, -Math.PI / 2, 9);          // right side
  // Seated silhouettes: torso + head, slightly varied.
  const bodies = new THREE.InstancedMesh(person, crowdMat, spots.length);
  const heads = new THREE.InstancedMesh(new THREE.SphereGeometry(0.1, 8, 6), crowdMat, spots.length);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
  spots.forEach((p, i) => {
    const k = 0.9 + Math.random() * 0.2;
    sc.set(k, k, k);
    bodies.setMatrixAt(i, m4.compose(new THREE.Vector3(p[0], p[1] - 0.12, p[2]), q, sc));
    heads.setMatrixAt(i, m4.compose(new THREE.Vector3(p[0] + (Math.random() - 0.5) * 0.04, p[1] + 0.2 * k, p[2]), q, sc));
  });
  scene.add(bodies, heads);

  // Lights: overhead lamp housings + spotlights make the pool of light on the table.
  const lt = THEME.lights;
  scene.add(new THREE.AmbientLight(0xffffff, lt.ambient));
  scene.add(new THREE.HemisphereLight(lt.hemi.sky, lt.hemi.ground, lt.hemi.intensity));
  const housingMat = new THREE.MeshStandardMaterial({ color: '#111', emissive: v.lampColor, emissiveIntensity: 2.2 });
  for (const lamp of lt.lamps) {
    const spot = new THREE.SpotLight(v.lampColor, lamp.intensity, 12, lt.lampAngle, lt.lampPenumbra, 1.6);
    spot.position.set(...lamp.pos);
    spot.target.position.set(lamp.pos[0], 0, lamp.pos[2] * 0.6);
    spot.castShadow = true;
    spot.shadow.mapSize.set(1024, 1024);
    spot.shadow.bias = -0.0004;
    spot.shadow.radius = 5;
    scene.add(spot, spot.target);
    const housing = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.06, 0.3), housingMat);
    housing.position.set(lamp.pos[0], lamp.pos[1] + 0.05, lamp.pos[2]);
    scene.add(housing);
  }
  // A row of distant ceiling lights for depth.
  for (let i = -2; i <= 2; i++) {
    const far = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.05, 0.25),
      new THREE.MeshBasicMaterial({ color: '#8f8a80' }));
    far.position.set(i * 3.2, 6.5, -9);
    scene.add(far);
  }
  const rim = new THREE.DirectionalLight(lt.rim.color, lt.rim.intensity);
  rim.position.set(...lt.rim.pos);
  scene.add(rim);
  const fill = new THREE.DirectionalLight(lt.fill.color, lt.fill.intensity);
  fill.position.set(...lt.fill.pos);
  scene.add(fill);
}

export function buildArena(scene) {
  buildVenue(scene);
  const t = THEME.table;
  const table = new THREE.Group();

  const top = new THREE.Mesh(new THREE.BoxGeometry(W, THICK, L),
    new THREE.MeshStandardMaterial({ color: t.top, roughness: t.roughness, metalness: t.metalness }));
  top.position.y = TABLE_H - THICK / 2;
  top.receiveShadow = true;
  top.castShadow = true;
  table.add(top);

  // White lines: edges + center line, as thin boxes just above the surface.
  const lineMat = new THREE.MeshStandardMaterial({ color: t.lines, roughness: 0.6 });
  const addLine = (w, l, x, z) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, 0.002, l), lineMat);
    m.position.set(x, TABLE_H + 0.001, z);
    m.receiveShadow = true;
    table.add(m);
  };
  const lw = 0.02;
  addLine(lw, L, -W / 2 + lw / 2, 0);
  addLine(lw, L, W / 2 - lw / 2, 0);
  addLine(W, lw, 0, -L / 2 + lw / 2);
  addLine(W, lw, 0, L / 2 - lw / 2);
  addLine(0.003, L, 0, 0);

  // Undercarriage: two folding frames.
  const legMat = new THREE.MeshStandardMaterial({ color: t.legs, roughness: 0.5, metalness: 0.4 });
  for (const sz of [-1, 1]) {
    for (const sx of [-1, 1]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.04, TABLE_H - THICK, 0.04), legMat);
      leg.position.set(sx * (W / 2 - 0.15), (TABLE_H - THICK) / 2, sz * (L / 2 - 0.3));
      leg.castShadow = true;
      table.add(leg);
    }
    const bar = new THREE.Mesh(new THREE.BoxGeometry(W - 0.3, 0.04, 0.04), legMat);
    bar.position.set(0, 0.2, sz * (L / 2 - 0.3));
    table.add(bar);
  }

  // Net: dark mesh with a white tape on top, plus posts.
  const span = W + 2 * OVERHANG;
  const net = new THREE.Mesh(new THREE.PlaneGeometry(span, NET_H),
    new THREE.MeshStandardMaterial({ color: t.net, transparent: true, opacity: t.netOpacity,
      side: THREE.DoubleSide, roughness: 1 }));
  net.position.set(0, TABLE_H + NET_H / 2, 0);
  table.add(net);
  const tape = new THREE.Mesh(new THREE.BoxGeometry(span, 0.014, 0.004),
    new THREE.MeshStandardMaterial({ color: t.netTape, roughness: 0.5 }));
  tape.position.set(0, TABLE_H + NET_H - 0.007, 0);
  table.add(tape);
  for (const sx of [-1, 1]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.02, NET_H + 0.02, 0.03),
      new THREE.MeshStandardMaterial({ color: t.posts, metalness: 0.5, roughness: 0.4 }));
    post.position.set(sx * span / 2, TABLE_H + NET_H / 2, 0);
    table.add(post);
  }
  scene.add(table);

  return { table, net, top };
}

// Faint half-table highlight used for the forehand/backhand hint.
export function buildSideHint(scene) {
  const mat = new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0, depthWrite: false });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(W / 2, L / 2), mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set(0, TABLE_H + 0.003, L / 4);
  scene.add(mesh);
  return mesh;
}
