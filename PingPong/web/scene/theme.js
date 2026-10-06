// theme.js -- every color, material setting and visual size in one place.
// Restyle the game by editing this object; no game logic lives in web/.
//
// Look: a tribute to Rockstar Games' Table Tennis (2006) -- a dark, sparsely
// designed venue lit by pools of overhead light, a matte blue table on a red
// sports floor behind court barriers, a realistic athlete opponent, and a ball
// that glows in the color of its spin (the Xbox 360 face-button colors the game
// used: A green = topspin, Y yellow = backspin, X blue = left, B red = right).
// All 3D assets are original (built from primitives). The opponents carry the
// names of three characters from that game as a tribute -- fine for a private
// class project, but swap the names out before publishing anything.

export const THEME = {
  background: { color: '#050507', fog: '#050507', fogNear: 9, fogFar: 26 },

  venue: {
    floor: '#3a1714',             // court mat (deep red sports floor)
    concrete: '#0c0c0f',          // outside the barriers
    barrier: '#0f1a2e',
    barrierText: '#c9d3e6',
    barrierWords: ['TABLE TENNIS', 'ME193', 'LEGO PADDLE', 'WORLD SERIES'],
    courtHalfWidth: 3.6,          // barrier positions (m)
    courtHalfLength: 6.2,
    wall: '#0a0a0d',
    standsRows: 7,
    crowd: '#17171c',
    lampColor: '#fff3df',
  },

  table: {
    top: '#183c70',
    lines: '#f4f6fb',
    legs: '#1c1c20',
    net: '#0d0d10',
    netOpacity: 0.78,
    netTape: '#f4f6fb',
    posts: '#2a2a2e',
    roughness: 0.75,
    metalness: 0.0,
  },

  // Spin colors: the ball glows with these.
  spin: {
    topspin: '#3fd14a',           // A
    backspin: '#ffd21f',          // Y
    left: '#2f8dff',              // X
    right: '#ff3b2f',             // B
    none: '#ffffff',
  },

  ball: {
    color: '#fbfaf5',
    emissive: '#ffffff',
    emissiveIntensity: 0.15,
    stripe: '#d9d6cc',            // faint seam so rotation is still readable
    visualRadius: 0.024,
    glowSize: 0.16,
    trailLength: 18,
  },

  // Shared opponent settings. The per-speed characters are in OPPONENTS below.
  opponent: {
    paddleRubber: '#b81d2c',
    paddleBack: '#151515',
    paddleWood: '#d9b48a',
    paddleHandle: '#b8895a',
    distanceBehindEnd: 0.55,
  },

  playerPaddle: {
    rubber: '#b81d2c',
    back: '#151515',
    blade: '#d9b48a',
    handle: '#b8895a',
    opacity: 0.88,
    flash: '#ffffff',
  },

  effects: {
    bounceMark: '#ffffff',
    hitFlash: '#ffffff',
    netFlash: '#ffffff',
    targetRing: '#ffffff',
    targetRingOpen: '#3fd14a',
    hintForehand: '#ffffff',
    hintBackhand: '#ffffff',
  },

  lights: {
    ambient: 0.08,
    hemi: { sky: '#3a3f55', ground: '#200b08', intensity: 0.35 },
    // Overhead lamps above the table -- the "pool of light" look.
    lamps: [
      { pos: [0, 4.2, 1.4], intensity: 38 },
      { pos: [0, 4.2, -1.4], intensity: 38 },
    ],
    lampAngle: 0.62,
    lampPenumbra: 0.75,
    rim: { color: '#9fb4ff', intensity: 0.5, pos: [-4, 3, -6] },
    // Soft frontal fill from behind the camera so faces and the paddle read.
    fill: { color: '#ffe9d6', intensity: 0.9, pos: [0, 2.6, 7] },
  },

  // Bloom starts off (V toggles it): in software-rendered browsers it can tint the scene.
  bloom: { enabledByDefault: false, strength: 0.35, radius: 0.5, threshold: 0.85 },

  camera: {
    fov: 42,
    height: 0.95,                 // above the table top
    distance: 3.45,               // camera z (scene coords; the player's end is +z)
    lookAtZ: -0.6,
    sway: 0.05,                   // fraction of ball x the camera drifts by
  },
};

// Physics table top is y = 0; in the scene the table top sits at this height.
export const TABLE_H = 0.76;

// Ball glow color for a spin state. topspin/sidespin in -1..1 (sidespin > 0 curves
// the ball toward screen right). The dominant component picks the color.
export function spinColor(topspin, sidespin, out) {
  const s = THEME.spin;
  const aTop = Math.abs(topspin), aSide = Math.abs(sidespin);
  if (aTop < 0.05 && aSide < 0.05) return out.set(s.none);
  out.set(aTop >= aSide ? (topspin >= 0 ? s.topspin : s.backspin) : (sidespin >= 0 ? s.right : s.left));
  return out;
}

// One opponent per speed setting, named after characters from Rockstar's game.
// Purely visual: difficulty still comes from config.py's speed settings.
//   heightM     standing height (the whole model scales with it)
//   hair        'short' | 'ponytail' | 'buns'
//   bottom      'shorts' | 'skirt'
//   style       crouch (thigh angle, rad), knee bend, bounce Hz / amplitude (m),
//               lean (torso pitch), footwork (shuffle speed multiplier)
export const OPPONENTS = {
  slow: {
    name: 'CASSIDY', country: 'IRL', tagline: 'THE DEFENDER',
    heightM: 1.70,
    skin: '#f1cdb0', hair: '#8c3a1a', hairStyle: 'ponytail', brows: '#7a3418',
    shirt: '#1d7a47', trim: '#f4f6fb', accent: '#f4b33d',
    bottom: 'skirt', bottomColor: '#f4f6fb',
    socks: '#f4f6fb', shoes: '#f4f6fb', shoeAccent: '#1d7a47',
    wristband: null,
    style: { crouch: -0.32, knee: 0.6, bounceHz: 1.6, bounceAmp: 0.006, lean: 0.2, footwork: 0.9 },
  },
  medium: {
    name: 'LIU PING', country: 'CHN', tagline: 'THE ATTACKER',
    heightM: 1.75,
    skin: '#d8a47a', hair: '#141110', hairStyle: 'short', brows: '#141110',
    shirt: '#c41e2a', trim: '#ffcc33', accent: '#ffcc33',
    bottom: 'shorts', bottomColor: '#14161c',
    socks: '#f4f6fb', shoes: '#1a1a1e', shoeAccent: '#c41e2a',
    wristband: '#f4f6fb',
    style: { crouch: -0.55, knee: 0.95, bounceHz: 2.4, bounceAmp: 0.008, lean: 0.34, footwork: 1.0 },
  },
  fast: {
    name: 'KUMI', country: 'JPN', tagline: 'THE SPEEDSTER',
    heightM: 1.57,
    skin: '#ecc6a2', hair: '#17110e', hairStyle: 'buns', brows: '#17110e',
    shirt: '#f4f6fb', trim: '#d0263a', accent: '#d0263a',
    bottom: 'shorts', bottomColor: '#1b2a5c',
    socks: '#f4f6fb', shoes: '#f4f6fb', shoeAccent: '#d0263a',
    wristband: '#d0263a',
    style: { crouch: -0.48, knee: 0.85, bounceHz: 4.2, bounceAmp: 0.014, lean: 0.28, footwork: 1.5 },
  },
};

export function opponentFor(speedSetting) {
  return OPPONENTS[speedSetting] || OPPONENTS.medium;
}
