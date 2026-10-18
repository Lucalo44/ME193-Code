// Sumo robot -- microcontroller half. A thin executor with no decisions of
// its own: it drives the two wheel motors at whatever signed speeds Python
// sends, and reports the paddle's two limit switches when Python asks.
// The decision tree runs on the laptop (sumo wrestling/sumo.py).
//
// Motor driver: same board and pins as mqtt-minifig-drive. Each motor gets
// two PWM pins and no direction pin -- PWM pin A for forward, pin B for
// reverse. Motor leads go to the driver's M1/M2 screw terminals, the motor
// battery to VB+/VB-.
//
// Limit switches: COM -> GND, NO -> the pin below. The pins use the internal
// pull-up, so they read HIGH when released and LOW when pressed. Never wire
// a switch to 5V.

#include <Arduino_RouterBridge.h>

// Motor 1 = left wheel, Motor 2 = right wheel; swap here if that is backwards.
const int MOTOR1_A_PIN = 5;   // M1A -- ~D5
const int MOTOR1_B_PIN = 6;   // M1B -- ~D6
const int MOTOR2_A_PIN = 9;   // M2A -- ~D9
const int MOTOR2_B_PIN = 10;  // M2B -- ~D10
const int MAX_SPEED = 255;    // must match MAX_DRIVE in sumo.py

const int SWITCH_LEFT_PIN = 2;   // left limit switch on the paddle
const int SWITCH_RIGHT_PIN = 4;  // right limit switch on the paddle
const unsigned long DEBOUNCE_MS = 5;

struct Switch {
  int pin;
  bool raw;               // last reading
  bool stable;            // debounced state
  bool hitSinceRead;      // pressed at some point since Python last asked
  unsigned long changedAt;
};

Switch switches[2] = {
  {SWITCH_LEFT_PIN, false, false, false, 0},
  {SWITCH_RIGHT_PIN, false, false, false, 0},
};

void setup() {
  pinMode(MOTOR1_A_PIN, OUTPUT);
  pinMode(MOTOR1_B_PIN, OUTPUT);
  pinMode(MOTOR2_A_PIN, OUTPUT);
  pinMode(MOTOR2_B_PIN, OUTPUT);
  drive(0, 0);

  for (Switch &s : switches) {
    pinMode(s.pin, INPUT_PULLUP);
  }

  Bridge.begin();
  Bridge.provide("drive", drive);
  Bridge.provide("read_switches", read_switches);
}

void loop() {
  unsigned long now = millis();
  for (Switch &s : switches) {
    bool pressed = digitalRead(s.pin) == LOW;
    if (pressed != s.raw) {
      s.raw = pressed;
      s.changedAt = now;
    } else if (now - s.changedAt >= DEBOUNCE_MS && s.stable != pressed) {
      s.stable = pressed;
      if (pressed) {
        s.hitSinceRead = true;
      }
    }
  }
  delay(1);
}

// speed: -MAX_SPEED..MAX_SPEED. If a motor spins the wrong way, swap pinA/pinB
// in drive() below (or swap that motor's two wires at the driver).
void driveMotor(int pinA, int pinB, int speed) {
  speed = constrain(speed, -MAX_SPEED, MAX_SPEED);
  if (speed >= 0) {
    analogWrite(pinA, speed);
    analogWrite(pinB, 0);
  } else {
    analogWrite(pinA, 0);
    analogWrite(pinB, -speed);
  }
}

// Called from Python with signed wheel speeds (-MAX_SPEED..MAX_SPEED).
void drive(int leftSpeed, int rightSpeed) {
  driveMotor(MOTOR1_A_PIN, MOTOR1_B_PIN, leftSpeed);
  driveMotor(MOTOR2_A_PIN, MOTOR2_B_PIN, rightSpeed);
}

// Called from Python. Bit 0 = left switch, bit 1 = right switch. A switch
// counts as pressed if it is held now OR was tapped since the last call, so
// a hit shorter than Python's polling period is never missed.
int read_switches() {
  int bits = 0;
  for (int i = 0; i < 2; i++) {
    if (switches[i].stable || switches[i].hitSinceRead) {
      bits |= 1 << i;
    }
    switches[i].hitSinceRead = false;
  }
  return bits;
}
