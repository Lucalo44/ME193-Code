// Ground Control -- microcontroller (STM32) half.
//
// A thin, safe executor. All planning, text rendering and choreography run in
// Python on the Linux side. This side only:
//   * draws 8x13 frames on the LED matrix      <- Bridge "draw" (brightness 0-7, row by row)
//   * drives the two wheel motors               <- Bridge "drive" (left, right: -255..255)
//   * keeps a heartbeat watchdog                <- Bridge "heartbeat" (Python calls it every ~200 ms)
//   * answers a round-trip probe                <- Bridge "ping" (returns its argument + 1)
//
// Watchdog: if Python is silent for more than WATCHDOG_MS, the wheels stop and
// the matrix shows an X. That covers a crashed app, a hung LLM call that took
// the main loop with it, or someone pulling the plug on the Linux side.
// The X stays until the next heartbeat arrives.
//
// Motor driver (same board and pins as the sumo robot): each motor gets two PWM
// pins and no direction pin -- pin A for forward, pin B for reverse.
//   M1A/M1B = D5/D6 (left wheel), M2A/M2B = D9/D10 (right wheel).

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>

const int MOTOR1_A_PIN = 5;
const int MOTOR1_B_PIN = 6;
const int MOTOR2_A_PIN = 9;
const int MOTOR2_B_PIN = 10;
const int MAX_SPEED = 255;

const unsigned long WATCHDOG_MS = 1000;

const uint8_t FRAME_ROWS = 8;
const uint8_t FRAME_COLS = 13;
const uint8_t FRAME_SIZE = FRAME_ROWS * FRAME_COLS;

Arduino_LED_Matrix matrix;
uint8_t frame[FRAME_SIZE] = {0};
volatile bool frameDirty = true;

volatile unsigned long lastBeat = 0;
volatile bool everBeat = false;
bool tripped = false;

// A 7x7 X centred on the matrix, brightness 7.
const uint8_t X_ICON[FRAME_ROWS][FRAME_COLS] = {
  {0,0,0,0,0,0,0,0,0,0,0,0,0},
  {0,0,0,7,0,0,0,0,0,7,0,0,0},
  {0,0,0,0,7,0,0,0,7,0,0,0,0},
  {0,0,0,0,0,7,0,7,0,0,0,0,0},
  {0,0,0,0,0,0,7,0,0,0,0,0,0},
  {0,0,0,0,0,7,0,7,0,0,0,0,0},
  {0,0,0,0,7,0,0,0,7,0,0,0,0},
  {0,0,0,7,0,0,0,0,0,7,0,0,0},
};

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

void stopMotors() {
  driveMotor(MOTOR1_A_PIN, MOTOR1_B_PIN, 0);
  driveMotor(MOTOR2_A_PIN, MOTOR2_B_PIN, 0);
}

void setup() {
  pinMode(MOTOR1_A_PIN, OUTPUT);
  pinMode(MOTOR1_B_PIN, OUTPUT);
  pinMode(MOTOR2_A_PIN, OUTPUT);
  pinMode(MOTOR2_B_PIN, OUTPUT);
  stopMotors();

  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();

  Bridge.begin();
  Bridge.provide("draw", draw);
  Bridge.provide("drive", drive);
  Bridge.provide("heartbeat", heartbeat);
  Bridge.provide("ping", ping);
}

void loop() {
  unsigned long now = millis();

  // Watchdog. Only armed after the first heartbeat, so a slow app start
  // doesn't show the X before Python has even connected.
  if (everBeat && !tripped && now - lastBeat > WATCHDOG_MS) {
    tripped = true;
    stopMotors();
    memcpy(frame, X_ICON, FRAME_SIZE);
    frameDirty = true;
  }

  if (frameDirty) {
    frameDirty = false;
    matrix.draw(frame);
  }
  delay(1);
}

// 8 x 13 brightness values (0-7), row by row. Ignored while the watchdog is
// tripped, so the X can't be painted over by a stale frame.
void draw(std::vector<uint8_t> newFrame) {
  if (tripped) return;
  size_t len = min(newFrame.size(), (size_t)FRAME_SIZE);
  memcpy(frame, newFrame.data(), len);
  frameDirty = true;
}

// Signed wheel speeds. Refused while the watchdog is tripped.
void drive(int leftSpeed, int rightSpeed) {
  if (tripped) {
    stopMotors();
    return;
  }
  driveMotor(MOTOR1_A_PIN, MOTOR1_B_PIN, leftSpeed);
  driveMotor(MOTOR2_A_PIN, MOTOR2_B_PIN, rightSpeed);
}

// Python calls this every ~200 ms. Returns 1 if the watchdog had tripped
// (so Python knows to redraw the matrix), else 0.
int heartbeat() {
  lastBeat = millis();
  everBeat = true;
  if (tripped) {
    tripped = false;
    return 1;
  }
  return 0;
}

int ping(int x) {
  return x + 1;
}
