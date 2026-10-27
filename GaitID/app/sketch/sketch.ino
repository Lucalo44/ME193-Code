// UNO Q - microcontroller (STM32) side
// Samples the 6-axis IMU at SAMPLE_HZ and queues the readings for Linux.
// Linux drains the queue with Bridge.call("get_imu") and shows the predicted
// activity with Bridge.call("show", "W").
//
// ASSUMPTION: the "5060" is an MPU-6050 (I2C 0x68) on the header SDA/SCL pins (Wire).
// If yours is on the Qwiic connector use Wire1; if it is a different chip, change
// imu_init() / imu_read() only -- everything else stays the same.

#include <Arduino_RouterBridge.h>
#include <Wire.h>
#include <zephyr/kernel.h>
#include "ArduinoGraphics.h"      // must be included BEFORE Arduino_LED_Matrix.h
#include "Arduino_LED_Matrix.h"

#define IMU_WIRE   Wire
#define IMU_ADDR   0x68
#define SAMPLE_HZ  50             // keep equal to FS in gaitid/config.py
#define ACC_LSB_G  4096.0f        // +/-8 g
#define GYR_LSB_DPS 32.8f         // +/-1000 deg/s
#define QUEUE_MAX  100            // samples held between Linux polls (2 s)

Arduino_LED_Matrix matrix;
struct k_mutex q_lock;
String queue_str;
int queue_n = 0;
unsigned long next_us = 0;

void imu_init() {
  IMU_WIRE.begin();
  IMU_WIRE.beginTransmission(IMU_ADDR); IMU_WIRE.write(0x6B); IMU_WIRE.write(0x00); IMU_WIRE.endTransmission(); // wake
  IMU_WIRE.beginTransmission(IMU_ADDR); IMU_WIRE.write(0x1A); IMU_WIRE.write(0x03); IMU_WIRE.endTransmission(); // DLPF ~44 Hz
  IMU_WIRE.beginTransmission(IMU_ADDR); IMU_WIRE.write(0x1B); IMU_WIRE.write(0x10); IMU_WIRE.endTransmission(); // gyro +/-1000
  IMU_WIRE.beginTransmission(IMU_ADDR); IMU_WIRE.write(0x1C); IMU_WIRE.write(0x10); IMU_WIRE.endTransmission(); // accel +/-8 g
}

bool imu_read(float v[6]) {
  IMU_WIRE.beginTransmission(IMU_ADDR);
  IMU_WIRE.write(0x3B);
  if (IMU_WIRE.endTransmission(false) != 0) return false;
  if (IMU_WIRE.requestFrom(IMU_ADDR, 14) != 14) return false;
  int16_t r[7];
  for (int i = 0; i < 7; i++) {
    int16_t hi = IMU_WIRE.read();
    r[i] = (hi << 8) | IMU_WIRE.read();
  }
  // A missing sensor can read back as all zeros instead of an error; a real
  // accelerometer always feels ~1 g, so treat that as a failed read.
  if (r[0] == 0 && r[1] == 0 && r[2] == 0) return false;
  v[0] = r[0] / ACC_LSB_G; v[1] = r[1] / ACC_LSB_G; v[2] = r[2] / ACC_LSB_G;  // r[3] is temperature
  v[3] = r[4] / GYR_LSB_DPS; v[4] = r[5] / GYR_LSB_DPS; v[5] = r[6] / GYR_LSB_DPS;
  return true;
}

// Called from Linux: returns "ms,ax,ay,az,gx,gy,gz;ms,..." and clears the queue
String get_imu() {
  k_mutex_lock(&q_lock, K_FOREVER);
  String out = queue_str;
  queue_str = "";
  queue_n = 0;
  k_mutex_unlock(&q_lock);
  return out;
}

// DEBUG: called from Linux, lists responding I2C addresses on both buses
String i2c_scan() {
  String out = "Wire:";
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) out += " 0x" + String(a, HEX);
  }
  Wire1.begin();
  out += " | Wire1:";
  for (uint8_t a = 1; a < 127; a++) {
    Wire1.beginTransmission(a);
    if (Wire1.endTransmission() == 0) out += " 0x" + String(a, HEX);
  }
  IMU_WIRE.beginTransmission(IMU_ADDR);                    // WHO_AM_I: 0x68 on an MPU-6050
  IMU_WIRE.write(0x75);
  IMU_WIRE.endTransmission(false);
  IMU_WIRE.requestFrom(IMU_ADDR, 1);
  out += " | WHO_AM_I=0x" + String(IMU_WIRE.read(), HEX);
  float v[6];
  out += imu_read(v) ? " | imu_read OK az=" + String(v[2]) : " | imu_read FAILED";
  return out;
}

// Called from Linux: shows one character (non-blocking, so sampling never stalls)
void show(String s) {
  matrix.beginDraw();
  matrix.stroke(0xFFFFFFFF);
  matrix.textFont(Font_5x7);
  matrix.beginText(3, 0, 0xFFFFFF);
  matrix.print(s.c_str());
  matrix.endText();
  matrix.endDraw();
}

void setup() {
  k_mutex_init(&q_lock);
  matrix.begin();
  imu_init();
  Bridge.begin();
  Bridge.provide("get_imu", get_imu);
  Bridge.provide("show", show);
  Bridge.provide("i2c_scan", i2c_scan);   // DEBUG
  next_us = micros();
}

void loop() {
  unsigned long now = micros();
  if ((long)(now - next_us) < 0) return;
  next_us += 1000000UL / SAMPLE_HZ;

  float v[6];
  if (!imu_read(v)) return;
  char line[96];
  snprintf(line, sizeof(line), "%lu,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f;",
           millis(), v[0], v[1], v[2], v[3], v[4], v[5]);

  k_mutex_lock(&q_lock, K_FOREVER);
  if (queue_n >= QUEUE_MAX) { queue_str = ""; queue_n = 0; }   // Linux fell behind: drop oldest chunk
  queue_str += line;
  queue_n++;
  k_mutex_unlock(&q_lock);
}
