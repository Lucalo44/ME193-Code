// Campus Navigator -- microcontroller (STM32) side. A thin executor; the step counter, heading
// filter and navigator all run in Python on the Linux side (python/main.py).
//   * samples the ankle MPU-6050 at 50 Hz and queues it -> Bridge "get_imu"
//     (same sampling and reply format as ArduinoUNOq/gaitid, which StepNet was trained on)
//   * shows the next maneuver on the LED matrix          <- Bridge "draw"
//
// Wiring: MPU-6050 on the header SDA/SCL pins (Wire, I2C 0x68), 3.3V and GND, strapped to the
// right ankle in the same orientation as for the step counter.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>
#include <zephyr/kernel.h>
#include <vector>

#define IMU_WIRE    Wire
#define IMU_ADDR    0x68
#define SAMPLE_HZ   50            // keep equal to FS in Navigation/config.py
#define ACC_LSB_G   4096.0f       // +/-8 g
#define GYR_LSB_DPS 32.8f         // +/-1000 deg/s
#define QUEUE_MAX   100           // samples held between Linux polls (2 s)

const uint8_t FRAME_SIZE = 8 * 13;

Arduino_LED_Matrix matrix;
uint8_t frame[FRAME_SIZE] = {0};
volatile bool frameDirty = true;

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
  // all zeros = the MPU-6050 is asleep (loose wire, power-up): re-wake and skip
  if (r[0] == 0 && r[1] == 0 && r[2] == 0) { imu_init(); return false; }
  v[0] = r[0] / ACC_LSB_G; v[1] = r[1] / ACC_LSB_G; v[2] = r[2] / ACC_LSB_G;   // r[3] = temperature
  v[3] = r[4] / GYR_LSB_DPS; v[4] = r[5] / GYR_LSB_DPS; v[5] = r[6] / GYR_LSB_DPS;
  return true;
}

// Linux drains this every ~0.1 s: "ms,ax,ay,az,gx,gy,gz;ms,..." and the queue is cleared
String get_imu() {
  k_mutex_lock(&q_lock, K_FOREVER);
  String out = queue_str;
  queue_str = "";
  queue_n = 0;
  k_mutex_unlock(&q_lock);
  return out;
}

// 8 x 13 brightness values (0-7), row by row (same protocol as the safe and color-car)
void draw(std::vector<uint8_t> newFrame) {
  size_t len = min(newFrame.size(), (size_t)FRAME_SIZE);
  memcpy(frame, newFrame.data(), len);
  frameDirty = true;
}

void setup() {
  k_mutex_init(&q_lock);
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();
  Bridge.begin();                 // before the sensor: a stuck I2C bus must not silence the Bridge
  Bridge.provide("get_imu", get_imu);
  Bridge.provide("draw", draw);
  imu_init();
  next_us = micros();
}

void loop() {
  if (frameDirty) {               // redraw only when Python sent a new arrow
    frameDirty = false;
    matrix.draw(frame);
  }
  unsigned long now = micros();
  if ((long)(now - next_us) < 0) return;
  next_us += 1000000UL / SAMPLE_HZ;

  float v[6];
  if (!imu_read(v)) return;
  char line[96];
  snprintf(line, sizeof(line), "%lu,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f;",
           millis(), v[0], v[1], v[2], v[3], v[4], v[5]);
  k_mutex_lock(&q_lock, K_FOREVER);
  if (queue_n >= QUEUE_MAX) { queue_str = ""; queue_n = 0; }   // Linux fell behind: drop the old chunk
  queue_str += line;
  queue_n++;
  k_mutex_unlock(&q_lock);
}
