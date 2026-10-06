"""
server/mqtt_publisher.py -- publishes the continuous-hit record over MQTT.

Topic MQTT_TOPIC (ME193/Rogers/Luca), payload the record as a floating-point
number in plain text, e.g. "7.0". Same broker as the rest of ME193
(test.mosquitto.org:1883, see 9_22/mqttlib.py). Messages are retained, so
anyone who subscribes later immediately gets the current record.

The connection runs in paho's background thread and reconnects on its own;
if the network is down the game carries on and the latest record is sent as
soon as the broker is reachable.
"""

from __future__ import annotations

import threading
from typing import Optional

import config as C

try:
    import paho.mqtt.client as mqtt
except ImportError:  # the game still runs without it
    mqtt = None


class ScorePublisher:
    def __init__(self, broker: str = C.MQTT_BROKER, port: int = C.MQTT_PORT,
                 topic: str = C.MQTT_TOPIC, enabled: bool = True):
        self.topic = topic
        self.enabled = enabled and mqtt is not None
        self.status = "off" if enabled else "disabled"
        if enabled and mqtt is None:
            self.status = "paho-mqtt not installed"
        self._latest: Optional[str] = None
        self._lock = threading.Lock()
        self._client = None
        if not self.enabled:
            return
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.status = "connecting"
        try:
            self._client.connect_async(broker, port, keepalive=30)
            self._client.loop_start()
        except Exception as exc:  # noqa: BLE001 -- never let MQTT stop the game
            self.status = f"error: {exc}"

    @staticmethod
    def payload(record: float) -> str:
        return str(float(record))

    def publish_record(self, record: float) -> None:
        """Thread-safe. Remembers the value and sends it now (or on connect)."""
        msg = self.payload(record)
        with self._lock:
            self._latest = msg
        if self._client is not None and self.status == "connected":
            self._client.publish(self.topic, msg, qos=1, retain=True)

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if not reason_code.is_failure:
            self.status = "connected"
            with self._lock:
                latest = self._latest
            if latest is not None:
                client.publish(self.topic, latest, qos=1, retain=True)
        else:
            self.status = f"refused: {reason_code}"

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.status = "reconnecting"

    def close(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
