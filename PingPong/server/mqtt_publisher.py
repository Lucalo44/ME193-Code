"""
server/mqtt_publisher.py -- publishes the game over MQTT.

  MQTT_TOPIC (ME193/Rogers/Luca)               the continuous-hit record of the
                                               current game, as a float in plain
                                               text, e.g. "7.0"
  MQTT_SCORE_TOPIC (ME193/Rogers/Luca/score)   the game score after every point
                                               (and 0-0 when a game starts), as JSON

Same broker as the rest of ME193 (test.mosquitto.org:1883, see
9_22/mqttlib.py). Messages are retained, so anyone who subscribes later
immediately gets the current record and score.

The connection runs in paho's background thread and reconnects on its own;
if the network is down the game carries on and the latest record is sent as
soon as the broker is reachable.
"""

from __future__ import annotations

import json
import threading
from typing import Dict, Optional

import config as C

try:
    import paho.mqtt.client as mqtt
except ImportError:  # the game still runs without it
    mqtt = None


class ScorePublisher:
    def __init__(self, broker: str = C.MQTT_BROKER, port: int = C.MQTT_PORT,
                 topic: str = C.MQTT_TOPIC, enabled: bool = True, score_topic: Optional[str] = None):
        self.topic = topic
        self.score_topic = score_topic or f"{topic}/score"
        self.enabled = enabled and mqtt is not None
        self.status = "off" if enabled else "disabled"
        if enabled and mqtt is None:
            self.status = "paho-mqtt not installed"
        self._latest: Dict[str, str] = {}     # topic -> last payload (re-sent on reconnect)
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

    @staticmethod
    def score_payload(score: dict) -> str:
        return json.dumps(score, separators=(",", ":"))

    def publish_record(self, record: float) -> None:
        """Thread-safe. Remembers the value and sends it now (or on connect)."""
        self._publish(self.topic, self.payload(record))

    def publish_score(self, score: dict) -> None:
        """Thread-safe. The game score (see game/loop.py: Game._publish_score)."""
        self._publish(self.score_topic, self.score_payload(score))

    def _publish(self, topic: str, msg: str) -> None:
        with self._lock:
            self._latest[topic] = msg
        if self._client is not None and self.status == "connected":
            self._client.publish(topic, msg, qos=1, retain=True)

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if not reason_code.is_failure:
            self.status = "connected"
            with self._lock:
                latest = dict(self._latest)
            for topic, msg in latest.items():
                client.publish(topic, msg, qos=1, retain=True)
        else:
            self.status = f"refused: {reason_code}"

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.status = "reconnecting"

    def close(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
