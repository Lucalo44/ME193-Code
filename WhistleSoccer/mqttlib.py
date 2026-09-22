"""Shared MQTTClient wrapper for ME193 examples.

Usage:
    from mqttlib import MQTTClient

    with MQTTClient() as client:
        client.subscribe(TOPIC, on_message)
        client.publish(TOPIC, "hello world")
"""

import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"  # shared public broker used in class
PORT = 1883


class MQTTClient:
    def __init__(self, broker=BROKER, port=PORT):
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._callbacks = {}
        self._client.on_message = self._dispatch
        self._client.connect(broker, port)
        self._client.loop_start()

    def _dispatch(self, client, userdata, msg):
        callback = self._callbacks.get(msg.topic)
        if callback:
            callback(msg.topic, msg.payload.decode())

    def subscribe(self, topic, callback):
        self._callbacks[topic] = callback
        self._client.subscribe(topic)

    def publish(self, topic, payload):
        self._client.publish(topic, payload)

    def close(self):
        self._client.loop_stop()
        self._client.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
