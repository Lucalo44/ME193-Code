import time

from mqttlib import MQTTClient

TOPIC = "ME193"


def on_message(topic, payload):
    print(f"[{topic}] {payload}")


with MQTTClient() as client:
    client.subscribe(TOPIC, on_message)
    print(f"Listening on '{TOPIC}' (Ctrl+C to quit)...")
    while True:
        time.sleep(1)
