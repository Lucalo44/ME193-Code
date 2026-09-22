import paho.mqtt.client as mqtt

BROKER = "localhost"  # update to match your MQTT broker's address
PORT = 1883
TOPIC = "ME193"


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected to {BROKER}:{PORT} (reason code {reason_code})")
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    print(f"[{msg.topic}] {msg.payload.decode()}")


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT)
    client.loop_forever()


if __name__ == "__main__":
    main()
