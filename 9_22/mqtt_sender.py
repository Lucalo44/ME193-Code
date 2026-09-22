import paho.mqtt.client as mqtt

BROKER = "localhost"  # update to match your MQTT broker's address
PORT = 1883
TOPIC = "ME193"


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER, PORT)
    client.loop_start()

    print(f"Publishing to '{TOPIC}' on {BROKER}:{PORT}. Type a message and press Enter (Ctrl+C to quit).")
    try:
        while True:
            message = input("> ")
            client.publish(TOPIC, message)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
