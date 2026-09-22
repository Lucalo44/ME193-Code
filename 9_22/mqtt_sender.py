from mqttlib import MQTTClient

TOPIC = "ME193"

with MQTTClient() as client:
    print(f"Publishing to '{TOPIC}' on test.mosquitto.org. Type a message and press Enter (Ctrl+C to quit).")
    try:
        while True:
            message = input("> ")
            client.publish(TOPIC, message)
    except KeyboardInterrupt:
        pass
