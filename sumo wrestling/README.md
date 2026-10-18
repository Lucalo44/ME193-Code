# Sumo Wrestling

LEGO robot that flips opponents out of a blue-tape ring. Limit switches on the
paddle, read by the UnoQ, report contact over MQTT; the laptop runs the policy
and drives the LEGO double motor. If pushed out of the ring, the robot stops
and plays a sad song.

Constraints: <= 1 kg, fits in a 30 cm cube, decision policy (behavior tree /
decision tree / KNN / K-means), >= 2 sensors (>= 1 autonomous, >= 1 teleop).

## Setup

```bash
cd "sumo wrestling"
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
