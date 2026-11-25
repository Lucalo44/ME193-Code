"""Real board: Bridge RPC to sketch/sketch.ino."""
from hal import Hal


class BridgeHal(Hal):
    kind = "bridge"

    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        ok, _ = self.ping()
        self.connected = bool(ok)

    def _draw(self, frame):
        self.bridge.call("draw", bytes(frame))

    def _drive(self, left, right):
        self.bridge.call("drive", left, right)

    def _heartbeat(self):
        return int(self.bridge.call("heartbeat") or 0) == 1

    def _probe_pins(self):
        return str(self.bridge.call("probe_pins") or "")

    def _i2c_scan(self):
        return str(self.bridge.call("i2c_scan") or "")

    def _ping(self):
        try:
            return self.bridge.call("ping", 41) == 42
        except Exception:  # noqa: BLE001
            return False
