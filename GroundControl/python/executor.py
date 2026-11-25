"""Runs a validated plan step by step.

- {ns.key} placeholders in string args are filled from earlier step results
  ("yaw is {imu.yaw} deg"). A bare {key} also works ({yaw}).
- Each step has a hard timeout; the job's cancel flag (e-stop) is checked
  between steps and inside motor loops.
- If the plan never published anything, the executor still replies "done"
  on the reply topic (the assignment: "or just that it is done").
"""
import re
import threading
import time

from skills.context import Cancelled, Ctx, SkillError
from skills.registry import ArgError, validate_plan

PH = re.compile(r"\{(\w+)(?:\.(\w+))?\}")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.1f}"
    if v is None:
        return "?"
    return str(v)


def resolve(value, results):
    if isinstance(value, str):
        def sub(m):
            a, b = m.group(1), m.group(2)
            if b is not None:
                if a in results and b in results[a]:
                    return _fmt(results[a][b])
                return m.group(0)
            for ns in reversed(list(results.values())):
                if a in ns:
                    return _fmt(ns[a])
            return m.group(0)
        return PH.sub(sub, value)
    if isinstance(value, dict):
        return {k: resolve(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, results) for v in value]
    return value


class Executor:
    def __init__(self, make_ctx, emit=None, status=None):
        self.make_ctx = make_ctx            # (job) -> Ctx
        self.emit = emit or (lambda *a, **k: None)
        self.status = status

    def run(self, job, plan):
        ctx: Ctx = self.make_ctx(job)
        steps = validate_plan(plan)
        results, errors = {}, []
        self.step_times = []
        n = len(steps)
        for i, (sk, args) in enumerate(steps):
            name = sk.name if sk else "?"
            if job.cancel.is_set():
                for j in range(i, n):
                    self.emit("plan.step", {"id": job.id, "i": j, "state": "cancelled"})
                errors.append("stopped")
                break
            if isinstance(args, ArgError):
                errors.append(f"{name}: {args}")
                self.emit("plan.step", {"id": job.id, "i": i, "skill": name, "state": "fail",
                                        "error": str(args)})
                continue
            if "imu" not in results and "{imu." in str(args):
                results["imu"] = ctx.imu() or {}          # sensor placeholders read on demand
            args = resolve(args, results)
            if self.status:
                self.status.update(job={"id": job.id, "step": i + 1, "of": n, "skill": name},
                                   doing=f"Step {i + 1}/{n}: {name}")
            self.emit("plan.step", {"id": job.id, "i": i, "skill": name, "args": args, "state": "running"})
            t0 = time.time()
            ctx.step_timeout.clear()
            timer = threading.Timer(sk.timeout_s, ctx.step_timeout.set)
            timer.daemon = True
            timer.start()
            try:
                r = sk.fn(ctx, **args)
                if sk.ns and isinstance(r, dict):
                    results[sk.ns] = r
                results[f"step{i + 1}"] = r if isinstance(r, dict) else {}
                self.emit("plan.step", {"id": job.id, "i": i, "skill": name, "args": args, "state": "ok",
                                        "result": r, "t_ms": round((time.time() - t0) * 1000)})
            except (SkillError, Cancelled, ValueError) as e:
                errors.append(f"{name}: {e}")
                self.emit("plan.step", {"id": job.id, "i": i, "skill": name, "args": args, "state": "fail",
                                        "error": str(e), "t_ms": round((time.time() - t0) * 1000)})
                if isinstance(e, Cancelled) and job.cancel.is_set():
                    for j in range(i + 1, n):
                        self.emit("plan.step", {"id": job.id, "i": j, "state": "cancelled"})
                    break
            except Exception as e:  # noqa: BLE001  a skill bug must not kill the worker
                errors.append(f"{name}: internal error {e!r}")
                self.emit("plan.step", {"id": job.id, "i": i, "skill": name, "state": "fail",
                                        "error": repr(e)})
            finally:
                timer.cancel()
                self.step_times.append({"skill": name, "t0": t0, "t1": time.time()})
            if sk.motor and job.cancel.is_set():
                ctx.hal.stop()

        if job.cancel.is_set():
            reply = "stopped"
            ctx.publish(ctx.reply_to, reply)
        elif not steps:
            why = "I don't have a skill for that"
            reply = f"Sorry, I can't do that: {why}"
            ctx.publish(ctx.reply_to, reply)
        elif errors and len(errors) == n:
            reply = f"Sorry, I can't do that: {'; '.join(errors)}"
            ctx.publish(ctx.reply_to, reply)
        elif not ctx.published:
            reply = "done" if not errors else f"done, with problems: {'; '.join(errors)}"
            ctx.publish(ctx.reply_to, reply)
        else:
            reply = "done" if not errors else f"done, with problems: {'; '.join(errors)}"
        return {"reply": reply, "errors": errors, "results": results, "published": ctx.published,
                "steps": self.step_times}
