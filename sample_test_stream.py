import asyncio
from typing import Callable, Awaitable, Any

Emit = Callable[[str, Any, str | None], Awaitable[None]]

async def main_app2(total_steps: int, delay_sec: float, emit: Emit):
    """
    Orchestrates the nested calls. Everyone gets the same `emit`.
    """
    # Example: a long loop with progress
    for step in range(1, total_steps + 1):
        # do some work...
        await asyncio.sleep(delay_sec * 0.1)

        # call downstream agent; pass emit through
        # await agent_invoke(step=i, total=total_steps, emit=emit)

        # also emit your own step-level progress
        pct = round(step * 100 / total_steps, 1)
        await emit("progress", {
            "step": step,
            "total": total_steps,
            "percent": round(step * 100 / total_steps, 2),
            "message": f"Processed {step}/{total_steps}"
        }, id_=str(step))
