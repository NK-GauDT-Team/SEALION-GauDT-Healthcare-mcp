from fastapi import FastAPI, Query
from sealionapp import main_app
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
from typing import AsyncIterator, Any, Dict, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sample_test_stream import main_app2
import contextlib

app = FastAPI()


# Allow your frontend origin during dev. Use a specific origin in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ProcessIn(BaseModel):
  query: str
  
@app.get("/")
async def homepage():
  return {"result": "Can load pages"}

@app.post("/process/")
def process(body: ProcessIn):
  result = main_app(body.query)
  return {"result": result}

@app.post("/test_process/")
def test_process(body: ProcessIn):
  return {
          "result": {
              "medicine_details": [
                  {
                      "medicine_name": "ginger tea",
                      "medicine_instruction": "Boil sliced ginger in water (often with honey and/or lime).",
                      "dosage": None
                  },
                  {
                      "medicine_name": "turmeric tea",
                      "medicine_instruction": "Combine turmeric powder or grated fresh turmeric with ginger in hot water.",
                      "dosage": None
                  },
                  {
                      "medicine_name": "lemongrass tea",
                      "medicine_instruction": "Steep lemongrass stalks in hot water.",
                      "dosage": None
                  },
                  {
                      "medicine_name": "galangal tea",
                      "medicine_instruction": "Similar preparation to ginger tea.",
                      "dosage": None
                  },
                  {
                      "medicine_name": "ginger-garlic decoction",
                      "medicine_instruction": "Combine ginger and garlic in a decoction.",
                      "dosage": None
                  },
                  {
                      "medicine_name": "warm broth",
                      "medicine_instruction": "Consume chicken or vegetable broth.",
                      "dosage": None
                  }
              ],
              "non_pharmacologic_methods": [
                  {
                      "method_name": "rest",
                      "instructions": "Get plenty of rest.",
                      "frequency": None,
                      "duration": None,
                      "notes": None
                  },
                  {
                      "method_name": "hydration",
                      "instructions": "Drink plenty of warm fluids.",
                      "frequency": None,
                      "duration": None,
                      "notes": None
                  },
                  {
                      "method_name": "warm clothing",
                      "instructions": "Dress warmly, especially if you feel chilled.",
                      "frequency": None,
                      "duration": None,
                      "notes": None
                  },
                  {
                      "method_name": "steam inhalation",
                      "instructions": "Inhale steam infused with lemongrass or ginger.",
                      "frequency": None,
                      "duration": None,
                      "notes": "Be careful to avoid burns."
                  },
                  {
                      "method_name": "warm compress",
                      "instructions": "Apply warm compresses to the chest or back.",
                      "frequency": None,
                      "duration": None,
                      "notes": None
                  }
              ],
              "analysis": "The text describes traditional remedies for cold and flu symptoms common in Indonesian and Vietnamese cultures, focusing on warming the body and expelling 'cold'. Recommendations include herbal teas (ginger, turmeric, lemongrass, galangal), warm broths, and lifestyle measures like rest, hydration, and warm clothing. The remedies are generally supportive and aim to alleviate symptoms.",
              "severity": "low"
          }
      }
  

def sse_pack(data, event=None, id_=None):
    lines = []
    if event:
        lines.append(f"event: {event}")
    if id_:
        lines.append(f"id: {id_}")
    payload = data if isinstance(data, str) else json.dumps(data)
    for line in str(payload).splitlines():
        lines.append(f"data: {line}")
    # 👇 TWO blank lines to terminate the event properly
    return "\n".join(lines) + "\n\n"

async def progress_event_stream(total_steps: int = 30) -> AsyncIterator[str]:
    """
    Wraps the long job and yields SSE-formatted messages.
    """
    try:
        i = 0
        # Optional: initial hello so client can render instantly
        yield sse_pack({"status": "started"}, event="status", id_="0")

        async for update in long_running_job(total_steps=total_steps, delay_sec=1.0):
            i += 1
            # Send a "progress" event
            yield sse_pack(update, event="progress", id_=str(i))

            # Optional: keep-alive every ~15s if your iterations can be sparse
            # if i % 15 == 0:
            #     yield ":\n\n"  # comment line is a valid SSE heartbeat

        # Final event
        yield sse_pack({"status": "done"}, event="complete", id_=str(i + 1))

    except asyncio.CancelledError:
        # Client disconnected; clean up if needed
        # e.g., cancel your underlying task
        raise

@app.get("/stream")
async def stream(query:str = None):
    """
    Open this URL from the frontend via EventSource.
    Use the 'total' query param to control total steps (for testing).
    """
    q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)
    async def emit(event:str,data,id_:str):
        """
        Downstream-friendly async emitter.
        Convert to SSE frame here so children only think in (event, data).
        """
        await q.put(sse_pack(data,event=event,id_=id_))
    
    async def runner():
        try:
            # Kick things off (initial hello so UI renders immediately)
            await emit("status", {"status": "started"}, id_="0")
            # Run your app with the emitter wired in
            await main_app(query=query,emit=emit)
            await emit("complete", {"status": "done"},id_="9999")
        except asyncio.CancelledError:
            # Client disconnected; allow graceful cleanup
            raise
        finally:
            # Signal the generator to stop
            await q.put(None)
    # Start the worker
    task = asyncio.create_task(runner())
    
    async def gen():
        # Optional: reconnection delay for EventSource
        yield "retry: 1500\n\n"
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                yield item
        finally:
            # Ensure the background task is cancelled if the client goes away
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

@app.get("/health")
async def health():
    return {"ok": True}