import asyncio
import json
import logging
from typing import Set
from queue import Queue, Empty

import websockets
from websockets.server import WebSocketServerProtocol


async def _broadcast_loop(queue: Queue, clients: Set[WebSocketServerProtocol]):
    """Forward messages from queue to all connected clients."""
    while True:
        try:
            data = queue.get_nowait()
        except Empty:
            await asyncio.sleep(0.1)
            continue

        message: str
        if isinstance(data, str):
            message = data
        else:
            try:
                message = json.dumps(data)
            except Exception:
                message = json.dumps({"data": str(data)})

        stale = set()
        for ws in clients:
            try:
                await ws.send(message)
            except websockets.exceptions.ConnectionClosed:
                stale.add(ws)
        for ws in stale:
            clients.discard(ws)


async def _client_handler(websocket: WebSocketServerProtocol, clients: Set[WebSocketServerProtocol]):
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)


async def _run(queue: Queue, host: str, port: int):
    clients: Set[WebSocketServerProtocol] = set()
    async with websockets.serve(lambda ws, _: _client_handler(ws, clients), host, port):
        logging.info("WebSocket server running on %s:%d", host, port)
        await _broadcast_loop(queue, clients)


def run_ws_server(queue: Queue, host: str = "0.0.0.0", port: int = 8765) -> None:
    """Start a simple WebSocket server forwarding queue messages as JSON."""
    try:
        asyncio.run(_run(queue, host, port))
    except KeyboardInterrupt:
        pass
