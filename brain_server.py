#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_server.py
===============
WebSocket-сервер симуляции мозга дрозофилы со статической структурой весов,
кратковременной памятью (leaky integrator) и расширенными сенсорными входами.
"""

import asyncio
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HOST = "127.0.0.1"
PORT = 8765
GRAPH_PATH = Path(__file__).resolve().parent / "brain_subgraph_v1" / "graph.json"

class FlyBrainEngine:
    def __init__(self, graph_file: Path):
        self.graph_file = graph_file
        self.input_vision_nodes = []  
        self.input_smell_nodes = []   
        self.input_touch_nodes = []    
        self.output_nodes = []        
        
        # Фиксированные веса (генерируются один раз при старте)
        self.W_vis = None
        self.W_smell = None
        self.W_touch = None
        self.W_out = None
        self.W_vis_hidden = None
        self.W_smell_hidden = None
        self.W_touch_hidden = None
        self.W_recurrent = None       # Матрица внутренней памяти (рекуррентные связи)
        
        self.hidden_state = None      # Внутреннее состояние памяти мозга
        
        self._load_and_init_graph()

    def _load_and_init_graph(self):
        if not self.graph_file.exists():
            logging.warning(f"Файл графа не найден. Используется резервный режим.")
            self._init_dummy()
            return

        with open(self.graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        
        self.input_vision_nodes = [n["id"] for n in nodes if n.get("base_region") == "LP" or "LP" in n.get("regions", "")]
        if not self.input_vision_nodes:
            self.input_vision_nodes = [n["id"] for n in nodes[:32]]

        self.input_smell_nodes = [n["id"] for n in nodes if n.get("base_region") == "AL" or "AL" in n.get("regions", "")]
        if not self.input_smell_nodes:
            self.input_smell_nodes = [n["id"] for n in nodes[32:40]]

        self.input_touch_nodes = [n["id"] for n in nodes if n.get("base_region") == "MB" or "MB" in n.get("regions", "")]
        if not self.input_touch_nodes:
            self.input_touch_nodes = [n["id"] for n in nodes[40:48]]

        self.output_nodes = [n["id"] for n in nodes if n.get("is_descending") or n.get("super_class") == "descending"]
        if not self.output_nodes:
            self.output_nodes = [n["id"] for n in nodes[-16:]]

        logging.info(f"Зрительные нейроны (LP): {len(self.input_vision_nodes)}")
        logging.info(f"Обонятельные нейроны (AL): {len(self.input_smell_nodes)}")
        logging.info(f"Тактильные нейроны (MB): {len(self.input_touch_nodes)}")
        logging.info(f"Моторные нейроны (DN): {len(self.output_nodes)}")

        self._init_weights()

    def _init_dummy(self):
        self.input_vision_nodes = [f"vis_{i}" for i in range(32)]
        self.input_smell_nodes = [f"smell_{i}" for i in range(4)]
        self.input_touch_nodes = [f"touch_{i}" for i in range(4)]
        self.output_nodes = [f"out_{i}" for i in range(16)]
        self._init_weights()

    def _init_weights(self):
        # Генерация всех весов строго ОДИН РАЗ при старте
        num_out = len(self.output_nodes)
        self.W_vis = np.random.randn(32, len(self.input_vision_nodes)) * 0.1
        self.W_smell = np.random.randn(4, len(self.input_smell_nodes)) * 0.3  # Расширено под подробный запах
        self.W_touch = np.random.randn(4, len(self.input_touch_nodes)) * 0.05
        
        self.W_vis_hidden = np.random.randn(len(self.input_vision_nodes), num_out) * 0.05
        self.W_smell_hidden = np.random.randn(len(self.input_smell_nodes), num_out) * 0.1
        self.W_touch_hidden = np.random.randn(len(self.input_touch_nodes), num_out) * 0.05
        
        # Рекуррентные веса памяти для скрытого слоя моторики (сохранение динамики сессии)
        self.W_recurrent = np.random.randn(num_out, num_out) * 0.04
        self.hidden_state = np.zeros(num_out, dtype=np.float32)

        self.W_out = np.random.randn(num_out, 2) * 0.1

    def process_step(self, ray_distances: list[float], ray_origins: list[list[float]], smell_data: list[float], touch_data: list[float]) -> list[float]:
        rays = np.array(ray_distances, dtype=np.float32)
        if len(rays) != 32:
            rays = np.pad(rays, (0, max(0, 32 - len(rays))))[:32]

        smell = np.array(smell_data, dtype=np.float32)
        if len(smell) != 4:
            smell = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        touch = np.array(touch_data, dtype=np.float32)
        if len(touch) != 4:
            touch = np.zeros(4, dtype=np.float32)

        # Активность сенсорных слоев с учетом геометрии лучей (ray_origins учитывают анатомическое смещение)
        vis_act = np.tanh(np.dot(rays, self.W_vis))
        smell_act = np.tanh(np.dot(smell, self.W_smell))
        touch_act = np.tanh(np.dot(touch, self.W_touch))

        # Проекции к моторике через статические веса
        h_vis = np.dot(vis_act, self.W_vis_hidden)
        h_smell = np.dot(smell_act, self.W_smell_hidden)
        h_touch = np.dot(touch_act, self.W_touch_hidden)

        # Реалистичная память (Leaky Integrator с рекуррентной динамикой сессии)
        # 0.7 - коэффициент затухания (памяти), позволяющий удерживать контекст прошлой секунды
        instantaneous_input = h_vis + h_smell + h_touch
        self.hidden_state = np.tanh(0.7 * self.hidden_state + 0.3 * instantaneous_input + np.dot(self.hidden_state, self.W_recurrent))

        output_activation = np.tanh(self.hidden_state)
        raw_vector = np.dot(output_activation, self.W_out)
        vector = np.clip(raw_vector, -1.0, 1.0)

        return [float(vector[0]), float(vector[1])]

async def handle_client(websocket, brain_engine: FlyBrainEngine):
    try:
        async for message in websocket:
            data = json.loads(message)
            rays = data.get("rays", [])
            ray_origins = data.get("ray_origins", [])
            smell = data.get("smell", [0.0, 0.0, 0.0, 0.0])
            touch = data.get("touch", [0.0, 0.0, 0.0, 0.0])
            
            vx, vy = brain_engine.process_step(rays, ray_origins, smell, touch)
            await websocket.send(json.dumps({"status": "ok", "vector": [vx, vy]}))
    except Exception as e:
        logging.info(f"Соединение закрыто: {e}")

async def main():
    try:
        import websockets
    except ImportError:
        print("Установите библиотеку websockets: pip install websockets")
        return

    brain = FlyBrainEngine(GRAPH_PATH)
    logging.info(f"Запуск сервера мозга на ws://{HOST}:{PORT}")
    async with websockets.serve(lambda ws: handle_client(ws, brain), HOST, PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
