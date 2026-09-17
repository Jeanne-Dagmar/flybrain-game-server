import asyncio
import json
import math
import sys
import threading
import pygame
import websockets

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
FPS = 60

PLAYER_WIDTH = 44
PLAYER_HEIGHT = 12
PLAYER_SPEED = 2.0
RAY_COUNT = 32
RAY_MAX_DIST = 100.0  

SMELL_MAX_DIST = 400.0  

SERVER_URI = "ws://127.0.0.1:8765"

COLOR_BG = (15, 15, 20)
COLOR_WALL = (70, 80, 100)
COLOR_PLAYER = (255, 215, 0)
COLOR_PLAYER_BUMP = (255, 50, 50)
COLOR_RAY = (0, 255, 150)
COLOR_SMELL_SOURCE = (255, 105, 180)
COLOR_TEXT = (200, 200, 200)

def get_perimeter_ray(rect: pygame.Rect, ray_index: int, total_rays: int):
    angle = (2 * math.pi / total_rays) * ray_index
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)

    cx, cy = rect.centerx, rect.centery
    half_w = rect.width / 2.0
    half_h = rect.height / 2.0

    dx = dir_x if abs(dir_x) > 1e-6 else 1e-6
    dy = dir_y if abs(dir_y) > 1e-6 else 1e-6

    tx = half_w / abs(dx)
    ty = half_h / abs(dy)

    t_edge = min(tx, ty)
    start_x = cx + dx * t_edge
    start_y = cy + dy * t_edge

    local_offset = [start_x - cx, start_y - cy]
    return (start_x, start_y), local_offset, (dir_x, dir_y)

def cast_rays(player_rect: pygame.Rect, walls: list[pygame.Rect], smell_pos: tuple[float, float]):
    ray_distances = []
    ray_origins = []
    ray_lines = []
    ray_colors = []

    sx, sy = smell_pos
    smell_radius = 12.0  

    for i in range(RAY_COUNT):
        (start_x, start_y), local_offset, (dir_x, dir_y) = get_perimeter_ray(player_rect, i, RAY_COUNT)
        
        hit_dist = RAY_MAX_DIST
        end_x = start_x + dir_x * RAY_MAX_DIST
        end_y = start_y + dir_y * RAY_MAX_DIST
        hit_type = "none"

        step = 3.0
        curr_dist = 0.0
        while curr_dist < RAY_MAX_DIST:
            px = start_x + dir_x * curr_dist
            py = start_y + dir_y * curr_dist

            if px <= 0 or px >= SCREEN_WIDTH or py <= 0 or py >= SCREEN_HEIGHT:
                hit_dist = curr_dist
                end_x, end_y = px, py
                hit_type = "wall"
                break

            if math.hypot(px - sx, py - sy) <= smell_radius:
                hit_dist = curr_dist
                end_x, end_y = px, py
                hit_type = "smell"
                break

            hit_wall = False
            for wall in walls:
                if wall.collidepoint(px, py):
                    hit_dist = curr_dist
                    end_x, end_y = px, py
                    hit_wall = True
                    break
            if hit_wall:
                hit_type = "wall"
                break

            curr_dist += step

        if hit_type == "smell":
            norm_dist = -0.5  
            ray_colors.append((255, 105, 180))
        else:
            norm_dist = hit_dist / RAY_MAX_DIST
            ray_colors.append(COLOR_RAY)

        ray_distances.append(norm_dist)
        ray_origins.append(local_offset)
        ray_lines.append(((start_x, start_y), (end_x, end_y)))

    return ray_distances, ray_origins, ray_lines, ray_colors

def calculate_detailed_smell(player_rect: pygame.Rect, smell_pos: tuple[float, float]) -> list[float]:
    cx, cy = player_rect.centerx, player_rect.centery
    
    left_antenna = (cx - player_rect.width / 4, cy)
    right_antenna = (cx + player_rect.width / 4, cy)

    sx, sy = smell_pos

    dist_left = math.hypot(sx - left_antenna[0], sy - left_antenna[1])
    dist_right = math.hypot(sx - right_antenna[0], sy - right_antenna[1])
    dist_center = math.hypot(sx - cx, sy - cy)

    smell_l = max(0.0, 1.0 - (dist_left / SMELL_MAX_DIST))
    smell_r = max(0.0, 1.0 - (dist_right / SMELL_MAX_DIST))
    
    grad_x = (sx - cx) / SMELL_MAX_DIST
    grad_y = (sy - cy) / SMELL_MAX_DIST

    return [smell_l, smell_r, grad_x, grad_y]

def build_maze() -> list[pygame.Rect]:
    return [
        pygame.Rect(0, 0, SCREEN_WIDTH, 10),
        pygame.Rect(0, SCREEN_HEIGHT - 10, SCREEN_WIDTH, 10),
        pygame.Rect(0, 0, 10, SCREEN_HEIGHT),
        pygame.Rect(SCREEN_WIDTH - 10, 0, 10, SCREEN_HEIGHT),
        
        pygame.Rect(100, 100, 20, 280),
        pygame.Rect(200, 100, 240, 20),
        pygame.Rect(200, 220, 150, 20),
        pygame.Rect(420, 200, 20, 180),
        pygame.Rect(500, 80, 20, 260),
        pygame.Rect(280, 320, 140, 20),
        pygame.Rect(100, 380, 220, 20),
    ]

class GameState:
    def __init__(self):
        self.player_rect = pygame.Rect(40, 40, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.smell_source = (50, 120)  
        self.move_vector = [0.0, 0.0]
        self.ray_distances = [1.0] * RAY_COUNT
        self.ray_origins = [[0.0, 0.0]] * RAY_COUNT
        self.smell_values = [0.0, 0.0, 0.0, 0.0]
        self.touch_values = [0.0, 0.0, 0.0, 0.0]
        self.connected = False
        self.running = True

async def network_loop(state: GameState):
    while state.running:
        try:
            async with websockets.connect(SERVER_URI) as websocket:
                state.connected = True
                print("[Network] Подключено к серверу мозга!")
                
                while state.running:
                    payload = json.dumps({
                        "rays": state.ray_distances,
                        "ray_origins": state.ray_origins,
                        "smell": state.smell_values,
                        "touch": state.touch_values
                    })
                    await websocket.send(payload)

                    response = json.loads(await websocket.recv())
                    if "vector" in response:
                        state.move_vector = response["vector"]

                    await asyncio.sleep(0.033)
        except (websockets.exceptions.ConnectionClosed, OSError):
            state.connected = False
            await asyncio.sleep(1.0)

def start_network_thread(state: GameState):
    threading.Thread(target=lambda: asyncio.run(network_loop(state)), daemon=True).start()

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("FlyBrain")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14)

    walls = build_maze()
    state = GameState()
    start_network_thread(state)

    while state.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state.running = False

        distances, origins, ray_lines, ray_colors = cast_rays(state.player_rect, walls, state.smell_source)
        state.ray_distances = distances
        state.ray_origins = origins
        state.smell_values = calculate_detailed_smell(state.player_rect, state.smell_source)

        vx, vy = state.move_vector[0], state.move_vector[1]
        touch_top, touch_bottom, touch_left, touch_right = 0.0, 0.0, 0.0, 0.0

        if vx != 0:
            state.player_rect.x += int(vx * PLAYER_SPEED)
            for wall in walls:
                if state.player_rect.colliderect(wall):
                    if vx > 0:
                        state.player_rect.right = wall.left
                        touch_right = 0.3
                    elif vx < 0:
                        state.player_rect.left = wall.right
                        touch_left = 0.3

        if vy != 0:
            state.player_rect.y += int(vy * PLAYER_SPEED)
            for wall in walls:
                if state.player_rect.colliderect(wall):
                    if vy > 0:
                        state.player_rect.bottom = wall.top
                        touch_bottom = 0.3
                    elif vy < 0:
                        state.player_rect.top = wall.bottom
                        touch_top = 0.3

        state.touch_values = [touch_top, touch_bottom, touch_left, touch_right]
        state.player_rect.clamp_ip(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        
        screen.fill(COLOR_BG)

        for wall in walls:
            pygame.draw.rect(screen, COLOR_WALL, wall)

        pygame.draw.circle(screen, COLOR_SMELL_SOURCE, state.smell_source, 10)

        for idx, (start_pt, end_pt) in enumerate(ray_lines):
            pygame.draw.line(screen, ray_colors[idx], start_pt, end_pt, 1)

        is_touching = any(state.touch_values)
        player_color = COLOR_PLAYER_BUMP if is_touching else COLOR_PLAYER
        pygame.draw.rect(screen, player_color, state.player_rect)

        status_color = (0, 255, 0) if state.connected else (255, 50, 50)
        screen.blit(font.render("SERVER: " + ("CONNECTED" if state.connected else "SEARCHING"), True, status_color), (15, 15))
        screen.blit(font.render(f"Smell Vector: [{state.smell_values[0]:.2f}, {state.smell_values[1]:.2f}, {state.smell_values[2]:.2f}, {state.smell_values[3]:.2f}]", True, COLOR_SMELL_SOURCE), (15, 32))
        screen.blit(font.render(f"Vector: [{vx:.2f}, {vy:.2f}]", True, COLOR_TEXT), (15, 49))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
