"""Walkable-surface mapper and autonomous explorer for PAL.

Builds a sparse 2-D occupancy grid using dead-reckoning odometry
(direction + duration → estimated displacement).  No VRChat position data is
available, so the grid is in "step-units" calibrated to approximate VRChat
walk speed (~2 step-units per second).

Grid cell states:
  UNKNOWN (0) — never visited
  FREE    (1) — bot successfully walked here (scene changed after move)
  WALL    (2) — bot appears blocked here (scene didn't change after move)

Frontier exploration continuously selects unknown cells adjacent to FREE cells
and navigates toward them, producing a systematic sweep of the world rather
than random wandering.

Usage from navigator / main::
    # create once (or retrieve from navigator)
    ex = Explorer()
    ex.set_world('Forest World')
    ex.load_grids(world_knowledge)          # restore from disk

    # before issuing any move command
    ex.notify_move(direction, duration, current_scene)

    # at start of next think cycle, after scene is captured
    stuck = ex.evaluate_stuck(new_scene)    # updates grid; True = wall hit

    # decide next exploration move
    move = ex.next_move()                   # (direction, dur) | None

    # inject into think prompt
    people_ctx += ex.context()

    # before saving world_map to disk
    ex.save_grids(world_knowledge)
"""

import math
import random
import re
import time
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELL_SIZE           = 2.0    # step-units per grid cell edge
WALK_SPEED          = 2.0    # step-units per second (forward/backward/strafe)
TURN_RATE           = 180.0  # degrees per second via /input/LookLeft|LookRight
STUCK_THRESHOLD     = 0.68   # Jaccard word-similarity above this → stuck / no movement
EXPLORE_COOLDOWN    = 2.5    # seconds between explorer-generated moves
SURVEY_TURN_DUR     = 0.5    # 90° turn at default VRChat turn rate
MAP_READY_CELLS     = 12     # free cells before continuous cruise mode activates
ROUTE_STEPS         = 10     # moves to pre-plan per cruise route

UNKNOWN = 0
FREE    = 1
WALL    = 2


# ---------------------------------------------------------------------------
# Scene similarity helper
# ---------------------------------------------------------------------------

def _words(scene: str) -> set:
    return set(re.findall(r'[a-z]{3,}', scene.lower()))


def _similarity(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ---------------------------------------------------------------------------
# Dead-reckoning pose
# ---------------------------------------------------------------------------

class PoseTracker:
    """Tracks estimated (x, y, heading) using issued locomotion commands."""

    def __init__(self) -> None:
        self.x:       float = 0.0
        self.y:       float = 0.0
        self.heading: float = 0.0   # degrees; 0 = initial forward direction

    def reset(self) -> None:
        self.x = self.y = self.heading = 0.0

    def apply(self, direction: str, duration: float) -> None:
        """Update pose after a locomotion command is issued."""
        d   = direction.lower().strip()
        rad = math.radians(self.heading)
        dist = WALK_SPEED * duration

        if d == 'forward':
            self.x +=  dist * math.sin(rad)
            self.y +=  dist * math.cos(rad)
        elif d in ('backward', 'back'):
            self.x -= dist * math.sin(rad)
            self.y -= dist * math.cos(rad)
        elif d in ('left', 'strafe_left'):
            self.x +=  dist * math.cos(rad)
            self.y -= dist * math.sin(rad)
        elif d in ('right', 'strafe_right'):
            self.x -= dist * math.cos(rad)
            self.y +=  dist * math.sin(rad)
        elif d == 'turn_left':
            self.heading = (self.heading - TURN_RATE * duration) % 360
        elif d == 'turn_right':
            self.heading = (self.heading + TURN_RATE * duration) % 360

    @property
    def cell(self) -> Tuple[int, int]:
        """Current grid cell key."""
        return (int(self.x / CELL_SIZE), int(self.y / CELL_SIZE))

    def __repr__(self) -> str:
        return f'PoseTracker(x={self.x:.1f}, y={self.y:.1f}, hdg={self.heading:.0f}°, cell={self.cell})'


# ---------------------------------------------------------------------------
# Sparse occupancy grid
# ---------------------------------------------------------------------------

class OccupancyGrid:
    """Sparse 2-D grid: cell (gx, gy) → UNKNOWN / FREE / WALL."""

    def __init__(self) -> None:
        self._cells: Dict[Tuple[int, int], int] = {}

    def mark(self, cell: Tuple[int, int], state: int) -> None:
        existing = self._cells.get(cell, UNKNOWN)
        # Walls are permanent; FREE promotes UNKNOWN but can't downgrade WALL
        if existing == WALL and state != WALL:
            return
        self._cells[cell] = state

    def get(self, cell: Tuple[int, int]) -> int:
        return self._cells.get(cell, UNKNOWN)

    def free_cells(self) -> List[Tuple[int, int]]:
        return [c for c, s in self._cells.items() if s == FREE]

    def frontier(self) -> List[Tuple[int, int]]:
        """Unknown cells that border at least one FREE cell."""
        result: set = set()
        for (gx, gy), state in list(self._cells.items()):
            if state != FREE:
                continue
            for nb in ((gx+1, gy), (gx-1, gy), (gx, gy+1), (gx, gy-1)):
                if self._cells.get(nb, UNKNOWN) == UNKNOWN:
                    result.add(nb)
        return list(result)

    def nearest_frontier(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        candidates = self.frontier()
        if not candidates:
            return None
        return min(candidates, key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)

    def stats(self) -> Dict[str, int]:
        total = len(self._cells)
        free  = sum(1 for s in self._cells.values() if s == FREE)
        walls = total - free
        return {'total': total, 'free': free, 'walls': walls, 'frontier': len(self.frontier())}

    def to_dict(self) -> Dict[str, int]:
        return {f'{k[0]},{k[1]}': v for k, v in self._cells.items()}

    def from_dict(self, d: Dict) -> None:
        self._cells.clear()
        for k, v in d.items():
            parts = k.split(',')
            if len(parts) == 2:
                try:
                    self._cells[(int(parts[0]), int(parts[1]))] = int(v)
                except ValueError:
                    pass


# ---------------------------------------------------------------------------
# Explorer — coordinates pose + grid + frontier navigation
# ---------------------------------------------------------------------------

class Explorer:
    """Autonomous world mapper combining dead reckoning and frontier exploration."""

    def __init__(self) -> None:
        self.pose:    PoseTracker  = PoseTracker()
        self._grids:  Dict[str, OccupancyGrid] = {}
        self._world:  str          = ''
        self._grid:   OccupancyGrid = OccupancyGrid()

        # Pending move — evaluate stuck on next cycle
        self._pending_dir:   str   = ''
        self._pending_dur:   float = 0.0
        self._pre_scene:     str   = ''
        self._has_pending:   bool  = False

        # Frontier navigation target
        self._target: Optional[Tuple[int, int]] = None

        # Timing
        self._last_explore_t: float = 0.0

        # Survey spin state (used when frontier is empty)
        self._survey_steps: int  = 0
        self._survey_max:   int  = 8   # full 360° survey in ~8 turns of ~45°

        # Respawn / teleport detection
        self._spawn_scene:   str  = ''   # first scene observed in this world (at origin)
        self._scene_history: list = []   # ring buffer of last 5 scenes for continuity check

    # -----------------------------------------------------------------------
    # World switching
    # -----------------------------------------------------------------------

    def set_world(self, world_name: str) -> None:
        """Switch to (or create) the grid for *world_name*.  Resets pose."""
        if world_name == self._world:
            return
        # Stash current grid
        if self._world:
            self._grids[self._world] = self._grid
        self._world  = world_name
        self._grid   = self._grids.get(world_name, OccupancyGrid())
        self.pose.reset()
        self._target      = None
        self._pre_scene   = ''
        self._has_pending = False
        self._survey_steps = 0
        self._spawn_scene   = ''
        self._scene_history = []
        print(f'  🗺  EXPLORER: switched to world "{world_name}"', flush=True)

    def load_grids(self, world_map: Dict) -> None:
        """Restore grids from world_map JSON (call on startup)."""
        for wname, entry in world_map.items():
            gdata = entry.get('occupancy_grid')
            if gdata:
                g = OccupancyGrid()
                g.from_dict(gdata)
                self._grids[wname] = g
                s = g.stats()
                print(f'  🗺  EXPLORER: loaded grid for "{wname}" '
                      f'({s["free"]} free, {s["walls"]} wall cells)', flush=True)
        if self._world in self._grids:
            self._grid = self._grids[self._world]

    def save_grids(self, world_map: Dict) -> None:
        """Persist grids into *world_map* dict (caller must then wm.save())."""
        if self._world:
            self._grids[self._world] = self._grid
        for wname, grid in self._grids.items():
            entry = world_map.setdefault(wname, {})
            entry['occupancy_grid'] = grid.to_dict()

    # -----------------------------------------------------------------------
    # Stuck detection pipeline
    # -----------------------------------------------------------------------

    def notify_move(self, direction: str, duration: float, scene_before: str) -> None:
        """Record that a movement command was issued.

        Call immediately after firing locomotion.move_async().
        """
        self._pending_dir  = direction
        self._pending_dur  = duration
        self._pre_scene    = scene_before
        self._has_pending  = True

    def evaluate_stuck(self, scene_after: str) -> bool:
        """Compare scene after movement to scene before.

        Call at the start of the next think cycle, once *scene_after* is ready.
        Returns True if the bot appears stuck (wall / boundary hit).
        Updates the occupancy grid accordingly and applies pose update.
        """
        if not self._has_pending:
            return False
        self._has_pending = False

        direction = self._pending_dir
        duration  = self._pending_dur

        # Apply pose update regardless of stuck status
        self.pose.apply(direction, duration)
        cell_after = self.pose.cell

        translational = direction in ('forward', 'backward', 'back',
                                      'left',    'right',
                                      'strafe_left', 'strafe_right')

        sim = _similarity(self._pre_scene, scene_after)

        if translational and sim > STUCK_THRESHOLD:
            # Scene barely changed → wall / boundary
            self._grid.mark(cell_after, WALL)
            # Invalidate current target so we pick a new one
            self._target = None
            self._pre_scene = ''
            print(f'  🧱 WALL at cell {cell_after} (scene sim={sim:.2f})', flush=True)
            return True
        else:
            self._grid.mark(cell_after, FREE)
            self._pre_scene = ''
            return False

    def apply_cruise_move(self, direction: str, duration: float) -> None:
        """Apply a cruise-mode move immediately — no deferred stuck detection.

        Drains any held deferred pending move (applies its pose; skips stuck
        comparison), then applies *this* move and marks the new cell FREE.
        Called by the navigator between think cycles for continuous walking.
        """
        if self._has_pending:
            self.pose.apply(self._pending_dir, self._pending_dur)
            self._has_pending = False
            self._pre_scene   = ''
        self.pose.apply(direction, duration)
        self._grid.mark(self.pose.cell, FREE)
        self._last_explore_t = time.time()

    # -----------------------------------------------------------------------
    # Passive observation + respawn detection
    # -----------------------------------------------------------------------

    def observe(self, scene: str) -> None:
        """Mark current cell FREE every cycle and maintain scene history.

        Call once per think cycle with the current scene description.
        Records the spawn scene on first observation, maintains a 5-scene
        ring buffer used by check_respawn().
        """
        cell = self.pose.cell
        self._grid.mark(cell, FREE)
        # Record spawn scene (very first observation in this world)
        if not self._spawn_scene:
            self._spawn_scene = scene
            print(f'  🗺  EXPLORER: spawn scene recorded at cell {cell}', flush=True)
        # Maintain history
        self._scene_history.append(scene)
        if len(self._scene_history) > 5:
            self._scene_history.pop(0)

    def check_respawn(self, current_scene: str) -> bool:
        """Detect if PAL was teleported back to spawn.

        Two signals:
        1. Current scene is very similar to spawn scene AND pose is far from origin.
        2. Scene changed completely in one cycle (sudden jump) AND pose is far from origin.

        Returns True and resets pose to origin if respawn is detected.
        """
        if not self._spawn_scene:
            return False
        cx, cy = self.pose.cell
        dist_from_origin = math.sqrt(cx * cx + cy * cy)
        if dist_from_origin < 1.0:
            return False  # Already at origin — not a respawn

        spawn_sim = _similarity(current_scene, self._spawn_scene)

        # Signal 1: back at spawn location
        if spawn_sim > 0.45:
            self._handle_respawn(current_scene, f'spawn match (sim={spawn_sim:.2f})')
            return True

        # Signal 2: sudden scene discontinuity (teleport / fall)
        if len(self._scene_history) >= 2:
            prev_scene = self._scene_history[-2]
            continuity = _similarity(prev_scene, current_scene)
            if continuity < 0.15 and dist_from_origin > 3.0:
                self._handle_respawn(current_scene, f'scene jump (continuity={continuity:.2f})')
                return True

        return False

    def _handle_respawn(self, scene: str, reason: str) -> None:
        """Reset pose to origin and update spawn scene after a respawn."""
        prev_cell = self.pose.cell
        self.pose.reset()
        self._has_pending  = False
        self._pre_scene    = ''
        self._target       = None
        self._survey_steps = 0
        self._grid._cells.clear()        # wipe old map — after respawn the world starts fresh
        self._grid.mark((0, 0), FREE)
        print(f'  🔄 RESPAWN ({reason}) — was at cell {prev_cell}, reset to origin', flush=True)
        # Refresh spawn scene (spawn may look slightly different each time)
        if _similarity(scene, self._spawn_scene) < 0.85:
            self._spawn_scene = scene

    def check_external_move(self, current_scene: str) -> bool:
        """Detect if the player was moved externally (by another player, portal, world mechanic).

        Call AFTER observe() and check_respawn().  Only fires when PAL did NOT issue a move
        itself (caller must gate on _pre_move_scene / pal_moved_prev_cycle).

        Detection: significant scene discontinuity in one cycle without PAL locomotion.
        Returns True when an unexpected relocation is detected; invalidates nav target.
        """
        if len(self._scene_history) < 2:
            return False
        prev_scene = self._scene_history[-2]   # scene from the previous think cycle
        continuity = _similarity(prev_scene, current_scene)
        if continuity < 0.20:
            self._target = None   # stale nav target is meaningless after relocation
            print(f'  📡 EXTERNAL MOVE detected (scene continuity={continuity:.2f})', flush=True)
            return True
        return False

    # -----------------------------------------------------------------------
    # ASCII map rendering
    # -----------------------------------------------------------------------

    def render_ascii(self, width: int = 52, height: int = 20) -> str:
        """Render the occupancy grid as a fixed-size ASCII art with X/Y axes.

        Legend:  @=current position  O=origin (0,0)  .=walkable  #=wall  space=unknown
        Y axis increases upward (north/forward); X axis increases rightward (east/right).
        """
        cells = self._grid._cells
        cx, cy = self.pose.cell

        all_coords: list = list(cells.keys())
        if not all_coords:
            all_coords = [(0, 0)]
        all_coords.append((cx, cy))
        all_coords.append((0, 0))  # always include origin

        xs = [c[0] for c in all_coords]
        ys = [c[1] for c in all_coords]
        min_x, max_x = min(xs) - 2, max(xs) + 2
        min_y, max_y = min(ys) - 2, max(ys) + 2

        range_x = max(max_x - min_x, 1)
        range_y = max(max_y - min_y, 1)

        # Y-axis label prefix width (e.g. "+12 " or "-4  ")
        YLABEL_W = 5

        rows: List[str] = []
        for row_i in range(height):
            # Invert y axis: row 0 = top = largest world-y (north/forward)
            world_y = max_y - row_i * range_y / max(height - 1, 1)
            gy = round(world_y)

            # Y-axis tick label — only on rows that are close to an integer step unit
            if abs(world_y - round(world_y)) < (range_y / height) * 0.6:
                wy_label = f'{gy * CELL_SIZE:+.0f}'.rjust(YLABEL_W - 1) + ' '
            else:
                wy_label = ' ' * YLABEL_W

            row: List[str] = []
            for col_i in range(width):
                world_x = min_x + col_i * range_x / max(width - 1, 1)
                gx = round(world_x)
                if gx == cx and gy == cy:
                    row.append('@')
                elif gx == 0 and gy == 0:
                    # Origin marker — show only when not occupied by @
                    row.append('O')
                else:
                    state = cells.get((gx, gy), UNKNOWN)
                    if state == FREE:
                        row.append('.')
                    elif state == WALL:
                        row.append('#')
                    else:
                        row.append(' ')
            rows.append(wy_label + ''.join(row))

        # X-axis tick labels below the map
        x_label_row_1 = ' ' * YLABEL_W
        x_label_row_2 = ' ' * YLABEL_W
        tick_every = max(1, width // 8)
        for col_i in range(0, width, tick_every):
            world_x = min_x + col_i * range_x / max(width - 1, 1)
            gx = round(world_x)
            wx_val = f'{gx * CELL_SIZE:+.0f}'
            # Place the label centred on col_i in x_label_row_2
            insert_at = col_i + YLABEL_W - len(wx_val) // 2
            pad = insert_at - len(x_label_row_1)
            if pad >= 0:
                x_label_row_1 += ' ' * pad + '|'
                x_label_row_2 += ' ' * (insert_at - len(x_label_row_2)) + wx_val

        # Stats + legend
        s    = self._grid.stats()
        hdg  = int(self.pose.heading) % 360
        rx   = self.pose.x
        ry   = self.pose.y
        legend = (
            f'@ you ({rx:+.1f}, {ry:+.1f}) hdg{hdg}°  '
            f'O origin (0,0)  . walkable  # wall\n'
            f'explored {s["free"]} cells · {s["walls"]} walls · '
            f'{s["frontier"]} unexplored edges · '
            f'X=east/right  Y=north/forward'
        )
        return '\n'.join(rows) + '\n' + x_label_row_1 + '\n' + x_label_row_2 + '\n' + legend

    # -----------------------------------------------------------------------
    # Frontier navigation
    # -----------------------------------------------------------------------

    def _direction_to_cell(self, target: Tuple[int, int]) -> Tuple[str, float]:
        """Single move step toward *target* cell."""
        cx, cy = self.pose.cell
        tx, ty = target
        if (cx, cy) == (tx, ty):
            return ('forward', 0.8)

        # Desired world angle (heading 0 = +Y is forward, clockwise)
        desired = math.degrees(math.atan2(tx - cx, ty - cy)) % 360
        current = self.pose.heading % 360
        error   = (desired - current + 540) % 360 - 180   # −180…+180

        if abs(error) > 20:
            # Turn toward target first — 90° = 0.5 s
            turn_dur = min(abs(error) / TURN_RATE, 0.5)
            return ('turn_right' if error > 0 else 'turn_left', round(turn_dur, 2))
        else:
            dist     = math.sqrt((tx-cx)**2 + (ty-cy)**2) * CELL_SIZE
            walk_dur = min(max(0.4, dist / WALK_SPEED), 1.0)
            return ('forward', round(walk_dur, 1))

    def next_move(self) -> Optional[Tuple[str, float]]:
        """Return next exploration move or None if on cooldown / not ready.

        Prioritises frontier cells (unexplored edges of known-free area).
        Falls back to a survey spin when no frontier is known yet.
        """
        now = time.time()
        if now - self._last_explore_t < EXPLORE_COOLDOWN:
            return None

        # Always mark current cell as FREE (we're standing here)
        self._grid.mark(self.pose.cell, FREE)

        # Refresh target if it has been reached or turned into a wall
        if self._target is not None:
            state = self._grid.get(self._target)
            if state != UNKNOWN:
                self._target = None

        if self._target is None:
            cx, cy = self.pose.cell
            frontiers = list(self._grid.frontier())
            if frontiers:
                r = random.random()
                if r < 0.35:
                    # Farthest frontier — forces long-range exploration in neglected areas
                    self._target = max(frontiers, key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)
                elif r < 0.70:
                    # Non-forward frontier — avoids the "always walk ahead" bias
                    self._target = self._pick_diverse_frontier(frontiers, cx, cy)
                else:
                    # Nearest frontier — original greedy behaviour
                    self._target = self._grid.nearest_frontier(cx, cy)

        if self._target is None:
            # No frontier yet — survey all cardinal directions to seed new cells.
            # 8-step cycle: 0.5s turn = 90° (TURN_RATE=180°/s), one turn per cardinal.
            # Steps: face E → seed E → face S → seed S → face W → seed W → face N → vary
            step = self._survey_steps % 8
            self._survey_steps += 1
            if step == 0:            # turn 90° right → face east
                direction, duration = 'turn_right', 0.5
            elif step == 1:          # walk forward to seed east cells
                direction, duration = 'forward', 1.0
            elif step == 2:          # turn 90° right → face south
                direction, duration = 'turn_right', 0.5
            elif step == 3:          # walk forward to seed south cells
                direction, duration = 'forward', 1.0
            elif step == 4:          # turn 90° right → face west
                direction, duration = 'turn_right', 0.5
            elif step == 5:          # walk forward to seed west cells
                direction, duration = 'forward', 1.0
            elif step == 6:          # turn 90° right → back to north (full 360°)
                direction, duration = 'turn_right', 0.5
            else:                    # step 7 — lateral variation to break symmetry
                direction, duration = random.choice([('left', 0.8), ('right', 0.8), ('forward', 1.0)])
            self._last_explore_t = now
            return (direction, duration)

        # Navigate one step toward frontier target
        direction, duration = self._direction_to_cell(self._target)
        self._last_explore_t = now
        return (direction, duration)

    def _pick_diverse_frontier(
        self, frontiers: list, cx: int, cy: int
    ) -> Tuple[int, int]:
        """Return a frontier that is at least 60° off current heading, else random."""
        hdg = self.pose.heading
        non_forward = []
        for tx, ty in frontiers:
            angle = math.degrees(math.atan2(tx - cx, ty - cy)) % 360
            diff  = abs((angle - hdg + 180) % 360 - 180)
            if diff > 60:
                non_forward.append((tx, ty))
        return random.choice(non_forward) if non_forward else random.choice(frontiers)

    @property
    def map_ready(self) -> bool:
        """True once the grid has enough free cells for meaningful cruise routing."""
        return len(self._grid.free_cells()) >= MAP_READY_CELLS

    def plan_route(self, steps: int = ROUTE_STEPS) -> List[Tuple[str, float]]:
        """Pre-plan *steps* moves toward frontier(s) by simulating pose advancement.

        Purely advisory — does NOT modify real state. Returns a list of
        (direction, duration) pairs for sequential execution via apply_cruise_move().
        """
        frontiers = list(self._grid.frontier())
        if not frontiers:
            return []

        # Snapshot current pose for simulation
        sim_x       = self.pose.x
        sim_y       = self.pose.y
        sim_heading = self.pose.heading

        # Pick initial target using diverse selection
        cx, cy = int(sim_x / CELL_SIZE), int(sim_y / CELL_SIZE)
        r = random.random()
        if r < 0.35:
            sim_target: Tuple[int, int] = max(
                frontiers, key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)
        elif r < 0.70:
            sim_target = self._pick_diverse_frontier(frontiers, cx, cy)
        else:
            sim_target = min(frontiers, key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)

        remaining = list(frontiers)
        moves: List[Tuple[str, float]] = []

        for _ in range(steps):
            sim_cx = int(sim_x / CELL_SIZE)
            sim_cy = int(sim_y / CELL_SIZE)
            if (sim_cx, sim_cy) == sim_target:
                remaining = [f for f in remaining if f != sim_target]
                if not remaining:
                    break
                sim_target = self._pick_diverse_frontier(remaining, sim_cx, sim_cy)

            tx, ty  = sim_target
            desired = math.degrees(math.atan2(tx - sim_cx, ty - sim_cy)) % 360
            error   = (desired - sim_heading % 360 + 540) % 360 - 180

            if abs(error) > 20:
                turn_dur  = min(abs(error) / TURN_RATE, 0.5)
                direction = 'turn_right' if error > 0 else 'turn_left'
                duration  = round(turn_dur, 2)
            else:
                dist     = math.sqrt((tx - sim_cx)**2 + (ty - sim_cy)**2) * CELL_SIZE
                walk_dur = min(max(0.4, dist / WALK_SPEED), 1.0)
                direction = 'forward'
                duration  = round(walk_dur, 1)

            moves.append((direction, duration))

            # Advance simulated pose
            rad    = math.radians(sim_heading)
            dist_s = WALK_SPEED * duration
            if direction == 'forward':
                sim_x       +=  dist_s * math.sin(rad)
                sim_y       +=  dist_s * math.cos(rad)
            elif direction == 'turn_right':
                sim_heading  = (sim_heading + TURN_RATE * duration) % 360
            elif direction == 'turn_left':
                sim_heading  = (sim_heading - TURN_RATE * duration) % 360

        return moves

    # -----------------------------------------------------------------------
    # Context and status
    # -----------------------------------------------------------------------

    def context(self) -> str:
        """Short context block for injecting into the think prompt."""
        if not self._world:
            return ''
        s = self._grid.stats()
        rx, ry = self.pose.x, self.pose.y
        hdg    = int(self.pose.heading) % 360
        if s['total'] == 0:
            return (f'[Explorer: world is new — starting to map walkable surface. '
                    f'Current position: X={rx:+.1f} Y={ry:+.1f} (origin is spawn point 0,0)]')
        cx, cy = self.pose.cell
        parts = [
            f'[Explorer map: pos X={rx:+.1f} Y={ry:+.1f} hdg={hdg}° | '
            f'{s["free"]} walkable cells, {s["walls"]} walls, '
            f'{s["frontier"]} unexplored edges — '
        ]
        if self._target:
            tx, ty = self._target
            dist = math.sqrt((tx-cx)**2 + (ty-cy)**2) * CELL_SIZE
            parts.append(f'heading toward frontier ~{dist:.1f} units away]')
        else:
            parts.append('scanning for new areas to explore]')
        return ''.join(parts)

    @property
    def status(self) -> str:
        s    = self._grid.stats()
        rx, ry = self.pose.x, self.pose.y
        hdg  = int(self.pose.heading) % 360
        tgt  = f'→({self._target[0]*CELL_SIZE:+.0f},{self._target[1]*CELL_SIZE:+.0f})' if self._target else '→?'
        return (
            f'🗺  {s["free"]}free/{s["walls"]}wall/{s["frontier"]}frontier | '
            f'X={rx:+.1f} Y={ry:+.1f} hdg{hdg}° {tgt}'
        )
