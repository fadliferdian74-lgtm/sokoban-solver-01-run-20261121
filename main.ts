// sokoban-solver: a push-optimal Sokoban solver.
//
// Two ideas carry the search:
//
// Decomposition. The levels this program answers are built as rooms off a hallway, each room
// joined to the rest of the board through a single door square, with as many goals as boxes
// inside. A box can never leave its room (pushed through the door it could never come back),
// so a level's push optimum is the sum of its rooms' optima, and each room can be solved on
// its own: find a square whose removal seals off a part of the board holding boxes and an
// equal number of goals but not the player, solve that part with the player starting on the
// hallway square outside the door and required to be able to walk back to it, then seal the
// part off and move on. A narrow passage inside a room splits it in the same way, and the
// piece cut off there may not solve on its own, so candidate doors are tried in order —
// gaps in a wall band and room-sized pieces first — and a door whose piece will not solve
// is struck out and the search moves on. Whatever the split leaves behind is solved whole.
//
// Push search. Each (sub)level is solved by A* over pushes, not steps: a state is the set of
// box squares plus the player's reachable region (named by its smallest square, since only
// the region matters for what can be pushed next). The cost so far is pushes spent, so the
// first goal state settled is push-optimal. The estimate is a minimum-cost matching between
// boxes and goals, where pairing a box with a goal costs the pushes a lone box needs to
// travel between the squares (breadth-first search in push space, walls only): each remaining
// push moves one box one square, so the matching lower-bounds the pushes still needed, and
// when no finite matching exists the position can never be solved. Two deadlock tests prune
// states before they are queued: dead squares (a lone box there can reach no goal at all)
// and frozen boxes (boxes that can never move again — the greatest set closed under "every
// axis blocked by a wall or another frozen box" — while not on a goal).
//
// Budgets are counted in expansions, never in time, so two runs on the same input write the
// same bytes. If the optimal search exhausts its budget, a greedy best-first search (same
// machinery, estimate only) still tries to answer with some solution rather than none.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Direction order is fixed (u, d, l, r) so every traversal is deterministic.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];
const INF = 1_000_000_000;

// ---------------------------------------------------------------------------
// Level
// ---------------------------------------------------------------------------

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array; // 1 on goal squares
  goals: number[]; // goal squares, ascending
  boxes: number[]; // box squares, ascending
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h);
  const goal = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  for (let y = 0; y < h; y++) {
    const row = rows[y]!;
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') wall[i] = 1;
      if (ch === '.' || ch === '*' || ch === '+') {
        goal[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, goals, boxes, player };
}

// Breadth-first walk of the player from `from` to `to` over squares that are neither wall
// nor box; returns the step letters, or null when no walk exists.
function walkPath(lv: Level, boxes: Set<number>, from: number, to: number): string | null {
  if (from === to) return '';
  const { w, wall } = lv;
  const prev = new Map<number, [number, string]>();
  const queue = [from];
  prev.set(from, [-1, '']);
  for (let head = 0; head < queue.length; head++) {
    const c = queue[head]!;
    if (c === to) break;
    for (let d = 0; d < 4; d++) {
      const nx = (c % w) + DX[d]!;
      const ny = Math.floor(c / w) + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
      const q = ny * w + nx;
      if (wall[q] || boxes.has(q) || prev.has(q)) continue;
      prev.set(q, [c, LETTER[d]!]);
      queue.push(q);
    }
  }
  if (!prev.has(to)) return null;
  let path = '';
  for (let c = to; c !== from; ) {
    const [p, letter] = prev.get(c)!;
    path = letter + path;
    c = p;
  }
  return path;
}

// Replay a move string on a level; returns its push count when every step is legal and the
// replay ends with every box on a goal, else -1.
function replay(lv: Level, moves: string): number {
  const { w, wall, goal } = lv;
  const boxes = new Set(lv.boxes);
  let player = lv.player;
  let pushes = 0;
  for (const ch of moves) {
    const d = LETTER.indexOf(ch.toLowerCase());
    if (d === -1) return -1;
    const step = DY[d]! * w + DX[d]!;
    const dest = player + step;
    if (dest < 0 || dest >= wall.length || wall[dest]) return -1;
    if (DX[d] !== 0 && Math.floor(dest / w) !== Math.floor(player / w)) return -1;
    if (boxes.has(dest)) {
      const beyond = dest + step;
      if (beyond < 0 || beyond >= wall.length || wall[beyond] || boxes.has(beyond)) return -1;
      if (DX[d] !== 0 && Math.floor(beyond / w) !== Math.floor(dest / w)) return -1;
      boxes.delete(dest);
      boxes.add(beyond);
      pushes++;
    }
    player = dest;
  }
  for (const b of boxes) if (!goal[b]) return -1;
  return pushes;
}

// ---------------------------------------------------------------------------
// Push distances and dead squares
// ---------------------------------------------------------------------------

// pushDist[g][s]: pushes a lone box needs to go from square s to goal g (walls only), or INF.
// Computed by walking the push graph backwards from the goal: a box now on t came from t - d
// when it was pushed along d, which needed the square t - 2d free for the player to stand on.
function computePushDist(lv: Level): Int32Array[] {
  const { w, wall } = lv;
  const n = lv.w * lv.h;
  const table: Int32Array[] = [];
  for (const g of lv.goals) {
    const dist = new Int32Array(n).fill(INF);
    dist[g] = 0;
    const queue = [g];
    for (let head = 0; head < queue.length; head++) {
      const t = queue[head]!;
      for (let d = 0; d < 4; d++) {
        const from = t - (DY[d]! * w + DX[d]!);
        const stand = t - 2 * (DY[d]! * w + DX[d]!);
        if (from < 0 || from >= n || stand < 0 || stand >= n) continue;
        if (wall[from] || wall[stand]) continue;
        if (DX[d] !== 0 && Math.floor(from / w) !== Math.floor(t / w)) continue;
        if (dist[from] === INF) {
          dist[from] = dist[t]! + 1;
          queue.push(from);
        }
      }
    }
    table.push(dist);
  }
  return table;
}

// ---------------------------------------------------------------------------
// Minimum-cost matching (Hungarian method, square cost matrix)
// ---------------------------------------------------------------------------

// cost[i][j]: pairing box i with goal j. Returns the total of the cheapest perfect matching,
// or INF when some pairing it would need is impossible.
function matchingCost(cost: Int32Array[], n: number): number {
  const u = new Float64Array(n + 1);
  const v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1).fill(-1);
  const way = new Int32Array(n + 1).fill(-1);
  for (let i = 0; i < n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(n + 1).fill(Infinity);
    const used = new Uint8Array(n + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0]!;
      let delta = Infinity;
      let j1 = -1;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0]![j - 1]! - u[i0]! - v[j]!;
        if (cur < minv[j]!) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j]! < delta) {
          delta = minv[j]!;
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]!] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== -1);
    do {
      const j1 = way[j0]!;
      p[j0] = p[j1]!;
      j0 = j1;
    } while (j0 !== 0);
  }
  let total = 0;
  for (let j = 1; j <= n; j++) {
    if (p[j] === -1) return INF;
    const c = cost[p[j]!]![j - 1]!;
    if (c >= INF) return INF;
    total += c;
  }
  return total;
}

// ---------------------------------------------------------------------------
// Push search on one (sub)level
// ---------------------------------------------------------------------------

interface NodeRec {
  boxes: number[];
  player: number; // canonical: smallest square of the player's reachable region
  g: number;
  parent: number;
  pushFrom: number; // square the pushed box left (-1 at the root)
  pushTo: number; // square it entered
  pushDir: number; // direction of the push
}

class Solver {
  private lv: Level;
  private pushDist: Int32Array[];
  private dead: Uint8Array;
  private stamp: Int32Array;
  private stampNow = 0;
  private hCache = new Map<string, number>();
  private mustReach: number; // a goal state must also let the player walk to this square (-1: no)
  expansions = 0;

  constructor(lv: Level, mustReach = -1) {
    this.lv = lv;
    this.mustReach = mustReach;
    this.pushDist = computePushDist(lv);
    const n = lv.w * lv.h;
    this.dead = new Uint8Array(n);
    for (let s = 0; s < n; s++) {
      if (lv.wall[s] || lv.goal[s]) continue;
      let live = false;
      for (const dist of this.pushDist) {
        if (dist[s]! < INF) {
          live = true;
          break;
        }
      }
      if (!live) this.dead[s] = 1;
    }
    this.stamp = new Int32Array(n);
  }

  // Flood fill from `start` over squares that are neither wall nor box; returns the smallest
  // square reached (the region's canonical name). Squares reached carry the current stamp.
  private flood(start: number, boxSet: Set<number>): number {
    const { w, wall } = this.lv;
    this.stampNow++;
    const stamp = this.stampNow;
    let min = start;
    const queue = [start];
    this.stamp[start] = stamp;
    for (let head = 0; head < queue.length; head++) {
      const c = queue[head]!;
      if (c < min) min = c;
      for (let d = 0; d < 4; d++) {
        const nx = (c % w) + DX[d]!;
        const ny = Math.floor(c / w) + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= this.lv.w || ny >= this.lv.h) continue;
        const q = ny * w + nx;
        if (wall[q] || boxSet.has(q) || this.stamp[q] === stamp) continue;
        this.stamp[q] = stamp;
        queue.push(q);
      }
    }
    return min;
  }

  // The matching estimate for a box placement, memoised on the placement alone.
  private estimate(boxes: number[]): number {
    const key = String.fromCharCode(...boxes);
    const hit = this.hCache.get(key);
    if (hit !== undefined) return hit;
    const n = boxes.length;
    const cost: Int32Array[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const row = new Int32Array(n).fill(INF);
      for (let j = 0; j < this.pushDist.length; j++) row[j] = this.pushDist[j]![boxes[i]!]!;
      cost[i] = row;
    }
    const value = matchingCost(cost, n);
    this.hCache.set(key, value);
    return value;
  }

  // True when some box can never move again and is not on a goal. The frozen set is the
  // greatest fixed point of "every axis is blocked by a wall or by another frozen box":
  // start from all boxes and repeatedly strike out any box that could be pushed along some
  // axis if the boxes still in the set stayed put (a push needs its landing square and its
  // standing square both free of walls and of set members). What remains can never move:
  // the first push of any of them would need another of them to have moved already. A plain
  // recursion that treats a box met mid-decision as frozen is NOT this set — it condemns
  // boxes whose only blocker is a box that turns out movable — so the iteration matters.
  private hasFrozenBox(boxes: number[]): boolean {
    const { w, h, wall, goal } = this.lv;
    const boxAt = new Map<number, number>();
    boxes.forEach((b, i) => boxAt.set(b, i));
    const inSet = new Uint8Array(boxes.length).fill(1);
    const free = (x: number, y: number): boolean => {
      if (x < 0 || y < 0 || x >= w || y >= h) return false;
      const cell = y * w + x;
      if (wall[cell]) return false;
      const j = boxAt.get(cell);
      return j === undefined || inSet[j] === 0;
    };
    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < boxes.length; i++) {
        if (!inSet[i]) continue;
        const b = boxes[i]!;
        const bx = b % w;
        const by = Math.floor(b / w);
        let movable = false;
        for (let d = 0; d < 4 && !movable; d++) {
          const dx = DX[d]!;
          const dy = DY[d]!;
          if (free(bx + dx, by + dy) && free(bx - dx, by - dy)) movable = true;
        }
        if (movable) {
          inSet[i] = 0;
          changed = true;
        }
      }
    }
    for (let i = 0; i < boxes.length; i++) {
      if (inSet[i] && !goal[boxes[i]!]) return true;
    }
    return false;
  }

  private solved(boxes: number[]): boolean {
    return boxes.every((b) => this.lv.goal[b] === 1);
  }

  private goalState(nd: NodeRec): boolean {
    if (!this.solved(nd.boxes)) return false;
    if (this.mustReach < 0) return true;
    this.flood(nd.player, new Set(nd.boxes));
    return this.stamp[this.mustReach] === this.stampNow;
  }

  // A* (greedy=false) or best-first (greedy=true) over pushes. Returns the winning node index
  // in `nodes`, or -1 when the open list empties or the expansion budget runs out.
  private search(nodes: NodeRec[], budget: number, greedy: boolean): number {
    const { w, wall } = this.lv;
    const startBoxes = this.lv.boxes;
    const startSet = new Set(startBoxes);
    const startPlayer = this.flood(this.lv.player, startSet);
    const h0 = this.estimate(startBoxes);
    if (h0 >= INF) return -1;
    nodes.push({ boxes: startBoxes, player: startPlayer, g: 0, parent: -1, pushFrom: -1, pushTo: -1, pushDir: -1 });
    const best = new Map<string, number>();
    best.set(String.fromCharCode(startPlayer, ...startBoxes), 0);
    // Binary heap of node indices on (priority, then deeper g first, then insertion order).
    const heap: number[] = [0];
    const prioArr: number[] = [h0];
    const seq = new Map<number, number>([[0, 0]]);
    let counter = 1;
    const less = (a: number, b: number): boolean => {
      const pa = prioArr[a]!;
      const pb = prioArr[b]!;
      if (pa !== pb) return pa < pb;
      const ga = nodes[a]!.g;
      const gb = nodes[b]!.g;
      if (ga !== gb) return ga > gb;
      return seq.get(a)! < seq.get(b)!;
    };
    const heapPush = (idx: number): void => {
      seq.set(idx, counter++);
      const nd = nodes[idx]!;
      const h = this.estimate(nd.boxes);
      prioArr[idx] = greedy ? h : nd.g + h;
      let i = heap.length;
      heap.push(idx);
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (!less(heap[i]!, heap[p]!)) break;
        [heap[i], heap[p]] = [heap[p]!, heap[i]!];
        i = p;
      }
    };
    const heapPop = (): number => {
      const top = heap[0]!;
      const last = heap.pop()!;
      if (heap.length > 0) {
        heap[0] = last;
        for (let i = 0; ; ) {
          let c = 2 * i + 1;
          if (c >= heap.length) break;
          if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
          if (!less(heap[c]!, heap[i]!)) break;
          [heap[i], heap[c]] = [heap[c]!, heap[i]!];
          i = c;
        }
      }
      return top;
    };
    let used = 0;
    while (heap.length > 0) {
      const idx = heapPop();
      const nd = nodes[idx]!;
      const key = String.fromCharCode(nd.player, ...nd.boxes);
      if (best.get(key) !== nd.g) continue; // a cheaper route to this state was settled already
      if (this.goalState(nd)) return idx;
      if (used >= budget || globalExpansions >= GLOBAL_CAP) return -1;
      used++;
      this.expansions++;
      globalExpansions++;
      const boxSet = new Set(nd.boxes);
      this.flood(nd.player, boxSet);
      // Snapshot the player's reachable region: generating a child floods again for the
      // child's own region, which would overwrite the stamps this loop still reads.
      const reachStamp = this.stampNow;
      const reach = this.stamp.slice();
      for (let bi = 0; bi < nd.boxes.length; bi++) {
        const b = nd.boxes[bi]!;
        for (let d = 0; d < 4; d++) {
          const step = DY[d]! * w + DX[d]!;
          const dest = b + step;
          const stand = b - step;
          if (dest < 0 || dest >= wall.length || stand < 0 || stand >= wall.length) continue;
          if (DX[d] !== 0 && Math.floor(dest / w) !== Math.floor(b / w)) continue;
          if (wall[dest] || boxSet.has(dest) || this.dead[dest]) continue;
          if (reach[stand] !== reachStamp) continue;
          const childBoxes = nd.boxes.slice();
          childBoxes[bi] = dest;
          childBoxes.sort((a, z) => a - z);
          if (this.hasFrozenBox(childBoxes)) continue;
          const h = this.estimate(childBoxes);
          if (h >= INF) continue;
          const childPlayer = this.flood(b, new Set(childBoxes));
          const cKey = String.fromCharCode(childPlayer, ...childBoxes);
          const cg = nd.g + 1;
          const seen = best.get(cKey);
          if (seen !== undefined && seen <= cg) continue;
          best.set(cKey, cg);
          nodes.push({ boxes: childBoxes, player: childPlayer, g: cg, parent: idx, pushFrom: b, pushTo: dest, pushDir: d });
          heapPush(nodes.length - 1);
        }
      }
    }
    return -1;
  }

  // Replay the winning chain of pushes from the start, walking the player between pushes,
  // and return the full move string (lowercase steps, uppercase pushes).
  private render(nodes: NodeRec[], win: number): string {
    const chain: NodeRec[] = [];
    for (let t = win; t > 0; t = nodes[t]!.parent) chain.push(nodes[t]!);
    chain.reverse();
    const boxes = new Set(this.lv.boxes);
    let player = this.lv.player;
    let out = '';
    for (const nd of chain) {
      const stand = nd.pushFrom - (DY[nd.pushDir]! * this.lv.w + DX[nd.pushDir]!);
      const walk = walkPath(this.lv, boxes, player, stand);
      if (walk === null) return '';
      out += walk;
      out += LETTER[nd.pushDir]!.toUpperCase();
      boxes.delete(nd.pushFrom);
      boxes.add(nd.pushTo);
      player = nd.pushFrom;
    }
    return out;
  }

  // Solve this (sub)level push-optimally within the budget; returns '' when no solution is
  // found (greedy fallback included).
  solve(budget: number): string {
    const nodes: NodeRec[] = [];
    const win = this.search(nodes, budget, false);
    if (win >= 0) return this.render(nodes, win);
    const greedyNodes: NodeRec[] = [];
    const gWin = this.search(greedyNodes, budget, true);
    if (gWin >= 0) return this.render(greedyNodes, gWin);
    return '';
  }
}

// ---------------------------------------------------------------------------
// Decomposition into door-sealed parts
// ---------------------------------------------------------------------------

interface Part {
  door: number;
  cells: number[]; // the sealed component (door not included)
  boxes: number[];
  goals: number[];
  band: boolean; // the door looks like a gap in a wall band: walls on both sides of it
}

// Every part of the board that a single door square seals off from the player, holding at
// least one box and exactly as many goals as boxes. Not every such split is a room of the
// level's construction — a narrow passage inside a room splits it too, and the piece cut
// off there may be unsolvable on its own or worth more pushes than it costs inside the
// whole — so the candidates come back sorted for trial (wall-band doors and room-sized
// box counts first) and the caller strikes out the doors whose part will not solve.
function findParts(lv: Level, wall: Uint8Array, boxes: Set<number>, player: number, banned: Set<number>): Part[] {
  const { w, h } = lv;
  const n = w * h;
  const parts: Part[] = [];
  for (let door = 0; door < n; door++) {
    if (wall[door] || boxes.has(door) || door === player || banned.has(door)) continue;
    // Label the components of the floor with the door removed.
    const label = new Int32Array(n).fill(-1);
    for (let s = 0; s < n; s++) {
      if (wall[s] || s === door || label[s] !== -1) continue;
      const cells: number[] = [];
      const queue = [s];
      label[s] = 1;
      for (let head = 0; head < queue.length; head++) {
        const c = queue[head]!;
        cells.push(c);
        for (let d = 0; d < 4; d++) {
          const nx = (c % w) + DX[d]!;
          const ny = Math.floor(c / w) + DY[d]!;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const q = ny * w + nx;
          if (wall[q] || q === door || label[q] !== -1) continue;
          label[q] = 1;
          queue.push(q);
        }
      }
      if (cells.includes(player)) continue;
      const cBoxes = cells.filter((c) => boxes.has(c));
      if (cBoxes.length === 0) continue;
      const cGoals = cells.filter((c) => lv.goal[c] === 1);
      if (cGoals.length !== cBoxes.length) continue;
      const dx = door % w;
      const band =
        dx > 0 && dx < w - 1 && wall[door - 1] === 1 && wall[door + 1] === 1 && !wall[door - w] && !wall[door + w];
      parts.push({ door, cells, boxes: cBoxes, goals: cGoals, band });
    }
  }
  const roomSized = (p: Part): number => (p.boxes.length >= 3 && p.boxes.length <= 4 ? 0 : 1);
  parts.sort(
    (a, b) =>
      Number(b.band) - Number(a.band) ||
      roomSized(a) - roomSized(b) ||
      a.boxes.length - b.boxes.length ||
      a.cells.length - b.cells.length ||
      a.door - b.door,
  );
  return parts;
}

// Solve a level by splitting it into door-sealed parts, solving each part on its own and
// walking the player between them; returns the move string, or null when the split fails
// (a part unsolved, a walk missing) so the caller can fall back to a whole-level search.
function solveByParts(lv: Level, budgetPerPart: number): { moves: string; expansions: number } | null {
  const wall = lv.wall.slice();
  const boxes = new Set(lv.boxes);
  const banned = new Set<number>();
  let player = lv.player;
  let moves = '';
  let expansions = 0;
  const work: Level = { w: lv.w, h: lv.h, wall, goal: lv.goal, goals: lv.goals, boxes: lv.boxes, player };
  for (;;) {
    // Take the first candidate part that solves on its own; strike out the doors whose
    // parts do not (an interior split of a room, typically) and look again.
    let solvedPart: Part | null = null;
    let solvedMoves = '';
    let solvedEntry = -1;
    for (const cand of findParts(work, wall, boxes, player, banned)) {
      // The part is entered from the square outside its door (the hallway square the door
      // opens onto): a box standing in the doorway can only be pushed back inside by a
      // player standing there, so that square belongs to the part's arena for the player
      // (a box pushed onto it is lost, exactly as in the level, and the search prunes it).
      const cellSet = new Set(cand.cells);
      let entry = cand.door;
      for (let d = 0; d < 4; d++) {
        const q = cand.door + DY[d]! * lv.w + DX[d]!;
        if (q < 0 || q >= wall.length || wall[q] || cellSet.has(q)) continue;
        if (DX[d] !== 0 && Math.floor(q / lv.w) !== Math.floor(cand.door / lv.w)) continue;
        entry = q;
        break;
      }
      // The part's own level: everything outside the part, its door and its entry is wall.
      const subWall = new Uint8Array(lv.w * lv.h).fill(1);
      subWall[cand.door] = 0;
      subWall[entry] = 0;
      for (const c of cand.cells) subWall[c] = lv.wall[c]!;
      const sub: Level = {
        w: lv.w,
        h: lv.h,
        wall: subWall,
        goal: lv.goal,
        goals: cand.goals.slice().sort((a, b) => a - b),
        boxes: cand.boxes.slice().sort((a, b) => a - b),
        player: entry,
      };
      const solver = new Solver(sub, entry);
      const subMoves = solver.solve(budgetPerPart);
      expansions += solver.expansions;
      if (subMoves === '') {
        banned.add(cand.door);
        continue;
      }
      solvedPart = cand;
      solvedMoves = subMoves;
      solvedEntry = entry;
      break;
    }
    if (solvedPart === null) break;
    const part = solvedPart;
    const entry = solvedEntry;
    // Walk to the entry square on the working board, then play the part's solution.
    const walkIn = walkPath(work, boxes, player, entry);
    if (walkIn === null) return null;
    moves += walkIn + solvedMoves;
    // Where does the player end inside the part? Replay the part alone to find out.
    let p = entry;
    const pBoxes = new Set(part.boxes);
    for (const ch of solvedMoves) {
      const d = LETTER.indexOf(ch.toLowerCase());
      const step = DY[d]! * lv.w + DX[d]!;
      if (pBoxes.has(p + step)) {
        pBoxes.delete(p + step);
        pBoxes.add(p + 2 * step);
      }
      p += step;
    }
    const walkOut = walkPath(
      work,
      new Set([...boxes].filter((b) => !part.boxes.includes(b)).concat([...pBoxes])),
      p,
      entry,
    );
    if (walkOut === null) return null;
    moves += walkOut;
    player = entry;
    // Seal the part off and take its boxes out of the working set.
    for (const c of part.cells) wall[c] = 1;
    for (const b of part.boxes) boxes.delete(b);
  }
  if (boxes.size > 0) {
    // Whatever the split left (boxes outside any sealed part) is solved whole, in place;
    // its goals are the ones not sealed inside the parts already solved.
    const rest: Level = {
      w: lv.w,
      h: lv.h,
      wall,
      goal: lv.goal,
      goals: lv.goals.filter((g) => wall[g] === 0),
      boxes: [...boxes].sort((a, b) => a - b),
      player,
    };
    const solver = new Solver(rest);
    const restMoves = solver.solve(budgetPerPart);
    expansions += solver.expansions;
    if (restMoves === '') return null;
    moves += restMoves;
  }
  if (replay(lv, moves) < 0) return null;
  return { moves, expansions };
}

// ---------------------------------------------------------------------------
// Entry point: one run answers every instance in the input directory.
// ---------------------------------------------------------------------------

// Expansion budgets. Every search counts its expansions against one global cap for the
// whole run, so the total work — and with it the wall time on one CPU — stays bounded no
// matter how the levels split. Counting work, never time, keeps the run deterministic.
const GLOBAL_CAP = 3_500_000;
const PART_BUDGET = 250_000;
const LEVEL_BUDGET = 1_200_000;
let globalExpansions = 0;

function solveLevel(lv: Level, budget: number): string {
  const byParts = solveByParts(lv, PART_BUDGET);
  if (byParts !== null) return byParts.moves;
  const solver = new Solver(lv);
  return solver.solve(budget);
}

function main(): void {
  const ids = readdirSync(I).sort();
  for (const id of ids) {
    const text = readFileSync(join(I, id, 'level.txt'), 'utf8');
    const lv = parseLevel(text);
    const moves = solveLevel(lv, LEVEL_BUDGET);
    mkdirSync(join(O, id), { recursive: true });
    writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
  }
}

main();
