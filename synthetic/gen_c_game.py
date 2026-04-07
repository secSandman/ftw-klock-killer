"""
gen_c_game.py — Synthetic C game corpus generator
===================================================
Generates a small, NEVER-BEFORE-SEEN C game (TOXOID: a simple arcade shooter)
designed specifically to stress-test the KLOC-KILLER compression pipeline.

Why synthetic?
  - Chocolate Doom is in Claude's training data — it answers questions about it
    even WITHOUT compression. That's not a real test.
  - TOXOID is generated fresh each run with slight randomization — the LLM has
    never seen it. Compression quality is the only thing standing between the
    LLM and a correct answer.

Corpus design for maximum compression signal:
  - 8 pseudo-modules (player.c, enemy.c, render.c, etc.)
  - Every file has the same copyright header           → comment_block KQ hit
  - Every file includes the same 5 system headers      → include KQ hit
  - Repeated boilerplate macros across files           → define KQ hit
  - Simple predictable functions (PI >= 0.70)          → skeleton candidates
  - A few complex state-machine functions (PI < 0.70)  → NOT skeletonized
  - Slight randomization per run (magic numbers, ids)  → not memorisable

Usage:
    python synthetic/gen_c_game.py                         # write to synthetic/toxoid/
    python synthetic/gen_c_game.py --seed 42 --out /tmp/toxoid
    python synthetic/gen_c_game.py --list                  # show file list only

After generating:
    python rag/c_etl.py --repo synthetic/toxoid --dry-run  # measure TER
    python kloc.py pipeline --file synthetic/toxoid/enemy.c --question "explain enemy AI"

TODO (slice placeholder — future work):
  - Add procedurally generated level maps (arrays of int32_t)
  - Add multi-file header cross-references to test include masking depth
  - Add randomly-named but structurally identical helper functions to test
    identifier-frequency masking (future CPatternMiner v2 feature)
  - Pipe output into the benchmark suite as a third corpus alongside
    synthetic Python and Doom C
  - Add --complexity flag to tune PI distribution of generated functions
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from textwrap import dedent
from typing import List, Tuple

# ── Randomisation seed (fixed default = reproducible baseline) ─────────────────
DEFAULT_SEED = 7

# ── Shared boilerplate that appears in EVERY file (drives KQ pattern mining) ───

COPYRIGHT = """\
/*
 * TOXOID — A synthetic arcade shooter for compression benchmarking.
 * Copyright (C) 2026 FTW-KLOC-KILLER Project. All rights reserved.
 * This file is AUTO-GENERATED. Do not edit manually.
 * License: MIT
 */
"""

COMMON_INCLUDES = """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
"""

COMMON_DEFINES = """\
#define TOXOID_VERSION  "1.0.0-synthetic"
#define MAX_ENTITIES    256
#define SCREEN_WIDTH    320
#define SCREEN_HEIGHT   200
#define FIXED_SCALE     65536
#define DEBUG_LOG(msg)  fprintf(stderr, "[TOXOID] %s\\n", (msg))
"""

# ── Module generators ──────────────────────────────────────────────────────────

def _header(module: str, rng: random.Random) -> str:
    """Every file gets the same copyright + includes + defines (KQ hits)."""
    magic_comment = f"/* module id: {rng.randint(0x1000, 0xFFFF):04X} */"
    return f"{COPYRIGHT}\n{magic_comment}\n{COMMON_INCLUDES}\n{COMMON_DEFINES}\n"


def _simple_fn(name: str, body_lines: List[str], ret_type: str = "void") -> str:
    """
    Generate a simple function (PI >= 0.70 → skeleton candidate).
    body_lines should be pure assignments/returns (boilerplate bonus).
    """
    body = "\n".join(f"    {l}" for l in body_lines)
    return dedent(f"""\
        {ret_type} {name}(void)
        {{
        {body}
        }}
    """)


def _complex_fn(name: str, cases: List[Tuple[str, List[str]]], ret_type: str = "void") -> str:
    """
    Generate a complex function with a switch statement (PI < 0.70 → NOT skeletonized).
    Each case is (label, [body_line, ...]).
    """
    inner = []
    for label, stmts in cases:
        inner.append(f"    case {label}:")
        for s in stmts:
            inner.append(f"        {s}")
        inner.append(f"        break;")
    body = "\n".join(inner)
    return dedent(f"""\
        {ret_type} {name}(int state)
        {{
            switch (state) {{
        {body}
            default:
                DEBUG_LOG("unknown state");
                break;
            }}
        }}
    """)


# ── Individual module generators ───────────────────────────────────────────────

def gen_types_h(rng: random.Random) -> str:
    seed_val = rng.randint(100, 999)
    return f"""{_header('types', rng)}
/* Core type definitions for TOXOID */

typedef int32_t  fixed_t;
typedef uint16_t angle_t;
typedef uint8_t  byte_t;

typedef struct {{
    fixed_t x, y;
    fixed_t vx, vy;
    int     health;
    int     state;
    int     flags;
    byte_t  type;
    byte_t  _pad[{seed_val % 7 + 1}];
}} Entity;

typedef struct {{
    int      width;
    int      height;
    byte_t  *pixels;
}} Surface;

typedef enum {{
    STATE_IDLE    = 0,
    STATE_CHASE   = 1,
    STATE_ATTACK  = 2,
    STATE_PAIN    = 3,
    STATE_DEAD    = 4,
}} EntityState;

#define ENTITY_FLAG_ACTIVE  0x01
#define ENTITY_FLAG_VISIBLE 0x02
#define ENTITY_FLAG_HOSTILE 0x04
"""


def gen_player_c(rng: random.Random) -> str:
    speed = rng.randint(3, 8)
    hp    = rng.randint(80, 120)
    return f"""{_header('player', rng)}
#include "types.h"

static Entity g_player;
static int    g_score = 0;

void player_init(void)
{{
    g_player.x      = SCREEN_WIDTH  * FIXED_SCALE / 2;
    g_player.y      = SCREEN_HEIGHT * FIXED_SCALE / 2;
    g_player.vx     = 0;
    g_player.vy     = 0;
    g_player.health = {hp};
    g_player.state  = STATE_IDLE;
    g_player.flags  = ENTITY_FLAG_ACTIVE | ENTITY_FLAG_VISIBLE;
    g_player.type   = 0;
    g_score         = 0;
}}

void player_move(int dx, int dy)
{{
    g_player.vx = dx * {speed} * FIXED_SCALE;
    g_player.vy = dy * {speed} * FIXED_SCALE;
    g_player.x += g_player.vx;
    g_player.y += g_player.vy;
}}

void player_clamp(void)
{{
    if (g_player.x < 0)                         g_player.x = 0;
    if (g_player.x > SCREEN_WIDTH  * FIXED_SCALE) g_player.x = SCREEN_WIDTH  * FIXED_SCALE;
    if (g_player.y < 0)                         g_player.y = 0;
    if (g_player.y > SCREEN_HEIGHT * FIXED_SCALE) g_player.y = SCREEN_HEIGHT * FIXED_SCALE;
}}

void player_take_damage(int dmg)
{{
    g_player.health -= dmg;
    g_player.state   = STATE_PAIN;
    if (g_player.health <= 0) {{
        g_player.health = 0;
        g_player.state  = STATE_DEAD;
        g_player.flags &= ~ENTITY_FLAG_ACTIVE;
        DEBUG_LOG("player died");
    }}
}}

int player_score(void)
{{
    return g_score;
}}

void player_add_score(int pts)
{{
    g_score += pts;
}}

Entity *player_get(void)
{{
    return &g_player;
}}
"""


def gen_enemy_c(rng: random.Random) -> str:
    speed    = rng.randint(1, 4)
    sight    = rng.randint(60, 120)
    atk_dmg  = rng.randint(5, 20)
    return f"""{_header('enemy', rng)}
#include "types.h"

static Entity g_enemies[MAX_ENTITIES];
static int    g_enemy_count = 0;

/* ── Simple init / lifecycle (PI >= 0.70, skeleton candidates) ─────────────── */

void enemy_init_all(void)
{{
    memset(g_enemies, 0, sizeof(g_enemies));
    g_enemy_count = 0;
}}

Entity *enemy_spawn(fixed_t x, fixed_t y, byte_t type)
{{
    if (g_enemy_count >= MAX_ENTITIES) return NULL;
    Entity *e  = &g_enemies[g_enemy_count++];
    e->x       = x;
    e->y       = y;
    e->vx      = 0;
    e->vy      = 0;
    e->health  = 30 + type * 10;
    e->state   = STATE_IDLE;
    e->flags   = ENTITY_FLAG_ACTIVE | ENTITY_FLAG_HOSTILE;
    e->type    = type;
    return e;
}}

void enemy_kill(Entity *e)
{{
    e->health = 0;
    e->state  = STATE_DEAD;
    e->flags &= ~(ENTITY_FLAG_ACTIVE | ENTITY_FLAG_HOSTILE);
}}

int enemy_count(void)
{{
    return g_enemy_count;
}}

/* ── Complex AI state machine (PI < 0.70, NOT skeletonized) ─────────────────── */

void enemy_update(Entity *e, Entity *player)
{{
    if (!(e->flags & ENTITY_FLAG_ACTIVE)) return;

    fixed_t dx    = player->x - e->x;
    fixed_t dy    = player->y - e->y;
    fixed_t dist  = (fixed_t)sqrt((double)(dx*dx + dy*dy));

    switch (e->state) {{
        case STATE_IDLE:
            if (dist < {sight} * FIXED_SCALE) {{
                e->state = STATE_CHASE;
                DEBUG_LOG("enemy spotted player");
            }}
            break;

        case STATE_CHASE:
            if (dist > 0) {{
                e->vx = dx * {speed} * FIXED_SCALE / dist;
                e->vy = dy * {speed} * FIXED_SCALE / dist;
                e->x += e->vx;
                e->y += e->vy;
            }}
            if (dist < 8 * FIXED_SCALE) {{
                e->state = STATE_ATTACK;
            }}
            if (dist > {sight * 2} * FIXED_SCALE) {{
                e->state = STATE_IDLE;
            }}
            break;

        case STATE_ATTACK:
            player->health -= {atk_dmg};
            if (player->health <= 0) {{
                player->state = STATE_DEAD;
                DEBUG_LOG("player killed by enemy");
            }}
            e->state = STATE_CHASE;
            break;

        case STATE_PAIN:
            e->state = STATE_CHASE;
            break;

        case STATE_DEAD:
            e->flags &= ~ENTITY_FLAG_ACTIVE;
            break;

        default:
            e->state = STATE_IDLE;
            break;
    }}
}}

void enemy_update_all(Entity *player)
{{
    for (int i = 0; i < g_enemy_count; i++) {{
        enemy_update(&g_enemies[i], player);
    }}
}}
"""


def gen_render_c(rng: random.Random) -> str:
    bg_color = rng.randint(0, 40)
    return f"""{_header('render', rng)}
#include "types.h"

static Surface g_screen;

void render_init(void)
{{
    g_screen.width  = SCREEN_WIDTH;
    g_screen.height = SCREEN_HEIGHT;
    g_screen.pixels = (byte_t *)malloc(SCREEN_WIDTH * SCREEN_HEIGHT);
    if (!g_screen.pixels) {{
        DEBUG_LOG("render_init: alloc failed");
        return;
    }}
    memset(g_screen.pixels, {bg_color}, SCREEN_WIDTH * SCREEN_HEIGHT);
}}

void render_clear(void)
{{
    memset(g_screen.pixels, {bg_color}, SCREEN_WIDTH * SCREEN_HEIGHT);
}}

void render_pixel(int x, int y, byte_t color)
{{
    if (x < 0 || x >= SCREEN_WIDTH)  return;
    if (y < 0 || y >= SCREEN_HEIGHT) return;
    g_screen.pixels[y * SCREEN_WIDTH + x] = color;
}}

void render_entity(Entity *e, byte_t color)
{{
    int sx = (int)(e->x / FIXED_SCALE);
    int sy = (int)(e->y / FIXED_SCALE);
    render_pixel(sx,     sy,     color);
    render_pixel(sx + 1, sy,     color);
    render_pixel(sx,     sy + 1, color);
    render_pixel(sx + 1, sy + 1, color);
}}

void render_shutdown(void)
{{
    free(g_screen.pixels);
    g_screen.pixels = NULL;
}}
"""


def gen_collision_c(rng: random.Random) -> str:
    radius = rng.randint(6, 14)
    return f"""{_header('collision', rng)}
#include "types.h"

/*
 * Axis-aligned bounding box collision detection.
 * All coordinates in fixed-point units.
 */

#define PLAYER_RADIUS  ({radius} * FIXED_SCALE)
#define ENEMY_RADIUS   ({radius} * FIXED_SCALE)
#define BULLET_RADIUS  (3  * FIXED_SCALE)

static int _sq_dist(fixed_t ax, fixed_t ay, fixed_t bx, fixed_t by)
{{
    fixed_t dx = ax - bx;
    fixed_t dy = ay - by;
    return (int)((dx / 256) * (dx / 256) + (dy / 256) * (dy / 256));
}}

int collision_circle(Entity *a, Entity *b, int radius)
{{
    int dist2 = _sq_dist(a->x, a->y, b->x, b->y);
    int r2    = (radius / 256) * (radius / 256);
    return dist2 < r2;
}}

int collision_player_enemy(Entity *player, Entity *enemy)
{{
    return collision_circle(player, enemy, PLAYER_RADIUS + ENEMY_RADIUS);
}}

void collision_resolve_all(Entity *player, Entity *enemies, int count)
{{
    for (int i = 0; i < count; i++) {{
        if (!(enemies[i].flags & ENTITY_FLAG_ACTIVE)) continue;
        if (collision_player_enemy(player, &enemies[i])) {{
            player->health -= 1;
            enemies[i].x  += {rng.randint(2,8)} * FIXED_SCALE;
            enemies[i].y  += {rng.randint(2,8)} * FIXED_SCALE;
        }}
    }}
}}
"""


def gen_score_c(rng: random.Random) -> str:
    hi_init = rng.randint(0, 1000)
    return f"""{_header('score', rng)}
#include "types.h"

static int g_highscore = {hi_init};
static int g_level     = 1;
static int g_lives     = 3;

void score_reset(void)
{{
    g_level = 1;
    g_lives = 3;
}}

int score_get_highscore(void)
{{
    return g_highscore;
}}

void score_update_highscore(int current)
{{
    if (current > g_highscore) {{
        g_highscore = current;
        DEBUG_LOG("new highscore");
    }}
}}

int score_level(void)
{{
    return g_level;
}}

void score_next_level(void)
{{
    g_level += 1;
    DEBUG_LOG("next level");
}}

int score_lives(void)
{{
    return g_lives;
}}

void score_lose_life(void)
{{
    g_lives -= 1;
    if (g_lives < 0) g_lives = 0;
}}
"""


def gen_main_c(rng: random.Random) -> str:
    frame_ms = rng.randint(14, 20)
    return f"""{_header('main', rng)}
#include "types.h"

/* Forward declarations */
void player_init(void);
void enemy_init_all(void);
void render_init(void);
void render_clear(void);
void render_shutdown(void);
void enemy_update_all(Entity *player);
void collision_resolve_all(Entity *player, Entity *enemies, int count);
void score_reset(void);
Entity *player_get(void);
int   enemy_count(void);
int   score_lives(void);

static int g_running = 1;

/* Simple game loop — {frame_ms}ms per tick */
int main(int argc, char *argv[])
{{
    (void)argc; (void)argv;

    player_init();
    enemy_init_all();
    render_init();
    score_reset();

    DEBUG_LOG("TOXOID starting");

    while (g_running) {{
        render_clear();

        Entity *player = player_get();
        if (!(player->flags & ENTITY_FLAG_ACTIVE)) {{
            g_running = 0;
            break;
        }}

        enemy_update_all(player);
        collision_resolve_all(player, NULL, enemy_count());

        if (score_lives() <= 0) {{
            g_running = 0;
            DEBUG_LOG("game over");
        }}
    }}

    render_shutdown();
    DEBUG_LOG("TOXOID exiting");
    return 0;
}}
"""


# ── Corpus assembly ────────────────────────────────────────────────────────────

MODULES = [
    ("types.h",      gen_types_h),
    ("player.c",     gen_player_c),
    ("enemy.c",      gen_enemy_c),
    ("render.c",     gen_render_c),
    ("collision.c",  gen_collision_c),
    ("score.c",      gen_score_c),
    ("main.c",       gen_main_c),
]


def generate(out_dir: Path, seed: int = DEFAULT_SEED) -> List[Path]:
    """
    Generate the TOXOID corpus and write to out_dir.
    Returns list of written file paths.
    """
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for filename, generator in MODULES:
        content = generator(rng)
        dest    = out_dir / filename
        dest.write_text(content, encoding="utf-8")
        written.append(dest)

    return written


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Generate synthetic TOXOID C game corpus")
    parser.add_argument("--out",  default="synthetic/toxoid", help="Output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed")
    parser.add_argument("--list", action="store_true", help="List files without writing")
    args = parser.parse_args()

    if args.list:
        for name, _ in MODULES:
            print(f"  {name}")
        sys.exit(0)

    out_dir = Path(args.out)
    files   = generate(out_dir, seed=args.seed)

    total_lines = 0
    total_bytes = 0
    print(f"\n  TOXOID corpus generated in {out_dir}/")
    print(f"  Seed: {args.seed}")
    print()
    for f in files:
        lines = f.read_text(encoding="utf-8").count("\n")
        size  = f.stat().st_size
        total_lines += lines
        total_bytes += size
        print(f"    {f.name:<20} {lines:>4} lines  {size:>6} bytes")

    print(f"\n    {'TOTAL':<20} {total_lines:>4} lines  {total_bytes:>6} bytes")
    print(f"\n  Next steps:")
    print(f"    python rag/c_etl.py --repo {out_dir} --dry-run")
    print(f"    python kloc.py pipeline --file {out_dir}/enemy.c --question \"explain enemy AI\"")
