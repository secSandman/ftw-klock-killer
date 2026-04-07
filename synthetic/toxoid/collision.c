/*
 * TOXOID — A synthetic arcade shooter for compression benchmarking.
 * Copyright (C) 2026 FTW-KLOC-KILLER Project. All rights reserved.
 * This file is AUTO-GENERATED. Do not edit manually.
 * License: MIT
 */

/* module id: 2600 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define TOXOID_VERSION  "1.0.0-synthetic"
#define MAX_ENTITIES    256
#define SCREEN_WIDTH    320
#define SCREEN_HEIGHT   200
#define FIXED_SCALE     65536
#define DEBUG_LOG(msg)  fprintf(stderr, "[TOXOID] %s\n", (msg))


#include "types.h"

/*
 * Axis-aligned bounding box collision detection.
 * All coordinates in fixed-point units.
 */

#define PLAYER_RADIUS  (6 * FIXED_SCALE)
#define ENEMY_RADIUS   (6 * FIXED_SCALE)
#define BULLET_RADIUS  (3  * FIXED_SCALE)

static int _sq_dist(fixed_t ax, fixed_t ay, fixed_t bx, fixed_t by)
{
    fixed_t dx = ax - bx;
    fixed_t dy = ay - by;
    return (int)((dx / 256) * (dx / 256) + (dy / 256) * (dy / 256));
}

int collision_circle(Entity *a, Entity *b, int radius)
{
    int dist2 = _sq_dist(a->x, a->y, b->x, b->y);
    int r2    = (radius / 256) * (radius / 256);
    return dist2 < r2;
}

int collision_player_enemy(Entity *player, Entity *enemy)
{
    return collision_circle(player, enemy, PLAYER_RADIUS + ENEMY_RADIUS);
}

void collision_resolve_all(Entity *player, Entity *enemies, int count)
{
    for (int i = 0; i < count; i++) {
        if (!(enemies[i].flags & ENTITY_FLAG_ACTIVE)) continue;
        if (collision_player_enemy(player, &enemies[i])) {
            player->health -= 1;
            enemies[i].x  += 5 * FIXED_SCALE;
            enemies[i].y  += 5 * FIXED_SCALE;
        }
    }
}
