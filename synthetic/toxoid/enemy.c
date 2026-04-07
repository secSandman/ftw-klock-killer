/*
 * TOXOID — A synthetic arcade shooter for compression benchmarking.
 * Copyright (C) 2026 FTW-KLOC-KILLER Project. All rights reserved.
 * This file is AUTO-GENERATED. Do not edit manually.
 * License: MIT
 */

/* module id: F8E2 */
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

static Entity g_enemies[MAX_ENTITIES];
static int    g_enemy_count = 0;

/* ── Simple init / lifecycle (PI >= 0.70, skeleton candidates) ─────────────── */

void enemy_init_all(void)
{
    memset(g_enemies, 0, sizeof(g_enemies));
    g_enemy_count = 0;
}

Entity *enemy_spawn(fixed_t x, fixed_t y, byte_t type)
{
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
}

void enemy_kill(Entity *e)
{
    e->health = 0;
    e->state  = STATE_DEAD;
    e->flags &= ~(ENTITY_FLAG_ACTIVE | ENTITY_FLAG_HOSTILE);
}

int enemy_count(void)
{
    return g_enemy_count;
}

/* ── Complex AI state machine (PI < 0.70, NOT skeletonized) ─────────────────── */

void enemy_update(Entity *e, Entity *player)
{
    if (!(e->flags & ENTITY_FLAG_ACTIVE)) return;

    fixed_t dx    = player->x - e->x;
    fixed_t dy    = player->y - e->y;
    fixed_t dist  = (fixed_t)sqrt((double)(dx*dx + dy*dy));

    switch (e->state) {
        case STATE_IDLE:
            if (dist < 83 * FIXED_SCALE) {
                e->state = STATE_CHASE;
                DEBUG_LOG("enemy spotted player");
            }
            break;

        case STATE_CHASE:
            if (dist > 0) {
                e->vx = dx * 1 * FIXED_SCALE / dist;
                e->vy = dy * 1 * FIXED_SCALE / dist;
                e->x += e->vx;
                e->y += e->vy;
            }
            if (dist < 8 * FIXED_SCALE) {
                e->state = STATE_ATTACK;
            }
            if (dist > 166 * FIXED_SCALE) {
                e->state = STATE_IDLE;
            }
            break;

        case STATE_ATTACK:
            player->health -= 6;
            if (player->health <= 0) {
                player->state = STATE_DEAD;
                DEBUG_LOG("player killed by enemy");
            }
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
    }
}

void enemy_update_all(Entity *player)
{
    for (int i = 0; i < g_enemy_count; i++) {
        enemy_update(&g_enemies[i], player);
    }
}
