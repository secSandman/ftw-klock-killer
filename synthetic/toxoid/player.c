/*
 * TOXOID — A synthetic arcade shooter for compression benchmarking.
 * Copyright (C) 2026 FTW-KLOC-KILLER Project. All rights reserved.
 * This file is AUTO-GENERATED. Do not edit manually.
 * License: MIT
 */

/* module id: 228B */
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

static Entity g_player;
static int    g_score = 0;

void player_init(void)
{
    g_player.x      = SCREEN_WIDTH  * FIXED_SCALE / 2;
    g_player.y      = SCREEN_HEIGHT * FIXED_SCALE / 2;
    g_player.vx     = 0;
    g_player.vy     = 0;
    g_player.health = 83;
    g_player.state  = STATE_IDLE;
    g_player.flags  = ENTITY_FLAG_ACTIVE | ENTITY_FLAG_VISIBLE;
    g_player.type   = 0;
    g_score         = 0;
}

void player_move(int dx, int dy)
{
    g_player.vx = dx * 6 * FIXED_SCALE;
    g_player.vy = dy * 6 * FIXED_SCALE;
    g_player.x += g_player.vx;
    g_player.y += g_player.vy;
}

void player_clamp(void)
{
    if (g_player.x < 0)                         g_player.x = 0;
    if (g_player.x > SCREEN_WIDTH  * FIXED_SCALE) g_player.x = SCREEN_WIDTH  * FIXED_SCALE;
    if (g_player.y < 0)                         g_player.y = 0;
    if (g_player.y > SCREEN_HEIGHT * FIXED_SCALE) g_player.y = SCREEN_HEIGHT * FIXED_SCALE;
}

void player_take_damage(int dmg)
{
    g_player.health -= dmg;
    g_player.state   = STATE_PAIN;
    if (g_player.health <= 0) {
        g_player.health = 0;
        g_player.state  = STATE_DEAD;
        g_player.flags &= ~ENTITY_FLAG_ACTIVE;
        DEBUG_LOG("player died");
    }
}

int player_score(void)
{
    return g_score;
}

void player_add_score(int pts)
{
    g_score += pts;
}

Entity *player_get(void)
{
    return &g_player;
}
