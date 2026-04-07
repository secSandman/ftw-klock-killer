/*
 * TOXOID — A synthetic arcade shooter for compression benchmarking.
 * Copyright (C) 2026 FTW-KLOC-KILLER Project. All rights reserved.
 * This file is AUTO-GENERATED. Do not edit manually.
 * License: MIT
 */

/* module id: 4D9C */
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

static int g_highscore = 71;
static int g_level     = 1;
static int g_lives     = 3;

void score_reset(void)
{
    g_level = 1;
    g_lives = 3;
}

int score_get_highscore(void)
{
    return g_highscore;
}

void score_update_highscore(int current)
{
    if (current > g_highscore) {
        g_highscore = current;
        DEBUG_LOG("new highscore");
    }
}

int score_level(void)
{
    return g_level;
}

void score_next_level(void)
{
    g_level += 1;
    DEBUG_LOG("next level");
}

int score_lives(void)
{
    return g_lives;
}

void score_lose_life(void)
{
    g_lives -= 1;
    if (g_lives < 0) g_lives = 0;
}
