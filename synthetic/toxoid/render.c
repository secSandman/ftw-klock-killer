/*
 * TOXOID — A synthetic arcade shooter for compression benchmarking.
 * Copyright (C) 2026 FTW-KLOC-KILLER Project. All rights reserved.
 * This file is AUTO-GENERATED. Do not edit manually.
 * License: MIT
 */

/* module id: 46F6 */
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

static Surface g_screen;

void render_init(void)
{
    g_screen.width  = SCREEN_WIDTH;
    g_screen.height = SCREEN_HEIGHT;
    g_screen.pixels = (byte_t *)malloc(SCREEN_WIDTH * SCREEN_HEIGHT);
    if (!g_screen.pixels) {
        DEBUG_LOG("render_init: alloc failed");
        return;
    }
    memset(g_screen.pixels, 32, SCREEN_WIDTH * SCREEN_HEIGHT);
}

void render_clear(void)
{
    memset(g_screen.pixels, 32, SCREEN_WIDTH * SCREEN_HEIGHT);
}

void render_pixel(int x, int y, byte_t color)
{
    if (x < 0 || x >= SCREEN_WIDTH)  return;
    if (y < 0 || y >= SCREEN_HEIGHT) return;
    g_screen.pixels[y * SCREEN_WIDTH + x] = color;
}

void render_entity(Entity *e, byte_t color)
{
    int sx = (int)(e->x / FIXED_SCALE);
    int sy = (int)(e->y / FIXED_SCALE);
    render_pixel(sx,     sy,     color);
    render_pixel(sx + 1, sy,     color);
    render_pixel(sx,     sy + 1, color);
    render_pixel(sx + 1, sy + 1, color);
}

void render_shutdown(void)
{
    free(g_screen.pixels);
    g_screen.pixels = NULL;
}
