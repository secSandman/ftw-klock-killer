/*
 * TOXOID — A synthetic arcade shooter for compression benchmarking.
 * Copyright (C) 2026 FTW-KLOC-KILLER Project. All rights reserved.
 * This file is AUTO-GENERATED. Do not edit manually.
 * License: MIT
 */

/* module id: 9D11 */
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

/* Simple game loop — 14ms per tick */
int main(int argc, char *argv[])
{
    (void)argc; (void)argv;

    player_init();
    enemy_init_all();
    render_init();
    score_reset();

    DEBUG_LOG("TOXOID starting");

    while (g_running) {
        render_clear();

        Entity *player = player_get();
        if (!(player->flags & ENTITY_FLAG_ACTIVE)) {
            g_running = 0;
            break;
        }

        enemy_update_all(player);
        collision_resolve_all(player, NULL, enemy_count());

        if (score_lives() <= 0) {
            g_running = 0;
            DEBUG_LOG("game over");
        }
    }

    render_shutdown();
    DEBUG_LOG("TOXOID exiting");
    return 0;
}
