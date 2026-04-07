/*
 * TOXOID — A synthetic arcade shooter for compression benchmarking.
 * Copyright (C) 2026 FTW-KLOC-KILLER Project. All rights reserved.
 * This file is AUTO-GENERATED. Do not edit manually.
 * License: MIT
 */

/* module id: 369E */
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


/* Core type definitions for TOXOID */

typedef int32_t  fixed_t;
typedef uint16_t angle_t;
typedef uint8_t  byte_t;

typedef struct {
    fixed_t x, y;
    fixed_t vx, vy;
    int     health;
    int     state;
    int     flags;
    byte_t  type;
    byte_t  _pad[5];
} Entity;

typedef struct {
    int      width;
    int      height;
    byte_t  *pixels;
} Surface;

typedef enum {
    STATE_IDLE    = 0,
    STATE_CHASE   = 1,
    STATE_ATTACK  = 2,
    STATE_PAIN    = 3,
    STATE_DEAD    = 4,
} EntityState;

#define ENTITY_FLAG_ACTIVE  0x01
#define ENTITY_FLAG_VISIBLE 0x02
#define ENTITY_FLAG_HOSTILE 0x04
