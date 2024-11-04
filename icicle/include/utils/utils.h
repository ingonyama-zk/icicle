#pragma once
#ifndef ICICLE_UTILS_H
#define ICICLE_UTILS_H

#define CONCAT_DIRECT(a, b) a##_##b
#define CONCAT_EXPAND(a, b) CONCAT_DIRECT(a, b) // expand a,b before concatenation

static unsigned int next_pow_of_two(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

#endif // ICICLE_UTILS_H