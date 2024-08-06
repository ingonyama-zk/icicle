#pragma once
#ifndef ICICLE_UTILS_H
#define ICICLE_UTILS_H

#define CONCAT_DIRECT(a, b) a##_##b
#define CONCAT_EXPAND(a, b) CONCAT_DIRECT(a, b) // expand a,b before concatenation
#define UNIQUE(a)           CONCAT_EXPAND(a, __LINE__)

#endif // ICICLE_UTILS_H