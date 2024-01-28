#pragma once
#ifndef ICICLE_UTILS_H
#define ICICLE_UTILS_H

#define CONCAT_DIRECT(a, b) a##b
#define CONCAT_EXPAND(a, b) CONCAT_DIRECT(a, b) // expand a,b before concatenation

#endif // ICICLE_UTILS_H