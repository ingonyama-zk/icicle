#pragma once

#include <array>
#include <cstdint>
#include <tuple>
#include <arm_neon.h>

using mp_128_t = std::array<uint64_t, 2>;
using mp_256_t = std::array<uint64_t, 4>;
using mp_320_t = std::array<uint64_t, 5>;
using mp_384_t = std::array<uint64_t, 6>;
using mp_448_t = std::array<uint64_t, 7>;
using mp_512_t = std::array<uint64_t, 8>;


struct bn254_mont_t {
#define INLINE __attribute__((always_inline))
    static INLINE mp_512_t add(const mp_512_t &a, const mp_512_t &b) {
        unsigned long long c;
        mp_512_t res;
        res[0] = __builtin_addcll(a[0], b[0], 0, &c);
        res[1] = __builtin_addcll(a[1], b[1], c, &c);
        res[2] = __builtin_addcll(a[2], b[2], c, &c);
        res[3] = __builtin_addcll(a[3], b[3], c, &c);
        res[4] = __builtin_addcll(a[4], b[4], c, &c);
        res[5] = __builtin_addcll(a[5], b[5], c, &c);
        res[6] = __builtin_addcll(a[6], b[6], c, &c);
        res[7] = __builtin_addcll(a[7], b[7], c, &c);
        return res;
    }

    static INLINE mp_256_t add(const mp_256_t &a, const mp_256_t &b) {
        unsigned long long c;
        mp_256_t res;
        res[0] = __builtin_addcll(a[0], b[0], 0, &c);
        res[1] = __builtin_addcll(a[1], b[1], c, &c);
        res[2] = __builtin_addcll(a[2], b[2], c, &c);
        res[3] = __builtin_addcll(a[3], b[3], c, &c);
        return res;
    }

    static INLINE mp_256_t mul_2(const mp_256_t &a) {
        unsigned long long c;
        mp_256_t res;
        res[0] = __builtin_addcll(a[0], a[0], 0, &c);
        res[1] = __builtin_addcll(a[1], a[1], c, &c);
        res[2] = __builtin_addcll(a[2], a[2], c, &c);
        res[3] = __builtin_addcll(a[3], a[3], c, &c);
        return res;
    }

    static INLINE std::pair<mp_128_t, bool> wadd(const uint64_t lo, const uint64_t hi, const mp_128_t acc, bool c_in) {
        unsigned long long c;
        mp_128_t res;
        res[0] = __builtin_addcll(acc[0], lo, c_in, &c);
        res[1] = __builtin_addcll(acc[1], hi, c, &c);
        return {res, static_cast<bool>(c)};
    }

    static INLINE std::pair<uint64_t, uint64_t> addc(const uint64_t a, const uint64_t b) {
        unsigned long long c;
        uint64_t r = __builtin_addcll(a, b, 0, &c);
        return {r, c};
    }

    static INLINE mp_512_t sub(const mp_512_t &a, const mp_512_t &b) {
        unsigned long long c;
        mp_512_t res;
        res[0] = __builtin_subcll(a[0], b[0], 0, &c);
        res[1] = __builtin_subcll(a[1], b[1], c, &c);
        res[2] = __builtin_subcll(a[2], b[2], c, &c);
        res[3] = __builtin_subcll(a[3], b[3], c, &c);
        res[4] = __builtin_subcll(a[4], b[4], c, &c);
        res[5] = __builtin_subcll(a[5], b[5], c, &c);
        res[6] = __builtin_subcll(a[6], b[6], c, &c);
        res[7] = __builtin_subcll(a[7], b[7], c, &c);
        return res;
    }

    static INLINE mp_256_t sub(const mp_256_t &a, const mp_256_t &b) {
        unsigned long long c;
        mp_256_t res;
        res[0] = __builtin_subcll(a[0], b[0], 0, &c);
        res[1] = __builtin_subcll(a[1], b[1], c, &c);
        res[2] = __builtin_subcll(a[2], b[2], c, &c);
        res[3] = __builtin_subcll(a[3], b[3], c, &c);
        return res;
    }

    static INLINE std::pair<mp_256_t, bool> subc(const mp_256_t &a, const mp_256_t &b) {
        unsigned long long c;
        mp_256_t res;
        res[0] = __builtin_subcll(a[0], b[0], 0, &c);
        res[1] = __builtin_subcll(a[1], b[1], c, &c);
        res[2] = __builtin_subcll(a[2], b[2], c, &c);
        res[3] = __builtin_subcll(a[3], b[3], c, &c);
        return {res, c};
    }

    static INLINE std::pair<mp_320_t, bool> subc(const mp_320_t &a, const mp_320_t &b) {
        unsigned long long c;
        mp_320_t res;
        res[0] = __builtin_subcll(a[0], b[0], 0, &c);
        res[1] = __builtin_subcll(a[1], b[1], c, &c);
        res[2] = __builtin_subcll(a[2], b[2], c, &c);
        res[3] = __builtin_subcll(a[3], b[3], c, &c);
        res[4] = __builtin_subcll(a[4], b[4], c, &c);
        return {res, c};
    }

    static INLINE std::pair<mp_512_t, bool> subc(const mp_512_t &a, const mp_512_t &b) {
        unsigned long long c;
        mp_512_t res;
        res[0] = __builtin_subcll(a[0], b[0], 0, &c);
        res[1] = __builtin_subcll(a[1], b[1], c, &c);
        res[2] = __builtin_subcll(a[2], b[2], c, &c);
        res[3] = __builtin_subcll(a[3], b[3], c, &c);
        res[4] = __builtin_subcll(a[4], b[4], c, &c);
        res[5] = __builtin_subcll(a[5], b[5], c, &c);
        res[6] = __builtin_subcll(a[6], b[6], c, &c);
        res[7] = __builtin_subcll(a[7], b[7], c, &c);
        return {res, c};
    }

    static INLINE std::pair<uint64_t, uint64_t> madd_with_c_in(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t c_in) {
        __uint128_t r = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b) + static_cast<__uint128_t>(c) + static_cast<__uint128_t>(c_in);
        return {static_cast<uint64_t>(r), static_cast<uint64_t>(r >> 64)};
    }

    static INLINE std::pair<uint64_t, uint64_t> mul_wide(const uint64_t a, const uint64_t b) {
        __uint128_t r = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
        return {static_cast<uint64_t>(r), static_cast<uint64_t>(r >> 64)};
    }

    INLINE static mp_512_t schoolbook_mul(const mp_256_t &a, const mp_256_t &b) {
        mp_512_t ab = {0, 0, 0, 0, 0, 0, 0, 0};
        uint64_t c;

        c = 0;
        std::tie(ab[0], c) = madd_with_c_in(a[0], b[0], ab[0], c);
        std::tie(ab[1], c) = madd_with_c_in(a[0], b[1], ab[1], c);
        std::tie(ab[2], c) = madd_with_c_in(a[0], b[2], ab[2], c);
        std::tie(ab[3], c) = madd_with_c_in(a[0], b[3], ab[3], c);
        ab[4] = c;

        c = 0;
        std::tie(ab[1], c) = madd_with_c_in(a[1], b[0], ab[1], c);
        std::tie(ab[2], c) = madd_with_c_in(a[1], b[1], ab[2], c);
        std::tie(ab[3], c) = madd_with_c_in(a[1], b[2], ab[3], c);
        std::tie(ab[4], c) = madd_with_c_in(a[1], b[3], ab[4], c);
        ab[5] = c;

        c = 0;
        std::tie(ab[2], c) = madd_with_c_in(a[2], b[0], ab[2], c);
        std::tie(ab[3], c) = madd_with_c_in(a[2], b[1], ab[3], c);
        std::tie(ab[4], c) = madd_with_c_in(a[2], b[2], ab[4], c);
        std::tie(ab[5], c) = madd_with_c_in(a[2], b[3], ab[5], c);
        ab[6] = c;

        c = 0;
        std::tie(ab[3], c) = madd_with_c_in(a[3], b[0], ab[3], c);
        std::tie(ab[4], c) = madd_with_c_in(a[3], b[1], ab[4], c);
        std::tie(ab[5], c) = madd_with_c_in(a[3], b[2], ab[5], c);
        std::tie(ab[6], c) = madd_with_c_in(a[3], b[3], ab[6], c);
        ab[7] = c;

        return ab;
    }

    INLINE static mp_320_t smul(const uint64_t s, const mp_256_t &v) {
        mp_320_t ab = {0, 0, 0, 0, 0};
        std::tie(ab[0], ab[1]) = madd_with_c_in(s, v[0], ab[0], 0);
        std::tie(ab[1], ab[2]) = madd_with_c_in(s, v[1], ab[1], 0);
        std::tie(ab[2], ab[3]) = madd_with_c_in(s, v[2], ab[2], 0);
        std::tie(ab[3], ab[4]) = madd_with_c_in(s, v[3], ab[3], 0);
        return ab;
    }

    INLINE static mp_320_t add(const mp_320_t &a, const mp_320_t &b) {
        unsigned long long c;
        mp_320_t res;
        res[0] = __builtin_addcll(a[0], b[0], 0, &c);
        res[1] = __builtin_addcll(a[1], b[1], c, &c);
        res[2] = __builtin_addcll(a[2], b[2], c, &c);
        res[3] = __builtin_addcll(a[3], b[3], c, &c);
        res[4] = __builtin_addcll(a[4], b[4], c, &c);
        return res;
    }

    INLINE static std::tuple<mp_256_t, mp_256_t, mp_256_t> mul_block_3(
            const mp_256_t &a_0,
            const mp_256_t &b_0,
            const mp_256_t &a_1,
            const mp_256_t &b_1,
            const mp_256_t &a_2,
            const mp_256_t &b_2) {

        std::array<uint64x2_t, 4> av{};
        av[0] = vcombine_u64(vcreate_u64(a_1[0]), vcreate_u64(a_1[1]));
        av[1] = vcombine_u64(vcreate_u64(a_1[2]), vcreate_u64(a_1[3]));
        av[2] = vcombine_u64(vcreate_u64(a_2[0]), vcreate_u64(a_2[1]));
        av[3] = vcombine_u64(vcreate_u64(a_2[2]), vcreate_u64(a_2[3]));

        std::array<uint64x2_t, 4> bv{};
        bv[0] = vcombine_u64(vcreate_u64(b_1[0]), vcreate_u64(b_1[1]));
        bv[1] = vcombine_u64(vcreate_u64(b_1[2]), vcreate_u64(b_1[3]));
        bv[2] = vcombine_u64(vcreate_u64(b_2[0]), vcreate_u64(b_2[1]));
        bv[3] = vcombine_u64(vcreate_u64(b_2[2]), vcreate_u64(b_2[3]));

        auto [out_arr, outv] = mont_interleaved_3(a_0, b_0, av, bv);

        mp_256_t out_0 = {out_arr[0], out_arr[1], out_arr[2], out_arr[3]};

        mp_256_t out_1, out_2;
        out_1[0] = vgetq_lane_u64(outv[0], 0);
        out_1[1] = vgetq_lane_u64(outv[0], 1);
        out_1[2] = vgetq_lane_u64(outv[1], 0);
        out_1[3] = vgetq_lane_u64(outv[1], 1);

        out_2[0] = vgetq_lane_u64(outv[2], 0);
        out_2[1] = vgetq_lane_u64(outv[2], 1);
        out_2[2] = vgetq_lane_u64(outv[3], 0);
        out_2[3] = vgetq_lane_u64(outv[3], 1);

        return {out_0, out_1, out_2};
    }

    __attribute__((noinline)) static std::pair<std::array<uint64_t, 4>, std::array<uint64x2_t, 4>> mont_interleaved_3(
            const std::array<uint64_t, 4> &a,
            const std::array<uint64_t, 4> &b,
            const std::array<uint64x2_t, 4> &av,
            const std::array<uint64x2_t, 4> &bv) {
        std::array<uint64_t, 4> out = {0, 0, 0, 0};
        std::array<uint64x2_t, 4> outv = {
                vdupq_n_u64(0),
                vdupq_n_u64(0),
                vdupq_n_u64(0),
                vdupq_n_u64(0)};

        asm volatile(
                "  mov x8, #4503599627370495\n"
                "  dup.2d v8, x8\n"
                "  mul x9, x0, x4\n"
                "  mov x10, #5075556780046548992\n"
                "  dup.2d v9, x10\n"
                "  mov x10, #1\n"
                "  umulh x11, x0, x4\n"
                "  movk x10, #18032, lsl 48\n"
                "  dup.2d v10, x10\n"
                "  shl.2d v11, v1, #14\n"
                "  mul x10, x1, x4\n"
                "  shl.2d v12, v2, #26\n"
                "  shl.2d v13, v3, #38\n"
                "  ushr.2d v3, v3, #14\n"
                "  umulh x12, x1, x4\n"
                "  shl.2d v14, v0, #2\n"
                "  usra.2d v11, v0, #50\n"
                "  adds x10, x10, x11\n"
                "  cinc x11, x12, hs\n"
                "  usra.2d v12, v1, #38\n"
                "  usra.2d v13, v2, #26\n"
                "  and.16b v0, v14, v8\n"
                "  mul x12, x2, x4\n"
                "  and.16b v1, v11, v8\n"
                "  and.16b v2, v12, v8\n"
                "  and.16b v11, v13, v8\n"
                "  umulh x13, x2, x4\n"
                "  shl.2d v12, v5, #14\n"
                "  shl.2d v13, v6, #26\n"
                "  shl.2d v14, v7, #38\n"
                "  adds x11, x12, x11\n"
                "  cinc x12, x13, hs\n"
                "  ushr.2d v7, v7, #14\n"
                "  shl.2d v15, v4, #2\n"
                "  mul x13, x3, x4\n"
                "  usra.2d v12, v4, #50\n"
                "  usra.2d v13, v5, #38\n"
                "  usra.2d v14, v6, #26\n"
                "  umulh x4, x3, x4\n"
                "  and.16b v4, v15, v8\n"
                "  and.16b v5, v12, v8\n"
                "  and.16b v6, v13, v8\n"
                "  adds x12, x13, x12\n"
                "  cinc x4, x4, hs\n"
                "  and.16b v12, v14, v8\n"
                "  mov x13, #13605374474286268416\n"
                "  dup.2d v13, x13\n"
                "  mul x13, x0, x5\n"
                "  mov x14, #6440147467139809280\n"
                "  dup.2d v14, x14\n"
                "  umulh x14, x0, x5\n"
                "  mov x15, #3688448094816436224\n"
                "  dup.2d v15, x15\n"
                "  mov x15, #9209861237972664320\n"
                "  adds x10, x13, x10\n"
                "  cinc x13, x14, hs\n"
                "  dup.2d v16, x15\n"
                "  mov x14, #12218265789056155648\n"
                "  dup.2d v17, x14\n"
                "  mul x14, x1, x5\n"
                "  mov x15, #17739678932212383744\n"
                "  dup.2d v18, x15\n"
                "  mov x15, #2301339409586323456\n"
                "  umulh x16, x1, x5\n"
                "  dup.2d v19, x15\n"
                "  mov x15, #7822752552742551552\n"
                "  adds x13, x14, x13\n"
                "  cinc x14, x16, hs\n"
                "  dup.2d v20, x15\n"
                "  mov x15, #5071053180419178496\n"
                "  dup.2d v21, x15\n"
                "  adds x11, x13, x11\n"
                "  cinc x13, x14, hs\n"
                "  mov x14, #16352570246982270976\n"
                "  dup.2d v22, x14\n"
                "  ucvtf.2d v0, v0\n"
                "  mul x14, x2, x5\n"
                "  ucvtf.2d v1, v1\n"
                "  ucvtf.2d v2, v2\n"
                "  ucvtf.2d v11, v11\n"
                "  umulh x15, x2, x5\n"
                "  ucvtf.2d v3, v3\n"
                "  ucvtf.2d v4, v4\n"
                "  adds x13, x14, x13\n"
                "  cinc x14, x15, hs\n"
                "  ucvtf.2d v5, v5\n"
                "  ucvtf.2d v6, v6\n"
                "  ucvtf.2d v12, v12\n"
                "  adds x12, x13, x12\n"
                "  cinc x13, x14, hs\n"
                "  ucvtf.2d v7, v7\n"
                "  mov.16b v23, v9\n"
                "  fmla.2d v23, v0, v4\n"
                "  mul x14, x3, x5\n"
                "  fsub.2d v24, v10, v23\n"
                "  fmla.2d v24, v0, v4\n"
                "  add.2d v15, v15, v23\n"
                "  umulh x5, x3, x5\n"
                "  add.2d v13, v13, v24\n"
                "  mov.16b v23, v9\n"
                "  adds x13, x14, x13\n"
                "  cinc x5, x5, hs\n"
                "  fmla.2d v23, v0, v5\n"
                "  fsub.2d v24, v10, v23\n"
                "  fmla.2d v24, v0, v5\n"
                "  adds x4, x13, x4\n"
                "  cinc x5, x5, hs\n"
                "  add.2d v17, v17, v23\n"
                "  add.2d v15, v15, v24\n"
                "  mov.16b v23, v9\n"
                "  mul x13, x0, x6\n"
                "  fmla.2d v23, v0, v6\n"
                "  fsub.2d v24, v10, v23\n"
                "  fmla.2d v24, v0, v6\n"
                "  umulh x14, x0, x6\n"
                "  add.2d v19, v19, v23\n"
                "  add.2d v17, v17, v24\n"
                "  adds x11, x13, x11\n"
                "  cinc x13, x14, hs\n"
                "  mov.16b v23, v9\n"
                "  fmla.2d v23, v0, v12\n"
                "  fsub.2d v24, v10, v23\n"
                "  mul x14, x1, x6\n"
                "  fmla.2d v24, v0, v12\n"
                "  add.2d v21, v21, v23\n"
                "  add.2d v19, v19, v24\n"
                "  umulh x15, x1, x6\n"
                "  mov.16b v23, v9\n"
                "  fmla.2d v23, v0, v7\n"
                "  fsub.2d v24, v10, v23\n"
                "  adds x13, x14, x13\n"
                "  cinc x14, x15, hs\n"
                "  fmla.2d v24, v0, v7\n"
                "  add.2d v0, v22, v23\n"
                "  adds x12, x13, x12\n"
                "  cinc x13, x14, hs\n"
                "  add.2d v21, v21, v24\n"
                "  mov.16b v22, v9\n"
                "  fmla.2d v22, v1, v4\n"
                "  mul x14, x2, x6\n"
                "  fsub.2d v23, v10, v22\n"
                "  fmla.2d v23, v1, v4\n"
                "  add.2d v17, v17, v22\n"
                "  umulh x15, x2, x6\n"
                "  add.2d v15, v15, v23\n"
                "  mov.16b v22, v9\n"
                "  fmla.2d v22, v1, v5\n"
                "  adds x13, x14, x13\n"
                "  cinc x14, x15, hs\n"
                "  fsub.2d v23, v10, v22\n"
                "  fmla.2d v23, v1, v5\n"
                "  adds x4, x13, x4\n"
                "  cinc x13, x14, hs\n"
                "  add.2d v19, v19, v22\n"
                "  add.2d v17, v17, v23\n"
                "  mov.16b v22, v9\n"
                "  mul x14, x3, x6\n"
                "  fmla.2d v22, v1, v6\n"
                "  fsub.2d v23, v10, v22\n"
                "  fmla.2d v23, v1, v6\n"
                "  umulh x6, x3, x6\n"
                "  add.2d v21, v21, v22\n"
                "  add.2d v19, v19, v23\n"
                "  mov.16b v22, v9\n"
                "  adds x13, x14, x13\n"
                "  cinc x6, x6, hs\n"
                "  fmla.2d v22, v1, v12\n"
                "  fsub.2d v23, v10, v22\n"
                "  fmla.2d v23, v1, v12\n"
                "  adds x5, x13, x5\n"
                "  cinc x6, x6, hs\n"
                "  add.2d v0, v0, v22\n"
                "  add.2d v21, v21, v23\n"
                "  mul x13, x0, x7\n"
                "  mov.16b v22, v9\n"
                "  fmla.2d v22, v1, v7\n"
                "  fsub.2d v23, v10, v22\n"
                "  umulh x0, x0, x7\n"
                "  fmla.2d v23, v1, v7\n"
                "  add.2d v1, v20, v22\n"
                "  add.2d v0, v0, v23\n"
                "  adds x12, x13, x12\n"
                "  cinc x0, x0, hs\n"
                "  mov.16b v20, v9\n"
                "  fmla.2d v20, v2, v4\n"
                "  fsub.2d v22, v10, v20\n"
                "  mul x13, x1, x7\n"
                "  fmla.2d v22, v2, v4\n"
                "  add.2d v19, v19, v20\n"
                "  umulh x1, x1, x7\n"
                "  add.2d v17, v17, v22\n"
                "  mov.16b v20, v9\n"
                "  fmla.2d v20, v2, v5\n"
                "  adds x0, x13, x0\n"
                "  cinc x1, x1, hs\n"
                "  fsub.2d v22, v10, v20\n"
                "  fmla.2d v22, v2, v5\n"
                "  add.2d v20, v21, v20\n"
                "  adds x0, x0, x4\n"
                "  cinc x1, x1, hs\n"
                "  add.2d v19, v19, v22\n"
                "  mov.16b v21, v9\n"
                "  fmla.2d v21, v2, v6\n"
                "  mul x4, x2, x7\n"
                "  fsub.2d v22, v10, v21\n"
                "  fmla.2d v22, v2, v6\n"
                "  umulh x2, x2, x7\n"
                "  add.2d v0, v0, v21\n"
                "  add.2d v20, v20, v22\n"
                "  mov.16b v21, v9\n"
                "  adds x1, x4, x1\n"
                "  cinc x2, x2, hs\n"
                "  fmla.2d v21, v2, v12\n"
                "  fsub.2d v22, v10, v21\n"
                "  fmla.2d v22, v2, v12\n"
                "  adds x1, x1, x5\n"
                "  cinc x2, x2, hs\n"
                "  add.2d v1, v1, v21\n"
                "  add.2d v0, v0, v22\n"
                "  mov.16b v21, v9\n"
                "  mul x4, x3, x7\n"
                "  fmla.2d v21, v2, v7\n"
                "  fsub.2d v22, v10, v21\n"
                "  umulh x3, x3, x7\n"
                "  fmla.2d v22, v2, v7\n"
                "  add.2d v2, v18, v21\n"
                "  add.2d v1, v1, v22\n"
                "  adds x2, x4, x2\n"
                "  cinc x3, x3, hs\n"
                "  mov.16b v18, v9\n"
                "  fmla.2d v18, v11, v4\n"
                "  fsub.2d v21, v10, v18\n"
                "  adds x2, x2, x6\n"
                "  cinc x3, x3, hs\n"
                "  fmla.2d v21, v11, v4\n"
                "  add.2d v18, v20, v18\n"
                "  add.2d v19, v19, v21\n"
                "  mov x4, #48718\n"
                "  mov.16b v20, v9\n"
                "  fmla.2d v20, v11, v5\n"
                "  movk x4, #4732, lsl 16\n"
                "  fsub.2d v21, v10, v20\n"
                "  fmla.2d v21, v11, v5\n"
                "  add.2d v0, v0, v20\n"
                "  movk x4, #45078, lsl 32\n"
                "  add.2d v18, v18, v21\n"
                "  mov.16b v20, v9\n"
                "  fmla.2d v20, v11, v6\n"
                "  movk x4, #39852, lsl 48\n"
                "  fsub.2d v21, v10, v20\n"
                "  fmla.2d v21, v11, v6\n"
                "  add.2d v1, v1, v20\n"
                "  mov x5, #16676\n"
                "  add.2d v0, v0, v21\n"
                "  mov.16b v20, v9\n"
                "  movk x5, #12692, lsl 16\n"
                "  fmla.2d v20, v11, v12\n"
                "  fsub.2d v21, v10, v20\n"
                "  fmla.2d v21, v11, v12\n"
                "  movk x5, #20986, lsl 32\n"
                "  add.2d v2, v2, v20\n"
                "  add.2d v1, v1, v21\n"
                "  mov.16b v20, v9\n"
                "  movk x5, #2848, lsl 48\n"
                "  fmla.2d v20, v11, v7\n"
                "  fsub.2d v21, v10, v20\n"
                "  fmla.2d v21, v11, v7\n"
                "  mov x6, #51052\n"
                "  add.2d v11, v16, v20\n"
                "  add.2d v2, v2, v21\n"
                "  movk x6, #24721, lsl 16\n"
                "  mov.16b v16, v9\n"
                "  fmla.2d v16, v3, v4\n"
                "  fsub.2d v20, v10, v16\n"
                "  movk x6, #61092, lsl 32\n"
                "  fmla.2d v20, v3, v4\n"
                "  add.2d v0, v0, v16\n"
                "  add.2d v4, v18, v20\n"
                "  movk x6, #45156, lsl 48\n"
                "  mov.16b v16, v9\n"
                "  fmla.2d v16, v3, v5\n"
                "  fsub.2d v18, v10, v16\n"
                "  mov x7, #3197\n"
                "  fmla.2d v18, v3, v5\n"
                "  add.2d v1, v1, v16\n"
                "  movk x7, #18936, lsl 16\n"
                "  add.2d v0, v0, v18\n"
                "  mov.16b v5, v9\n"
                "  fmla.2d v5, v3, v6\n"
                "  movk x7, #10922, lsl 32\n"
                "  fsub.2d v16, v10, v5\n"
                "  fmla.2d v16, v3, v6\n"
                "  add.2d v2, v2, v5\n"
                "  movk x7, #11014, lsl 48\n"
                "  add.2d v1, v1, v16\n"
                "  mov.16b v5, v9\n"
                "  fmla.2d v5, v3, v12\n"
                "  mul x13, x4, x9\n"
                "  fsub.2d v6, v10, v5\n"
                "  fmla.2d v6, v3, v12\n"
                "  umulh x4, x4, x9\n"
                "  add.2d v5, v11, v5\n"
                "  add.2d v2, v2, v6\n"
                "  mov.16b v6, v9\n"
                "  adds x12, x13, x12\n"
                "  cinc x4, x4, hs\n"
                "  fmla.2d v6, v3, v7\n"
                "  fsub.2d v11, v10, v6\n"
                "  fmla.2d v11, v3, v7\n"
                "  mul x13, x5, x9\n"
                "  add.2d v3, v14, v6\n"
                "  add.2d v5, v5, v11\n"
                "  usra.2d v15, v13, #52\n"
                "  umulh x5, x5, x9\n"
                "  usra.2d v17, v15, #52\n"
                "  usra.2d v19, v17, #52\n"
                "  usra.2d v4, v19, #52\n"
                "  adds x4, x13, x4\n"
                "  cinc x5, x5, hs\n"
                "  and.16b v6, v13, v8\n"
                "  and.16b v7, v15, v8\n"
                "  adds x0, x4, x0\n"
                "  cinc x4, x5, hs\n"
                "  and.16b v11, v17, v8\n"
                "  and.16b v8, v19, v8\n"
                "  ucvtf.2d v6, v6\n"
                "  mul x5, x6, x9\n"
                "  mov x13, #37864\n"
                "  movk x13, #1815, lsl 16\n"
                "  movk x13, #28960, lsl 32\n"
                "  umulh x6, x6, x9\n"
                "  movk x13, #17153, lsl 48\n"
                "  dup.2d v12, x13\n"
                "  mov.16b v13, v9\n"
                "  adds x4, x5, x4\n"
                "  cinc x5, x6, hs\n"
                "  fmla.2d v13, v6, v12\n"
                "  fsub.2d v14, v10, v13\n"
                "  adds x1, x4, x1\n"
                "  cinc x4, x5, hs\n"
                "  fmla.2d v14, v6, v12\n"
                "  add.2d v0, v0, v13\n"
                "  add.2d v4, v4, v14\n"
                "  mul x5, x7, x9\n"
                "  mov x6, #46128\n"
                "  movk x6, #29964, lsl 16\n"
                "  movk x6, #7587, lsl 32\n"
                "  umulh x7, x7, x9\n"
                "  movk x6, #17161, lsl 48\n"
                "  dup.2d v12, x6\n"
                "  mov.16b v13, v9\n"
                "  adds x4, x5, x4\n"
                "  cinc x5, x7, hs\n"
                "  fmla.2d v13, v6, v12\n"
                "  fsub.2d v14, v10, v13\n"
                "  adds x2, x4, x2\n"
                "  cinc x4, x5, hs\n"
                "  fmla.2d v14, v6, v12\n"
                "  add.2d v1, v1, v13\n"
                "  add.2d v0, v0, v14\n"
                "  add x3, x3, x4\n"
                "  mov x4, #52826\n"
                "  movk x4, #57790, lsl 16\n"
                "  movk x4, #55431, lsl 32\n"
                "  mov x5, #56431\n"
                "  movk x4, #17196, lsl 48\n"
                "  dup.2d v12, x4\n"
                "  mov.16b v13, v9\n"
                "  movk x5, #30457, lsl 16\n"
                "  fmla.2d v13, v6, v12\n"
                "  fsub.2d v14, v10, v13\n"
                "  movk x5, #30012, lsl 32\n"
                "  fmla.2d v14, v6, v12\n"
                "  add.2d v2, v2, v13\n"
                "  add.2d v1, v1, v14\n"
                "  movk x5, #6382, lsl 48\n"
                "  mov x4, #31276\n"
                "  movk x4, #21262, lsl 16\n"
                "  movk x4, #2304, lsl 32\n"
                "  mov x6, #59151\n"
                "  movk x4, #17182, lsl 48\n"
                "  dup.2d v12, x4\n"
                "  mov.16b v13, v9\n"
                "  movk x6, #41769, lsl 16\n"
                "  fmla.2d v13, v6, v12\n"
                "  fsub.2d v14, v10, v13\n"
                "  movk x6, #32276, lsl 32\n"
                "  fmla.2d v14, v6, v12\n"
                "  add.2d v5, v5, v13\n"
                "  add.2d v2, v2, v14\n"
                "  movk x6, #21677, lsl 48\n"
                "  mov x4, #28672\n"
                "  movk x4, #24515, lsl 16\n"
                "  movk x4, #54929, lsl 32\n"
                "  mov x7, #34015\n"
                "  movk x4, #17064, lsl 48\n"
                "  dup.2d v12, x4\n"
                "  mov.16b v13, v9\n"
                "  movk x7, #20342, lsl 16\n"
                "  fmla.2d v13, v6, v12\n"
                "  fsub.2d v14, v10, v13\n"
                "  movk x7, #13935, lsl 32\n"
                "  fmla.2d v14, v6, v12\n"
                "  add.2d v3, v3, v13\n"
                "  add.2d v5, v5, v14\n"
                "  movk x7, #11030, lsl 48\n"
                "  ucvtf.2d v6, v7\n"
                "  mov x4, #44768\n"
                "  movk x4, #51919, lsl 16\n"
                "  mov x9, #13689\n"
                "  movk x4, #6346, lsl 32\n"
                "  movk x4, #17133, lsl 48\n"
                "  dup.2d v7, x4\n"
                "  movk x9, #8159, lsl 16\n"
                "  mov.16b v12, v9\n"
                "  fmla.2d v12, v6, v7\n"
                "  movk x9, #215, lsl 32\n"
                "  fsub.2d v13, v10, v12\n"
                "  fmla.2d v13, v6, v7\n"
                "  add.2d v0, v0, v12\n"
                "  movk x9, #4913, lsl 48\n"
                "  add.2d v4, v4, v13\n"
                "  mov x4, #47492\n"
                "  movk x4, #23630, lsl 16\n"
                "  mul x13, x5, x10\n"
                "  movk x4, #49985, lsl 32\n"
                "  movk x4, #17168, lsl 48\n"
                "  dup.2d v7, x4\n"
                "  umulh x4, x5, x10\n"
                "  mov.16b v12, v9\n"
                "  fmla.2d v12, v6, v7\n"
                "  adds x5, x13, x12\n"
                "  cinc x4, x4, hs\n"
                "  fsub.2d v13, v10, v12\n"
                "  fmla.2d v13, v6, v7\n"
                "  add.2d v1, v1, v12\n"
                "  mul x12, x6, x10\n"
                "  add.2d v0, v0, v13\n"
                "  mov x13, #57936\n"
                "  movk x13, #54828, lsl 16\n"
                "  umulh x6, x6, x10\n"
                "  movk x13, #18292, lsl 32\n"
                "  movk x13, #17197, lsl 48\n"
                "  dup.2d v7, x13\n"
                "  adds x4, x12, x4\n"
                "  cinc x6, x6, hs\n"
                "  mov.16b v12, v9\n"
                "  fmla.2d v12, v6, v7\n"
                "  adds x0, x4, x0\n"
                "  cinc x4, x6, hs\n"
                "  fsub.2d v13, v10, v12\n"
                "  fmla.2d v13, v6, v7\n"
                "  add.2d v2, v2, v12\n"
                "  mul x6, x7, x10\n"
                "  add.2d v1, v1, v13\n"
                "  mov x12, #17708\n"
                "  movk x12, #43915, lsl 16\n"
                "  umulh x7, x7, x10\n"
                "  movk x12, #64348, lsl 32\n"
                "  movk x12, #17188, lsl 48\n"
                "  dup.2d v7, x12\n"
                "  adds x4, x6, x4\n"
                "  cinc x6, x7, hs\n"
                "  mov.16b v12, v9\n"
                "  fmla.2d v12, v6, v7\n"
                "  fsub.2d v13, v10, v12\n"
                "  adds x1, x4, x1\n"
                "  cinc x4, x6, hs\n"
                "  fmla.2d v13, v6, v7\n"
                "  add.2d v5, v5, v12\n"
                "  mul x6, x9, x10\n"
                "  add.2d v2, v2, v13\n"
                "  mov x7, #29184\n"
                "  movk x7, #20789, lsl 16\n"
                "  umulh x9, x9, x10\n"
                "  movk x7, #19197, lsl 32\n"
                "  movk x7, #17083, lsl 48\n"
                "  dup.2d v7, x7\n"
                "  adds x4, x6, x4\n"
                "  cinc x6, x9, hs\n"
                "  mov.16b v12, v9\n"
                "  fmla.2d v12, v6, v7\n"
                "  fsub.2d v13, v10, v12\n"
                "  adds x2, x4, x2\n"
                "  cinc x4, x6, hs\n"
                "  fmla.2d v13, v6, v7\n"
                "  add.2d v3, v3, v12\n"
                "  add x3, x3, x4\n"
                "  add.2d v5, v5, v13\n"
                "  ucvtf.2d v6, v11\n"
                "  mov x4, #58856\n"
                "  mov x6, #61005\n"
                "  movk x4, #14953, lsl 16\n"
                "  movk x4, #15155, lsl 32\n"
                "  movk x4, #17181, lsl 48\n"
                "  movk x6, #58262, lsl 16\n"
                "  dup.2d v7, x4\n"
                "  mov.16b v11, v9\n"
                "  fmla.2d v11, v6, v7\n"
                "  movk x6, #32851, lsl 32\n"
                "  fsub.2d v12, v10, v11\n"
                "  fmla.2d v12, v6, v7\n"
                "  movk x6, #11582, lsl 48\n"
                "  add.2d v0, v0, v11\n"
                "  add.2d v4, v4, v12\n"
                "  mov x4, #35392\n"
                "  mov x7, #37581\n"
                "  movk x4, #12477, lsl 16\n"
                "  movk x4, #56780, lsl 32\n"
                "  movk x4, #17142, lsl 48\n"
                "  movk x7, #43836, lsl 16\n"
                "  dup.2d v7, x4\n"
                "  mov.16b v11, v9\n"
                "  fmla.2d v11, v6, v7\n"
                "  movk x7, #36286, lsl 32\n"
                "  fsub.2d v12, v10, v11\n"
                "  fmla.2d v12, v6, v7\n"
                "  movk x7, #51783, lsl 48\n"
                "  add.2d v1, v1, v11\n"
                "  add.2d v0, v0, v12\n"
                "  mov x4, #9848\n"
                "  mov x9, #10899\n"
                "  movk x4, #54501, lsl 16\n"
                "  movk x4, #31540, lsl 32\n"
                "  movk x4, #17170, lsl 48\n"
                "  movk x9, #30709, lsl 16\n"
                "  dup.2d v7, x4\n"
                "  mov.16b v11, v9\n"
                "  fmla.2d v11, v6, v7\n"
                "  movk x9, #61551, lsl 32\n"
                "  fsub.2d v12, v10, v11\n"
                "  fmla.2d v12, v6, v7\n"
                "  movk x9, #45784, lsl 48\n"
                "  add.2d v2, v2, v11\n"
                "  add.2d v1, v1, v12\n"
                "  mov x4, #9584\n"
                "  mov x10, #36612\n"
                "  movk x4, #63883, lsl 16\n"
                "  movk x4, #18253, lsl 32\n"
                "  movk x4, #17190, lsl 48\n"
                "  movk x10, #63402, lsl 16\n"
                "  dup.2d v7, x4\n"
                "  mov.16b v11, v9\n"
                "  fmla.2d v11, v6, v7\n"
                "  movk x10, #47623, lsl 32\n"
                "  fsub.2d v12, v10, v11\n"
                "  fmla.2d v12, v6, v7\n"
                "  movk x10, #9430, lsl 48\n"
                "  add.2d v5, v5, v11\n"
                "  add.2d v2, v2, v12\n"
                "  mov x4, #51712\n"
                "  mul x12, x6, x11\n"
                "  movk x4, #16093, lsl 16\n"
                "  movk x4, #30633, lsl 32\n"
                "  movk x4, #17068, lsl 48\n"
                "  umulh x6, x6, x11\n"
                "  dup.2d v7, x4\n"
                "  mov.16b v11, v9\n"
                "  fmla.2d v11, v6, v7\n"
                "  adds x4, x12, x5\n"
                "  cinc x5, x6, hs\n"
                "  fsub.2d v12, v10, v11\n"
                "  fmla.2d v12, v6, v7\n"
                "  mul x6, x7, x11\n"
                "  add.2d v3, v3, v11\n"
                "  add.2d v5, v5, v12\n"
                "  ucvtf.2d v6, v8\n"
                "  umulh x7, x7, x11\n"
                "  mov x12, #34724\n"
                "  movk x12, #40393, lsl 16\n"
                "  movk x12, #23752, lsl 32\n"
                "  adds x5, x6, x5\n"
                "  cinc x6, x7, hs\n"
                "  movk x12, #17184, lsl 48\n"
                "  dup.2d v7, x12\n"
                "  mov.16b v8, v9\n"
                "  adds x0, x5, x0\n"
                "  cinc x5, x6, hs\n"
                "  fmla.2d v8, v6, v7\n"
                "  fsub.2d v11, v10, v8\n"
                "  mul x6, x9, x11\n"
                "  fmla.2d v11, v6, v7\n"
                "  add.2d v0, v0, v8\n"
                "  add.2d v4, v4, v11\n"
                "  umulh x7, x9, x11\n"
                "  mov x9, #25532\n"
                "  movk x9, #31025, lsl 16\n"
                "  movk x9, #10002, lsl 32\n"
                "  adds x5, x6, x5\n"
                "  cinc x6, x7, hs\n"
                "  movk x9, #17199, lsl 48\n"
                "  dup.2d v7, x9\n"
                "  mov.16b v8, v9\n"
                "  adds x1, x5, x1\n"
                "  cinc x5, x6, hs\n"
                "  fmla.2d v8, v6, v7\n"
                "  fsub.2d v11, v10, v8\n"
                "  mul x6, x10, x11\n"
                "  fmla.2d v11, v6, v7\n"
                "  add.2d v1, v1, v8\n"
                "  add.2d v0, v0, v11\n"
                "  umulh x7, x10, x11\n"
                "  mov x9, #18830\n"
                "  movk x9, #2465, lsl 16\n"
                "  movk x9, #36348, lsl 32\n"
                "  adds x5, x6, x5\n"
                "  cinc x6, x7, hs\n"
                "  movk x9, #17194, lsl 48\n"
                "  dup.2d v7, x9\n"
                "  mov.16b v8, v9\n"
                "  adds x2, x5, x2\n"
                "  cinc x5, x6, hs\n"
                "  fmla.2d v8, v6, v7\n"
                "  fsub.2d v11, v10, v8\n"
                "  fmla.2d v11, v6, v7\n"
                "  add x3, x3, x5\n"
                "  add.2d v2, v2, v8\n"
                "  add.2d v1, v1, v11\n"
                "  mov x5, #65535\n"
                "  mov x6, #21566\n"
                "  movk x6, #43708, lsl 16\n"
                "  movk x6, #57685, lsl 32\n"
                "  movk x5, #61439, lsl 16\n"
                "  movk x6, #17185, lsl 48\n"
                "  dup.2d v7, x6\n"
                "  mov.16b v8, v9\n"
                "  movk x5, #62867, lsl 32\n"
                "  fmla.2d v8, v6, v7\n"
                "  fsub.2d v11, v10, v8\n"
                "  fmla.2d v11, v6, v7\n"
                "  movk x5, #49889, lsl 48\n"
                "  add.2d v5, v5, v8\n"
                "  add.2d v2, v2, v11\n"
                "  mul x5, x5, x4\n"
                "  mov x6, #3072\n"
                "  movk x6, #8058, lsl 16\n"
                "  movk x6, #46097, lsl 32\n"
                "  mov x7, #1\n"
                "  movk x6, #17047, lsl 48\n"
                "  dup.2d v7, x6\n"
                "  mov.16b v8, v9\n"
                "  movk x7, #61440, lsl 16\n"
                "  fmla.2d v8, v6, v7\n"
                "  fsub.2d v11, v10, v8\n"
                "  fmla.2d v11, v6, v7\n"
                "  movk x7, #62867, lsl 32\n"
                "  add.2d v3, v3, v8\n"
                "  add.2d v5, v5, v11\n"
                "  movk x7, #17377, lsl 48\n"
                "  mov x6, #65535\n"
                "  movk x6, #61439, lsl 16\n"
                "  movk x6, #62867, lsl 32\n"
                "  mov x9, #28817\n"
                "  movk x6, #1, lsl 48\n"
                "  umov x10, v4.d[0]\n"
                "  umov x11, v4.d[1]\n"
                "  movk x9, #31161, lsl 16\n"
                "  mul x10, x10, x6\n"
                "  mul x6, x11, x6\n"
                "  and x10, x10, x8\n"
                "  movk x9, #59464, lsl 32\n"
                "  and x6, x6, x8\n"
                "  ins v6.d[0], x10\n"
                "  ins v6.d[1], x6\n"
                "  movk x9, #10291, lsl 48\n"
                "  ucvtf.2d v6, v6\n"
                "  mov x6, #16\n"
                "  movk x6, #22847, lsl 32\n"
                "  mov x8, #22621\n"
                "  movk x6, #17151, lsl 48\n"
                "  dup.2d v7, x6\n"
                "  mov.16b v8, v9\n"
                "  movk x8, #33153, lsl 16\n"
                "  fmla.2d v8, v6, v7\n"
                "  fsub.2d v11, v10, v8\n"
                "  fmla.2d v11, v6, v7\n"
                "  movk x8, #17846, lsl 32\n"
                "  add.2d v0, v0, v8\n"
                "  add.2d v4, v4, v11\n"
                "  movk x8, #47184, lsl 48\n"
                "  mov x6, #20728\n"
                "  movk x6, #23588, lsl 16\n"
                "  movk x6, #7790, lsl 32\n"
                "  mov x10, #41001\n"
                "  movk x6, #17170, lsl 48\n"
                "  dup.2d v7, x6\n"
                "  mov.16b v8, v9\n"
                "  movk x10, #57649, lsl 16\n"
                "  fmla.2d v8, v6, v7\n"
                "  fsub.2d v11, v10, v8\n"
                "  fmla.2d v11, v6, v7\n"
                "  movk x10, #20082, lsl 32\n"
                "  add.2d v1, v1, v8\n"
                "  add.2d v0, v0, v11\n"
                "  movk x10, #12388, lsl 48\n"
                "  mov x6, #16000\n"
                "  movk x6, #53891, lsl 16\n"
                "  movk x6, #5509, lsl 32\n"
                "  mul x11, x7, x5\n"
                "  movk x6, #17144, lsl 48\n"
                "  dup.2d v7, x6\n"
                "  mov.16b v8, v9\n"
                "  umulh x6, x7, x5\n"
                "  fmla.2d v8, v6, v7\n"
                "  fsub.2d v11, v10, v8\n"
                "  fmla.2d v11, v6, v7\n"
                "  cmn x11, x4\n"
                "  cinc x6, x6, hs\n"
                "  add.2d v2, v2, v8\n"
                "  add.2d v7, v1, v11\n"
                "  mul x4, x9, x5\n"
                "  mov x7, #46800\n"
                "  movk x7, #2568, lsl 16\n"
                "  movk x7, #1335, lsl 32\n"
                "  umulh x9, x9, x5\n"
                "  movk x7, #17188, lsl 48\n"
                "  dup.2d v1, x7\n"
                "  mov.16b v8, v9\n"
                "  adds x4, x4, x6\n"
                "  cinc x6, x9, hs\n"
                "  fmla.2d v8, v6, v1\n"
                "  fsub.2d v11, v10, v8\n"
                "  fmla.2d v11, v6, v1\n"
                "  adds x0, x4, x0\n"
                "  cinc x4, x6, hs\n"
                "  add.2d v1, v5, v8\n"
                "  add.2d v5, v2, v11\n"
                "  mul x6, x8, x5\n"
                "  mov x7, #39040\n"
                "  movk x7, #14704, lsl 16\n"
                "  movk x7, #12839, lsl 32\n"
                "  umulh x8, x8, x5\n"
                "  movk x7, #17096, lsl 48\n"
                "  dup.2d v2, x7\n"
                "  mov.16b v8, v9\n"
                "  adds x4, x6, x4\n"
                "  cinc x6, x8, hs\n"
                "  fmla.2d v8, v6, v2\n"
                "  fsub.2d v9, v10, v8\n"
                "  fmla.2d v9, v6, v2\n"
                "  adds x1, x4, x1\n"
                "  cinc x4, x6, hs\n"
                "  add.2d v6, v3, v8\n"
                "  add.2d v8, v1, v9\n"
                "  mul x6, x10, x5\n"
                "  ssra.2d v0, v4, #52\n"
                "  ssra.2d v7, v0, #52\n"
                "  ssra.2d v5, v7, #52\n"
                "  umulh x5, x10, x5\n"
                "  ssra.2d v8, v5, #52\n"
                "  ssra.2d v6, v8, #52\n"
                "  ushr.2d v1, v7, #12\n"
                "  adds x4, x6, x4\n"
                "  cinc x5, x5, hs\n"
                "  ushr.2d v2, v5, #24\n"
                "  ushr.2d v3, v8, #36\n"
                "  sli.2d v0, v7, #52\n"
                "  adds x2, x4, x2\n"
                "  cinc x4, x5, hs\n"
                "  sli.2d v1, v5, #40\n"
                "  sli.2d v2, v8, #28\n"
                "  sli.2d v3, v6, #16\n"
                "  add x3, x3, x4\n"

                :                                                         // Output operands
                "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]),   // x0-x3 outputs
                "=w"(outv[0]), "=w"(outv[1]), "=w"(outv[2]), "=w"(outv[3])// v0-v3 outputs

                :                                              // Input operands
                "0"(a[0]), "1"(a[1]), "2"(a[2]), "3"(a[3]),    // x0-x3 inputs (a)
                "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),    // x4-x7 inputs (b)
                "4"(av[0]), "5"(av[1]), "6"(av[2]), "7"(av[3]),// v0-v3 inputs (av)
                "w"(bv[0]), "w"(bv[1]), "w"(bv[2]), "w"(bv[3]) // v4-v7 inputs (bv)

                :// Clobbered registers
                "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
                "x12", "x13", "x14", "x15", "x16", "lr",
                "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                "v20", "v21", "v22", "v23", "v24");

        return std::make_pair(out, outv);
    }

    static constexpr mp_256_t ZERO = {0, 0, 0, 0};
    static constexpr mp_256_t P = {0x3c208c16d87cfd47, 0x97816a916871ca8d, 0xb85045b68181585d, 0x30644e72e131a029};
    static constexpr mp_256_t P2 = {0x7841182db0f9fa8e, 0x2f02d522d0e3951a, 0x70a08b6d0302b0bb, 0x60c89ce5c2634053};
    static constexpr mp_256_t P3 = {0xb461a4448976f7d5, 0xc6843fb439555fa7, 0x28f0d12384840918, 0x912ceb58a394e07d};
    static constexpr mp_256_t P4 = {0xf082305b61f3f51c, 0x5e05aa45a1c72a34, 0xe14116da06056176, 0xc19139cb84c680a6};
    static constexpr mp_512_t PP = {0x3b5458a2275d69b1, 0xa602072d09eac101, 0x4a50189c6d96cadc, 0x4689e957a1242c8, 0x26edfa5c34c6b38d, 0xb00b855116375606, 0x599a6f7c0348d21c, 0x925c4b8763cbf9c};
    static constexpr mp_256_t I1 = {0x327d7c1b18f7bd41, 0xdb8ed52f824ed32f, 0x29b67b05eb29a6a1, 0x19ac99126b459dda};
    static constexpr mp_256_t I2 = {0x1da790e434ade680, 0x27a2f342f9905883, 0xb5ab34890dfa3d61, 0x1e07f71b064ef9b1};
    static constexpr mp_256_t I3 = {0xb334aa7264874f53, 0x62a52db096edbc9e, 0x235878f5c0a1dafe, 0x28f5dd496ed1da9d};
    static constexpr uint64_t MU = 0x87d20782e4866389;

    //    static constexpr mp_256_t P = {0x43e1f593f0000001, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029};
    //    static constexpr mp_256_t P2 = {0x87c3eb27e0000002, 0x5067d090f372e122, 0x70a08b6d0302b0ba, 0x60c89ce5c2634053};
    //    static constexpr mp_256_t P3 = {0xcba5e0bbd0000003, 0x789bb8d96d2c51b3, 0x28f0d12384840917, 0x912ceb58a394e07d};
    //    static constexpr mp_256_t I1 = {0x2d3e8053e396ee4d, 0xca478dbeab3c92cd, 0xb2d8f06f77f52a93, 0x24d6ba07f7aa8f04};
    //    static constexpr mp_256_t I2 = {0x18ee753c76f9dc6f, 0x54ad7e14a329e70f, 0x2b16366f4f7684df, 0x133100d71fdf3579};
    //    static constexpr mp_256_t I3 = {0x9BACB016127CBE4E, 0x0B2051FA31944124, 0xB064EEA46091C76C, 0x2B062AAA49F80C7D};
    //    static constexpr uint64_t MU = 0xc2e1f593efffffff;

    static mp_256_t mul(const mp_256_t &a, const mp_256_t &b) {
        auto [c00lo, c00hi] = mul_wide(a[0], b[0]);
        auto [c01lo, c01hi] = mul_wide(a[0], b[1]);
        auto [c02lo, c02hi] = mul_wide(a[0], b[2]);
        auto [c03lo, c03hi] = mul_wide(a[0], b[3]);
        auto [c10lo, c10hi] = mul_wide(a[1], b[0]);
        auto [c11lo, c11hi] = mul_wide(a[1], b[1]);
        auto [c12lo, c12hi] = mul_wide(a[1], b[2]);
        auto [c13lo, c13hi] = mul_wide(a[1], b[3]);
        auto [c20lo, c20hi] = mul_wide(a[2], b[0]);
        auto [c21lo, c21hi] = mul_wide(a[2], b[1]);
        auto [c22lo, c22hi] = mul_wide(a[2], b[2]);
        auto [c23lo, c23hi] = mul_wide(a[2], b[3]);
        auto [c30lo, c30hi] = mul_wide(a[3], b[0]);
        auto [c31lo, c31hi] = mul_wide(a[3], b[1]);
        auto [c32lo, c32hi] = mul_wide(a[3], b[2]);
        auto [c33lo, c33hi] = mul_wide(a[3], b[3]);

        bool c = false;
        mp_128_t r0 = {0, 0};
        mp_128_t r1 = {0, 0};
        mp_128_t r2 = {0, 0};
        mp_128_t r3 = {0, 0};

        std::tie(r0, std::ignore) = wadd(c00lo, c00hi, r0, false);

        std::tie(r0, c) = wadd(0, c01lo, r0, false);
        std::tie(r1, std::ignore) = wadd(c11lo, c11hi, r1, c);

        std::tie(r0, c) = wadd(0, c10lo, r0, false);
        std::tie(r1, c) = wadd(c01hi, c12lo, r1, c);
        std::tie(r2, std::ignore) = wadd(c12hi, 0, r2, c);

        std::tie(r1, c) = wadd(c10hi, c21lo, r1, false);
        std::tie(r2, std::ignore) = wadd(c21hi, 0, r2, c);

        std::tie(r1, c) = wadd(c02lo, c02hi, r1, false);
        std::tie(r2, c) = wadd(c13lo, c13hi, r2, c);

        std::tie(r1, c) = wadd(c20lo, c20hi, r1, false);
        std::tie(r2, c) = wadd(c31lo, c31hi, r2, c);

        std::tie(r1, c) = wadd(0, c03lo, r1, false);
        std::tie(r2, c) = wadd(c03hi, c23lo, r2, c);
        std::tie(r3, std::ignore) = wadd(c23hi, 0, r3, c);

        std::tie(r1, c) = wadd(0, c30lo, r1, false);
        std::tie(r2, c) = wadd(c30hi, c32lo, r2, c);
        std::tie(r3, std::ignore) = wadd(c32hi, 0, r3, c);

        uint64_t r0lo = r0[0];
        uint64_t r0hi = r0[1];
        auto [ir000lo, ir000hi] = mul_wide(r0lo, I2[0]);
        auto [ir001lo, ir001hi] = mul_wide(r0lo, I2[1]);
        auto [ir002lo, ir002hi] = mul_wide(r0lo, I2[2]);
        auto [ir003lo, ir003hi] = mul_wide(r0lo, I2[3]);
        auto [ir010lo, ir010hi] = mul_wide(r0hi, I2[0]);
        auto [ir011lo, ir011hi] = mul_wide(r0hi, I2[1]);
        auto [ir012lo, ir012hi] = mul_wide(r0hi, I2[2]);
        auto [ir013lo, ir013hi] = mul_wide(r0hi, I2[3]);

        std::tie(r1, c) = wadd(ir000lo, ir000hi, r1, false);
        std::tie(r2, c) = wadd(c22lo, c22hi, r2, c);
        std::tie(r3, std::ignore) = wadd(c33lo, c33hi, r3, c);

        std::tie(r1, c) = wadd(0, ir001lo, r1, false);
        std::tie(r2, c) = wadd(ir002lo, ir002hi, r2, c);
        std::tie(r3, std::ignore) = wadd(ir003hi, 0, r3, c);

        std::tie(r1, c) = wadd(0, ir010lo, r1, false);
        std::tie(r2, c) = wadd(ir001hi, ir003lo, r2, c);
        std::tie(r3, std::ignore) = wadd(ir012hi, 0, r3, c);

        uint64_t r1lo = r1[0];
        auto [ir100lo, ir100hi] = mul_wide(r1lo, I1[0]);
        auto [ir101lo, ir101hi] = mul_wide(r1lo, I1[1]);
        auto [ir102lo, ir102hi] = mul_wide(r1lo, I1[2]);
        auto [ir103lo, ir103hi] = mul_wide(r1lo, I1[3]);

        std::tie(r1, c) = wadd(0, ir100lo, r1, false);
        std::tie(r2, c) = wadd(ir010hi, ir012lo, r2, c);
        std::tie(r3, std::ignore) = wadd(ir013lo, ir013hi, r3, c);

        uint64_t m = MU * r1[1];
        auto [m0lo, m0hi] = mul_wide(m, P[0]);
        auto [m1lo, m1hi] = mul_wide(m, P[1]);
        auto [m2lo, m2hi] = mul_wide(m, P[2]);
        auto [m3lo, m3hi] = mul_wide(m, P[3]);

        std::tie(std::ignore, c) = wadd(0, m0lo, r1, false);
        std::tie(r2, c) = wadd(ir011lo, ir011hi, r2, c);
        std::tie(r3, std::ignore) = wadd(ir102hi, 0, r3, c);

        std::tie(r2, c) = wadd(ir100hi, ir102lo, r2, false);
        std::tie(r3, std::ignore) = wadd(ir103lo, ir103hi, r3, c);

        std::tie(r2, c) = wadd(ir101lo, ir101hi, r2, false);
        std::tie(r3, std::ignore) = wadd(m2hi, 0, r3, c);

        std::tie(r2, c) = wadd(m0hi, m2lo, r2, false);
        std::tie(r3, std::ignore) = wadd(m3lo, m3hi, r3, c);

        std::tie(r2, c) = wadd(m1lo, m1hi, r2, false);
        std::tie(r3, std::ignore) = wadd(0, 0, r3, c);

        mp_256_t r = {r2[0], r2[1], r3[0], r3[1]};

        const uint8_t top_bits = r[3] >> 62;
        if (top_bits == 0b11) {
            r = sub(r, P3);
        } else if (top_bits == 0b10) {
            r = sub(r, P2);
        } else if (top_bits == 0b01) {
            r = sub(r, P);
        }
        auto [rr, cc] = subc(r, P);
        r = cc ? r : rr;

        return r;
    }

    INLINE static mp_256_t reduce(const mp_512_t &ab) {
        mp_320_t r1 = smul(ab[0], I3);
        mp_320_t r2 = smul(ab[1], I2);
        mp_320_t r3 = smul(ab[2], I1);
        mp_320_t s = add(add({ab[3], ab[4], ab[5], ab[6], ab[7]}, r1), add(r2, r3));
        uint64_t m = MU * s[0];
        mp_320_t mp = smul(m, P);
        s = add(s, mp);
        mp_256_t r = {s[1], s[2], s[3], s[4]};
        const uint8_t top_bits = r[3] >> 62;
        if (top_bits == 0b11) {
            r = sub(r, P3);
        } else if (top_bits == 0b10) {
            r = sub(r, P2);
        } else if (top_bits == 0b01) {
            r = sub(r, P);
        }
        auto [rr, cc] = subc(r, P);
        r = cc ? r : rr;

        return r;
    }

    static mp_256_t reduce(const mp_256_t &a) {
        mp_320_t r1 = smul(a[0], I3);
        mp_320_t r2 = smul(a[1], I2);
        mp_320_t r3 = smul(a[2], I1);
        mp_320_t s = add(add({a[3], 0, 0, 0, 0}, r1), add(r2, r3));
        uint64_t m = MU * s[0];
        mp_320_t mp = smul(m, P);
        s = add(s, mp);
        mp_256_t r = {s[1], s[2], s[3], s[4]};
        const uint8_t top_bits = r[3] >> 62;
        if (top_bits == 0b11) {
            r = sub(r, P3);
        } else if (top_bits == 0b10) {
            r = sub(r, P2);
        } else if (top_bits == 0b01) {
            r = sub(r, P);
        }
        auto [rr, cc] = subc(r, P);
        r = cc ? r : rr;

        return r;
    }

    static INLINE mp_256_t add_red(const mp_256_t &a, const mp_256_t &b) {
        mp_256_t r = add(a, b);
        auto [rr, cc] = subc(r, P);
        r = cc ? r : rr;
        return r;
    }

    static INLINE mp_512_t add_red(const mp_512_t &a, const mp_512_t &b) {
        mp_512_t r = add(a, b);
        auto [rr, cc] = subc(r, PP);
        r = cc ? r : rr;
        return r;
    }

    static INLINE mp_256_t sub_red(const mp_256_t &a, const mp_256_t &b) {
        auto [r, c] = subc(a, b);
        mp_256_t rr = add(r, P);
        r = c ? rr : r;
        return r;
    }

    static INLINE mp_512_t sub_red(const mp_512_t &a, const mp_512_t &b) {
        auto [r, c] = subc(a, b);
        mp_512_t rr = add(r, PP);
        r = c ? rr : r;
        return r;
    }
};
