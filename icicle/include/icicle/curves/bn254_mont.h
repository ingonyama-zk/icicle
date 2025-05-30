#pragma once

#include <array>
#include <cstdint>
#include <tuple>

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

    INLINE static mp_320_t addv(const mp_320_t &a, const mp_320_t &b) {
        mp_320_t r;
        uint64_t carry = 0, tmp, c;

        std::tie(tmp, c) = addc(a[0], b[0]);
        std::tie(r[0], carry) = addc(tmp, carry);
        carry += c;

        std::tie(tmp, c) = addc(a[1], b[1]);
        std::tie(r[1], carry) = addc(tmp, carry);
        carry += c;

        std::tie(tmp, c) = addc(a[2], b[2]);
        std::tie(r[2], carry) = addc(tmp, carry);
        carry += c;

        std::tie(tmp, c) = addc(a[3], b[3]);
        std::tie(r[3], carry) = addc(tmp, carry);
        carry += c;

        std::tie(tmp, c) = addc(a[4], b[4]);
        std::tie(r[4], carry) = addc(tmp, carry);

        return r;
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
        mp_320_t s = addv(addv({ab[3], ab[4], ab[5], ab[6], ab[7]}, r1), addv(r2, r3));
        uint64_t m = MU * s[0];
        mp_320_t mp = smul(m, P);
        s = addv(s, mp);
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
        mp_320_t s = addv(addv({a[3], 0, 0, 0, 0}, r1), addv(r2, r3));
        uint64_t m = MU * s[0];
        mp_320_t mp = smul(m, P);
        s = addv(s, mp);
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
