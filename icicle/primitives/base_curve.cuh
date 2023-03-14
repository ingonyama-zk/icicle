#pragma once
#include "../utils/storage.cuh"
#include "../utils/ptx.cuh"
#include "../utils/host_math.cuh"
#include <random>
#include <iostream>

#define HOST_INLINE __host__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__

template <class CONFIG> class Field {
  public:
    static constexpr unsigned TLC = CONFIG::limbs_count;
    static constexpr unsigned NBITS = CONFIG::modulus_bits_count;

    static constexpr HOST_DEVICE_INLINE Field zero() {
      return Field { CONFIG::zero };
    }
    static constexpr HOST_DEVICE_INLINE Field one() {
      return Field { CONFIG::one };
    }
    static constexpr HOST_INLINE Field omega(uint32_t log_size) {
      storage<CONFIG::limbs_count> omega[32]={ {0x00000000, 0xffffffff, 0xfffe5bfe, 0x53bda402, 0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753}, {0x00000000, 0x00010000, 0x76030000, 0xec030002, 0x760304d0, 0x8d51ccce, 0x00000000, 0x00000000}, {0x688bc087, 0x8dd702cb, 0x78eaa4fe, 0xa0328240, 0x98ca5b22, 0xa733b23a, 0x25a31660, 0x3f96405d}, {0x0411fe73, 0x95df4b36, 0xebc1e1bb, 0x1ef4e672, 0x60afca4a, 0x6e92a9c4, 0x753e4fcc, 0x4f2c596e}, {0xba60eaa6, 0x9733f3a6, 0x77487ae7, 0xbd7fdf9c, 0xc8b6cc00, 0xd84f8612, 0x6162ffab, 0x476fa2fb}, {0xac5db47f, 0xd2fc5e69, 0x15d0b8e4, 0xa12a70a6, 0xbc8de5d9, 0x293b1d67, 0x57f86f5e, 0x0e4840ac}, {0xab28e208, 0xb750da4c, 0x3be95635, 0x501dff64, 0xf0b4b276, 0x8cbe2437, 0xa94a946e, 0x07d0c802}, {0x2fe322b8, 0x2cabadec, 0x15412560, 0x752c84f3, 0x1a3b0aef, 0x32a732ae, 0xa33dcbf2, 0x2e95da59}, {0xfe0c65f4, 0x33811ea1, 0x687f28a2, 0x15c1ad4c, 0x42dee7f4, 0xecfbede3, 0x9a5d88b1, 0x1bb46667}, {0x2d010ff9, 0xd58a5af4, 0x570bf109, 0x79efd6b0, 0x6350721d, 0x3ed6d55a, 0x58f43cef, 0x2f27b098}, {0x8c130477, 0x74a1f671, 0xb61e0abe, 0xa534af14, 0x620890d7, 0xeb674a1a, 0xca252472, 0x43527a8b}, {0x7ea8ee05, 0x450d9f97, 0x37d56fc0, 0x565af171, 0x93f9e9ac, 0xe155cb48, 0xc8e9101b, 0x110cebd0}, {0x59a0be92, 0x23c91599, 0x7a027759, 0x87d188ce, 0xcab3c3cc, 0x70491431, 0xb3f7f8da, 0x0ac00eb8}, {0x69583404, 0x13e96ade, 0x5306243d, 0x82c05727, 0x29ca9f2a, 0x77e48bf5, 0x1fe19595, 0x50646ac8}, {0xa97eccd4, 0xe6a354dd, 0x88fbbc57, 0x39929d2e, 0xd6e7b1c8, 0xa22ba63d, 0xf5f07f43, 0x42c22911}, {0xcfc35f7a, 0x137b458a, 0x29c01b06, 0x0caba63a, 0x7a02402c, 0x0409ee98, 0x56aa725b, 0x6709c6cd}, {0x8831e03e, 0x10251f7d, 0x7ff858ec, 0x77d85a93, 0x4fb9ac5c, 0xebe905bd, 0xf8727901, 0x05deb333}, {0xb9009408, 0xbf87b689, 0xdd3ccc96, 0x4f730e7d, 0x4610300c, 0xfd7f05ba, 0x0b8ac903, 0x5ef5e8db}, {0x17cd0c14, 0x64996884, 0x68812f7f, 0xa6728673, 0x22cc3253, 0x2e1d9a19, 0xaa0a1d80, 0x3a689e83}, {0x41144dea, 0x20b53cbe, 0xc2f0fcbd, 0x870c46fa, 0x537d6971, 0x556c35f6, 0x5f686d91, 0x3436287f}, {0x436ba2e7, 0x007e082a, 0x9116e877, 0x67c6630f, 0xfb4460f7, 0x36f8f165, 0x7e7046e0, 0x6eee34d5}, {0xa53a56d1, 0xc5b670ee, 0x53037d7b, 0x127d1f42, 0xa722c2e2, 0x57d4257e, 0x33cbd838, 0x03ae26a3}, {0x76504cf8, 0x1e914848, 0xb63edd02, 0x55bbbf1e, 0x4e55aa02, 0xbcdafec8, 0x2dc0beb0, 0x5145c4cd}, {0x1ab70e2c, 0x5b90153a, 0x75fb0ab8, 0x8deffa31, 0x46900c95, 0xc553ae23, 0x6bd3118c, 0x1d31dcdc}, {0x59a2e8eb, 0x801c894c, 0xe12fc974, 0xbc535c5c, 0x47d39803, 0x95508d27, 0xac5d094f, 0x16d9d3cd}, {0xcca1d8be, 0x810fa372, 0x82e0bfa7, 0xc67b8c28, 0xe2d35bc2, 0xdbb4edf0, 0x5087c995, 0x712d1580}, {0xfd88f133, 0xeb162203, 0xf010ea74, 0xac96c38f, 0xe64cfc70, 0x4307987f, 0x37b7a114, 0x350fe98d}, {0x42f2a254, 0xaba2f518, 0xa71efc0c, 0x4d7f3c3a, 0xd274a80a, 0x97ae418d, 0x5e3e7682, 0x2967385d}, {0x575a0b79, 0x75c55c7b, 0x74a7ded1, 0x3ba4a157, 0xa04fccf3, 0xc3974d73, 0x4a939684, 0x705aba4f}, {0x14ebb608, 0x8409a9ea, 0x66bac611, 0xfad0084e, 0x811c1dfb, 0x04287254, 0x23b30c29, 0x086d072b}, {0x67e4756a, 0xb427c9b3, 0x02ebc38d, 0xc7537fb9, 0xcd6a205f, 0x51de21be, 0x7923597d, 0x6064ab72}, {0x0b912f1f, 0x1b788f50, 0x70b3e094, 0xc4024ff2, 0xd168d6c0, 0x0fd56dc8, 0x5b416b6f, 0x0212d79e}};
      return Field { omega[log_size-1] };
    }
    static constexpr HOST_INLINE Field omega_inv(uint32_t log_size) {
      storage<CONFIG::limbs_count> omega_inv[32]={ {0x00000000, 0xffffffff, 0xfffe5bfe, 0x53bda402, 0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753}, {0x00000001, 0xfffeffff, 0x89fb5bfe, 0x67baa400, 0x939ed334, 0xa5e80b39, 0x299d7d47, 0x73eda753}, {0xae99502e, 0x6037fe81, 0x94b04fd8, 0x8e749036, 0xca86bf65, 0xbabc5aff, 0x5ce11044, 0x1333b22e}, {0x7dc08d74, 0x7f847ee4, 0x04eeaf5a, 0xbd433896, 0x1832fc60, 0xd66c91d6, 0x607e449b, 0x551115b4}, {0x4e7773cb, 0xee5bcecc, 0xf6dab086, 0x45593d6f, 0x4016e2bd, 0xa3a95d2d, 0xaf96816f, 0x047cb16c}, {0x982b68c5, 0xb891fa3f, 0x1d426b52, 0xa41e8501, 0x882952d6, 0x566009b5, 0x7b3c79d6, 0x199cdaee}, {0xcf28601b, 0x571ba2fc, 0xac74db12, 0x166fb582, 0x3501370b, 0x51420be4, 0x52f970ba, 0x1996fa8d}, {0x6a2f777a, 0xe9561c17, 0x2393991b, 0xc03cae03, 0x5a5bfd4f, 0x91b00023, 0x272e58ee, 0x6d64ed25}, {0xf02a116e, 0xfb350dbe, 0xb4543a3e, 0x1c510ebf, 0x37ad4eca, 0xf675522e, 0x80f82b2d, 0x1907a56e}, {0x4eb71aa6, 0xb0ad8003, 0xaa67e0be, 0x50a32c41, 0x19141f44, 0x105f0672, 0xa3dad316, 0x2bcd9508}, {0x0f6fb2ac, 0x3dc9e560, 0x9aa58ff5, 0x3cc5bb32, 0x36f376e1, 0xdeae67bc, 0x65ba213e, 0x394fda0d}, {0x60b82267, 0x09f239f7, 0x8b24f123, 0x14180e0e, 0x45625d95, 0xad5a5340, 0x6d174692, 0x58c3ba63}, {0x348b416f, 0x0acf21c2, 0xbc086439, 0x798b6bf6, 0xb1ca111d, 0x222d411f, 0x30ba1e0f, 0x044107b7}, {0x014abe84, 0xa3b861b8, 0x427ed008, 0x37c017e4, 0xae0ff4f5, 0xae51f613, 0xcb1218d3, 0x1a2d00e1}, {0x4de7eb2b, 0x48aaa3bf, 0x6772057d, 0x4a58d54d, 0x7093b551, 0xce25f16c, 0xd206337c, 0x242150ac}, {0x9ed57ae5, 0xdf3ec9ae, 0x7166577f, 0xea7df73a, 0x022fbbe4, 0x6ca8d281, 0x151e3f6b, 0x5850c003}, {0x645e1cfa, 0x903a0a0c, 0x34788c37, 0xfbac54cb, 0x8cf73d78, 0xdc127d11, 0x975d3c82, 0x6d0b5c7c}, {0x14b1ba04, 0xb49d6b05, 0xf00b84f2, 0x56e466b4, 0x0b904f22, 0x30c390cf, 0x3ee254cc, 0x3e11cfb7}, {0xbe8201ab, 0x84dfa547, 0x530715d2, 0x3887ce8b, 0x3eed4ed7, 0xa4c719c6, 0x8f8007b4, 0x18c44950}, {0x7d813cd1, 0xdaf0346d, 0xf755beb1, 0xeccf6f9a, 0xe08143e3, 0x167fce38, 0x6f5d6dfa, 0x545ad9b2}, {0x577605de, 0x973f5466, 0x974f953c, 0x0ce8986e, 0x074382f9, 0x8941cf4b, 0x6fa2672c, 0x156cd7f6}, {0x33b66141, 0x24315404, 0x1992f584, 0x5d1375ab, 0x8b20ca1a, 0xf193ffa6, 0x2701a503, 0x47880cd5}, {0xe9f7b9af, 0xf7b6847d, 0x62c83ce2, 0x9a339673, 0x6e5e6f79, 0xfabf4537, 0x35af33a3, 0x0975acd9}, {0x0eddd248, 0x4fb4204a, 0xc9e509b3, 0x8c98706a, 0x2bb27eb1, 0xd0be8987, 0xc831438b, 0x6ec5f960}, {0x20238f62, 0xa13c95b7, 0x83b476b9, 0x130aa097, 0x14860881, 0x758a04e0, 0x97066493, 0x58e2f8d6}, {0xe8bff41e, 0x65b09c73, 0x37f1c6a3, 0x8b3280e8, 0x2846fb21, 0xe17b82ce, 0xb1ae27df, 0x476534bf}, {0xd5fdb757, 0x8480c0e7, 0x365bf9fd, 0x3644eea0, 0xb776be86, 0x4ca116ca, 0x8b58390c, 0x17b6395f}, {0x252eb0db, 0x2c811e9a, 0x7479e161, 0x1b7d960d, 0xb0a89a26, 0xb3afc7c1, 0x32b5e793, 0x6a2f9533}, {0x08b8a7ad, 0xe877b2c4, 0x341652b4, 0x68b0e8f0, 0xe8b6a2d9, 0x2d44da3b, 0xfd09be59, 0x092778ff}, {0x7988f244, 0x84a1aa6f, 0x24faf63f, 0xa164b3d9, 0xc1bbb915, 0x7aae9724, 0xf386c0d2, 0x24e5d287}, {0x41a1b30c, 0xa70a7efd, 0x39f0e511, 0xc49c55a5, 0x033bb323, 0xab307a8f, 0x17acbd7f, 0x0158abd6}, {0x0f642025, 0x2c228b30, 0x01bd882b, 0xb0878e8d, 0xd7377fea, 0xd862b255, 0xf0490536, 0x18ac3666}};
      return Field { omega_inv[log_size-1] };
    }
    static constexpr HOST_INLINE Field inv_log_size(uint32_t log_size) {
      storage<CONFIG::limbs_count> inv[32]={ {0x80000001, 0x7fffffff, 0x7fff2dff, 0xa9ded201, 0x04d0ec02, 0x199cec04, 0x94cebea4, 0x39f6d3a9}, {0x40000001, 0x3fffffff, 0x3ffec4ff, 0xfece3b02, 0x07396203, 0x266b6206, 0x5f361df6, 0x56f23d7e}, {0x20000001, 0x1fffffff, 0x9ffe907f, 0xa945ef82, 0x086d9d04, 0x2cd29d07, 0xc469cd9f, 0x656ff268}, {0x10000001, 0x0fffffff, 0xcffe763f, 0xfe81c9c2, 0x8907ba84, 0xb0063a87, 0xf703a573, 0x6caeccdd}, {0x08000001, 0x07ffffff, 0xe7fe691f, 0x291fb6e2, 0xc954c945, 0xf1a00947, 0x9050915d, 0x704e3a18}, {0x04000001, 0x03ffffff, 0xf3fe628f, 0x3e6ead72, 0xe97b50a5, 0x126cf0a7, 0xdcf70753, 0x721df0b5}, {0x02000001, 0x01ffffff, 0xf9fe5f47, 0x491628ba, 0xf98e9455, 0xa2d36457, 0x834a424d, 0x7305cc04}, {0x01000001, 0x00ffffff, 0xfcfe5da3, 0x4e69e65e, 0x0198362d, 0xeb069e30, 0xd673dfca, 0x7379b9ab}, {0x00800001, 0x007fffff, 0xfe7e5cd1, 0x5113c530, 0x059d0719, 0x8f203b1c, 0x8008ae89, 0x73b3b07f}, {0x00400001, 0x003fffff, 0xff3e5c68, 0x5268b499, 0x079f6f8f, 0xe12d0992, 0x54d315e8, 0x73d0abe9}, {0x00200001, 0x801fffff, 0x7f9e5c33, 0x53132c4e, 0x08a0a3ca, 0x8a3370cd, 0x3f384998, 0x73df299e}, {0x00100001, 0x400fffff, 0xbfce5c19, 0xd3686828, 0x89213de7, 0x5eb6a46a, 0xb46ae370, 0x73e66878}, {0x00080001, 0x2007ffff, 0xdfe65c0c, 0x93930615, 0x49618af6, 0x48f83e39, 0xef04305c, 0x73ea07e5}, {0x00040001, 0x9003ffff, 0x6ff25c05, 0xf3a8550c, 0xa981b17d, 0x3e190b20, 0x8c50d6d2, 0x73ebd79c}, {0x00020001, 0x4801ffff, 0xb7f85c02, 0xa3b2fc87, 0x5991c4c1, 0x38a97194, 0xdaf72a0d, 0x73ecbf77}, {0x00010001, 0xa400ffff, 0x5bfb5c00, 0x7bb85045, 0x3199ce63, 0xb5f1a4ce, 0x824a53aa, 0x73ed3365}, {0x00008001, 0xd2007fff, 0x2dfcdbff, 0x67bafa24, 0x1d9dd334, 0x7495be6b, 0x55f3e879, 0x73ed6d5c}, {0x00004001, 0x69003fff, 0x96fd9bff, 0xddbc4f13, 0x939fd59c, 0xd3e7cb39, 0xbfc8b2e0, 0x73ed8a57}, {0x00002001, 0x34801fff, 0x4b7dfbff, 0x18bcf98b, 0xcea0d6d1, 0x8390d1a0, 0x74b31814, 0x73ed98d5}, {0x00001001, 0x1a400fff, 0x25be2bff, 0x363d4ec7, 0x6c21576b, 0x5b6554d4, 0x4f284aae, 0x73eda014}, {0x00000801, 0x0d2007ff, 0x12de43ff, 0x44fd7965, 0x3ae197b8, 0x474f966e, 0xbc62e3fb, 0x73eda3b3}, {0x00000401, 0x069003ff, 0x096e4fff, 0xcc5d8eb4, 0x2241b7de, 0xbd44b73b, 0x730030a1, 0x73eda583}, {0x00000201, 0x034801ff, 0x84b655ff, 0x100d995b, 0x95f1c7f2, 0xf83f47a1, 0x4e4ed6f4, 0x73eda66b}, {0x00000101, 0x01a400ff, 0x425a58ff, 0xb1e59eaf, 0xcfc9cffb, 0x95bc8fd4, 0x3bf62a1e, 0x73eda6df}, {0x00000081, 0x00d2007f, 0x212c5a7f, 0x82d1a159, 0x6cb5d400, 0x647b33ee, 0x32c9d3b3, 0x73eda719}, {0x00000041, 0x0069003f, 0x10955b3f, 0xeb47a2ae, 0x3b2bd602, 0xcbda85fb, 0x2e33a87d, 0x73eda736}, {0x00000021, 0x0034801f, 0x8849db9f, 0x1f82a358, 0xa266d704, 0xff8a2f01, 0xabe892e2, 0x73eda744}, {0x00000011, 0x001a400f, 0xc4241bcf, 0xb9a023ad, 0xd6045784, 0x99620384, 0xeac30815, 0x73eda74b}, {0x00000009, 0x000d2007, 0x62113be7, 0x06aee3d8, 0x6fd317c5, 0xe64dedc6, 0x8a3042ae, 0x73eda74f}, {0x00000005, 0x00069003, 0xb107cbf3, 0x2d3643ed, 0x3cba77e5, 0x8cc3e2e7, 0x59e6dffb, 0x73eda751}, {0x00000003, 0x00034801, 0x588313f9, 0x4079f3f8, 0xa32e27f5, 0xdffedd77, 0x41c22ea1, 0x73eda752}, {0x00000002, 0x0001a400, 0xac40b7fc, 0x4a1bcbfd, 0xd667fffd, 0x099c5abf, 0xb5afd5f5, 0x73eda752}};
      return Field { inv[log_size-1] };
    }
    static constexpr HOST_DEVICE_INLINE Field modulus() {
      return Field { CONFIG::modulus };
    }


  private:
    typedef storage<TLC> ff_storage;
    typedef storage<2 * TLC> ff_wide_storage;

    static constexpr unsigned slack_bits = 32 * TLC - NBITS;

    struct wide {
      ff_wide_storage limbs_storage;

      friend HOST_DEVICE_INLINE wide operator+(wide xs, const wide& ys) {   
        // TODO: change the dummy implementation
        return xs;
      }

      Field HOST_DEVICE_INLINE get_lower() {
        Field out{};
      #ifdef __CUDA_ARCH__
      #pragma unroll
      #endif
        for (unsigned i = 0; i < TLC; i++)
          out.limbs_storage.limbs[i] = limbs_storage.limbs[i];
        return out;
      }

      Field HOST_DEVICE_INLINE get_higher_with_slack() {
        Field out{};
      #ifdef __CUDA_ARCH__
      #pragma unroll
      #endif
        for (unsigned i = 0; i < TLC; i++) {
        #ifdef __CUDA_ARCH__
          out.limbs_storage.limbs[i] = __funnelshift_lc(limbs_storage.limbs[i + TLC - 1], limbs_storage.limbs[i + TLC], slack_bits);
        #else
          out.limbs_storage.limbs[i] = (limbs_storage.limbs[i + TLC] << slack_bits) + (limbs_storage.limbs[i + TLC - 1] >> (32 - slack_bits));
        #endif
        }
        return out;
      }
    };

    // an incomplete impl that assumes that xs > ys
    friend HOST_DEVICE_INLINE wide operator-(wide xs, const wide& ys) {   
      wide rs = {};
      sub_limbs<false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      return rs;
    }

    // return modulus
    template <unsigned MULTIPLIER = 1> static constexpr HOST_DEVICE_INLINE ff_storage get_modulus() {
      switch (MULTIPLIER) {
        case 1:
          return CONFIG::modulus;
        case 2:
          return CONFIG::modulus_2;
        case 4:
          return CONFIG::modulus_4;
        default:
          return {};
      }
    }

    template <unsigned MULTIPLIER = 1> static constexpr HOST_DEVICE_INLINE ff_wide_storage modulus_wide() {
      return CONFIG::modulus_wide;
    }

    // return m
    static constexpr HOST_DEVICE_INLINE ff_storage get_m() {
      return CONFIG::m;
    }

    // return modulus^2, helpful for ab +/- cd
    template <unsigned MULTIPLIER = 1> static constexpr HOST_DEVICE_INLINE ff_wide_storage get_modulus_squared() {
      switch (MULTIPLIER) {
      case 1:
        return CONFIG::modulus_squared;
      case 2:
        return CONFIG::modulus_squared_2;
      case 4:
        return CONFIG::modulus_squared_4;
      default:
        return {};
      }
    }

    // add or subtract limbs
    template <bool SUBTRACT, bool CARRY_OUT> 
    static constexpr DEVICE_INLINE uint32_t add_sub_limbs_device(const ff_storage &xs, const ff_storage &ys, ff_storage &rs) {
      const uint32_t *x = xs.limbs;
      const uint32_t *y = ys.limbs;
      uint32_t *r = rs.limbs;
      r[0] = SUBTRACT ? ptx::sub_cc(x[0], y[0]) : ptx::add_cc(x[0], y[0]);
    #pragma unroll
      for (unsigned i = 1; i < (CARRY_OUT ? TLC : TLC - 1); i++)
        r[i] = SUBTRACT ? ptx::subc_cc(x[i], y[i]) : ptx::addc_cc(x[i], y[i]);
      if (!CARRY_OUT) {
        r[TLC - 1] = SUBTRACT ? ptx::subc(x[TLC - 1], y[TLC - 1]) : ptx::addc(x[TLC - 1], y[TLC - 1]);
        return 0;
      }
      return SUBTRACT ? ptx::subc(0, 0) : ptx::addc(0, 0);
    }

    template <bool SUBTRACT, bool CARRY_OUT> 
    static constexpr DEVICE_INLINE uint32_t add_sub_limbs_device(const ff_wide_storage &xs, const ff_wide_storage &ys, ff_wide_storage &rs) {
      const uint32_t *x = xs.limbs;
      const uint32_t *y = ys.limbs;
      uint32_t *r = rs.limbs;
      r[0] = SUBTRACT ? ptx::sub_cc(x[0], y[0]) : ptx::add_cc(x[0], y[0]);
    #pragma unroll
      for (unsigned i = 1; i < (CARRY_OUT ? 2 * TLC : 2 * TLC - 1); i++)
        r[i] = SUBTRACT ? ptx::subc_cc(x[i], y[i]) : ptx::addc_cc(x[i], y[i]);
      if (!CARRY_OUT) {
        r[2 * TLC - 1] = SUBTRACT ? ptx::subc(x[2 * TLC - 1], y[2 * TLC - 1]) : ptx::addc(x[2 * TLC - 1], y[2 * TLC - 1]);
        return 0;
      }
      return SUBTRACT ? ptx::subc(0, 0) : ptx::addc(0, 0);
    }

    template <bool SUBTRACT, bool CARRY_OUT>
    static constexpr HOST_INLINE uint32_t add_sub_limbs_host(const ff_storage &xs, const ff_storage &ys, ff_storage &rs) {
      const uint32_t *x = xs.limbs;
      const uint32_t *y = ys.limbs;
      uint32_t *r = rs.limbs;
      uint32_t carry = 0;
      host_math::carry_chain<TLC, false, CARRY_OUT> chain;
      for (unsigned i = 0; i < TLC; i++)
        r[i] = SUBTRACT ? chain.sub(x[i], y[i], carry) : chain.add(x[i], y[i], carry);
      return CARRY_OUT ? carry : 0;
    }

    template <bool SUBTRACT, bool CARRY_OUT>
    static constexpr HOST_INLINE uint32_t add_sub_limbs_host(const ff_wide_storage &xs, const ff_wide_storage &ys, ff_wide_storage &rs) {
      const uint32_t *x = xs.limbs;
      const uint32_t *y = ys.limbs;
      uint32_t *r = rs.limbs;
      uint32_t carry = 0;
      host_math::carry_chain<2 * TLC, false, CARRY_OUT> chain;
      for (unsigned i = 0; i < 2 * TLC; i++)
        r[i] = SUBTRACT ? chain.sub(x[i], y[i], carry) : chain.add(x[i], y[i], carry);
      return CARRY_OUT ? carry : 0;
    }

    template <bool CARRY_OUT, typename T> static constexpr HOST_DEVICE_INLINE uint32_t add_limbs(const T &xs, const T &ys, T &rs) {
    #ifdef __CUDA_ARCH__
      return add_sub_limbs_device<false, CARRY_OUT>(xs, ys, rs);
    #else
      return add_sub_limbs_host<false, CARRY_OUT>(xs, ys, rs);
    #endif
    }

    template <bool CARRY_OUT, typename T> static constexpr HOST_DEVICE_INLINE uint32_t sub_limbs(const T &xs, const T &ys, T &rs) {
    #ifdef __CUDA_ARCH__
      return add_sub_limbs_device<true, CARRY_OUT>(xs, ys, rs);
    #else
      return add_sub_limbs_host<true, CARRY_OUT>(xs, ys, rs);
    #endif
    }

    static DEVICE_INLINE void mul_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    #pragma unroll
      for (size_t i = 0; i < n; i += 2) {
        acc[i] = ptx::mul_lo(a[i], bi);
        acc[i + 1] = ptx::mul_hi(a[i], bi);
      }
    }

    static DEVICE_INLINE void cmad_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) {
      acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
      acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);
  #pragma unroll
      for (size_t i = 2; i < n; i += 2) {
        acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
        acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
      }
    }

    static DEVICE_INLINE void mad_row(uint32_t *odd, uint32_t *even, const uint32_t *a, uint32_t bi, size_t n = TLC) {
      cmad_n(odd, a + 1, bi, n - 2);
      odd[n - 2] = ptx::madc_lo_cc(a[n - 1], bi, 0);
      odd[n - 1] = ptx::madc_hi(a[n - 1], bi, 0);
      cmad_n(even, a, bi, n);
      odd[n - 1] = ptx::addc(odd[n - 1], 0);
    }

    static DEVICE_INLINE void multiply_raw_device(const ff_storage &as, const ff_storage &bs, ff_wide_storage &rs) {
      const uint32_t *a = as.limbs;
      const uint32_t *b = bs.limbs;
      uint32_t *even = rs.limbs;
      __align__(8) uint32_t odd[2 * TLC - 2];
      mul_n(even, a, b[0]);
      mul_n(odd, a + 1, b[0]);
      mad_row(&even[2], &odd[0], a, b[1]);
      size_t i;
    #pragma unroll
      for (i = 2; i < TLC - 1; i += 2) {
        mad_row(&odd[i], &even[i], a, b[i]);
        mad_row(&even[i + 2], &odd[i], a, b[i + 1]);
      }
      // merge |even| and |odd|
      even[1] = ptx::add_cc(even[1], odd[0]);
      for (i = 1; i < 2 * TLC - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
      even[i + 1] = ptx::addc(even[i + 1], 0);
    }

    static HOST_INLINE void multiply_raw_host(const ff_storage &as, const ff_storage &bs, ff_wide_storage &rs) {
      const uint32_t *a = as.limbs;
      const uint32_t *b = bs.limbs;
      uint32_t *r = rs.limbs;
      for (unsigned i = 0; i < TLC; i++) {
        uint32_t carry = 0;
        for (unsigned j = 0; j < TLC; j++)
          r[j + i] = host_math::madc_cc(a[j], b[i], r[j + i], carry);
        r[TLC + i] = carry;
      }
    }

    static HOST_DEVICE_INLINE void multiply_raw(const ff_storage &as, const ff_storage &bs, ff_wide_storage &rs) {
    #ifdef __CUDA_ARCH__
      return multiply_raw_device(as, bs, rs);
    #else
      return multiply_raw_host(as, bs, rs);
    #endif
    }

  public:
    ff_storage limbs_storage;

    HOST_DEVICE_INLINE uint32_t* export_limbs() {
       return (uint32_t *)limbs_storage.limbs;
    }

    HOST_DEVICE_INLINE unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width) {
      const uint32_t limb_lsb_idx = (digit_num*digit_width) / 32;
      const uint32_t shift_bits = (digit_num*digit_width) % 32;
      unsigned rv = limbs_storage.limbs[limb_lsb_idx] >> shift_bits;
      // printf("get_scalar_func digit %u rv %u\n",digit_num,rv);
      // if (shift_bits + digit_width > 32) {
      if ((shift_bits + digit_width > 32) && (limb_lsb_idx+1 < TLC)) {
        rv += limbs_storage.limbs[limb_lsb_idx + 1] << (32 - shift_bits);
      }
      rv &= ((1 << digit_width) - 1);
      return rv;
    }

    static HOST_INLINE Field rand_host() {
      std::random_device rd;
      std::mt19937_64 generator(rd());
      std::uniform_int_distribution<unsigned> distribution;
      Field value{};
      for (unsigned i = 0; i < TLC; i++)
        value.limbs_storage.limbs[i] = distribution(generator);
      while (lt(modulus(), value))
        value = value - modulus();
      return value;
    }

    static constexpr DEVICE_INLINE bool eq(const Field &xs, const Field &ys) {
      const uint32_t *x = xs.limbs_storage.limbs;
      const uint32_t *y = ys.limbs_storage.limbs;
      uint32_t limbs_or = x[0] ^ y[0];
  #pragma unroll
      for (unsigned i = 1; i < TLC; i++)
        limbs_or |= x[i] ^ y[i];
      return limbs_or == 0;
    }

    template <unsigned REDUCTION_SIZE = 1> static constexpr HOST_DEVICE_INLINE Field reduce(const Field &xs) {
      if (REDUCTION_SIZE == 0)
        return xs;
      const ff_storage modulus = get_modulus<REDUCTION_SIZE>();
      Field rs = {};
      return sub_limbs<true>(xs.limbs_storage, modulus, rs.limbs_storage) ? xs : rs;
    }

    friend std::ostream& operator<<(std::ostream& os, const Field& xs) {
      os << "{";
      for (int i = 0; i < TLC; i++)
        os << xs.limbs_storage.limbs[i] << ", ";
      os << "}";
      return os;
    }

    friend HOST_DEVICE_INLINE Field operator+(Field xs, const Field& ys) {   
      Field rs = {};
      add_limbs<false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      return reduce<1>(rs);
    }

    friend HOST_DEVICE_INLINE Field operator-(Field xs, const Field& ys) {   
      Field rs = {};
      uint32_t carry = sub_limbs<true>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      if (carry == 0)
        return rs;
      const ff_storage modulus = get_modulus<1>();
      add_limbs<false>(rs.limbs_storage, modulus, rs.limbs_storage);
      return rs;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE wide mul_wide(const Field& xs, const Field& ys) {
      wide rs = {};
      multiply_raw(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      return rs;
    }

    friend HOST_DEVICE_INLINE Field operator*(const Field& xs, const Field& ys) {
      wide xy = mul_wide(xs, ys);
      Field xy_hi = xy.get_higher_with_slack();
      wide l = {};
      multiply_raw(xy_hi.limbs_storage, get_m(), l.limbs_storage);
      Field l_hi = l.get_higher_with_slack();
      wide lp = {};
      multiply_raw(l_hi.limbs_storage, get_modulus(), lp.limbs_storage);
      wide r_wide = xy - lp;
      wide r_wide_reduced = {};
      uint32_t reduced = sub_limbs<true>(r_wide.limbs_storage, modulus_wide(), r_wide_reduced.limbs_storage);
      r_wide = reduced ? r_wide : r_wide_reduced;
      Field r = r_wide.get_lower();
      return reduce<1>(r);
    }

    friend HOST_DEVICE_INLINE bool operator==(const Field& xs, const Field& ys) {
    #ifdef __CUDA_ARCH__
      const uint32_t *x = xs.limbs_storage.limbs;
      const uint32_t *y = ys.limbs_storage.limbs;
      uint32_t limbs_or = x[0] ^ y[0];
  #pragma unroll
      for (unsigned i = 1; i < TLC; i++)
        limbs_or |= x[i] ^ y[i];
      return limbs_or == 0;
    #else
      for (unsigned i = 0; i < TLC; i++)
      if (xs.limbs_storage.limbs[i] != ys.limbs_storage.limbs[i])
        return false;
      return true;
    #endif
    }

    template <unsigned REDUCTION_SIZE = 1>
    static constexpr DEVICE_INLINE Field mul(const unsigned scalar, const Field &xs) {
      Field rs = {};
      Field temp = xs;
      unsigned l = scalar;
      bool is_zero = true;
  #pragma unroll
      for (unsigned i = 0; i < 32; i++) {
        if (l & 1) {
          rs = is_zero ? temp : (rs + temp);
          is_zero = false;
        }
        l >>= 1;
        if (l == 0)
          break;
        temp = temp + temp;
      }
      return rs;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr DEVICE_INLINE wide sqr_wide(const Field& xs) {
      // TODO: change to a more efficient squaring
      return mul_wide<MODULUS_MULTIPLE>(xs, xs);
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr DEVICE_INLINE Field sqr(const Field& xs) {
      // TODO: change to a more efficient squaring
      return xs * xs;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr DEVICE_INLINE Field neg(const Field& xs) {
      const ff_storage modulus = get_modulus<MODULUS_MULTIPLE>();
      Field rs = {};
      sub_limbs<false>(modulus, xs.limbs_storage, rs.limbs_storage);
      return rs;
    }

    template <unsigned MODULUS_MULTIPLE = 1> 
    static constexpr DEVICE_INLINE Field div2(const Field &xs) {
      const uint32_t *x = xs.limbs_storage.limbs;
      Field rs = {};
      uint32_t *r = rs.limbs_storage.limbs;
  #pragma unroll
      for (unsigned i = 0; i < TLC - 1; i++)
        r[i] = __funnelshift_rc(x[i], x[i + 1], 1);
      r[TLC - 1] = x[TLC - 1] >> 1;
      return reduce<MODULUS_MULTIPLE>(rs);
    }

    static constexpr HOST_DEVICE_INLINE bool lt(const Field &xs, const Field &ys) {
      ff_storage dummy = {};
      uint32_t carry = sub_limbs<true>(xs.limbs_storage, ys.limbs_storage, dummy);
      return carry;
    }

    static constexpr HOST_DEVICE_INLINE bool is_odd(const Field &xs) { 
      return xs.limbs_storage.limbs[0] & 1;
    }

    static constexpr HOST_DEVICE_INLINE bool is_even(const Field &xs) { 
      return ~xs.limbs_storage.limbs[0] & 1;
    }

    // inverse assumes that xs is nonzero
    static constexpr DEVICE_INLINE Field inverse(const Field& xs) {
      constexpr Field one = Field { CONFIG::one };
      constexpr ff_storage modulus = CONFIG::modulus;
      Field u = xs;
      Field v = Field { modulus };
      Field b = one;
      Field c = {};
      while (!(u == one) && !(v == one)) {
        while (is_even(u)) {
          u = div2(u);
          if (is_odd(b))
            add_limbs<false>(b.limbs_storage, modulus, b.limbs_storage);
          b = div2(b);
        }
        while (is_even(v)) {
          v = div2(v);
          if (is_odd(c))
            add_limbs<false>(c.limbs_storage, modulus, c.limbs_storage);
          c = div2(c);
        }
        if (lt(v, u)) {
          u = u - v;
          b = b - c;
        } else {
          v = v - u;
          c = c - b;
        }
      }
      return (u == one) ?  b : c;
    }
};
