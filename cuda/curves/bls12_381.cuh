#pragma once

#include "../utils/storage.cuh"


struct fp_config {
  // field structure size = 8 * 32 bit
  static constexpr unsigned limbs_count = 8;
  // modulus = 52435875175126190479447740508185965837690552500527637822603658699938581184513
  static constexpr storage<limbs_count> modulus = {0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402, 0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753};
  // modulus*2 = 104871750350252380958895481016371931675381105001055275645207317399877162369026
  static constexpr storage<limbs_count> modulus_2 = {0x00000002, 0xfffffffe, 0xfffcb7fd, 0xa77b4805, 0x1343b00a, 0x6673b010, 0x533afa90, 0xe7db4ea6};
  static constexpr storage<limbs_count> modulus_4 = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};

  static constexpr storage<2 * limbs_count> modulus_wide = {0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402, 0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753,
                                                            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
  // modulus^2
  static constexpr storage<2*limbs_count> modulus_sqared = {0x00000001, 0xfffffffe, 0xfffcb7fe, 0xa77e9007, 0x1cdbb005, 0x698ae002, 0x5433f7b8, 0x48aa415e, 
                                                            0x4aa9c661, 0xc2611f6f, 0x59934a1d, 0x0e9593f9, 0xef2cc20f, 0x520c13db, 0xf4bc2778, 0x347f60f3};
  // 2*modulus^2
  static constexpr storage<2*limbs_count> modulus_sqared_2 = {0x00000002, 0xfffffffc, 0xfff96ffd, 0x4efd200f, 0x39b7600b, 0xd315c004, 0xa867ef70, 0x915482bc, 
                                                              0x95538cc2, 0x84c23ede, 0xb326943b, 0x1d2b27f2, 0xde59841e, 0xa41827b7, 0xe9784ef0, 0x68fec1e7};
  static constexpr unsigned modulus_bits_count = 255;
  // m = floor(2^(2*modulus_bits_count) / modulus)
  static constexpr storage<limbs_count> m = {0x830358e4, 0x509cde80, 0x2f92eb5c, 0xd9410fad, 0xc1f823b4, 0xe2d772d, 0x7fb78ddf, 0x8d54253b};
  static constexpr storage<limbs_count> number_of_twiddles = {0x00000004, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
  
  static constexpr storage<limbs_count> omega = {0xa5d36306, 0xe206da11, 0x378fbf96, 0x0ad1347b, 0xe0f8245f, 0xfc3e8acf, 0xa0f704f4, 0x564c0a11};
  static constexpr storage<limbs_count> omega_inv = {3629396834, 2518295853, 1679307267, 1346818424, 3118225798, 1256349690, 3322524792, 958081110};

  static constexpr storage<limbs_count> one = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};

  static constexpr storage<limbs_count> inv_2 = {2147483649,2147483647,2147429887,2849952257,80800770,429714436,2496577188,972477353};
  static constexpr storage<limbs_count> inv_4 = {1073741825,1073741823,1073661183,4274928386,121201155,644571654,1597382134,1458716030};
  static constexpr storage<limbs_count> inv_32 = {134217729, 134217727, 3892209951, 689944290, 3377776965, 4053797191, 2421199197, 1884174872};
  static constexpr storage<limbs_count> inv_256 = {16777217,16777215,4244528547,1315563102,26752557,3943079472,3597918154,1937357227};
  static constexpr storage<limbs_count> inv_512 = {8388609,8388607,4269694161,1360250160,94177049,2401254172,2148052617,1941155967};
  static constexpr storage<limbs_count> inv_4096 = {1048577,1074790399,3217972249,3546834984,2300657127,1589027946,3026903920,1944479864};
};

struct fq_config {
  // field structure size = 12 * 32 bit
  static constexpr unsigned limbs_count = 12;
  // modulus = 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
  static constexpr storage<limbs_count> modulus = {0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe, 0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84, 0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea};
  // modulus*2 = 8004819110443334786835579651471808313113765639878015770664116272248063300981675728885375258258031328075788545119574
  static constexpr storage<limbs_count> modulus_2 = {0xffff5556, 0x73fdffff, 0x62a7ffff, 0x3d57fffd, 0xed61ec48, 0xce61a541, 0xe70a257e, 0xc8ee9709, 0x869759ae, 0x96374f6c, 0x72ffcd34, 0x340223d4};
  // modulus*4 = 16009638220886669573671159302943616626227531279756031541328232544496126601963351457770750516516062656151577090239148
  static constexpr storage<limbs_count> modulus_4 = {0xfffeaaac, 0xe7fbffff, 0xc54ffffe, 0x7aaffffa, 0xdac3d890, 0x9cc34a83, 0xce144afd, 0x91dd2e13, 0xd2eb35d, 0x2c6e9ed9, 0xe5ff9a69, 0x680447a8};
  
  static constexpr storage<2*limbs_count> modulus_wide = {0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe, 0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84, 
                                                          0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                          0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};

  // modulus^2
  static constexpr storage<2*limbs_count> modulus_sqared = {0x1c718e39, 0x26aa0000, 0x76382eab, 0x7ced6b1d, 0x62113cfd, 0x162c3383, 0x3e71b743, 0x66bf91ed, 
                                                            0x7091a049, 0x292e85a8, 0x86185c7b, 0x1d68619c, 0x0978ef01, 0xf5314933, 0x16ddca6e, 0x50a62cfd, 
                                                            0x349e8bd0, 0x66e59e49, 0x0e7046b4, 0xe2dc90e5, 0xa22f25e9, 0x4bd278ea, 0xb8c35fc7, 0x02a437a4};
  // 2*modulus^2
  static constexpr storage<2*limbs_count> modulus_sqared_2 = {0x38e31c72, 0x4d540000, 0xec705d56, 0xf9dad63a, 0xc42279fa, 0x2c586706, 0x7ce36e86, 0xcd7f23da, 
                                                              0xe1234092, 0x525d0b50, 0x0c30b8f6, 0x3ad0c339, 0x12f1de02, 0xea629266, 0x2dbb94dd, 0xa14c59fa, 
                                                              0x693d17a0, 0xcdcb3c92, 0x1ce08d68, 0xc5b921ca, 0x445e4bd3, 0x97a4f1d5, 0x7186bf8e, 0x05486f49};
  // 4*modulus^2
  static constexpr storage<2*limbs_count> modulus_sqared_4 = {0x71c638e4, 0x9aa80000, 0xd8e0baac, 0xf3b5ac75, 0x8844f3f5, 0x58b0ce0d, 0xf9c6dd0c, 0x9afe47b4, 
                                                              0xc2468125, 0xa4ba16a1, 0x186171ec, 0x75a18672, 0x25e3bc04, 0xd4c524cc, 0x5b7729bb, 0x4298b3f4, 
                                                              0xd27a2f41, 0x9b967924, 0x39c11ad1, 0x8b724394, 0x88bc97a7, 0x2f49e3aa, 0xe30d7f1d, 0x0a90de92};
  static constexpr unsigned modulus_bits_count = 381;
  // m = floor(2^(2*modulus_bits_count) / modulus)
  static constexpr storage<limbs_count> m = {0xd59646e8, 0xec4f881f, 0x8163c701, 0x4e65c59e, 0x80a19de7, 0x2f7d1dc7, 0x7fda82a5, 0xa46e09d0, 0x331e9ae8, 0x38a0406c, 0xcf327917, 0x2760d74b};
  static constexpr storage<limbs_count> one = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
};

static constexpr unsigned weierstrass_b = 4;
