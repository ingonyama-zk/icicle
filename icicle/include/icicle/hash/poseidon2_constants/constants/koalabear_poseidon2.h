#pragma once
#ifndef KOALABEAR_POSEIDON2_H
  #define KOALABEAR_POSEIDON2_H

  #include <string>

namespace poseidon2_constants_koalabear {

  /**
   * This inner namespace contains constants for running Poseidon2.
   * The number in the name corresponds to the arity of hash function
   */

  int full_rounds_2 = 12;
  int half_full_rounds_2 = 6;
  int partial_rounds_2 = 34;
  int alpha_2 = 3;

  static const std::string rounds_constants_2[] = {
    "0x524e0068", "0x152e723a", "0x200c7780", "0x4610bd42", "0x7e00ae51", "0x2d05586a", "0x7577d6f2", "0x54d9fd72",
    "0x45a66853", "0x6afeb322", "0x536d8281", "0x667c7cd9", "0x3135fdc",  "0x3d813a9b", "0x2d72c4b8", "0x4d3cdec",
    "0x41a38f00", "0x3be6d258", "0x441a8e98", "0x1d53d9f2", "0x4f8b2027", "0x60665a1b", "0x4f4dbc45", "0x498815f8",
    "0x2d4de954", "0x36398694", "0x2a50d97c", "0x63ecbbf2", "0x473ecf92", "0x1fe4ccaa", "0x4d305fa5", "0x23e55a39",
    "0x1e742c24", "0x344540b0", "0x11955354", "0x2e829f60", "0xe7f49b9",  "0x2b926a59", "0x3bbbdbfe", "0x173c8844",
    "0x75aa4bca", "0x7494872d", "0x435e003d", "0x6c70b66c", "0x10eb546c", "0x67247579", "0x459bbbc7", "0x1034f4e1",
    "0x5caae541", "0x7d30b81d", "0x4c143998", "0x4920186",  "0x5db943cf", "0x1d5c372a", "0x5cc66ffc", "0x241e2335",
    "0x1248e87f", "0x53ac47b0"};

  static const std::string mds_matrix_2[] = {"0x2", "0x1", "0x1", "0x2"};

  static const std::string partial_matrix_diagonal_2[] = {"0x2", "0x3"};

  int full_rounds_3 = 12;
  int half_full_rounds_3 = 6;
  int partial_rounds_3 = 24;
  int alpha_3 = 3;

  static const std::string rounds_constants_3[] = {
    "0x5b935d11", "0x33574cd8", "0x44c92c",   "0x1dbdf45c", "0x5f986fee", "0x4472e296", "0x26eded79", "0x64cbcd54",
    "0x5618d75f", "0x14a08952", "0x3af6f35c", "0x6a0f4e7f", "0x149f6b30", "0xbf3aee4",  "0x44396c88", "0x2e5f24fa",
    "0x6d71ba35", "0x2b5dabe0", "0x4dc5592a", "0x6708fad4", "0x75c41756", "0x11baba7e", "0x1cb3e5d1", "0x6eac5322",
    "0x274fdfff", "0x1f5645dc", "0x3d9d3d22", "0x34750734", "0x3f7ac03b", "0x1415fb",   "0x3d7f1885", "0x7a7a5434",
    "0x2332f922", "0x5c3124fe", "0x7c113e92", "0x372e6da1", "0x62f13aa3", "0x2f97638b", "0x20c1b3a3", "0x7375a644",
    "0x7a506ad9", "0x14148a14", "0x76165fc2", "0x3583b5a2", "0x2f0c0cf7", "0x7d4f249e", "0x1d3c73b2", "0xfe85624",
    "0xdf73505",  "0x67ede608", "0x546d5bbb", "0x4032e948", "0x4f54d61f", "0x58e7001",  "0x67c71091", "0x632cbf64",
    "0x5bc24a8b", "0x74fe3f8e", "0x1c9597ff", "0x36b8f459"};

  static const std::string mds_matrix_3[] = {"0x2", "0x1", "0x1", "0x1", "0x2", "0x1", "0x1", "0x1", "0x2"};

  static const std::string partial_matrix_diagonal_3[] = {"0x2", "0x2", "0x3"};

  int full_rounds_4 = 8;
  int half_full_rounds_4 = 4;
  int partial_rounds_4 = 27;
  int alpha_4 = 3;

  static const std::string rounds_constants_4[] = {
    "0x7b0bf911", "0x332721d5", "0x1f222d9d", "0x7d3124b9", "0x73e502e8", "0x47b31ace", "0x5966f4b7", "0x1348b683",
    "0x75993259", "0x69dceb5a", "0x26afde90", "0x58f43de8", "0x24c3f358", "0x22602505", "0x6fe2256a", "0x7c5b391d",
    "0x18d5f7b8", "0x133efad5", "0x27c3c3e9", "0xd5f1384",  "0x424021d7", "0x55dfc4d4", "0x55141da1", "0x41da664c",
    "0x7e12f98b", "0x6760b6da", "0x14f7bd5e", "0x6af5df52", "0x7e502223", "0x25adbaf0", "0xd8515e",   "0x1254fc06",
    "0x773497bc", "0x790d5d9d", "0x77c56a40", "0x3940c586", "0x6d568469", "0x10a782bc", "0x49b3928b", "0x632c2c74",
    "0x68ba1c5f", "0x435ede47", "0x6d3d6c06", "0x61a778df", "0x58508fc7", "0x1b38503e", "0xc6ad8a4",  "0x30884220",
    "0x1c134f45", "0x72234f4e", "0x8d4a3c8",  "0x7d9daea9", "0x382d17c2", "0x6e755917", "0x442a48e9", "0x19e15954",
    "0x4f8ca130", "0x1dafcddc", "0x1d53f8af"};

  static const std::string mds_matrix_4[] = {"0x5", "0x7", "0x1", "0x3", "0x4", "0x6", "0x1", "0x1",
                                             "0x1", "0x3", "0x5", "0x7", "0x1", "0x1", "0x4", "0x6"};

  static const std::string partial_matrix_diagonal_4[] = {"0x7432c18f", "0x3f1107e2", "0x455ba830", "0x7ca1b7b6"};

  int full_rounds_8 = 8;
  int half_full_rounds_8 = 4;
  int partial_rounds_8 = 19;
  int alpha_8 = 3;

  static const std::string rounds_constants_8[] = {
    "0x3dacc911", "0x4f507ed6", "0x6e1a1622", "0x8d361ed",  "0x73a072ee", "0x12a2edfe", "0x272f83e8", "0x6527863d",
    "0x6af43e6e", "0x6da9583b", "0x1937a6ea", "0x2ee651fa", "0x6978a4c4", "0x54a40fa0", "0x10a0af4b", "0x4fed28e0",
    "0x326113c1", "0x594e0ac5", "0x3d783053", "0x15a20f6",  "0x4b9c1618", "0x4f813987", "0x3c411634", "0x199d128e",
    "0x16efc1af", "0x4fe64831", "0x14b8120d", "0x1a343491", "0x53185364", "0x38b7dbb9", "0x5daa5e18", "0x69f3b149",
    "0x3f6c02d1", "0x48bbd183", "0x40ff4954", "0x660b0657", "0x1fee8396", "0x3ffcc2a6", "0x229cce1a", "0x174545c9",
    "0x499dc30c", "0x3ad10faf", "0x7ca44dfb", "0x50bc23da", "0x2b395809", "0x241c4da7", "0x40e82068", "0x6095140a",
    "0x43908c74", "0x4b9a35c6", "0x5fa912c9", "0x563eaa36", "0xc18bcd4",  "0x28a25543", "0x61515541", "0x597d6eb6",
    "0x5940913a", "0x3228268e", "0x425e053d", "0x77cc599a", "0x2b369bae", "0xf934464",  "0x3433af",   "0x4b36cae",
    "0x31b0639a", "0x3b94118c", "0x27cbee39", "0x4f06ca49", "0x364b19c7", "0x5524623b", "0x6d66f7d4", "0x77044492",
    "0x5c85a1",   "0x10d86358", "0x45aab3e9", "0x68583c99", "0x1ba5a3d3", "0x305f0db1", "0x27bff8a",  "0x1de46250",
    "0xe261c27",  "0x131ee530", "0x58950fd5"};

  static const std::string mds_matrix_8[] = {
    "0xa", "0xe", "0x2", "0x6", "0x5", "0x7", "0x1", "0x3", "0x8", "0xc", "0x2", "0x2", "0x4", "0x6", "0x1", "0x1",
    "0x2", "0x6", "0xa", "0xe", "0x1", "0x3", "0x5", "0x7", "0x2", "0x2", "0x8", "0xc", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0xa", "0xe", "0x2", "0x6", "0x4", "0x6", "0x1", "0x1", "0x8", "0xc", "0x2", "0x2",
    "0x1", "0x3", "0x5", "0x7", "0x2", "0x6", "0xa", "0xe", "0x1", "0x1", "0x4", "0x6", "0x2", "0x2", "0x8", "0xc"};

  static const std::string partial_matrix_diagonal_8[] = {"0x48018e49", "0x564044ae", "0x3710cdd5", "0x75b29",
                                                          "0x6b5708e9", "0x4d78c3b2", "0xa4ef4f6",  "0x7be770ee"};

  int full_rounds_12 = 8;
  int half_full_rounds_12 = 4;
  int partial_rounds_12 = 20;
  int alpha_12 = 3;

  static const std::string rounds_constants_12[] = {
    "0x552714b7", "0x5764a7d1", "0x46e7b5a1", "0x1901a8e2", "0x75c9d5b6", "0x39acd1fb", "0x254a0c81", "0x2f044d3e",
    "0x582feb3f", "0x547ec090", "0x30a2f56b", "0x5179c161", "0x25535ca5", "0x3932decf", "0x95b4aa3",  "0xbac5b3e",
    "0x5715482c", "0x1add16a5", "0x7d91733a", "0x770f9ec4", "0x2a8276ba", "0x20496a82", "0x3e6941c2", "0x5b83ae95",
    "0x79d393b7", "0x63f4aa4d", "0x18d0ccb",  "0x5f12ac40", "0x442661a6", "0x20e8a612", "0x1114090f", "0x56a8aeb8",
    "0x579fab79", "0x36c9b5a",  "0x3bff50e9", "0x1b2e7a16", "0x4d369049", "0x5a92728",  "0xa762f3",   "0x808e73e",
    "0x582381dd", "0x60103cf8", "0x6bd15c04", "0x57232471", "0x171f6b8d", "0x316eea92", "0x4c682389", "0x772eec85",
    "0x597e84c9", "0x7821b1d3", "0x45e27638", "0x7873968f", "0x78345401", "0x53ae8b39", "0xddea5b7",  "0x46f28939",
    "0x344857e0", "0x245cb84a", "0x6315963c", "0x42581dc4", "0x1e8799c0", "0x7b3887a1", "0x495957a5", "0x4d97c6c9",
    "0x999b968",  "0x2f30aa69", "0x9f93478",  "0xf6cfbd6",  "0x1cff6baa", "0x7c85c852", "0x76a6c4b0", "0x39f7f2c6",
    "0x6d224f31", "0x47f01bb1", "0x7cd45ac8", "0x7359ac50", "0x797b8722", "0x3e9d72b1", "0x60ac8b29", "0x38d3fb6c",
    "0x2dc5121d", "0x16244ebe", "0x4a20b9f6", "0x19424208", "0x349c6cbe", "0x4a3f03bb", "0x186cd89a", "0x1ed4e582",
    "0x28a362aa", "0x10479773", "0x394ba17c", "0x41c6d39f", "0x8773ced",  "0x32b32ae6", "0x166b459c", "0x4ecaba93",
    "0x3200ca87", "0x1f3f426",  "0x213ff084", "0x66313b58", "0x2ff10fce", "0x2e5c5b49", "0x1bb4f054", "0x1f7feefc",
    "0xf4bbf58",  "0xe2437d3",  "0x3b903d08", "0x6a7b9783", "0x207e7895", "0x12283a5d", "0xfe14d6d",  "0x57df2c15",
    "0x5b388f00", "0x1d90b129", "0x4d7845b8", "0x2ed78094"};

  static const std::string mds_matrix_12[] = {
    "0xa", "0xe", "0x2", "0x6", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x8", "0xc", "0x2", "0x2",
    "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x2", "0x6", "0xa", "0xe", "0x1", "0x3", "0x5", "0x7",
    "0x1", "0x3", "0x5", "0x7", "0x2", "0x2", "0x8", "0xc", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0xa", "0xe", "0x2", "0x6", "0x5", "0x7", "0x1", "0x3", "0x4", "0x6", "0x1", "0x1",
    "0x8", "0xc", "0x2", "0x2", "0x4", "0x6", "0x1", "0x1", "0x1", "0x3", "0x5", "0x7", "0x2", "0x6", "0xa", "0xe",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x1", "0x4", "0x6", "0x2", "0x2", "0x8", "0xc", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0xa", "0xe", "0x2", "0x6", "0x4", "0x6", "0x1", "0x1",
    "0x4", "0x6", "0x1", "0x1", "0x8", "0xc", "0x2", "0x2", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7",
    "0x2", "0x6", "0xa", "0xe", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x2", "0x2", "0x8", "0xc"};

  static const std::string partial_matrix_diagonal_12[] = {"0x1e9114a6", "0x701d8b0d", "0x5273af9b", "0x387e3f71",
                                                           "0x4176d043", "0x456235f1", "0x79b7c7b5", "0x3d80c80f",
                                                           "0x156e5d78", "0x42dddc6c", "0x611b754e", "0x10dc67fe"};

  int full_rounds_16 = 8;
  int half_full_rounds_16 = 4;
  int partial_rounds_16 = 20;
  int alpha_16 = 3;

  static const std::string rounds_constants_16[] = {
    "0x7ee56a48", "0x11367045", "0x12e41941", "0x7ebbc12b", "0x1970b7d5", "0x662b60e8", "0x3e4990c6", "0x679f91f5",
    "0x350813bb", "0x874ad4",   "0x28a0081a", "0x18fa5872", "0x5f25b071", "0x5e5d5998", "0x5e6fd3e7", "0x5b2e2660",
    "0x6f1837bf", "0x3fe6182b", "0x1edd7ac5", "0x57470d00", "0x43d486d5", "0x1982c70f", "0xea53af9",  "0x61d6165b",
    "0x51639c00", "0x2dec352c", "0x2950e531", "0x2d2cb947", "0x8256cef",  "0x1a0109f6", "0x1f51faf3", "0x5cef1c62",
    "0x3d65e50e", "0x33d91626", "0x133d5a1e", "0xff49b0d",  "0x38900cd1", "0x2c22cc3f", "0x28852bb2", "0x6c65a02",
    "0x7b2cf7bc", "0x68016e1a", "0x15e16bc0", "0x5248149a", "0x6dd212a0", "0x18d6830a", "0x5001be82", "0x64dac34e",
    "0x5902b287", "0x426583a0", "0xc921632",  "0x3fe028a5", "0x245f8e49", "0x43bb297e", "0x7873dbd9", "0x3cc987df",
    "0x286bb4ce", "0x640a8dcd", "0x512a8e36", "0x3a4cf55",  "0x481837a2", "0x3d6da84",  "0x73726ac7", "0x760e7fdf",
    "0x54dfeb5d", "0x7d40afd6", "0x722cb316", "0x106a4573", "0x45a7ccdb", "0x44061375", "0x154077a5", "0x45744faa",
    "0x4eb5e5ee", "0x3794e83f", "0x47c7093c", "0x5694903c", "0x69cb6299", "0x373df84c", "0x46a0df58", "0x46b8758a",
    "0x3241ebcb", "0xb09d233",  "0x1af42357", "0x1e66cec2", "0x43e7dc24", "0x259a5d61", "0x27e85a3b", "0x1b9133fa",
    "0x343e5628", "0x485cd4c2", "0x16e269f5", "0x165b60c6", "0x25f683d9", "0x124f81f9", "0x174331f9", "0x77344dc5",
    "0x5a821dba", "0x5fc4177f", "0x54153bf5", "0x5e3f1194", "0x3bdbf191", "0x88c84a3",  "0x68256c9b", "0x3c90bbc6",
    "0x6846166a", "0x3f4238d",  "0x463335fb", "0x5e3d3551", "0x6e59ae6f", "0x32d06cc0", "0x596293f3", "0x6c87edb2",
    "0x8fc60b5",  "0x34bcca80", "0x24f007f3", "0x62731c6f", "0x1e1db6c6", "0xca409bb",  "0x585c1e78", "0x56e94edc",
    "0x16d22734", "0x18e11467", "0x7b2c3730", "0x770075e4", "0x35d1b18c", "0x22be3db5", "0x4fb1fbb7", "0x477cb3ed",
    "0x7d5311c6", "0x5b62ae7d", "0x559c5fa8", "0x77f15048", "0x3211570b", "0x490fef6a", "0x77ec311f", "0x2247171b",
    "0x4e0ac711", "0x2edf69c9", "0x3b5a8850", "0x65809421", "0x5619b4aa", "0x362019a7", "0x6bf9d4ed", "0x5b413dff",
    "0x617e181e", "0x5e7ab57b", "0x33ad7833", "0x3466c7ca"};

  static const std::string mds_matrix_16[] = {
    "0xa", "0xe", "0x2", "0x6", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3",
    "0x8", "0xc", "0x2", "0x2", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1",
    "0x2", "0x6", "0xa", "0xe", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7",
    "0x2", "0x2", "0x8", "0xc", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0xa", "0xe", "0x2", "0x6", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3",
    "0x4", "0x6", "0x1", "0x1", "0x8", "0xc", "0x2", "0x2", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1",
    "0x1", "0x3", "0x5", "0x7", "0x2", "0x6", "0xa", "0xe", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7",
    "0x1", "0x1", "0x4", "0x6", "0x2", "0x2", "0x8", "0xc", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0xa", "0xe", "0x2", "0x6", "0x5", "0x7", "0x1", "0x3",
    "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x8", "0xc", "0x2", "0x2", "0x4", "0x6", "0x1", "0x1",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x2", "0x6", "0xa", "0xe", "0x1", "0x3", "0x5", "0x7",
    "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x2", "0x2", "0x8", "0xc", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0xa", "0xe", "0x2", "0x6",
    "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x8", "0xc", "0x2", "0x2",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x2", "0x6", "0xa", "0xe",
    "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x2", "0x2", "0x8", "0xc"};

  static const std::string partial_matrix_diagonal_16[] = {
    "0x1c9a12f2", "0x3f6fd40d", "0xe1d4ec4",  "0x1361c64d", "0x9a8f470",  "0x3d23a40", "0x109ad290", "0x28c2fb88",
    "0x3b6498f2", "0x74d8be57", "0x6a4277d2", "0x18c2b3d4", "0x6252c30c", "0x7cc2560", "0x209fe15b", "0x52a55fac"};

  int full_rounds_20 = 8;
  int half_full_rounds_20 = 4;
  int partial_rounds_20 = 20;
  int alpha_20 = 3;

  static const std::string rounds_constants_20[] = {
    "0x5c273f69", "0x4301cfff", "0x4e6dea8d", "0x6c44b6f1", "0x2db6d758", "0x57afb62b", "0x5435d2",   "0x3ff7b219",
    "0x1e4a0b66", "0x37c003ce", "0x6e62692e", "0x6786a213", "0x114692d8", "0x54aa6fd5", "0x760d0bab", "0x630055a4",
    "0x3f50278c", "0xbe2c406",  "0x63074431", "0x38cf95ee", "0x4ca3c5e8", "0x133dc7bb", "0x58378184", "0x6e390e07",
    "0x3abe418",  "0x33ed3fa7", "0x75b6da92", "0x1d736f0a", "0x3c4ed487", "0x737713ea", "0xd810703",  "0x24bd333d",
    "0x3bc54f7b", "0x129b8959", "0x347dfd77", "0x2851abf5", "0x73383763", "0x122d3b7e", "0x4681919b", "0x2deb614c",
    "0x4e8a4e03", "0x682b187a", "0x30382c8e", "0x5e764335", "0x78310686", "0x48787b39", "0x35f4e63d", "0x54cc9025",
    "0x6e2a9a4",  "0x7bb527a8", "0x681da524", "0x23f34a6",  "0xcf7185b",  "0x2992faa3", "0x6de5dada", "0x238b3b40",
    "0x80a9b43",  "0x5c3432b5", "0x4ba816af", "0x18a3482f", "0x14eae07f", "0x652f39ad", "0x34028b4b", "0x5a7c92f9",
    "0x75f0c2ae", "0xb0b1846",  "0x11c7b2f6", "0x25a1141d", "0x22b403f9", "0x736411b7", "0x155b8c52", "0xb237337",
    "0x4e48f6e7", "0x539441d8", "0xf53a696",  "0x4969380c", "0x6357944d", "0x49c629cc", "0x74f70a6d", "0x32dc4c0",
    "0x768dddad", "0x12b409a2", "0x1ea4e7f2", "0x584aa4be", "0x14eabcaa", "0x2c667f1f", "0x25058f19", "0x6ccf9dd",
    "0x796ec99b", "0x60aceafc", "0x4b48b64a", "0x12fa5acf", "0x2c0a4368", "0x2cb7eb6f", "0x6ed5271d", "0x4150764c",
    "0x8eb3562",  "0x14f09122", "0x3b3b3f6f", "0x46134a88", "0x61888bb2", "0x4ec281cf", "0xa1d7b72",  "0x9b86596",
    "0x4ab08cfb", "0x51303468", "0x54723c8e", "0x18752913", "0x68414af8", "0x1cdc7564", "0x1d820128", "0x7424efdc",
    "0x68398c35", "0x20cc4049", "0x5efa066d", "0xf37cca7",  "0x2da3871e", "0x76d972bc", "0x515a4049", "0x7b4da22a",
    "0x608450a4", "0x18dd8e6f", "0x32ee591",  "0x507dd05d", "0x293e62e3", "0x2bf9ab0a", "0x5cdf7ef",  "0x509d6fea",
    "0x5ab28ecc", "0x57f08d90", "0x4d0348",   "0x360e8ecb", "0x68132b70", "0x2b8da1c3", "0x6cc92608", "0x27f1ade0",
    "0x30cb0d7d", "0x6a50f6ac", "0xa09a9bd",  "0xa582f31",  "0x71865171", "0x6fba15d3", "0x7edbb0d5", "0x606f3136",
    "0x392526e6", "0x7a869e9f", "0x48d636d9", "0x47df6af3", "0x280ff722", "0x1a00e805", "0x3b8a9764", "0x6add78e2",
    "0x345386ab", "0x6cd0d9fb", "0x709d87e1", "0x36659109", "0x5e6f705a", "0x38cffe92", "0x6481b44d", "0x78d557a3",
    "0x4a101d6d", "0x5cff3c8b", "0x46c154de", "0x3de4c445", "0x49c639e0", "0x344dce2f", "0x13c0fedc", "0x60e58e31",
    "0xf124aea",  "0x58e92cd4", "0x79c8073c", "0x194bf6c7", "0x487b73b5", "0x33a2861e", "0x32276cdb", "0x34d7049e",
    "0x67fd6703", "0x50741966", "0x617c304f", "0x34aca14d"};

  static const std::string mds_matrix_20[] = {
    "0xa", "0xe", "0x2", "0x6", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3",
    "0x5", "0x7", "0x1", "0x3", "0x8", "0xc", "0x2", "0x2", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1",
    "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x2", "0x6", "0xa", "0xe", "0x1", "0x3", "0x5", "0x7",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x2", "0x2", "0x8", "0xc",
    "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0xa", "0xe", "0x2", "0x6", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3",
    "0x5", "0x7", "0x1", "0x3", "0x4", "0x6", "0x1", "0x1", "0x8", "0xc", "0x2", "0x2", "0x4", "0x6", "0x1", "0x1",
    "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x1", "0x3", "0x5", "0x7", "0x2", "0x6", "0xa", "0xe",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x1", "0x4", "0x6",
    "0x2", "0x2", "0x8", "0xc", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0xa", "0xe", "0x2", "0x6", "0x5", "0x7", "0x1", "0x3",
    "0x5", "0x7", "0x1", "0x3", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x8", "0xc", "0x2", "0x2",
    "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7",
    "0x2", "0x6", "0xa", "0xe", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x1", "0x4", "0x6",
    "0x1", "0x1", "0x4", "0x6", "0x2", "0x2", "0x8", "0xc", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0xa", "0xe", "0x2", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1",
    "0x8", "0xc", "0x2", "0x2", "0x4", "0x6", "0x1", "0x1", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7",
    "0x1", "0x3", "0x5", "0x7", "0x2", "0x6", "0xa", "0xe", "0x1", "0x3", "0x5", "0x7", "0x1", "0x1", "0x4", "0x6",
    "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x2", "0x2", "0x8", "0xc", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3",
    "0xa", "0xe", "0x2", "0x6", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1",
    "0x4", "0x6", "0x1", "0x1", "0x8", "0xc", "0x2", "0x2", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x2", "0x6", "0xa", "0xe", "0x1", "0x1", "0x4", "0x6",
    "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x2", "0x2", "0x8", "0xc"};

  static const std::string partial_matrix_diagonal_20[] = {
    "0x226d9fa6", "0x63dad412", "0x7e30379a", "0x1eccefca", "0x1788f77e", "0x2221a31b", "0x19c59031",
    "0x71fb4f36", "0x1efc2cbd", "0xd4900da",  "0x23326df1", "0x6617c287", "0x30bb41d3", "0x343f2eae",
    "0x77470d57", "0x5090f557", "0x3f7127bf", "0x2412d711", "0x753cd22",  "0x7847653f"};

  int full_rounds_24 = 8;
  int half_full_rounds_24 = 4;
  int partial_rounds_24 = 23;
  int alpha_24 = 3;

  static const std::string rounds_constants_24[] = {
    "0x1d0939dc", "0x6d050f8d", "0x628058ad", "0x2681385d", "0x3e3c62be", "0x32cfad8",  "0x5a91ba3c", "0x15a56e6",
    "0x696b889c", "0xdbcd780",  "0x5881b5c9", "0x2a076f2e", "0x55393055", "0x6513a085", "0x547ac78f", "0x4281c5b8",
    "0x3e7a3f6c", "0x34562c19", "0x2c04e679", "0xed78234",  "0x5f7a1aa9", "0x177640e",  "0xea4f8d1",  "0x15be7692",
    "0x6eafdd62", "0x71a572c6", "0x72416f0a", "0x31ce1ad3", "0x2136a0cf", "0x1507c0eb", "0x1eb6e07a", "0x3a0ccf7b",
    "0x38e4bf31", "0x44128286", "0x6b05e976", "0x244a9b92", "0x6e4b32a8", "0x78ee2496", "0x4761115b", "0x3d3a7077",
    "0x75d3c670", "0x396a2475", "0x26dd00b4", "0x7df50f59", "0xcb922df",  "0x568b190",  "0x5bd3fcd6", "0x1351f58e",
    "0x52191b5f", "0x119171b8", "0x1e8bb727", "0x27d21f26", "0x36146613", "0x1ee817a2", "0x71abe84e", "0x44b88070",
    "0x5dc04410", "0x2aeaa2f6", "0x2b7bb311", "0x6906884d", "0x522e053",  "0xc45a214",  "0x1b016998", "0x479b1052",
    "0x3acc89be", "0x776021a",  "0x7a34a1f5", "0x70f87911", "0x2caf9d9e", "0x26aff1b",  "0x2c42468e", "0x67726b45",
    "0x9b6f53c",  "0x73d76589", "0x5793eeb0", "0x29e720f3", "0x75fc8bdf", "0x4c2fae0e", "0x20b41db3", "0x7e491510",
    "0x2cadef18", "0x57fc24d6", "0x4d1ade4a", "0x36bf8e3c", "0x3511b63c", "0x64d8476f", "0x732ba706", "0x46634978",
    "0x521c17c",  "0x5ee69212", "0x3559cba9", "0x2b33df89", "0x653538d6", "0x5fde8344", "0x4091605d", "0x2933bdde",
    "0x1395d4ca", "0x5dbac049", "0x51fc2727", "0x13407399", "0x39ac6953", "0x45e8726c", "0x75a7311c", "0x599f82c9",
    "0x702cf13b", "0x26b8955",  "0x44e09bbc", "0x2211207f", "0x5128b4e3", "0x591c41af", "0x674f5c68", "0x3981d0d3",
    "0x2d82f898", "0x707cd267", "0x3b4cca45", "0x2ad0dc3c", "0xcb79b37",  "0x23f2f4e8", "0x3de4e739", "0x7d232359",
    "0x389d82f9", "0x259b2e6c", "0x45a94def", "0xd497380",  "0x5b049135", "0x3c268399", "0x78feb2f9", "0x300a3eec",
    "0x505165bb", "0x20300973", "0x2327c081", "0x1a45a2f4", "0x5b32ea2e", "0x2d5d1a70", "0x53e613e",  "0x5433e39f",
    "0x495529f0", "0x1eaa1aa9", "0x578f572a", "0x698ede71", "0x5a0f9dba", "0x398a2e96", "0xc7b2925",  "0x2e6b9564",
    "0x26b00de",  "0x7644c1e9", "0x5c23d0bd", "0x3470b5ef", "0x6013cf3a", "0x48747288", "0x13b7a543", "0x3eaebd44",
    "0x4e60c",    "0x1e8363a2", "0x2343259a", "0x69da0c2a", "0x6e3e4c4",  "0x1095018e", "0xdeea348",  "0x1f4c5513",
    "0x4f9a3a98", "0x3179112b", "0x524abb1f", "0x21615ba2", "0x23ab4065", "0x1202a1d1", "0x21d25b83", "0x6ed17c2f",
    "0x391e6b09", "0x5e4ed894", "0x6a2f58f2", "0x5d980d70", "0x3fa48c5e", "0x1f6366f7", "0x63540f5f", "0x6a8235ed",
    "0x14c12a78", "0x6edde1c9", "0x58ce1c22", "0x718588bb", "0x334313ad", "0x7478dbc7", "0x647ad52f", "0x39e82049",
    "0x6fee146a", "0x82c2f24",  "0x1f093015", "0x30173c18", "0x53f70c0d", "0x6028ab0c", "0x2f47a1ee", "0x26a6780e",
    "0x3540bc83", "0x1812b49f", "0x5149c827", "0x631dd925", "0x1f2dea",   "0x7dc05194", "0x3789672e", "0x7cabf72e",
    "0x242dbe2f", "0xb07a51d",  "0x38653650", "0x50785c4e", "0x60e8a7e0", "0x7464338",  "0x3482d6e1", "0x8a69f1e",
    "0x3f2aff24", "0x5814c30d", "0x13fecab2", "0x61cb291a", "0x68c8226f", "0x5c757eea", "0x289b4e1e"};

  static const std::string mds_matrix_24[] = {
    "0xa", "0xe", "0x2", "0x6", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x8", "0xc", "0x2", "0x2", "0x4", "0x6", "0x1", "0x1",
    "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1",
    "0x2", "0x6", "0xa", "0xe", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x2", "0x2", "0x8", "0xc", "0x1", "0x1", "0x4", "0x6",
    "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0xa", "0xe", "0x2", "0x6", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x4", "0x6", "0x1", "0x1", "0x8", "0xc", "0x2", "0x2",
    "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1",
    "0x1", "0x3", "0x5", "0x7", "0x2", "0x6", "0xa", "0xe", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x1", "0x4", "0x6", "0x2", "0x2", "0x8", "0xc",
    "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0xa", "0xe", "0x2", "0x6", "0x5", "0x7", "0x1", "0x3",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1",
    "0x8", "0xc", "0x2", "0x2", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x2", "0x6", "0xa", "0xe", "0x1", "0x3", "0x5", "0x7",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x2", "0x2", "0x8", "0xc", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0xa", "0xe", "0x2", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1",
    "0x4", "0x6", "0x1", "0x1", "0x8", "0xc", "0x2", "0x2", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x2", "0x6", "0xa", "0xe",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x1", "0x1", "0x4", "0x6", "0x2", "0x2", "0x8", "0xc", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3",
    "0xa", "0xe", "0x2", "0x6", "0x5", "0x7", "0x1", "0x3", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1",
    "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x8", "0xc", "0x2", "0x2", "0x4", "0x6", "0x1", "0x1",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7",
    "0x2", "0x6", "0xa", "0xe", "0x1", "0x3", "0x5", "0x7", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x2", "0x2", "0x8", "0xc", "0x1", "0x1", "0x4", "0x6",
    "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3",
    "0x5", "0x7", "0x1", "0x3", "0xa", "0xe", "0x2", "0x6", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1",
    "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x8", "0xc", "0x2", "0x2",
    "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7", "0x1", "0x3", "0x5", "0x7",
    "0x1", "0x3", "0x5", "0x7", "0x2", "0x6", "0xa", "0xe", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6",
    "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x1", "0x1", "0x4", "0x6", "0x2", "0x2", "0x8", "0xc"};

  static const std::string partial_matrix_diagonal_24[] = {
    "0x731b17fd", "0x201359bd", "0x22bf6499", "0x610f1a29", "0x3c73aa45", "0x6a092599", "0x1c7cb703", "0x79533459",
    "0x7ef62d86", "0x5ab925ab", "0x67722ab1", "0x33ca4cff", "0x7f7dce",   "0xeeac41e",  "0x4724bea7", "0x45eaf64f",
    "0x21a6c90f", "0x94b4150",  "0xd942630",  "0x18712c30", "0x3a470338", "0x6eba7720", "0x487827c8", "0x77013a6d"};

} // namespace poseidon2_constants_koalabear
#endif