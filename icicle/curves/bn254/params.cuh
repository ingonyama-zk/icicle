#pragma once
#include "../../utils/storage.cuh"
#include <array>
namespace PARAMS_BN254{
  struct fp_config{
    static constexpr unsigned limbs_count = 8;
    static constexpr unsigned omegas_count = 32;
    
    static constexpr storage<limbs_count> modulus = {0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    static constexpr storage<limbs_count> modulus_2 = {0xe0000002, 0x87c3eb27, 0xf372e122, 0x5067d090, 0x0302b0ba, 0x70a08b6d, 0xc2634053, 0x60c89ce5};
    static constexpr storage<limbs_count> modulus_4 = {0xc0000004, 0x0f87d64f, 0xe6e5c245, 0xa0cfa121, 0x06056174, 0xe14116da, 0x84c680a6, 0xc19139cb};
    static constexpr storage<2*limbs_count> modulus_wide = {0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<2*limbs_count> modulus_sqared = {0xe0000001, 0x08c3eb27, 0xdcb34000, 0xc7f26223, 0x68c9bb7f, 0xffe9a62c, 0xe821ddb0, 0xa6ce1975, 0x47b62fe7, 0x2c77527b, 0xd379d3df, 0x85f73bb0, 0x0348d21c, 0x599a6f7c, 0x763cbf9c, 0x0925c4b8};
    static constexpr storage<2*limbs_count> modulus_sqared_2 = {0xc0000002, 0x1187d64f, 0xb9668000, 0x8fe4c447, 0xd19376ff, 0xffd34c58, 0xd043bb61, 0x4d9c32eb, 0x8f6c5fcf, 0x58eea4f6, 0xa6f3a7be, 0x0bee7761, 0x0691a439, 0xb334def8, 0xec797f38, 0x124b8970};
    static constexpr storage<2*limbs_count> modulus_sqared_4 = {0x80000004, 0x230fac9f, 0x72cd0000, 0x1fc9888f, 0xa326edff, 0xffa698b1, 0xa08776c3, 0x9b3865d7, 0x1ed8bf9e, 0xb1dd49ed, 0x4de74f7c, 0x17dceec3, 0x0d234872, 0x6669bdf0, 0xd8f2fe71, 0x249712e1};
    static constexpr unsigned modulus_bits_count = 254;
    static constexpr storage<limbs_count> m = {0xbe1de925, 0x620703a6, 0x09e880ae, 0x71448520, 0x68073014, 0xab074a58, 0x623a04a7, 0x54a47462};
    static constexpr storage<limbs_count> one = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};

    static constexpr storage<limbs_count> omega1= {0xf0000000, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    static constexpr storage<limbs_count> omega2= {0x8f703636, 0x23120470, 0xfd736bec, 0x5cea24f6, 0x3fd84104, 0x048b6e19, 0xe131a029, 0x30644e72};
    static constexpr storage<limbs_count> omega3= {0xc1bd5e80, 0x948dad4a, 0xf8170a0a, 0x52627366, 0x96afef36, 0xec9b9e2f, 0xc8c14f22, 0x2b337de1};
    static constexpr storage<limbs_count> omega4= {0xe306460b, 0xb11509c6, 0x174efb98, 0x996dfbe1, 0x94dd508c, 0x1c6e4f45, 0x16cbbf4e, 0x21082ca2};
    static constexpr storage<limbs_count> omega5= {0x3bb512d0, 0x3eed4c53, 0x838eeb1d, 0x9c18d51b, 0x47c0b2a9, 0x9678200d, 0x306b93d2, 0x09c532c6};
    static constexpr storage<limbs_count> omega6= {0x118f023a, 0xdb94fb05, 0x26e324be, 0x46a6cb24, 0x49bdadf2, 0xc24cdb76, 0x5b080fca, 0x1418144d};
    static constexpr storage<limbs_count> omega7= {0xba9d1811, 0x9d0e470c, 0xb6f24c79, 0x1dcb5564, 0xe85943e0, 0xdf5ce19c, 0xad310991, 0x16e73dfd};
    static constexpr storage<limbs_count> omega8= {0x74a57a76, 0xc8936191, 0x6750f230, 0x61794254, 0x9f36ffb0, 0xf086204a, 0xa6148404, 0x07b0c561};
    static constexpr storage<limbs_count> omega9= {0x470157ce, 0x893a7fa1, 0xfc782d75, 0xe8302a41, 0xdd9b0675, 0xffc02c0e, 0xf6e72f5b, 0x0f1ded1e};
    static constexpr storage<limbs_count> omega10= {0xbc2e5912, 0x11f995e1, 0xa8d2d7ab, 0x39ba79c0, 0xb08771e3, 0xebbebc2b, 0x7017a420, 0x06fd19c1};
    static constexpr storage<limbs_count> omega11= {0x769a2ee2, 0xd00a58f9, 0x7494f0ca, 0xb8c12c17, 0xa5355d71, 0xb4027fd7, 0x99c5042b, 0x027a3584};
    static constexpr storage<limbs_count> omega12= {0x0042d43a, 0x1c477572, 0x6f039bb9, 0x76f169c7, 0xfd5a90a9, 0x01ddd073, 0xde2fd10f, 0x0931d596};
    static constexpr storage<limbs_count> omega13= {0x9bbdd310, 0x4aa49b8d, 0x8e3a2d76, 0xd31bf3e2, 0x78b2667b, 0x001deac8, 0xb869ae62, 0x006fab49};
    static constexpr storage<limbs_count> omega14= {0x617c6e85, 0xadaa01c2, 0x7420aae6, 0xb4a93ee1, 0x0ddca8a8, 0x1f4e51b8, 0xcdd9e481, 0x2d965651};
    static constexpr storage<limbs_count> omega15= {0x4e26ecfb, 0xa93458fd, 0x4115a009, 0x022a2a2d, 0x69ec2bd0, 0x017171fa, 0x5941dc91, 0x2d1ba66f};
    static constexpr storage<limbs_count> omega16= {0xdaac43b7, 0xd1628ba2, 0xe4347e7d, 0x16c8601d, 0xe081dcff, 0x649abebd, 0x5981ed45, 0x00eeb2cb};
    static constexpr storage<limbs_count> omega17= {0xec8af73d, 0x8d24de3c, 0xcf722b45, 0x50f778d4, 0x15bc7dd7, 0xf4506bc3, 0xf94a16e1, 0x0e43ba91};
    static constexpr storage<limbs_count> omega18= {0xd4405b8f, 0x0baa7b44, 0xee0f1394, 0xf8f3c7fe, 0xef0dfe6d, 0x46b153c0, 0x2dde6b95, 0x0ea2bcd9};
    static constexpr storage<limbs_count> omega19= {0x3d1fa34e, 0x5f4dc975, 0x15af81db, 0xc28e54ee, 0x04947d99, 0x83d9a55f, 0x54a2b488, 0x08ec7ccf};
    static constexpr storage<limbs_count> omega20= {0x0cac0ee8, 0x0d8fa7b3, 0x82ef38e4, 0x756284ed, 0xac8f90d2, 0x7014b194, 0x634e5d50, 0x092488f8};
    static constexpr storage<limbs_count> omega21= {0x6d34ed69, 0xd85399bf, 0x09e49cef, 0x4d9012ba, 0xca00ae5d, 0x020142ee, 0x3bdfebfd, 0x12772e57};
    static constexpr storage<limbs_count> omega22= {0x2eb41723, 0x676c8fc7, 0x5dd895bd, 0xe20380e2, 0x9bf22dde, 0x09dc8be8, 0x42638176, 0x12822f94};
    static constexpr storage<limbs_count> omega23= {0x81a6d2de, 0x1f1df770, 0xcf29c812, 0x5d33b2da, 0x134f0e7e, 0x1bf162de, 0x1e2877a8, 0x045162c4};
    static constexpr storage<limbs_count> omega24= {0xfecda1b6, 0x24f4503b, 0xded67d3c, 0x0e5d7ed3, 0x40cf20af, 0x2b7b7e5e, 0x4faad6af, 0x0d472650};
    static constexpr storage<limbs_count> omega25= {0x584b9eb1, 0xcc6c474c, 0x15a8d886, 0x47670804, 0xbb8654c5, 0x07736d2f, 0xeb207a4b, 0x0d14ce7a};
    static constexpr storage<limbs_count> omega26= {0xed25924a, 0xd1c6471c, 0x6bc312c3, 0xd98bb374, 0xfeae1a41, 0x50be0848, 0x3265c719, 0x04b07dea};
    static constexpr storage<limbs_count> omega27= {0x618241e3, 0xab13f73e, 0x166ca902, 0x571c9267, 0x5e828a6d, 0x8586443a, 0x6daba50b, 0x093fdf2f};
    static constexpr storage<limbs_count> omega28= {0xee11c34f, 0xe688e66b, 0xeacecf5a, 0xdc232eae, 0xb95ae685, 0x4fc35094, 0x7c1d31dc, 0x0273b5bd};
    static constexpr storage<limbs_count> omega29= {0x1a9057bd, 0x8a8a5a77, 0x41834fbb, 0xdcbfae1d, 0xb34ede6e, 0x534f5b97, 0xb78bbd3e, 0x07313ac5};
    static constexpr storage<limbs_count> omega30= {0x2be70731, 0x287abbb1, 0x7c35c5aa, 0x5cbcfd1e, 0x1671f4df, 0x7585b3fe, 0xb899c011, 0x08350ecf};
    static constexpr storage<limbs_count> omega31= {0x09f7c5e2, 0x3400c14e, 0x0a649ea1, 0xc112e60c, 0x067ce95e, 0xf7510758, 0xf9daf17c, 0x040a66a5};
    static constexpr storage<limbs_count> omega32= {0x43efecd3, 0x89d65957, 0x3bd6c318, 0x29246adc, 0xce01533c, 0xf9fb5ef6, 0x849078c3, 0x020410e4};

    static constexpr storage_array<omegas_count, limbs_count> omega = {
        omega1, omega2, omega3, omega4, omega5, omega6, omega7, omega8, 
        omega9, omega10, omega11, omega12, omega13, omega14, omega15, omega16,
        omega17, omega18, omega19, omega20, omega21, omega22, omega23, omega24,
        omega25, omega26, omega27, omega28, omega29, omega30, omega31, omega32,
    };

    static constexpr storage<limbs_count> omega_inv1= {0xf0000000, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    static constexpr storage<limbs_count> omega_inv2= {0x608fc9cb, 0x20cff123, 0x7c4604a5, 0xcb49c351, 0x41a91758, 0xb3c4d79d, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> omega_inv3= {0x07b95a9b, 0x8b11d9ab, 0x41671f56, 0x20710ead, 0x30f81dee, 0xfb3acaee, 0x9778465c, 0x130b1711};
    static constexpr storage<limbs_count> omega_inv4= {0x373428de, 0xb85a71e6, 0xaeb0337e, 0x74954d30, 0x303402b7, 0x2bfc85eb, 0x409556c0, 0x02e40daf};
    static constexpr storage<limbs_count> omega_inv5= {0xf210979d, 0x8c99980c, 0x34905b4d, 0xef8f3113, 0xdf25d8e7, 0x0aeaf3e7, 0x03bfbd79, 0x27247136};
    static constexpr storage<limbs_count> omega_inv6= {0x763d698f, 0x78ce6a0b, 0x1d3213ee, 0xd80396ec, 0x67a8a676, 0x035cdc75, 0xb2a13d3a, 0x26177cf2};
    static constexpr storage<limbs_count> omega_inv7= {0xc64427d7, 0xdddf985f, 0xa49e95bd, 0xaa4f964a, 0x5def8b04, 0x427c045f, 0x7969b732, 0x1641c053};
    static constexpr storage<limbs_count> omega_inv8= {0x0329f5d6, 0x692c553d, 0x8712848a, 0xa54cf8c6, 0x38e2b5e6, 0x64751ad9, 0x7422fad3, 0x204bd327};
    static constexpr storage<limbs_count> omega_inv9= {0xaf6b3e4e, 0x52f26c0f, 0xf0bcc0c8, 0x4c277a07, 0xe4fcfcab, 0x546875d5, 0xaa9995b3, 0x09d8f821};
    static constexpr storage<limbs_count> omega_inv10= {0xb2e5cc71, 0xcaa2e1e9, 0x6e43404e, 0xed42b68e, 0x7a2c7f0a, 0x6ed80915, 0xde3c86d6, 0x1c4042c7};
    static constexpr storage<limbs_count> omega_inv11= {0x579d71ae, 0x20a3a65d, 0x0adc4420, 0xfd7efed8, 0xfddabf54, 0x3bb6dcd7, 0xbc73d07b, 0x0fa9bb21};
    static constexpr storage<limbs_count> omega_inv12= {0xc79e0e57, 0xb6f70f8d, 0xa04e05ac, 0x269d3fde, 0x2ba088d9, 0xcf2e371c, 0x11b88d9c, 0x1af864d2};
    static constexpr storage<limbs_count> omega_inv13= {0xabd95dc9, 0x3b0b205a, 0x978188ca, 0xc8df74fa, 0x6a1cb6c8, 0x08e124db, 0xbfac6104, 0x1670ed58};
    static constexpr storage<limbs_count> omega_inv14= {0x641c8410, 0xf8eee934, 0x677771c0, 0xf40976b0, 0x558e6e8c, 0x11680d42, 0x06e7e9e9, 0x281c036f};
    static constexpr storage<limbs_count> omega_inv15= {0xb2dbc0b4, 0xc92a742f, 0x4d384e68, 0xc3f02842, 0x2fa43d0d, 0x22701b6f, 0xe4590b37, 0x05d33766};
    static constexpr storage<limbs_count> omega_inv16= {0x02d842d4, 0x922d5ac8, 0xc830e4c6, 0x91126414, 0x082f37e0, 0xe92338c0, 0x7fe704e8, 0x0b5d56b7};
    static constexpr storage<limbs_count> omega_inv17= {0xec8af73d, 0x8d24de3c, 0xcf722b45, 0x50f778d4, 0x15bc7dd7, 0xf4506bc3, 0xf94a16e1, 0x0e43ba91};
    static constexpr storage<limbs_count> omega_inv18= {0xd4405b8f, 0x0baa7b44, 0xee0f1394, 0xf8f3c7fe, 0xef0dfe6d, 0x46b153c0, 0x2dde6b95, 0x0ea2bcd9};
    static constexpr storage<limbs_count> omega_inv19= {0x3d1fa34e, 0x5f4dc975, 0x15af81db, 0xc28e54ee, 0x04947d99, 0x83d9a55f, 0x54a2b488, 0x08ec7ccf};
    static constexpr storage<limbs_count> omega_inv20= {0x0cac0ee8, 0x0d8fa7b3, 0x82ef38e4, 0x756284ed, 0xac8f90d2, 0x7014b194, 0x634e5d50, 0x092488f8};
    static constexpr storage<limbs_count> omega_inv21= {0x6d34ed69, 0xd85399bf, 0x09e49cef, 0x4d9012ba, 0xca00ae5d, 0x020142ee, 0x3bdfebfd, 0x12772e57};
    static constexpr storage<limbs_count> omega_inv22= {0x2eb41723, 0x676c8fc7, 0x5dd895bd, 0xe20380e2, 0x9bf22dde, 0x09dc8be8, 0x42638176, 0x12822f94};
    static constexpr storage<limbs_count> omega_inv23= {0x81a6d2de, 0x1f1df770, 0xcf29c812, 0x5d33b2da, 0x134f0e7e, 0x1bf162de, 0x1e2877a8, 0x045162c4};
    static constexpr storage<limbs_count> omega_inv24= {0xfecda1b6, 0x24f4503b, 0xded67d3c, 0x0e5d7ed3, 0x40cf20af, 0x2b7b7e5e, 0x4faad6af, 0x0d472650};
    static constexpr storage<limbs_count> omega_inv25= {0x584b9eb1, 0xcc6c474c, 0x15a8d886, 0x47670804, 0xbb8654c5, 0x07736d2f, 0xeb207a4b, 0x0d14ce7a};
    static constexpr storage<limbs_count> omega_inv26= {0xed25924a, 0xd1c6471c, 0x6bc312c3, 0xd98bb374, 0xfeae1a41, 0x50be0848, 0x3265c719, 0x04b07dea};
    static constexpr storage<limbs_count> omega_inv27= {0x618241e3, 0xab13f73e, 0x166ca902, 0x571c9267, 0x5e828a6d, 0x8586443a, 0x6daba50b, 0x093fdf2f};
    static constexpr storage<limbs_count> omega_inv28= {0xee11c34f, 0xe688e66b, 0xeacecf5a, 0xdc232eae, 0xb95ae685, 0x4fc35094, 0x7c1d31dc, 0x0273b5bd};
    static constexpr storage<limbs_count> omega_inv29= {0x1a9057bd, 0x8a8a5a77, 0x41834fbb, 0xdcbfae1d, 0xb34ede6e, 0x534f5b97, 0xb78bbd3e, 0x07313ac5};
    static constexpr storage<limbs_count> omega_inv30= {0x2be70731, 0x287abbb1, 0x7c35c5aa, 0x5cbcfd1e, 0x1671f4df, 0x7585b3fe, 0xb899c011, 0x08350ecf};
    static constexpr storage<limbs_count> omega_inv31= {0x09f7c5e2, 0x3400c14e, 0x0a649ea1, 0xc112e60c, 0x067ce95e, 0xf7510758, 0xf9daf17c, 0x040a66a5};
    static constexpr storage<limbs_count> omega_inv32= {0x43efecd3, 0x89d65957, 0x3bd6c318, 0x29246adc, 0xce01533c, 0xf9fb5ef6, 0x849078c3, 0x020410e4};

    static constexpr storage_array<omegas_count, limbs_count> omega_inv = {
        omega_inv1, omega_inv2, omega_inv3, omega_inv4, omega_inv5, omega_inv6, omega_inv7, omega_inv8, 
        omega_inv9, omega_inv10, omega_inv11, omega_inv12, omega_inv13, omega_inv14, omega_inv15, omega_inv16,
        omega_inv17, omega_inv18, omega_inv19, omega_inv20, omega_inv21, omega_inv22, omega_inv23, omega_inv24,
        omega_inv25, omega_inv26, omega_inv27, omega_inv28, omega_inv29, omega_inv30, omega_inv31, omega_inv32,
    };
    
    static constexpr storage<limbs_count> inv1= {0xf8000001, 0xa1f0fac9, 0x3cdcb848, 0x9419f424, 0x40c0ac2e, 0xdc2822db, 0x7098d014, 0x18322739};
    static constexpr storage<limbs_count> inv2= {0xf4000001, 0xf2e9782e, 0x5b4b146c, 0xde26ee36, 0xe1210245, 0x4a3c3448, 0x28e5381f, 0x244b3ad6};
    static constexpr storage<limbs_count> inv3= {0x72000001, 0x1b65b6e1, 0x6a82427f, 0x832d6b3f, 0xb1512d51, 0x81463cff, 0x850b6c24, 0x2a57c4a4};
    static constexpr storage<limbs_count> inv4= {0xb1000001, 0x2fa3d63a, 0xf21dd988, 0x55b0a9c3, 0x196942d7, 0x1ccb415b, 0xb31e8627, 0x2d5e098b};
    static constexpr storage<limbs_count> inv5= {0x50800001, 0xb9c2e5e7, 0x35eba50c, 0x3ef24906, 0xcd754d9a, 0x6a8dc388, 0x4a281328, 0x2ee12bff};
    static constexpr storage<limbs_count> inv6= {0xa0400001, 0xfed26dbd, 0x57d28ace, 0xb39318a7, 0xa77b52fb, 0x116f049f, 0x15acd9a9, 0x2fa2bd39};
    static constexpr storage<limbs_count> inv7= {0xc8200001, 0x215a31a8, 0xe8c5fdb0, 0x6de38077, 0x147e55ac, 0x64dfa52b, 0xfb6f3ce9, 0x300385d5};
    static constexpr storage<limbs_count> inv8= {0x5c100001, 0xb29e139e, 0x313fb720, 0xcb0bb460, 0xcaffd704, 0x8e97f570, 0x6e506e89, 0x3033ea24};
    static constexpr storage<limbs_count> inv9= {0x26080001, 0xfb400499, 0x557c93d8, 0xf99fce54, 0xa64097b0, 0xa3741d93, 0xa7c10759, 0x304c1c4b};
    static constexpr storage<limbs_count> inv10= {0x8b040001, 0x1f90fd16, 0x679b0235, 0x10e9db4e, 0x13e0f807, 0xade231a5, 0x447953c1, 0x3058355f};
    static constexpr storage<limbs_count> inv11= {0x3d820001, 0x31b97955, 0x70aa3963, 0x1c8ee1cb, 0xcab12832, 0xb3193bad, 0x12d579f5, 0x305e41e9};
    static constexpr storage<limbs_count> inv12= {0x96c10001, 0x3acdb774, 0xf531d4fa, 0xa2616509, 0x26194047, 0xb5b4c0b2, 0xfa038d0f, 0x3061482d};
    static constexpr storage<limbs_count> inv13= {0x43608001, 0xbf57d684, 0x3775a2c5, 0x654aa6a9, 0x53cd4c52, 0xb7028334, 0x6d9a969c, 0x3062cb50};
    static constexpr storage<limbs_count> inv14= {0x19b04001, 0x819ce60c, 0xd89789ab, 0xc6bf4778, 0x6aa75257, 0x37a96475, 0xa7661b63, 0x30638ce1};
    static constexpr storage<limbs_count> inv15= {0x04d82001, 0x62bf6dd0, 0xa9287d1e, 0x777997e0, 0xf614555a, 0x77fcd515, 0x444bddc6, 0x3063edaa};
    static constexpr storage<limbs_count> inv16= {0xfa6c1001, 0xd350b1b1, 0x9170f6d7, 0xcfd6c014, 0x3bcad6db, 0x18268d66, 0x92bebef8, 0x30641e0e};
    static constexpr storage<limbs_count> inv17= {0xec8af73d, 0x8d24de3c, 0xcf722b45, 0x50f778d4, 0x15bc7dd7, 0xf4506bc3, 0xf94a16e1, 0x0e43ba91};
    static constexpr storage<limbs_count> inv18= {0xd4405b8f, 0x0baa7b44, 0xee0f1394, 0xf8f3c7fe, 0xef0dfe6d, 0x46b153c0, 0x2dde6b95, 0x0ea2bcd9};
    static constexpr storage<limbs_count> inv19= {0x3d1fa34e, 0x5f4dc975, 0x15af81db, 0xc28e54ee, 0x04947d99, 0x83d9a55f, 0x54a2b488, 0x08ec7ccf};
    static constexpr storage<limbs_count> inv20= {0x0cac0ee8, 0x0d8fa7b3, 0x82ef38e4, 0x756284ed, 0xac8f90d2, 0x7014b194, 0x634e5d50, 0x092488f8};
    static constexpr storage<limbs_count> inv21= {0x6d34ed69, 0xd85399bf, 0x09e49cef, 0x4d9012ba, 0xca00ae5d, 0x020142ee, 0x3bdfebfd, 0x12772e57};
    static constexpr storage<limbs_count> inv22= {0x2eb41723, 0x676c8fc7, 0x5dd895bd, 0xe20380e2, 0x9bf22dde, 0x09dc8be8, 0x42638176, 0x12822f94};
    static constexpr storage<limbs_count> inv23= {0x81a6d2de, 0x1f1df770, 0xcf29c812, 0x5d33b2da, 0x134f0e7e, 0x1bf162de, 0x1e2877a8, 0x045162c4};
    static constexpr storage<limbs_count> inv24= {0xfecda1b6, 0x24f4503b, 0xded67d3c, 0x0e5d7ed3, 0x40cf20af, 0x2b7b7e5e, 0x4faad6af, 0x0d472650};
    static constexpr storage<limbs_count> inv25= {0x584b9eb1, 0xcc6c474c, 0x15a8d886, 0x47670804, 0xbb8654c5, 0x07736d2f, 0xeb207a4b, 0x0d14ce7a};
    static constexpr storage<limbs_count> inv26= {0xed25924a, 0xd1c6471c, 0x6bc312c3, 0xd98bb374, 0xfeae1a41, 0x50be0848, 0x3265c719, 0x04b07dea};
    static constexpr storage<limbs_count> inv27= {0x618241e3, 0xab13f73e, 0x166ca902, 0x571c9267, 0x5e828a6d, 0x8586443a, 0x6daba50b, 0x093fdf2f};
    static constexpr storage<limbs_count> inv28= {0xee11c34f, 0xe688e66b, 0xeacecf5a, 0xdc232eae, 0xb95ae685, 0x4fc35094, 0x7c1d31dc, 0x0273b5bd};
    static constexpr storage<limbs_count> inv29= {0x1a9057bd, 0x8a8a5a77, 0x41834fbb, 0xdcbfae1d, 0xb34ede6e, 0x534f5b97, 0xb78bbd3e, 0x07313ac5};
    static constexpr storage<limbs_count> inv30= {0x2be70731, 0x287abbb1, 0x7c35c5aa, 0x5cbcfd1e, 0x1671f4df, 0x7585b3fe, 0xb899c011, 0x08350ecf};
    static constexpr storage<limbs_count> inv31= {0x09f7c5e2, 0x3400c14e, 0x0a649ea1, 0xc112e60c, 0x067ce95e, 0xf7510758, 0xf9daf17c, 0x040a66a5};
    static constexpr storage<limbs_count> inv32= {0x43efecd3, 0x89d65957, 0x3bd6c318, 0x29246adc, 0xce01533c, 0xf9fb5ef6, 0x849078c3, 0x020410e4};

    static constexpr storage_array<omegas_count, limbs_count> inv = {
        inv1, inv2, inv3, inv4, inv5, inv6, inv7, inv8, 
        inv9, inv10, inv11, inv12, inv13, inv14, inv15, inv16,
        inv17, inv18, inv19, inv20, inv21, inv22, inv23, inv24,
        inv25, inv26, inv27, inv28, inv29, inv30, inv31, inv32,
    }; 
  };

  struct fq_config{
    static constexpr unsigned limbs_count = 8;
    static constexpr storage<limbs_count> modulus = {0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    static constexpr storage<limbs_count> modulus_2 = {0xb0f9fa8e, 0x7841182d, 0xd0e3951a, 0x2f02d522, 0x0302b0bb, 0x70a08b6d, 0xc2634053, 0x60c89ce5};
    static constexpr storage<limbs_count> modulus_4 = {0x61f3f51c, 0xf082305b, 0xa1c72a34, 0x5e05aa45, 0x06056176, 0xe14116da, 0x84c680a6, 0xc19139cb};
    static constexpr storage<2*limbs_count> modulus_wide = {0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<2*limbs_count> modulus_sqared = {0x275d69b1, 0x3b5458a2, 0x09eac101, 0xa602072d, 0x6d96cadc, 0x4a50189c, 0x7a1242c8, 0x04689e95, 0x34c6b38d, 0x26edfa5c, 0x16375606, 0xb00b8551, 0x0348d21c, 0x599a6f7c, 0x763cbf9c, 0x0925c4b8};
    static constexpr storage<2*limbs_count> modulus_sqared_2 = {0x4ebad362, 0x76a8b144, 0x13d58202, 0x4c040e5a, 0xdb2d95b9, 0x94a03138, 0xf4248590, 0x08d13d2a, 0x698d671a, 0x4ddbf4b8, 0x2c6eac0c, 0x60170aa2, 0x0691a439, 0xb334def8, 0xec797f38, 0x124b8970};
    static constexpr storage<2*limbs_count> modulus_sqared_4 = {0x9d75a6c4, 0xed516288, 0x27ab0404, 0x98081cb4, 0xb65b2b72, 0x29406271, 0xe8490b21, 0x11a27a55, 0xd31ace34, 0x9bb7e970, 0x58dd5818, 0xc02e1544, 0x0d234872, 0x6669bdf0, 0xd8f2fe71, 0x249712e1};
    static constexpr unsigned modulus_bits_count = 254;
    static constexpr storage<limbs_count> m = {0x19bf90e5, 0x6f3aed8a, 0x67cd4c08, 0xae965e17, 0x68073013, 0xab074a58, 0x623a04a7, 0x54a47462};
    static constexpr storage<limbs_count> one = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    // i^2, the square of the imaginary unit for the extension field
    static constexpr uint32_t i_squared = 1;
    // true if i^2 is negative
    static constexpr bool i_squared_is_negative = true;
    // G1 and G2 generators 
    static constexpr storage<limbs_count> generator_x = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> generator_y = {0x00000002, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> generator_x_re = {0xd992f6ed, 0x46debd5c, 0xf75edadd, 0x674322d4, 0x5e5c4479, 0x426a0066, 0x121f1e76, 0x1800deef};
    static constexpr storage<limbs_count> generator_x_im = {0xaef312c2, 0x97e485b7, 0x35a9e712, 0xf1aa4933, 0x31fb5d25, 0x7260bfb7, 0x920d483a, 0x198e9393};
    static constexpr storage<limbs_count> generator_y_re = {0x66fa7daa, 0x4ce6cc01, 0xc43d37b, 0xe3d1e769, 0x8dcb408f, 0x4aab7180, 0xdb8c6deb, 0x12c85ea5};
    static constexpr storage<limbs_count> generator_y_im = {0xd122975b, 0x55acdadc, 0x70b38ef3, 0xbc4b3133, 0x690c3395, 0xec9e99ad, 0x585ff075, 0x90689d0};
  };

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {0x00000003, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};

  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_re = {0x24a138e5, 0x3267e6dc, 0x59dbefa3, 0xb5b4c5e5, 0x1be06ac3, 0x81be1899, 0xceb8aaae, 0x2b149d40};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_im = {0x85c315d2, 0xe4a2bd06, 0xe52d1852, 0xa74fa084, 0xeed8fdf4, 0xcd2cafad, 0x3af0fed4, 0x9713b0};
}
