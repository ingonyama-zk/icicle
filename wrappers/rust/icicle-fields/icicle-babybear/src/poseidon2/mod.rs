use crate::field::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::hash::SpongeConfig;
use icicle_core::poseidon2::{DiffusionStrategy, MdsType, Poseidon2, Poseidon2Handle, Poseidon2Impl};
use icicle_core::traits::IcicleResultWrap;
use icicle_core::tree::{TreeBuilder, TreeBuilderConfig};
use icicle_core::{impl_poseidon2, impl_poseidon2_tree_builder};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::{DeviceSlice, HostOrDeviceSlice};

use core::mem::MaybeUninit;

impl_poseidon2!("babybear", babybear, ScalarField, ScalarCfg);
impl_poseidon2_tree_builder!("babybear", babybear_poseidon2_tb, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {

    use super::Poseidon2TreeBuilder;
    use crate::field::ScalarField;
    use icicle_core::impl_poseidon2_tests;
    use icicle_core::poseidon2::{tests::*, DiffusionStrategy, MdsType, Poseidon2};
    use icicle_core::traits::FieldImpl;
    use icicle_core::tree::tests::check_build_field_merkle_tree;
    use icicle_core::tree::{TreeBuilder, TreeBuilderConfig};
    use icicle_cuda_runtime::device_context::{self, DeviceContext};

    use icicle_cuda_runtime::memory::HostSlice;
    use p3_baby_bear::BabyBear;
    use p3_baby_bear::DiffusionMatrixBabyBear;
    use p3_commit::Mmcs;
    use p3_field::{AbstractField, Field, PrimeField32};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_poseidon2::{Poseidon2 as PlonkyPoseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::{PaddingFreeSponge, Permutation, TruncatedPermutation};

    impl_poseidon2_tests!(ScalarField);

    #[test]
    fn poseidon2_merkle_tree_test() {
        let ctx = device_context::DeviceContext::default();
        let sponge = Poseidon2::load(2, MdsType::Default, DiffusionStrategy::Default, &ctx).unwrap();

        check_build_field_merkle_tree::<_, _, Poseidon2TreeBuilder>(25, 2, &sponge, &sponge, ScalarField::zero());
    }

    #[test]
    fn test_poseidon2_kats() {
        let kats = [
            ScalarField::from_hex("0x2ed3e23d"),
            ScalarField::from_hex("0x12921fb0"),
            ScalarField::from_hex("0x0e659e79"),
            ScalarField::from_hex("0x61d81dc9"),
            ScalarField::from_hex("0x32bae33b"),
            ScalarField::from_hex("0x62486ae3"),
            ScalarField::from_hex("0x1e681b60"),
            ScalarField::from_hex("0x24b91325"),
            ScalarField::from_hex("0x2a2ef5b9"),
            ScalarField::from_hex("0x50e8593e"),
            ScalarField::from_hex("0x5bc818ec"),
            ScalarField::from_hex("0x10691997"),
            ScalarField::from_hex("0x35a14520"),
            ScalarField::from_hex("0x2ba6a3c5"),
            ScalarField::from_hex("0x279d47ec"),
            ScalarField::from_hex("0x55014e81"),
            ScalarField::from_hex("0x5953a67f"),
            ScalarField::from_hex("0x2f403111"),
            ScalarField::from_hex("0x6b8828ff"),
            ScalarField::from_hex("0x1801301f"),
            ScalarField::from_hex("0x2749207a"),
            ScalarField::from_hex("0x3dc9cf21"),
            ScalarField::from_hex("0x3c985ba2"),
            ScalarField::from_hex("0x57a99864"),
        ];

        let poseidon = init_poseidon::<ScalarField>(24, MdsType::Default, DiffusionStrategy::Default);
        check_poseidon_kats(24, &kats, &poseidon);
    }

    type PlonkyPoseidon2T16 = PlonkyPoseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;

    fn get_plonky3_poseidon2_t16() -> (Poseidon2<ScalarField>, PlonkyPoseidon2T16) {
        let rounds_p = 13;
        let rounds_f = 8;
        const ALPHA: u64 = 7;
        const WIDTH: usize = 16;

        let cnv = BabyBear::from_canonical_u32;
        let external_constants: Vec<[BabyBear; 16]> = vec![
            [
                cnv(1774958255),
                cnv(1185780729),
                cnv(1621102414),
                cnv(1796380621),
                cnv(588815102),
                cnv(1932426223),
                cnv(1925334750),
                cnv(747903232),
                cnv(89648862),
                cnv(360728943),
                cnv(977184635),
                cnv(1425273457),
                cnv(256487465),
                cnv(1200041953),
                cnv(572403254),
                cnv(448208942),
            ],
            [
                cnv(1215789478),
                cnv(944884184),
                cnv(953948096),
                cnv(547326025),
                cnv(646827752),
                cnv(889997530),
                cnv(1536873262),
                cnv(86189867),
                cnv(1065944411),
                cnv(32019634),
                cnv(333311454),
                cnv(456061748),
                cnv(1963448500),
                cnv(1827584334),
                cnv(1391160226),
                cnv(1348741381),
            ],
            [
                cnv(88424255),
                cnv(104111868),
                cnv(1763866748),
                cnv(79691676),
                cnv(1988915530),
                cnv(1050669594),
                cnv(359890076),
                cnv(573163527),
                cnv(222820492),
                cnv(159256268),
                cnv(669703072),
                cnv(763177444),
                cnv(889367200),
                cnv(256335831),
                cnv(704371273),
                cnv(25886717),
            ],
            [
                cnv(51754520),
                cnv(1833211857),
                cnv(454499742),
                cnv(1384520381),
                cnv(777848065),
                cnv(1053320300),
                cnv(1851729162),
                cnv(344647910),
                cnv(401996362),
                cnv(1046925956),
                cnv(5351995),
                cnv(1212119315),
                cnv(754867989),
                cnv(36972490),
                cnv(751272725),
                cnv(506915399),
            ],
            [
                cnv(1922082829),
                cnv(1870549801),
                cnv(1502529704),
                cnv(1990744480),
                cnv(1700391016),
                cnv(1702593455),
                cnv(321330495),
                cnv(528965731),
                cnv(183414327),
                cnv(1886297254),
                cnv(1178602734),
                cnv(1923111974),
                cnv(744004766),
                cnv(549271463),
                cnv(1781349648),
                cnv(542259047),
            ],
            [
                cnv(1536158148),
                cnv(715456982),
                cnv(503426110),
                cnv(340311124),
                cnv(1558555932),
                cnv(1226350925),
                cnv(742828095),
                cnv(1338992758),
                cnv(1641600456),
                cnv(1843351545),
                cnv(301835475),
                cnv(43203215),
                cnv(386838401),
                cnv(1520185679),
                cnv(1235297680),
                cnv(904680097),
            ],
            [
                cnv(1491801617),
                cnv(1581784677),
                cnv(913384905),
                cnv(247083962),
                cnv(532844013),
                cnv(107190701),
                cnv(213827818),
                cnv(1979521776),
                cnv(1358282574),
                cnv(1681743681),
                cnv(1867507480),
                cnv(1530706910),
                cnv(507181886),
                cnv(695185447),
                cnv(1172395131),
                cnv(1250800299),
            ],
            [
                cnv(1503161625),
                cnv(817684387),
                cnv(498481458),
                cnv(494676004),
                cnv(1404253825),
                cnv(108246855),
                cnv(59414691),
                cnv(744214112),
                cnv(890862029),
                cnv(1342765939),
                cnv(1417398904),
                cnv(1897591937),
                cnv(1066647396),
                cnv(1682806907),
                cnv(1015795079),
                cnv(1619482808),
            ],
        ];

        let internal_constants: Vec<BabyBear> = vec![
            cnv(1518359488),
            cnv(1765533241),
            cnv(945325693),
            cnv(422793067),
            cnv(311365592),
            cnv(1311448267),
            cnv(1629555936),
            cnv(1009879353),
            cnv(190525218),
            cnv(786108885),
            cnv(557776863),
            cnv(212616710),
            cnv(605745517),
        ];

        let plonky_poseidon2: PlonkyPoseidon2<
            BabyBear,
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            WIDTH,
            ALPHA,
        > = PlonkyPoseidon2::new(
            rounds_f,
            external_constants.clone(),
            Poseidon2ExternalMatrixGeneral::default(),
            rounds_p,
            internal_constants.clone(),
            DiffusionMatrixBabyBear::default(),
        );

        let ctx = DeviceContext::default();
        let mut round_constants = vec![ScalarField::zero(); rounds_f * WIDTH + rounds_p];
        let external_constants_flattened: Vec<ScalarField> = external_constants
            .into_iter()
            .flatten()
            .map(|c| ScalarField::from_u32(c.as_canonical_u32()))
            .collect();
        let internal_constants_icicle: Vec<ScalarField> = internal_constants
            .iter()
            .map(|&c| ScalarField::from_u32(c.as_canonical_u32()))
            .collect();

        (&mut round_constants[..rounds_f / 2 * WIDTH])
            .copy_from_slice(&external_constants_flattened[..rounds_f / 2 * WIDTH]);
        (&mut round_constants[rounds_f / 2 * WIDTH..rounds_f / 2 * WIDTH + rounds_p])
            .copy_from_slice(&internal_constants_icicle[..]);
        (&mut round_constants[rounds_p + rounds_f / 2 * WIDTH..])
            .copy_from_slice(&external_constants_flattened[rounds_f / 2 * WIDTH..]);

        let mut internal_matrix_diag = vec![
            ScalarField::from_u32(0x78000001 - 2),
            ScalarField::from_u32(1),
            ScalarField::from_u32(1 << 1),
            ScalarField::from_u32(1 << 2),
            ScalarField::from_u32(1 << 3),
            ScalarField::from_u32(1 << 4),
            ScalarField::from_u32(1 << 5),
            ScalarField::from_u32(1 << 6),
            ScalarField::from_u32(1 << 7),
            ScalarField::from_u32(1 << 8),
            ScalarField::from_u32(1 << 9),
            ScalarField::from_u32(1 << 10),
            ScalarField::from_u32(1 << 11),
            ScalarField::from_u32(1 << 12),
            ScalarField::from_u32(1 << 13),
            ScalarField::from_u32(1 << 15),
        ];

        let poseidon = Poseidon2::new(
            WIDTH,
            ALPHA as u32,
            rounds_p as u32,
            rounds_f as u32,
            &mut round_constants,
            &mut internal_matrix_diag,
            MdsType::Plonky,
            DiffusionStrategy::Montgomery,
            &ctx,
        )
        .unwrap();

        (poseidon, plonky_poseidon2)
    }

    #[test]
    fn test_poseidon2_plonky3_t16() {
        const WIDTH: usize = 16;

        let (poseidon, plonky_poseidon2) = get_plonky3_poseidon2_t16();

        let mut input: [BabyBear; WIDTH] = [BabyBear::zero(); WIDTH];
        for i in 0..WIDTH {
            input[i] = BabyBear::from_canonical_u32(i as u32);
        }

        let output = plonky_poseidon2.permute(input);

        let mut kats: [ScalarField; WIDTH] = [ScalarField::zero(); WIDTH];
        for i in 0..WIDTH {
            kats[i] = ScalarField::from_u32(output[i].as_canonical_u32());
        }

        check_poseidon_kats(WIDTH, &kats, &poseidon);
    }

    #[test]
    fn test_poseidon2_plonky3_t24() {
        let rounds_p = 21;
        let rounds_f = 8;
        const ALPHA: u64 = 7;
        const WIDTH: usize = 24;

        let cnv = BabyBear::from_canonical_u32;
        let external_constants: Vec<[BabyBear; 24]> = vec![
            [
                cnv(262278199),
                cnv(127253399),
                cnv(314968988),
                cnv(246143118),
                cnv(157582794),
                cnv(118043943),
                cnv(454905424),
                cnv(815798990),
                cnv(1004040026),
                cnv(1773108264),
                cnv(1066694495),
                cnv(1930780904),
                cnv(1180307149),
                cnv(1464793095),
                cnv(1660766320),
                cnv(1389166148),
                cnv(343354132),
                cnv(1307439985),
                cnv(638242172),
                cnv(525458520),
                cnv(1964135730),
                cnv(1751797115),
                cnv(1421525369),
                cnv(831813382),
            ],
            [
                cnv(695835963),
                cnv(1845603984),
                cnv(540703332),
                cnv(1333667262),
                cnv(1917861751),
                cnv(1170029417),
                cnv(1989924532),
                cnv(1518763784),
                cnv(1339793538),
                cnv(622609176),
                cnv(686842369),
                cnv(1737016378),
                cnv(1282239129),
                cnv(897025192),
                cnv(716894289),
                cnv(1997503974),
                cnv(395622276),
                cnv(1201063290),
                cnv(1917549072),
                cnv(1150912935),
                cnv(1687379185),
                cnv(1507936940),
                cnv(241306552),
                cnv(989176635),
            ],
            [
                cnv(1147522062),
                cnv(27129487),
                cnv(1257820264),
                cnv(142102402),
                cnv(217046702),
                cnv(1664590951),
                cnv(855276054),
                cnv(1215259350),
                cnv(946500736),
                cnv(552696906),
                cnv(1424297384),
                cnv(538103555),
                cnv(1608853840),
                cnv(162510541),
                cnv(623051854),
                cnv(1549062383),
                cnv(1908416316),
                cnv(1622328571),
                cnv(1079030649),
                cnv(1584033957),
                cnv(1099252725),
                cnv(1910423126),
                cnv(447555988),
                cnv(862495875),
            ],
            [
                cnv(128479034),
                cnv(1587822577),
                cnv(608401422),
                cnv(1290028279),
                cnv(342857858),
                cnv(825405577),
                cnv(427731030),
                cnv(1718628547),
                cnv(588764636),
                cnv(204228775),
                cnv(1454563174),
                cnv(1740472809),
                cnv(1338899225),
                cnv(1269493554),
                cnv(53007114),
                cnv(1647670797),
                cnv(306391314),
                cnv(172614232),
                cnv(51256176),
                cnv(1221257987),
                cnv(1239734761),
                cnv(273790406),
                cnv(1781980094),
                cnv(1291790245),
            ],
            [
                cnv(53041581),
                cnv(723038058),
                cnv(1439947916),
                cnv(1136469704),
                cnv(205609311),
                cnv(1883820770),
                cnv(14387587),
                cnv(720724951),
                cnv(1854174607),
                cnv(1629316321),
                cnv(530151394),
                cnv(1679178250),
                cnv(1549779579),
                cnv(48375137),
                cnv(976057819),
                cnv(463976218),
                cnv(875839332),
                cnv(1946596189),
                cnv(434078361),
                cnv(1878280202),
                cnv(1363837384),
                cnv(1470845646),
                cnv(1792450386),
                cnv(1040977421),
            ],
            [
                cnv(1209164052),
                cnv(714957516),
                cnv(390340387),
                cnv(1213686459),
                cnv(790726260),
                cnv(117294666),
                cnv(140621810),
                cnv(993455846),
                cnv(1889603648),
                cnv(78845751),
                cnv(925018226),
                cnv(708123747),
                cnv(1647665372),
                cnv(1649953458),
                cnv(942439428),
                cnv(1006235079),
                cnv(238616145),
                cnv(930036496),
                cnv(1401020792),
                cnv(989618631),
                cnv(1545325389),
                cnv(1715719711),
                cnv(755691969),
                cnv(150307788),
            ],
            [
                cnv(1567618575),
                cnv(1663353317),
                cnv(1950429111),
                cnv(1891637550),
                cnv(192082241),
                cnv(1080533265),
                cnv(1463323727),
                cnv(890243564),
                cnv(158646617),
                cnv(1402624179),
                cnv(59510015),
                cnv(1198261138),
                cnv(1065075039),
                cnv(1150410028),
                cnv(1293938517),
                cnv(76770019),
                cnv(1478577620),
                cnv(1748789933),
                cnv(457372011),
                cnv(1841795381),
                cnv(760115692),
                cnv(1042892522),
                cnv(1507649755),
                cnv(1827572010),
            ],
            [
                cnv(1206940496),
                cnv(1896271507),
                cnv(1003792297),
                cnv(738091882),
                cnv(1124078057),
                cnv(1889898),
                cnv(813674331),
                cnv(228520958),
                cnv(1832911930),
                cnv(781141772),
                cnv(459826664),
                cnv(202271745),
                cnv(1296144415),
                cnv(1111203133),
                cnv(1090783436),
                cnv(641665156),
                cnv(1393671120),
                cnv(1303271640),
                cnv(809508074),
                cnv(162506101),
                cnv(1262312258),
                cnv(1672219447),
                cnv(1608891156),
                cnv(1380248020),
            ],
        ];

        let internal_constants: Vec<BabyBear> = vec![
            cnv(497520322),
            cnv(1930103076),
            cnv(1052077299),
            cnv(1540960371),
            cnv(924863639),
            cnv(1365519753),
            cnv(1726563304),
            cnv(440300254),
            cnv(1891545577),
            cnv(822033215),
            cnv(1111544260),
            cnv(308575117),
            cnv(1708681573),
            cnv(1240419708),
            cnv(1199068823),
            cnv(1186174623),
            cnv(1551596046),
            cnv(1886977120),
            cnv(1327682690),
            cnv(1210751726),
            cnv(1810596765),
        ];

        let plonky_poseidon2: PlonkyPoseidon2<
            BabyBear,
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            WIDTH,
            ALPHA,
        > = PlonkyPoseidon2::new(
            rounds_f,
            external_constants.clone(),
            Poseidon2ExternalMatrixGeneral::default(),
            rounds_p,
            internal_constants.clone(),
            DiffusionMatrixBabyBear::default(),
        );

        let mut input: [BabyBear; WIDTH] = [BabyBear::zero(); WIDTH];
        for i in 0..WIDTH {
            input[i] = BabyBear::from_canonical_u32(i as u32);
        }

        let output = plonky_poseidon2.permute(input);

        let mut kats: [ScalarField; WIDTH] = [ScalarField::zero(); WIDTH];
        for i in 0..WIDTH {
            kats[i] = ScalarField::from_u32(output[i].as_canonical_u32());
        }

        let ctx = DeviceContext::default();
        let mut round_constants = vec![ScalarField::zero(); rounds_f * WIDTH + rounds_p];
        let external_constants_flattened: Vec<ScalarField> = external_constants
            .into_iter()
            .flatten()
            .map(|c| ScalarField::from_u32(c.as_canonical_u32()))
            .collect();
        let internal_constants_icicle: Vec<ScalarField> = internal_constants
            .iter()
            .map(|&c| ScalarField::from_u32(c.as_canonical_u32()))
            .collect();

        (&mut round_constants[..rounds_f / 2 * WIDTH])
            .copy_from_slice(&external_constants_flattened[..rounds_f / 2 * WIDTH]);
        (&mut round_constants[rounds_f / 2 * WIDTH..rounds_f / 2 * WIDTH + rounds_p])
            .copy_from_slice(&internal_constants_icicle[..]);
        (&mut round_constants[rounds_p + rounds_f / 2 * WIDTH..])
            .copy_from_slice(&external_constants_flattened[rounds_f / 2 * WIDTH..]);

        let mut internal_matrix_diag = vec![
            ScalarField::from_u32(0x78000001 - 2),
            ScalarField::from_u32(1),
            ScalarField::from_u32(1 << 1),
            ScalarField::from_u32(1 << 2),
            ScalarField::from_u32(1 << 3),
            ScalarField::from_u32(1 << 4),
            ScalarField::from_u32(1 << 5),
            ScalarField::from_u32(1 << 6),
            ScalarField::from_u32(1 << 7),
            ScalarField::from_u32(1 << 8),
            ScalarField::from_u32(1 << 9),
            ScalarField::from_u32(1 << 10),
            ScalarField::from_u32(1 << 11),
            ScalarField::from_u32(1 << 12),
            ScalarField::from_u32(1 << 13),
            ScalarField::from_u32(1 << 14),
            ScalarField::from_u32(1 << 15),
            ScalarField::from_u32(1 << 16),
            ScalarField::from_u32(1 << 18),
            ScalarField::from_u32(1 << 19),
            ScalarField::from_u32(1 << 20),
            ScalarField::from_u32(1 << 21),
            ScalarField::from_u32(1 << 22),
            ScalarField::from_u32(1 << 23),
        ];
        let poseidon = Poseidon2::new(
            WIDTH,
            ALPHA as u32,
            rounds_p as u32,
            rounds_f as u32,
            &mut round_constants,
            &mut internal_matrix_diag,
            MdsType::Plonky,
            DiffusionStrategy::Montgomery,
            &ctx,
        )
        .unwrap();
        check_poseidon_kats(WIDTH, &kats, &poseidon);
    }

    #[test]
    fn test_poseidon2_tree_plonky3() {
        const WIDTH: usize = 16;
        const ARITY: usize = 2;
        const HEIGHT: usize = 15;
        const ROWS: usize = 1 << HEIGHT;
        const COLS: usize = 8;

        let (poseidon, plonky_poseidon2) = get_plonky3_poseidon2_t16();

        type H = PaddingFreeSponge<PlonkyPoseidon2T16, WIDTH, COLS, COLS>;
        let h = H::new(plonky_poseidon2.clone());

        type C = TruncatedPermutation<PlonkyPoseidon2T16, ARITY, COLS, WIDTH>;
        let c = C::new(plonky_poseidon2.clone());

        type F = BabyBear;

        let mut input = vec![F::zero(); ROWS * COLS];
        let mut icicle_input = vec![ScalarField::zero(); ROWS * COLS];
        for i in 0..ROWS * COLS {
            input[i] = F::from_canonical_u32(i as u32);
            icicle_input[i] = ScalarField::from_u32(i as u32);
        }

        let matrix = RowMajorMatrix::new(input, COLS);
        let leaves = vec![matrix];

        let mmcs = FieldMerkleTreeMmcs::<<F as Field>::Packing, <F as Field>::Packing, H, C, 8>::new(h, c);

        let (commit, _data) = mmcs.commit(leaves);

        let mut config = TreeBuilderConfig::default();
        config.arity = ARITY as u32;
        config.keep_rows = 1;
        config.digest_elements = COLS as u32;
        let input_block_len = COLS;
        // let mut digests = vec![ScalarField::zero(); merkle_tree_digests_len(2 as u32, ARITY as u32, COLS as u32)];
        let mut digests = vec![ScalarField::zero(); COLS];

        let leaves_slice = HostSlice::from_slice(&icicle_input);
        let digests_slice = HostSlice::from_mut_slice(&mut digests);

        Poseidon2TreeBuilder::build_merkle_tree(
            leaves_slice,
            digests_slice,
            HEIGHT,
            input_block_len,
            &poseidon,
            &poseidon,
            &config,
        )
        .unwrap();

        let mut converted: [BabyBear; COLS] = [BabyBear::zero(); COLS];
        for i in 0..COLS {
            let mut scalar_bytes = [0u8; 4];
            scalar_bytes.copy_from_slice(&digests_slice[i].to_bytes_le());
            converted[i] = BabyBear::from_canonical_u32(u32::from_le_bytes(scalar_bytes));
        }
        assert_eq!(commit, converted);
    }
}
