use crate::field::{QuarticExtensionField, ScalarField};
use icicle_core::error::IcicleResult;
use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device::check_device;
use icicle_cuda_runtime::device_context::{DeviceContext, DEFAULT_DEVICE_ID};
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostSlice};

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CirclePoint<T: std::clone::Clone> {
    x: T,
    y: T,
}

pub type Point = CirclePoint<ScalarField>;
pub type SecurePoint = CirclePoint<QuarticExtensionField>;

/// Struct that encodes FRI parameters.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct QuotientConfig<'a> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,
    are_columns_on_device: bool,
    are_sample_points_on_device: bool,
    are_results_on_device: bool,
    /// Whether to run the vector operations asynchronously. If set to `true`, the functions will be non-blocking and you'd need to synchronize
    /// it explicitly by running `stream.synchronize()`. If set to false, the functions will block the current CPU thread.
    pub is_async: bool,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct ColumnSampleBatchInternal<F: std::clone::Clone> {
    pub point: *const CirclePoint<F>,
    pub columns: *const u32,
    pub values: *const F,
    pub size: u32,
}

#[derive(Debug, Clone)]
pub struct ColumnSampleBatch<F: std::clone::Clone> {
    pub point: CirclePoint<F>,
    pub columns: Vec<u32>,
    pub values: Vec<F>,
}

impl<F: std::clone::Clone> ColumnSampleBatch<F> {
    pub fn unpack(&self) -> ColumnSampleBatchInternal<F> {
        if self
            .columns
            .len()
            != self
                .values
                .len()
        {
            panic!(
                "Error in ColumnSampleBatch: columns.len() != values.len(), {} != {}",
                self.columns
                    .len(),
                self.values
                    .len()
            );
        }

        let result = unsafe {
            ColumnSampleBatchInternal {
                point: &self.point,
                columns: HostSlice::from_slice(
                    self.columns
                        .as_slice(),
                )
                .as_ptr(),
                values: HostSlice::from_slice(
                    self.values
                        .as_slice(),
                )
                .as_ptr(),
                size: self
                    .values
                    .len() as u32,
            }
        };

        result
    }
}

// impl<F: std::clone::Clone> TryInto<ColumnSampleBatchInternal<F>> for &ColumnSampleBatch<F> {
//     type Error = String;

//     fn try_into(self) -> Result<ColumnSampleBatchInternal<F>, Self::Error> {
//         if self
//             .columns
//             .len()
//             != self
//                 .values
//                 .len()
//         {
//             return Err(format!(
//                 "Error in ColumnSampleBatch: columns.len() != values.len(), {} != {}",
//                 self.columns
//                     .len(),
//                 self.values
//                     .len()
//             ));
//         }

//         let result = unsafe {
//             ColumnSampleBatchInternal {
//                 point: self
//                     .point
//                     .clone(),
//                 columns: HostSlice::from_slice(
//                     self.columns
//                         .as_slice(),
//                 )
//                 .as_ptr(),
//                 values: HostSlice::from_slice(
//                     self.values
//                         .as_slice(),
//                 )
//                 .as_ptr(),
//                 size: self.values.len() as u32,
//             }
//         };

//         Ok(result)
//     }
// }

impl<'a> Default for QuotientConfig<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> QuotientConfig<'a> {
    pub fn default_for_device(device_id: usize) -> Self {
        QuotientConfig {
            ctx: DeviceContext::default_for_device(device_id),
            are_columns_on_device: false,
            are_sample_points_on_device: false,
            are_results_on_device: false,
            is_async: false,
        }
    }
}

fn check_quotient_args<'a, F: std::clone::Clone, S>(
    columns: &(impl HostOrDeviceSlice<S> + ?Sized),
    domain_size: usize,
    sample_batches: &(impl HostOrDeviceSlice<ColumnSampleBatchInternal<F>> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &QuotientConfig<'a>,
) -> QuotientConfig<'a> {
    if columns.len() % domain_size != 0 {
        panic!(
            "Number of total column elements is not divisible by domain size; {} % {} = {}",
            columns.len(),
            domain_size,
            columns.len() % domain_size,
        );
    }

    let ctx_device_id = cfg
        .ctx
        .device_id;

    if let Some(device_id) = columns.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in columns and context are different"
        );
    }
    if let Some(device_id) = sample_batches.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in sample_batches and context are different"
        );
    }
    if let Some(device_id) = result.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in result and context are different"
        );
    }
    check_device(ctx_device_id);

    let mut res_cfg = cfg.clone();
    res_cfg.are_columns_on_device = columns.is_on_device();
    res_cfg.are_sample_points_on_device = sample_batches.is_on_device();
    res_cfg.are_results_on_device = result.is_on_device();
    res_cfg
}

pub fn accumulate_quotients(
    half_coset_initial_index: u32,
    half_coset_step_size: u32,
    domain_log_size: u32,
    columns: &(impl HostOrDeviceSlice<ScalarField> + ?Sized),
    random_coef: QuarticExtensionField,
    samples: &(impl HostOrDeviceSlice<ColumnSampleBatchInternal<QuarticExtensionField>> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<QuarticExtensionField> + ?Sized),
    flattened_line_coeffs_size: u32,
    cfg: &QuotientConfig,
) -> IcicleResult<()> {
    let cfg = check_quotient_args(columns, 1 << domain_log_size, samples, result, cfg);
    unsafe {
        _quotient::accumulate_quotients(
            half_coset_initial_index,
            half_coset_step_size,
            domain_log_size,
            columns.as_ptr(),
            (columns.len() / (1 << domain_log_size)) as u32,
            &random_coef as *const QuarticExtensionField,
            samples.as_ptr(),
            samples.len() as u32,
            flattened_line_coeffs_size,
            &cfg as *const QuotientConfig,
            result.as_mut_ptr(),
        )
        .wrap()
    }
}

pub fn accumulate_quotients_wrapped(
    half_coset_initial_index: u32,
    half_coset_step_size: u32,
    domain_log_size: u32,
    columns: &(impl HostOrDeviceSlice<ScalarField> + ?Sized),
    random_coef: QuarticExtensionField,
    samples: &[ColumnSampleBatch<QuarticExtensionField>],
    result: &mut (impl HostOrDeviceSlice<QuarticExtensionField> + ?Sized),
    cfg: &QuotientConfig,
) -> IcicleResult<()> {
    let internal_samples: Vec<ColumnSampleBatchInternal<QuarticExtensionField>> = samples
        .iter()
        .map(|x| x.unpack()) // Perform the TryInto conversion
        .collect();
    let flattened_line_coeffs_size: u32 = internal_samples
        .iter()
        .map(|x| x.size)
        .sum();
    println!("{:?}", internal_samples);
    accumulate_quotients(
        half_coset_initial_index,
        half_coset_step_size,
        domain_log_size,
        columns,
        random_coef,
        HostSlice::from_slice(&internal_samples),
        result,
        flattened_line_coeffs_size * 3,
        cfg,
    )
}

mod _quotient {
    use super::{ColumnSampleBatchInternal, CudaError, QuarticExtensionField, QuotientConfig, ScalarField};

    extern "C" {
        #[link_name = "m31_accumulate_quotients"]
        pub(crate) fn accumulate_quotients(
            half_coset_initial_index: u32,
            half_coset_step_size: u32,
            domain_log_size: u32,
            columns: *const ScalarField, // 2d number_of_columns * domain_size elements
            number_of_columns: u32,
            random_coefficient: *const QuarticExtensionField,
            samples: *const ColumnSampleBatchInternal<QuarticExtensionField>,
            sample_size: u32,
            flattened_line_coeffs_size: u32,
            cfg: *const QuotientConfig,
            result: *mut QuarticExtensionField,
        ) -> CudaError;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use icicle_core::traits::FieldImpl;

    #[test]
    fn test_quotients() {
        let half_coset_initial_index = 4194304;
        let half_coset_step_size = 16777216;
        let domain_log_size = 8;
        let columns_raw: Vec<ScalarField> = [
            1062746970, 1062779738, 93332098, 93364866, 284850447, 284883215, 2077028039, 2077060807, 2128136358,
            2128169126, 1900284744, 1900317512, 965928078, 965960846, 1634365123, 1634397891, 1059248251, 1059281019,
            1148138604, 1148171372, 2078782346, 2078815114, 1099245893, 1099278661, 1747310760, 1747343528, 72416711,
            72449479, 1619004176, 1619036944, 1946839808, 1946872576, 1230530610, 1230563378, 989329054, 989361822,
            2122406601, 2122439369, 1895560097, 1895592865, 956189727, 956222495, 522674687, 522707455, 738013901,
            738046669, 1987263360, 1987296128, 1219773718, 1219806486, 573570373, 573603141, 1927222976, 1927255744,
            1527622654, 1527655422, 639973013, 640005781, 1050034236, 1050067004, 92940447, 92973215, 110740291,
            110773059, 248648427, 248681195, 2137790649, 2137823417, 153975124, 154007892, 1846050924, 1846083692,
            924845847, 924878615, 575207283, 575240051, 2085063688, 2085096456, 1340698521, 1340731289, 1824221096,
            1824253864, 482176850, 482209618, 1561878562, 1561911330, 21881100, 21913868, 1557006499, 1557039267,
            1028957900, 1028990668, 954999593, 955032361, 1629028340, 1629061108, 376639012, 376671780, 1282307717,
            1282340485, 424633341, 424666109, 1258534123, 1258566891, 623175279, 623208047, 982151189, 982183957,
            1890336947, 1890369715, 1031524946, 1031557714, 1485947979, 1485980747, 2081510961, 2081543729, 1953675576,
            1953708344, 1395433763, 1395466531, 2085058118, 2085090886, 204373478, 204406246, 1668012958, 1668045726,
            1691116839, 1691149607, 587613431, 587580663, 1151144943, 1151112175, 127913026, 127880258, 1111529921,
            1111497153, 1002448000, 1002415232, 11137973, 11105205, 1995503144, 1995470376, 267622468, 267589700,
            1504865391, 1504832623, 853664182, 853631414, 1535461240, 1535428472, 172636603, 172603835, 1361835081,
            1361802313, 502626939, 502594171, 1796492183, 1796459415, 621096119, 621063351, 1048373892, 1048341124,
            1519587456, 1519554688, 1661094027, 1661061259, 774371362, 774338594, 1444540394, 1444507626, 1452806233,
            1452773465, 1249874958, 1249842190, 1058420247, 1058387479, 1903057295, 1903024527, 259720761, 259687993,
            1073783660, 1073750892, 1078388020, 1078355252, 79521857, 79489089, 963766845, 963734077, 1034799072,
            1034766304, 1714700794, 1714668026, 1233923633, 1233890865, 1839747196, 1839714428, 2098110105, 2098077337,
            1909389464, 1909356696, 2131996862, 2131964094, 1022249416, 1022216648, 1273649657, 1273616889, 1344690047,
            1344657279, 264392285, 264359517, 1499621640, 1499588872, 227675351, 227642583, 1714633095, 1714600327,
            2096465385, 2096432617, 221891518, 221858750, 1692972055, 1692939287, 628377223, 628344455, 1438178962,
            1438146194, 1804197682, 1804164914, 753056173, 753023405, 870658640, 870625872, 551566039, 551533271,
            1777937940, 1777905172, 54581589, 54548821, 681320678, 681287910, 1064240961, 1064208193, 1317642715,
            1317609947, 783373928, 783341160, 647341412, 647308644, 1145868462, 1145835694, 42467255, 42434487,
            1585507399, 1585474631, 82407060, 82374292, 2125493940, 2125559476, 186664196, 186729732, 569700894,
            569766430, 2006572431, 2006637967, 2108789069, 2108854605, 1653085841, 1653151377, 1931856156, 1931921692,
            1121246599, 1121312135, 2118496502, 2118562038, 148793561, 148859097, 2010081045, 2010146581, 51008139,
            51073675, 1347137873, 1347203409, 144833422, 144898958, 1090524705, 1090590241, 1746195969, 1746261505,
            313577573, 313643109, 1978658108, 1978723644, 2097329555, 2097395091, 1643636547, 1643702083, 1912379454,
            1912444990, 1045349374, 1045414910, 1476027802, 1476093338, 1827043073, 1827108609, 292063789, 292129325,
            1147140746, 1147206282, 1706962305, 1707027841, 907761661, 907827197, 1279946026, 1280011562, 2100068472,
            2100134008, 185880894, 185946430, 221480582, 221546118, 497296854, 497362390, 2128097651, 2128163187,
            307950248, 308015784, 1544618201, 1544683737, 1849691694, 1849757230, 1150414566, 1150480102, 2022643729,
            2022709265, 533913395, 533978931, 1500958545, 1501024081, 964353700, 964419236, 976273477, 976339013,
            43762200, 43827736, 966529351, 966594887, 2057915800, 2057981336, 1909999186, 1910064722, 1110573033,
            1110638569, 753278024, 753343560, 417131787, 417197323, 849266682, 849332218, 369584599, 369650135,
            1246350558, 1246416094, 1964302378, 1964367914, 1633190247, 1633255783, 2063049892, 2063115428, 824412311,
            824477847, 2015538275, 2015603811, 1759867505, 1759933041, 643383879, 643449415, 2022632589, 2022698125,
            408746956, 408812492, 1188542269, 1188607805, 1234750031, 1234815567, 1175226862, 1175161326, 154806239,
            154740703, 255826052, 255760516, 75576195, 75510659, 2004896000, 2004830464, 22275946, 22210410,
            1843522641, 1843457105, 535244936, 535179400, 862247135, 862181599, 1707328364, 1707262828, 923438833,
            923373297, 345273206, 345207670, 576186515, 576120979, 1005253878, 1005188342, 1445500719, 1445435183,
            1242192238, 1242126702, 2096747784, 2096682248, 891691265, 891625729, 1174704407, 1174638871, 1548742724,
            1548677188, 741597141, 741531605, 758128819, 758063283, 352266269, 352200733, 2116840494, 2116774958,
            1658630943, 1658565407, 519441522, 519375986, 83673, 18137, 9292393, 9226857, 159043714, 158978178,
            1927533690, 1927468154, 2069598144, 2069532608, 1281917941, 1281852405, 320363619, 320298083, 1532010745,
            1531945209, 2048736563, 2048671027, 1671295281, 1671229745, 2116510077, 2116444541, 2044498832, 2044433296,
            399815667, 399750131, 541896447, 541830911, 528784570, 528719034, 851759633, 851694097, 455350702,
            455285166, 1281782543, 1281717007, 2045447123, 2045381587, 443783036, 443717500, 1238460463, 1238394927,
            1256754446, 1256688910, 728874277, 728808741, 1460911717, 1460846181, 1506112346, 1506046810, 1741317280,
            1741251744, 1103132078, 1103066542, 1408392233, 1408326697, 109163178, 109097642, 1362641356, 1362575820,
            2128481922, 2128416386, 487801783, 487736247, 1566747856, 1566682320, 1294682824, 1294617288, 144253277,
            144187741, 84934510, 84868974, 1023531151, 1023465615, 164814120, 164748584,
        ]
        .iter()
        .map(|&x| ScalarField::from_u32(x))
        .collect();
        let columns = HostSlice::from_slice(&columns_raw);
        let rand_coef = QuarticExtensionField::from([1, 2, 3, 4]);
        let a = QuarticExtensionField::from([1133255250, 1096955590, 1717117068, 1572094903]);
        let b = QuarticExtensionField::from([119026853, 46427533, 1286750489, 996706159]);
        let indexes = vec![0, 1];
        let x = QuarticExtensionField::from([1, 0, 478637715, 513582971]);
        let y = QuarticExtensionField::from([992285211, 649143431, 740191619, 1186584352]);
        let point = SecurePoint {
            x: x.clone(),
            y: y.clone(),
        };
        let sample = ColumnSampleBatch {
            point,
            columns: indexes,
            values: vec![a, b],
        };
        let samples = vec![sample.clone(), sample.clone(), sample.clone()];
        println!("{:?}", &samples);
        let sample_im = [sample.unpack()];
        let sample_host = HostSlice::from_slice(&sample_im);
        println!("{:?}", sample_host);
        let mut result_raw = vec![QuarticExtensionField::zero(); 1 << 8];
        let result = HostSlice::from_mut_slice(result_raw.as_mut_slice());
        let cfg = QuotientConfig::default();
        let err = accumulate_quotients_wrapped(
            half_coset_initial_index,
            half_coset_step_size,
            domain_log_size,
            columns,
            rand_coef,
            &samples,
            result,
            &cfg,
        );
        assert!(err.is_ok());

        let golden_data = vec![
            QuarticExtensionField::from([1918624470, 761801338, 1494256289, 1749520612]),
            QuarticExtensionField::from([922033229, 1825710584, 1196209818, 629325460]),
            QuarticExtensionField::from([1073507565, 1691030021, 710430947, 828215691]),
            QuarticExtensionField::from([1003999890, 1182932600, 1160808680, 415990961]),
            QuarticExtensionField::from([62827172, 1154279940, 1777519446, 513592608]),
            QuarticExtensionField::from([575102475, 778935594, 853051860, 341407268]),
            QuarticExtensionField::from([1131038993, 1990034450, 1704184106, 1663696261]),
            QuarticExtensionField::from([940882507, 1131146637, 537282075, 98438852]),
            QuarticExtensionField::from([2134103599, 848126467, 664536756, 900504400]),
            QuarticExtensionField::from([1329786049, 855775657, 1290058638, 1547841857]),
            QuarticExtensionField::from([794268678, 311906948, 1146071044, 145305186]),
            QuarticExtensionField::from([622077128, 1260713863, 379706527, 592174090]),
            QuarticExtensionField::from([529903262, 1862572804, 1061267862, 1437047477]),
            QuarticExtensionField::from([1135441119, 349251075, 1993304517, 1901408921]),
            QuarticExtensionField::from([858539336, 624778635, 451239254, 902290806]),
            QuarticExtensionField::from([1186117315, 507128465, 687191535, 763380103]),
            QuarticExtensionField::from([448436975, 1071435175, 451186170, 1656336640]),
            QuarticExtensionField::from([1353838953, 777882060, 718853998, 1544871658]),
            QuarticExtensionField::from([740351701, 1200228429, 1611339294, 1616149811]),
            QuarticExtensionField::from([1792450270, 15116989, 1457672514, 393276787]),
            QuarticExtensionField::from([1837308454, 1442821966, 738731825, 1107690881]),
            QuarticExtensionField::from([1168063342, 1131341461, 1626777988, 1803436356]),
            QuarticExtensionField::from([680571947, 2000296274, 1341014875, 1671989804]),
            QuarticExtensionField::from([1224103421, 896585657, 808739098, 2090776473]),
            QuarticExtensionField::from([1818956226, 2001780992, 70379039, 1422919145]),
            QuarticExtensionField::from([346285248, 1265556758, 1550081566, 283362944]),
            QuarticExtensionField::from([1290936588, 423124762, 1592717468, 340030269]),
            QuarticExtensionField::from([853208020, 557451324, 1606480218, 476779417]),
            QuarticExtensionField::from([2045973427, 793565317, 1163134047, 1790506200]),
            QuarticExtensionField::from([2088373238, 1443247396, 1867433881, 1543363799]),
            QuarticExtensionField::from([887929693, 395637333, 567215855, 671226791]),
            QuarticExtensionField::from([273634458, 1477679831, 600009663, 1595758220]),
            QuarticExtensionField::from([500984329, 301400481, 2056579665, 1394543627]),
            QuarticExtensionField::from([629402281, 1709888242, 1503571803, 1307381680]),
            QuarticExtensionField::from([2144207031, 1530325865, 664648354, 134888169]),
            QuarticExtensionField::from([1782435455, 713285949, 1080195042, 178633824]),
            QuarticExtensionField::from([123374628, 127050538, 1022287598, 853476654]),
            QuarticExtensionField::from([813596144, 1042053111, 187764717, 1455603884]),
            QuarticExtensionField::from([432036563, 169542615, 1073812830, 1938643469]),
            QuarticExtensionField::from([1127889510, 349973725, 1422246731, 171515583]),
            QuarticExtensionField::from([159069933, 1813638929, 1974832614, 1255165846]),
            QuarticExtensionField::from([1176906141, 1112420106, 1188938690, 490055043]),
            QuarticExtensionField::from([414858147, 1354430805, 1903795379, 222857400]),
            QuarticExtensionField::from([1999080766, 166421759, 829837661, 1480145200]),
            QuarticExtensionField::from([1447073991, 855568350, 960360065, 1101863355]),
            QuarticExtensionField::from([1874613489, 629931165, 228831886, 1470020872]),
            QuarticExtensionField::from([711441840, 957888201, 1686711150, 663730387]),
            QuarticExtensionField::from([818074220, 1377018174, 1917739813, 823894031]),
            QuarticExtensionField::from([2021245544, 1903082893, 1871020011, 37185117]),
            QuarticExtensionField::from([491846700, 669469125, 566073531, 1341764123]),
            QuarticExtensionField::from([756123054, 1649392270, 892929449, 363674610]),
            QuarticExtensionField::from([1716200440, 2073486359, 55506147, 1709213263]),
            QuarticExtensionField::from([1721298341, 1269571575, 52943677, 1762250658]),
            QuarticExtensionField::from([1385100007, 1733256693, 410358959, 1655402958]),
            QuarticExtensionField::from([521448438, 1183179525, 2045287079, 1825803082]),
            QuarticExtensionField::from([1077499965, 1785070180, 529354150, 621593450]),
            QuarticExtensionField::from([990355074, 1862937657, 686853561, 1726477143]),
            QuarticExtensionField::from([1620130666, 743749211, 2050996325, 1888681169]),
            QuarticExtensionField::from([1265415080, 657368192, 1496231834, 2043623949]),
            QuarticExtensionField::from([1561677403, 1911262900, 1007991563, 320005898]),
            QuarticExtensionField::from([1405908338, 1566095915, 729345173, 1317429087]),
            QuarticExtensionField::from([1874541053, 993508276, 757636156, 1937950438]),
            QuarticExtensionField::from([1749603964, 1136894313, 601334362, 664360271]),
            QuarticExtensionField::from([1615424020, 1703379811, 1668980951, 1235858089]),
            QuarticExtensionField::from([7487977, 654774412, 1879912120, 265166508]),
            QuarticExtensionField::from([1002129399, 811455715, 1911253424, 1417928930]),
            QuarticExtensionField::from([2021198576, 1315810940, 1389850775, 2103325051]),
            QuarticExtensionField::from([2059425046, 1044911263, 44084118, 1825031071]),
            QuarticExtensionField::from([729761391, 1660707108, 1846424381, 1096515722]),
            QuarticExtensionField::from([1924747436, 1718929984, 1630720385, 103797521]),
            QuarticExtensionField::from([356254938, 1291454897, 1626041367, 959884056]),
            QuarticExtensionField::from([1615095910, 1847674158, 1045005667, 509508358]),
            QuarticExtensionField::from([611349264, 711551572, 1916426671, 325939611]),
            QuarticExtensionField::from([1817448508, 978304901, 228140952, 1997857583]),
            QuarticExtensionField::from([974649876, 940335108, 1502442465, 311556000]),
            QuarticExtensionField::from([1532929997, 953209120, 1049312108, 77789839]),
            QuarticExtensionField::from([1060086246, 303439976, 2029541796, 613794896]),
            QuarticExtensionField::from([207878461, 732807990, 1110490152, 1130577856]),
            QuarticExtensionField::from([24937716, 472179981, 447368847, 864196738]),
            QuarticExtensionField::from([2098518775, 597066049, 1347835525, 1562068805]),
            QuarticExtensionField::from([1604218553, 1433234473, 841047727, 164679216]),
            QuarticExtensionField::from([1969194966, 836795591, 1304851056, 1726159421]),
            QuarticExtensionField::from([1076130579, 999679476, 25989108, 512568329]),
            QuarticExtensionField::from([145969963, 1284191571, 153950318, 1927743226]),
            QuarticExtensionField::from([1553282235, 1008381631, 403805694, 53755324]),
            QuarticExtensionField::from([1423203590, 457115564, 1793520391, 1934983370]),
            QuarticExtensionField::from([941740115, 1411454695, 510279153, 465825514]),
            QuarticExtensionField::from([1632606480, 586506057, 310600359, 1999038657]),
            QuarticExtensionField::from([1344443036, 1574235860, 2048999351, 1400663695]),
            QuarticExtensionField::from([1628081103, 129690697, 688385016, 698152579]),
            QuarticExtensionField::from([385329647, 400246189, 1795017190, 1936926076]),
            QuarticExtensionField::from([387747889, 2140151816, 1931125256, 1531843699]),
            QuarticExtensionField::from([706794720, 154078363, 1707018697, 566486900]),
            QuarticExtensionField::from([1955029847, 302044995, 73319760, 169269004]),
            QuarticExtensionField::from([506994051, 289044613, 96787316, 1628667691]),
            QuarticExtensionField::from([953325758, 827126510, 541637873, 1624824843]),
            QuarticExtensionField::from([1619106863, 1890775957, 63999051, 1742789123]),
            QuarticExtensionField::from([1255122992, 598747135, 15888593, 1878692726]),
            QuarticExtensionField::from([524933342, 1368369659, 229437610, 23025917]),
            QuarticExtensionField::from([853114019, 1500001544, 1947991273, 1262409773]),
            QuarticExtensionField::from([831359653, 1681942920, 1635609199, 517599283]),
            QuarticExtensionField::from([611007140, 1664135082, 516916846, 1038968992]),
            QuarticExtensionField::from([504943525, 1005034451, 1499096173, 2058702794]),
            QuarticExtensionField::from([703928168, 1317745119, 39546288, 329695029]),
            QuarticExtensionField::from([1330769350, 1887664401, 1665494088, 2083915673]),
            QuarticExtensionField::from([1181011727, 496642509, 962799755, 458673628]),
            QuarticExtensionField::from([1074482158, 1318151565, 1209754409, 602484554]),
            QuarticExtensionField::from([1159252440, 531220567, 806047230, 459724481]),
            QuarticExtensionField::from([468542874, 1659518076, 549224746, 1786345403]),
            QuarticExtensionField::from([1031440272, 360128603, 419296742, 833160968]),
            QuarticExtensionField::from([1620313074, 347674600, 212782902, 651001460]),
            QuarticExtensionField::from([1786888599, 1761108641, 341730607, 1373172220]),
            QuarticExtensionField::from([27039022, 524742534, 814792345, 127691129]),
            QuarticExtensionField::from([1601568719, 1375387708, 1112790700, 200409820]),
            QuarticExtensionField::from([1544541755, 851178479, 1736845593, 320044822]),
            QuarticExtensionField::from([1439190735, 1586977663, 62722332, 1133681444]),
            QuarticExtensionField::from([601111425, 341499451, 1137453179, 885463894]),
            QuarticExtensionField::from([1656554330, 325054530, 117854124, 441702830]),
            QuarticExtensionField::from([42102689, 286088642, 952874687, 2145799399]),
            QuarticExtensionField::from([1578867646, 1863298905, 572180143, 550370015]),
            QuarticExtensionField::from([2114631308, 54570592, 959652233, 1347303826]),
            QuarticExtensionField::from([554913781, 272386993, 1985923679, 875943852]),
            QuarticExtensionField::from([1424435281, 887238006, 1870024845, 1496347261]),
            QuarticExtensionField::from([307147512, 362458382, 1436017488, 268708203]),
            QuarticExtensionField::from([1495708123, 1122668534, 149766476, 236957968]),
            QuarticExtensionField::from([139805602, 1254996531, 1424943280, 1069407759]),
            QuarticExtensionField::from([1264862582, 1262913776, 138139662, 643656826]),
            QuarticExtensionField::from([1303204475, 1186043635, 1899594205, 1288950539]),
            QuarticExtensionField::from([877594892, 1638281480, 2029396066, 377743645]),
            QuarticExtensionField::from([1821092124, 326810630, 1415720172, 50453286]),
            QuarticExtensionField::from([1946062643, 548995668, 871336575, 1307728760]),
            QuarticExtensionField::from([835752578, 173336400, 727472941, 1150223664]),
            QuarticExtensionField::from([1900208420, 1819528405, 1074366416, 2038270456]),
            QuarticExtensionField::from([275152789, 835175741, 109790318, 2119512393]),
            QuarticExtensionField::from([920061589, 599928663, 1377012175, 1860905991]),
            QuarticExtensionField::from([1552493810, 992250247, 642176955, 864542730]),
            QuarticExtensionField::from([179972512, 441327643, 1609308498, 1779914860]),
            QuarticExtensionField::from([1538136745, 399621182, 1033559052, 1584442957]),
            QuarticExtensionField::from([765541557, 1801069699, 764060118, 125581587]),
            QuarticExtensionField::from([1733795677, 305502848, 686942290, 1551244029]),
            QuarticExtensionField::from([176815448, 1736471542, 1492336822, 324383831]),
            QuarticExtensionField::from([1963235463, 215994837, 523244192, 1880840234]),
            QuarticExtensionField::from([1765510882, 452783068, 1948807934, 403904878]),
            QuarticExtensionField::from([631045078, 1530893872, 1869554695, 843321112]),
            QuarticExtensionField::from([1853049444, 899980558, 746565841, 1123803537]),
            QuarticExtensionField::from([970917397, 562179796, 1763004912, 956816550]),
            QuarticExtensionField::from([508277609, 498852363, 1910433806, 1503192390]),
            QuarticExtensionField::from([1986292742, 2010191043, 1253346262, 513901733]),
            QuarticExtensionField::from([490650176, 1060833994, 1172500488, 1701871715]),
            QuarticExtensionField::from([49554586, 1511360450, 1419156693, 1257750595]),
            QuarticExtensionField::from([797245405, 1128572452, 1788208931, 972281966]),
            QuarticExtensionField::from([1168699463, 58680411, 1832036717, 1886611600]),
            QuarticExtensionField::from([693018562, 1236257595, 436708491, 1405275070]),
            QuarticExtensionField::from([298055391, 1186047691, 1662227636, 74150862]),
            QuarticExtensionField::from([1818316773, 1638470581, 873716020, 2094394026]),
            QuarticExtensionField::from([1644066393, 1454612100, 819939998, 124089541]),
            QuarticExtensionField::from([1961645908, 950451518, 1607874430, 342777922]),
            QuarticExtensionField::from([1513498599, 1368920355, 744271559, 288201264]),
            QuarticExtensionField::from([1080496030, 151283737, 27585998, 1161080485]),
            QuarticExtensionField::from([1048466512, 1910918717, 1948576034, 1506966942]),
            QuarticExtensionField::from([2035478663, 154529039, 1810622493, 1956773457]),
            QuarticExtensionField::from([1360804866, 968452724, 637801132, 797436491]),
            QuarticExtensionField::from([651814562, 665940034, 942144263, 279532022]),
            QuarticExtensionField::from([1705839731, 1323313271, 1267360167, 52769292]),
            QuarticExtensionField::from([1966094340, 98697907, 1052090510, 329111666]),
            QuarticExtensionField::from([1616685967, 1401469342, 1485323049, 2130413924]),
            QuarticExtensionField::from([129000096, 139701208, 1089989756, 953306362]),
            QuarticExtensionField::from([1853303056, 797351937, 941912951, 784558348]),
            QuarticExtensionField::from([855202225, 1302064280, 1937145831, 974665314]),
            QuarticExtensionField::from([1608904915, 1821919482, 942100369, 1898005361]),
            QuarticExtensionField::from([1972333512, 122370364, 641210489, 1620052348]),
            QuarticExtensionField::from([715040110, 1012467976, 1368800962, 774136950]),
            QuarticExtensionField::from([1674521270, 1925175436, 1683806752, 1703329928]),
            QuarticExtensionField::from([405586049, 955216122, 395231874, 1283190097]),
            QuarticExtensionField::from([585879171, 979742359, 990296214, 659401201]),
            QuarticExtensionField::from([474329956, 522478150, 1442338092, 150770485]),
            QuarticExtensionField::from([286918005, 1449320497, 104911871, 888549696]),
            QuarticExtensionField::from([1082300105, 997479867, 1942589107, 2017873415]),
            QuarticExtensionField::from([1261755847, 1438434135, 1048322054, 1569971902]),
            QuarticExtensionField::from([1506677588, 2039732839, 1428051110, 904507734]),
            QuarticExtensionField::from([1223072319, 1144819072, 1759934323, 268760935]),
            QuarticExtensionField::from([1049482236, 1379941191, 712119671, 354887349]),
            QuarticExtensionField::from([1097722367, 460471787, 725132773, 1142004660]),
            QuarticExtensionField::from([666182744, 2024591106, 1269653270, 1489721639]),
            QuarticExtensionField::from([751022300, 1507060924, 1259220046, 186835864]),
            QuarticExtensionField::from([218972416, 729918549, 1150792576, 1271800225]),
            QuarticExtensionField::from([173756859, 1771395347, 1939801529, 1791538380]),
            QuarticExtensionField::from([420336754, 1062426754, 556495932, 460363610]),
            QuarticExtensionField::from([898476488, 281205668, 1138792738, 140830313]),
            QuarticExtensionField::from([390835881, 1550203560, 609621593, 207435049]),
            QuarticExtensionField::from([1682343017, 460640050, 1307173056, 918097326]),
            QuarticExtensionField::from([240701323, 1827211156, 10672104, 1729209499]),
            QuarticExtensionField::from([1116215327, 914176268, 641459635, 1855496762]),
            QuarticExtensionField::from([2052785472, 733255740, 318948503, 1144257786]),
            QuarticExtensionField::from([1026326720, 2026973129, 1156097768, 1193472736]),
            QuarticExtensionField::from([1333731425, 1754186243, 993351395, 1462484545]),
            QuarticExtensionField::from([1913684030, 1843585953, 789518453, 763782866]),
            QuarticExtensionField::from([1930506136, 379581355, 1516136727, 1803817838]),
            QuarticExtensionField::from([462022582, 1146946754, 887150801, 1211750447]),
            QuarticExtensionField::from([1319004921, 16830435, 2005693552, 1946012633]),
            QuarticExtensionField::from([873154094, 71888803, 478396410, 734945535]),
            QuarticExtensionField::from([482413625, 1538155645, 662153922, 1203659754]),
            QuarticExtensionField::from([2076183326, 1860894582, 942103424, 1058885109]),
            QuarticExtensionField::from([33741477, 410432622, 1952328560, 1257964203]),
            QuarticExtensionField::from([1765233261, 28509543, 1455147538, 1583241559]),
            QuarticExtensionField::from([1608015704, 874118111, 2000748243, 670329438]),
            QuarticExtensionField::from([2133473674, 366351394, 881775286, 2122409662]),
            QuarticExtensionField::from([574230943, 273768373, 2020015724, 1208153365]),
            QuarticExtensionField::from([436212495, 887323261, 1719082382, 182192]),
            QuarticExtensionField::from([1324223846, 1708556934, 424053431, 422312292]),
            QuarticExtensionField::from([1902244124, 1116097208, 2128440118, 803770959]),
            QuarticExtensionField::from([1631254947, 1086593403, 816574194, 419117370]),
            QuarticExtensionField::from([267923808, 371399314, 1956513668, 1428269486]),
            QuarticExtensionField::from([867621253, 734464114, 1510675753, 2124432451]),
            QuarticExtensionField::from([262730040, 1796758823, 1146123806, 1335253369]),
            QuarticExtensionField::from([2113562587, 1758160178, 1702549217, 1120932688]),
            QuarticExtensionField::from([926410249, 751933267, 947066646, 1010721228]),
            QuarticExtensionField::from([2925084, 762775638, 1932252091, 570599496]),
            QuarticExtensionField::from([1879742862, 1405100116, 627428297, 740950556]),
            QuarticExtensionField::from([497873628, 991411203, 2035042784, 1685953604]),
            QuarticExtensionField::from([1392594946, 84734891, 239966238, 1100134303]),
            QuarticExtensionField::from([860502012, 1273822642, 265038420, 541502111]),
            QuarticExtensionField::from([520355901, 1719230052, 1202420013, 1913444366]),
            QuarticExtensionField::from([1980750654, 67834589, 870920287, 66474991]),
            QuarticExtensionField::from([1153745488, 1841553357, 1678783671, 75631742]),
            QuarticExtensionField::from([1435288216, 354253984, 387954225, 106228371]),
            QuarticExtensionField::from([674441371, 1114809215, 1324106436, 311683215]),
            QuarticExtensionField::from([1602502619, 980466350, 986124760, 380262332]),
            QuarticExtensionField::from([635513485, 12390338, 13066332, 1390722195]),
            QuarticExtensionField::from([1826540105, 2118304867, 244935298, 974036541]),
            QuarticExtensionField::from([454232627, 966726748, 710707797, 1464927550]),
            QuarticExtensionField::from([1841024618, 712137923, 1081959297, 1759032127]),
            QuarticExtensionField::from([679792130, 1023671493, 904990910, 1851944173]),
            QuarticExtensionField::from([474499131, 887948466, 489312096, 3956892]),
            QuarticExtensionField::from([986883326, 137969330, 1761079190, 1106059137]),
            QuarticExtensionField::from([216572935, 1813289598, 1464350725, 368547196]),
            QuarticExtensionField::from([226529364, 1118651629, 120798314, 197088573]),
            QuarticExtensionField::from([452178301, 1148542694, 1159014645, 1724104093]),
            QuarticExtensionField::from([1792431461, 1581540092, 329365655, 1048879093]),
            QuarticExtensionField::from([1387128414, 1478598050, 236160429, 1659700207]),
            QuarticExtensionField::from([1211862055, 40320833, 1349010637, 1340770304]),
            QuarticExtensionField::from([1030900139, 1840125030, 66586326, 7504566]),
            QuarticExtensionField::from([1445010113, 604360748, 2099791157, 625417202]),
            QuarticExtensionField::from([2060595769, 1186048786, 1032629410, 1719277924]),
            QuarticExtensionField::from([1844918201, 275390495, 1750147095, 146657454]),
            QuarticExtensionField::from([373437184, 1662977908, 1223596031, 1593381642]),
            QuarticExtensionField::from([750157699, 375579794, 670924608, 1901927063]),
            QuarticExtensionField::from([431803923, 811993642, 922844133, 755127623]),
            QuarticExtensionField::from([100298373, 1940790941, 623431004, 859518984]),
            QuarticExtensionField::from([1465132695, 188583724, 396643333, 1045134065]),
            QuarticExtensionField::from([239286196, 948263895, 109085955, 1667415274]),
            QuarticExtensionField::from([1908257490, 1338248462, 1655620848, 338896255]),
            QuarticExtensionField::from([200498478, 98606084, 1381583592, 1583308823]),
            QuarticExtensionField::from([2045763890, 754871725, 1719995281, 704276593]),
            QuarticExtensionField::from([828781275, 455309901, 2117204438, 1622508949]),
            QuarticExtensionField::from([2090190726, 1125421132, 302519262, 1914456113]),
        ];
        assert_eq!(golden_data, result_raw);
    }
}
