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

pub fn to_internal_column_batch<F: std::clone::Clone>(samples: &[ColumnSampleBatch<F>]) -> Vec<ColumnSampleBatchInternal<F>> {
    samples.iter().map(|x| x.unpack()).collect()
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

fn check_quotient_args<'a, QF: std::clone::Clone, S, F>(
    columns: &(impl HostOrDeviceSlice<S> + ?Sized),
    sample_batches: &(impl HostOrDeviceSlice<ColumnSampleBatchInternal<QF>> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &QuotientConfig<'a>,
) -> QuotientConfig<'a> {
    // if columns.len() % domain_size != 0 {
    //     panic!(
    //         "Number of total column elements is not divisible by domain size; {} % {} = {}",
    //         columns.len(),
    //         domain_size,
    //         columns.len() % domain_size,
    //     );
    // }

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
    domain_log_size: u32,
    columns: &(impl HostOrDeviceSlice<*const ScalarField> + ?Sized),
    random_coef: QuarticExtensionField,
    samples: &(impl HostOrDeviceSlice<ColumnSampleBatchInternal<QuarticExtensionField>> + ?Sized),
    result1: &mut (impl HostOrDeviceSlice<ScalarField> + ?Sized),
    result2: &mut (impl HostOrDeviceSlice<ScalarField> + ?Sized),
    result3: &mut (impl HostOrDeviceSlice<ScalarField> + ?Sized),
    result4: &mut (impl HostOrDeviceSlice<ScalarField> + ?Sized),
    flattened_line_coeffs_size: u32,
    cfg: &QuotientConfig,
) -> IcicleResult<()> {
    let cfg = check_quotient_args(columns, samples, result1, cfg);
    unsafe {
        _quotient::accumulate_quotients(
            domain_log_size,
            columns.as_ptr() as *const *const ScalarField,
            columns.len() as u32,
            &random_coef as *const QuarticExtensionField,
            samples.as_ptr(),
            samples.len() as u32,
            flattened_line_coeffs_size,
            &cfg as *const QuotientConfig,
            result1.as_mut_ptr(),
            result2.as_mut_ptr(),
            result3.as_mut_ptr(),
            result4.as_mut_ptr(),
        )
        .wrap()
    }
}

pub fn accumulate_quotients_wrapped(
    domain_log_size: u32,
    columns: &(impl HostOrDeviceSlice<*const ScalarField> + ?Sized),
    random_coef: QuarticExtensionField,
    internal_samples: &[ColumnSampleBatchInternal<QuarticExtensionField>],
    result1: &mut (impl HostOrDeviceSlice<ScalarField> + ?Sized),
    result2: &mut (impl HostOrDeviceSlice<ScalarField> + ?Sized),
    result3: &mut (impl HostOrDeviceSlice<ScalarField> + ?Sized),
    result4: &mut (impl HostOrDeviceSlice<ScalarField> + ?Sized),
    cfg: &QuotientConfig,
) -> IcicleResult<()> {
    let flattened_line_coeffs_size: u32 = internal_samples
        .iter()
        .map(|x| x.size)
        .sum();
    accumulate_quotients(
        domain_log_size,
        columns,
        random_coef,
        HostSlice::from_slice(&internal_samples),
        result1,
        result2,
        result3,
        result4,
        flattened_line_coeffs_size * 3,
        cfg,
    )
}

mod _quotient {
    use super::{ColumnSampleBatchInternal, CudaError, QuarticExtensionField, QuotientConfig, ScalarField};

    extern "C" {
        #[link_name = "m31_accumulate_quotients"]
        pub(crate) fn accumulate_quotients(
            domain_log_size: u32,
            columns: *const *const ScalarField, // 2d number_of_columns * domain_size elements
            number_of_columns: u32,
            random_coefficient: *const QuarticExtensionField,
            samples: *const ColumnSampleBatchInternal<QuarticExtensionField>,
            sample_size: u32,
            flattened_line_coeffs_size: u32,
            cfg: *const QuotientConfig,
            result1: *mut ScalarField,
            result2: *mut ScalarField,
            result3: *mut ScalarField,
            result4: *mut ScalarField,
        ) -> CudaError;
    }
}

#[cfg(test)]
pub(crate) mod tests {


    use super::*;
    use icicle_core::traits::FieldImpl;
    use icicle_cuda_runtime::memory::DeviceVec;

    #[test]
    fn test_quotients() {
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

        let columns_chunked: Vec<_> = columns_raw.chunks(256).collect();
        let mut ptr_columns: Vec<DeviceVec<ScalarField>> = Vec::with_capacity(columns_chunked.len());
        for (i, chunk) in columns_chunked.iter().enumerate() {
            ptr_columns.push(DeviceVec::cuda_malloc(256).unwrap());

            ptr_columns[i].copy_from_host(HostSlice::from_slice(chunk)).unwrap();
        }
        let column_final: Vec<*const ScalarField> = ptr_columns.iter().map(|x| unsafe { x.as_ptr() }).collect();

        let columns = HostSlice::from_slice(&column_final);
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
        let internal_samples = to_internal_column_batch(&samples);
        
        let mut result_raw1 = vec![ScalarField::zero(); 1 << 8];
        let result1 = HostSlice::from_mut_slice(result_raw1.as_mut_slice());
        let mut result_raw2 = vec![ScalarField::zero(); 1 << 8];
        let result2 = HostSlice::from_mut_slice(result_raw2.as_mut_slice());
        let mut result_raw3 = vec![ScalarField::zero(); 1 << 8];
        let result3 = HostSlice::from_mut_slice(result_raw3.as_mut_slice());
        let mut result_raw4 = vec![ScalarField::zero(); 1 << 8];
        let result4 = HostSlice::from_mut_slice(result_raw4.as_mut_slice());

        let cfg = QuotientConfig::default();
        let err = accumulate_quotients_wrapped(domain_log_size, columns, rand_coef, &internal_samples, result1, result2, result3, result4, &cfg);
        assert!(err.is_ok());

        let result: Vec<ScalarField> = result_raw1.iter()
        .zip(result_raw2.iter())
        .zip(result_raw3.iter())
        .zip(result_raw4.iter())
        .flat_map(|(((a, b), c), d)| vec![*a, *b, *c, *d])
        .collect();

        let golden_data = vec![
            ScalarField::from_u32(1918624470), ScalarField::from_u32(761801338), ScalarField::from_u32(1494256289), ScalarField::from_u32(1749520612),
            ScalarField::from_u32(922033229), ScalarField::from_u32(1825710584), ScalarField::from_u32(1196209818), ScalarField::from_u32(629325460),
            ScalarField::from_u32(1073507565), ScalarField::from_u32(1691030021), ScalarField::from_u32(710430947), ScalarField::from_u32(828215691),
            ScalarField::from_u32(1003999890), ScalarField::from_u32(1182932600), ScalarField::from_u32(1160808680), ScalarField::from_u32(415990961),
            ScalarField::from_u32(62827172), ScalarField::from_u32(1154279940), ScalarField::from_u32(1777519446), ScalarField::from_u32(513592608),
            ScalarField::from_u32(575102475), ScalarField::from_u32(778935594), ScalarField::from_u32(853051860), ScalarField::from_u32(341407268),
            ScalarField::from_u32(1131038993), ScalarField::from_u32(1990034450), ScalarField::from_u32(1704184106), ScalarField::from_u32(1663696261),
            ScalarField::from_u32(940882507), ScalarField::from_u32(1131146637), ScalarField::from_u32(537282075), ScalarField::from_u32(98438852),
            ScalarField::from_u32(2134103599), ScalarField::from_u32(848126467), ScalarField::from_u32(664536756), ScalarField::from_u32(900504400),
            ScalarField::from_u32(1329786049), ScalarField::from_u32(855775657), ScalarField::from_u32(1290058638), ScalarField::from_u32(1547841857),
            ScalarField::from_u32(794268678), ScalarField::from_u32(311906948), ScalarField::from_u32(1146071044), ScalarField::from_u32(145305186),
            ScalarField::from_u32(622077128), ScalarField::from_u32(1260713863), ScalarField::from_u32(379706527), ScalarField::from_u32(592174090),
            ScalarField::from_u32(529903262), ScalarField::from_u32(1862572804), ScalarField::from_u32(1061267862), ScalarField::from_u32(1437047477),
            ScalarField::from_u32(1135441119), ScalarField::from_u32(349251075), ScalarField::from_u32(1993304517), ScalarField::from_u32(1901408921),
            ScalarField::from_u32(858539336), ScalarField::from_u32(624778635), ScalarField::from_u32(451239254), ScalarField::from_u32(902290806),
            ScalarField::from_u32(1186117315), ScalarField::from_u32(507128465), ScalarField::from_u32(687191535), ScalarField::from_u32(763380103),
            ScalarField::from_u32(448436975), ScalarField::from_u32(1071435175), ScalarField::from_u32(451186170), ScalarField::from_u32(1656336640),
            ScalarField::from_u32(1353838953), ScalarField::from_u32(777882060), ScalarField::from_u32(718853998), ScalarField::from_u32(1544871658),
            ScalarField::from_u32(740351701), ScalarField::from_u32(1200228429), ScalarField::from_u32(1611339294), ScalarField::from_u32(1616149811),
            ScalarField::from_u32(1792450270), ScalarField::from_u32(15116989), ScalarField::from_u32(1457672514), ScalarField::from_u32(393276787),
            ScalarField::from_u32(1837308454), ScalarField::from_u32(1442821966), ScalarField::from_u32(738731825), ScalarField::from_u32(1107690881),
            ScalarField::from_u32(1168063342), ScalarField::from_u32(1131341461), ScalarField::from_u32(1626777988), ScalarField::from_u32(1803436356),
            ScalarField::from_u32(680571947), ScalarField::from_u32(2000296274), ScalarField::from_u32(1341014875), ScalarField::from_u32(1671989804),
            ScalarField::from_u32(1224103421), ScalarField::from_u32(896585657), ScalarField::from_u32(808739098), ScalarField::from_u32(2090776473),
            ScalarField::from_u32(1818956226), ScalarField::from_u32(2001780992), ScalarField::from_u32(70379039), ScalarField::from_u32(1422919145),
            ScalarField::from_u32(346285248), ScalarField::from_u32(1265556758), ScalarField::from_u32(1550081566), ScalarField::from_u32(283362944),
            ScalarField::from_u32(1290936588), ScalarField::from_u32(423124762), ScalarField::from_u32(1592717468), ScalarField::from_u32(340030269),
            ScalarField::from_u32(853208020), ScalarField::from_u32(557451324), ScalarField::from_u32(1606480218), ScalarField::from_u32(476779417),
            ScalarField::from_u32(2045973427), ScalarField::from_u32(793565317), ScalarField::from_u32(1163134047), ScalarField::from_u32(1790506200),
            ScalarField::from_u32(2088373238), ScalarField::from_u32(1443247396), ScalarField::from_u32(1867433881), ScalarField::from_u32(1543363799),
            ScalarField::from_u32(887929693), ScalarField::from_u32(395637333), ScalarField::from_u32(567215855), ScalarField::from_u32(671226791),
            ScalarField::from_u32(273634458), ScalarField::from_u32(1477679831), ScalarField::from_u32(600009663), ScalarField::from_u32(1595758220),
            ScalarField::from_u32(500984329), ScalarField::from_u32(301400481), ScalarField::from_u32(2056579665), ScalarField::from_u32(1394543627),
            ScalarField::from_u32(629402281), ScalarField::from_u32(1709888242), ScalarField::from_u32(1503571803), ScalarField::from_u32(1307381680),
            ScalarField::from_u32(2144207031), ScalarField::from_u32(1530325865), ScalarField::from_u32(664648354), ScalarField::from_u32(134888169),
            ScalarField::from_u32(1782435455), ScalarField::from_u32(713285949), ScalarField::from_u32(1080195042), ScalarField::from_u32(178633824),
            ScalarField::from_u32(123374628), ScalarField::from_u32(127050538), ScalarField::from_u32(1022287598), ScalarField::from_u32(853476654),
            ScalarField::from_u32(813596144), ScalarField::from_u32(1042053111), ScalarField::from_u32(187764717), ScalarField::from_u32(1455603884),
            ScalarField::from_u32(432036563), ScalarField::from_u32(169542615), ScalarField::from_u32(1073812830), ScalarField::from_u32(1938643469),
            ScalarField::from_u32(1127889510), ScalarField::from_u32(349973725), ScalarField::from_u32(1422246731), ScalarField::from_u32(171515583),
            ScalarField::from_u32(159069933), ScalarField::from_u32(1813638929), ScalarField::from_u32(1974832614), ScalarField::from_u32(1255165846),
            ScalarField::from_u32(1176906141), ScalarField::from_u32(1112420106), ScalarField::from_u32(1188938690), ScalarField::from_u32(490055043),
            ScalarField::from_u32(414858147), ScalarField::from_u32(1354430805), ScalarField::from_u32(1903795379), ScalarField::from_u32(222857400),
            ScalarField::from_u32(1999080766), ScalarField::from_u32(166421759), ScalarField::from_u32(829837661), ScalarField::from_u32(1480145200),
            ScalarField::from_u32(1447073991), ScalarField::from_u32(855568350), ScalarField::from_u32(960360065), ScalarField::from_u32(1101863355),
            ScalarField::from_u32(1874613489), ScalarField::from_u32(629931165), ScalarField::from_u32(228831886), ScalarField::from_u32(1470020872),
            ScalarField::from_u32(711441840), ScalarField::from_u32(957888201), ScalarField::from_u32(1686711150), ScalarField::from_u32(663730387),
            ScalarField::from_u32(818074220), ScalarField::from_u32(1377018174), ScalarField::from_u32(1917739813), ScalarField::from_u32(823894031),
            ScalarField::from_u32(2021245544), ScalarField::from_u32(1903082893), ScalarField::from_u32(1871020011), ScalarField::from_u32(37185117),
            ScalarField::from_u32(491846700), ScalarField::from_u32(669469125), ScalarField::from_u32(566073531), ScalarField::from_u32(1341764123),
            ScalarField::from_u32(756123054), ScalarField::from_u32(1649392270), ScalarField::from_u32(892929449), ScalarField::from_u32(363674610),
            ScalarField::from_u32(1716200440), ScalarField::from_u32(2073486359), ScalarField::from_u32(55506147), ScalarField::from_u32(1709213263),
            ScalarField::from_u32(1721298341), ScalarField::from_u32(1269571575), ScalarField::from_u32(52943677), ScalarField::from_u32(1762250658),
            ScalarField::from_u32(1385100007), ScalarField::from_u32(1733256693), ScalarField::from_u32(410358959), ScalarField::from_u32(1655402958),
            ScalarField::from_u32(521448438), ScalarField::from_u32(1183179525), ScalarField::from_u32(2045287079), ScalarField::from_u32(1825803082),
            ScalarField::from_u32(1077499965), ScalarField::from_u32(1785070180), ScalarField::from_u32(529354150), ScalarField::from_u32(621593450),
            ScalarField::from_u32(990355074), ScalarField::from_u32(1862937657), ScalarField::from_u32(686853561), ScalarField::from_u32(1726477143),
            ScalarField::from_u32(1620130666), ScalarField::from_u32(743749211), ScalarField::from_u32(2050996325), ScalarField::from_u32(1888681169),
            ScalarField::from_u32(1265415080), ScalarField::from_u32(657368192), ScalarField::from_u32(1496231834), ScalarField::from_u32(2043623949),
            ScalarField::from_u32(1561677403), ScalarField::from_u32(1911262900), ScalarField::from_u32(1007991563), ScalarField::from_u32(320005898),
            ScalarField::from_u32(1405908338), ScalarField::from_u32(1566095915), ScalarField::from_u32(729345173), ScalarField::from_u32(1317429087),
            ScalarField::from_u32(1874541053), ScalarField::from_u32(993508276), ScalarField::from_u32(757636156), ScalarField::from_u32(1937950438),
            ScalarField::from_u32(1749603964), ScalarField::from_u32(1136894313), ScalarField::from_u32(601334362), ScalarField::from_u32(664360271),
            ScalarField::from_u32(1615424020), ScalarField::from_u32(1703379811), ScalarField::from_u32(1668980951), ScalarField::from_u32(1235858089),
            ScalarField::from_u32(7487977), ScalarField::from_u32(654774412), ScalarField::from_u32(1879912120), ScalarField::from_u32(265166508),
            ScalarField::from_u32(1002129399), ScalarField::from_u32(811455715), ScalarField::from_u32(1911253424), ScalarField::from_u32(1417928930),
            ScalarField::from_u32(2021198576), ScalarField::from_u32(1315810940), ScalarField::from_u32(1389850775), ScalarField::from_u32(2103325051),
            ScalarField::from_u32(2059425046), ScalarField::from_u32(1044911263), ScalarField::from_u32(44084118), ScalarField::from_u32(1825031071),
            ScalarField::from_u32(729761391), ScalarField::from_u32(1660707108), ScalarField::from_u32(1846424381), ScalarField::from_u32(1096515722),
            ScalarField::from_u32(1924747436), ScalarField::from_u32(1718929984), ScalarField::from_u32(1630720385), ScalarField::from_u32(103797521),
            ScalarField::from_u32(356254938), ScalarField::from_u32(1291454897), ScalarField::from_u32(1626041367), ScalarField::from_u32(959884056),
            ScalarField::from_u32(1615095910), ScalarField::from_u32(1847674158), ScalarField::from_u32(1045005667), ScalarField::from_u32(509508358),
            ScalarField::from_u32(611349264), ScalarField::from_u32(711551572), ScalarField::from_u32(1916426671), ScalarField::from_u32(325939611),
            ScalarField::from_u32(1817448508), ScalarField::from_u32(978304901), ScalarField::from_u32(228140952), ScalarField::from_u32(1997857583),
            ScalarField::from_u32(974649876), ScalarField::from_u32(940335108), ScalarField::from_u32(1502442465), ScalarField::from_u32(311556000),
            ScalarField::from_u32(1532929997), ScalarField::from_u32(953209120), ScalarField::from_u32(1049312108), ScalarField::from_u32(77789839),
            ScalarField::from_u32(1060086246), ScalarField::from_u32(303439976), ScalarField::from_u32(2029541796), ScalarField::from_u32(613794896),
            ScalarField::from_u32(207878461), ScalarField::from_u32(732807990), ScalarField::from_u32(1110490152), ScalarField::from_u32(1130577856),
            ScalarField::from_u32(24937716), ScalarField::from_u32(472179981), ScalarField::from_u32(447368847), ScalarField::from_u32(864196738),
            ScalarField::from_u32(2098518775), ScalarField::from_u32(597066049), ScalarField::from_u32(1347835525), ScalarField::from_u32(1562068805),
            ScalarField::from_u32(1604218553), ScalarField::from_u32(1433234473), ScalarField::from_u32(841047727), ScalarField::from_u32(164679216),
            ScalarField::from_u32(1969194966), ScalarField::from_u32(836795591), ScalarField::from_u32(1304851056), ScalarField::from_u32(1726159421),
            ScalarField::from_u32(1076130579), ScalarField::from_u32(999679476), ScalarField::from_u32(25989108), ScalarField::from_u32(512568329),
            ScalarField::from_u32(145969963), ScalarField::from_u32(1284191571), ScalarField::from_u32(153950318), ScalarField::from_u32(1927743226),
            ScalarField::from_u32(1553282235), ScalarField::from_u32(1008381631), ScalarField::from_u32(403805694), ScalarField::from_u32(53755324),
            ScalarField::from_u32(1423203590), ScalarField::from_u32(457115564), ScalarField::from_u32(1793520391), ScalarField::from_u32(1934983370),
            ScalarField::from_u32(941740115), ScalarField::from_u32(1411454695), ScalarField::from_u32(510279153), ScalarField::from_u32(465825514),
            ScalarField::from_u32(1632606480), ScalarField::from_u32(586506057), ScalarField::from_u32(310600359), ScalarField::from_u32(1999038657),
            ScalarField::from_u32(1344443036), ScalarField::from_u32(1574235860), ScalarField::from_u32(2048999351), ScalarField::from_u32(1400663695),
            ScalarField::from_u32(1628081103), ScalarField::from_u32(129690697), ScalarField::from_u32(688385016), ScalarField::from_u32(698152579),
            ScalarField::from_u32(385329647), ScalarField::from_u32(400246189), ScalarField::from_u32(1795017190), ScalarField::from_u32(1936926076),
            ScalarField::from_u32(387747889), ScalarField::from_u32(2140151816), ScalarField::from_u32(1931125256), ScalarField::from_u32(1531843699),
            ScalarField::from_u32(706794720), ScalarField::from_u32(154078363), ScalarField::from_u32(1707018697), ScalarField::from_u32(566486900),
            ScalarField::from_u32(1955029847), ScalarField::from_u32(302044995), ScalarField::from_u32(73319760), ScalarField::from_u32(169269004),
            ScalarField::from_u32(506994051), ScalarField::from_u32(289044613), ScalarField::from_u32(96787316), ScalarField::from_u32(1628667691),
            ScalarField::from_u32(953325758), ScalarField::from_u32(827126510), ScalarField::from_u32(541637873), ScalarField::from_u32(1624824843),
            ScalarField::from_u32(1619106863), ScalarField::from_u32(1890775957), ScalarField::from_u32(63999051), ScalarField::from_u32(1742789123),
            ScalarField::from_u32(1255122992), ScalarField::from_u32(598747135), ScalarField::from_u32(15888593), ScalarField::from_u32(1878692726),
            ScalarField::from_u32(524933342), ScalarField::from_u32(1368369659), ScalarField::from_u32(229437610), ScalarField::from_u32(23025917),
            ScalarField::from_u32(853114019), ScalarField::from_u32(1500001544), ScalarField::from_u32(1947991273), ScalarField::from_u32(1262409773),
            ScalarField::from_u32(831359653), ScalarField::from_u32(1681942920), ScalarField::from_u32(1635609199), ScalarField::from_u32(517599283),
            ScalarField::from_u32(611007140), ScalarField::from_u32(1664135082), ScalarField::from_u32(516916846), ScalarField::from_u32(1038968992),
            ScalarField::from_u32(504943525), ScalarField::from_u32(1005034451), ScalarField::from_u32(1499096173), ScalarField::from_u32(2058702794),
            ScalarField::from_u32(703928168), ScalarField::from_u32(1317745119), ScalarField::from_u32(39546288), ScalarField::from_u32(329695029),
            ScalarField::from_u32(1330769350), ScalarField::from_u32(1887664401), ScalarField::from_u32(1665494088), ScalarField::from_u32(2083915673),
            ScalarField::from_u32(1181011727), ScalarField::from_u32(496642509), ScalarField::from_u32(962799755), ScalarField::from_u32(458673628),
            ScalarField::from_u32(1074482158), ScalarField::from_u32(1318151565), ScalarField::from_u32(1209754409), ScalarField::from_u32(602484554),
            ScalarField::from_u32(1159252440), ScalarField::from_u32(531220567), ScalarField::from_u32(806047230), ScalarField::from_u32(459724481),
            ScalarField::from_u32(468542874), ScalarField::from_u32(1659518076), ScalarField::from_u32(549224746), ScalarField::from_u32(1786345403),
            ScalarField::from_u32(1031440272), ScalarField::from_u32(360128603), ScalarField::from_u32(419296742), ScalarField::from_u32(833160968),
            ScalarField::from_u32(1620313074), ScalarField::from_u32(347674600), ScalarField::from_u32(212782902), ScalarField::from_u32(651001460),
            ScalarField::from_u32(1786888599), ScalarField::from_u32(1761108641), ScalarField::from_u32(341730607), ScalarField::from_u32(1373172220),
            ScalarField::from_u32(27039022), ScalarField::from_u32(524742534), ScalarField::from_u32(814792345), ScalarField::from_u32(127691129),
            ScalarField::from_u32(1601568719), ScalarField::from_u32(1375387708), ScalarField::from_u32(1112790700), ScalarField::from_u32(200409820),
            ScalarField::from_u32(1544541755), ScalarField::from_u32(851178479), ScalarField::from_u32(1736845593), ScalarField::from_u32(320044822),
            ScalarField::from_u32(1439190735), ScalarField::from_u32(1586977663), ScalarField::from_u32(62722332), ScalarField::from_u32(1133681444),
            ScalarField::from_u32(601111425), ScalarField::from_u32(341499451), ScalarField::from_u32(1137453179), ScalarField::from_u32(885463894),
            ScalarField::from_u32(1656554330), ScalarField::from_u32(325054530), ScalarField::from_u32(117854124), ScalarField::from_u32(441702830),
            ScalarField::from_u32(42102689), ScalarField::from_u32(286088642), ScalarField::from_u32(952874687), ScalarField::from_u32(2145799399),
            ScalarField::from_u32(1578867646), ScalarField::from_u32(1863298905), ScalarField::from_u32(572180143), ScalarField::from_u32(550370015),
            ScalarField::from_u32(2114631308), ScalarField::from_u32(54570592), ScalarField::from_u32(959652233), ScalarField::from_u32(1347303826),
            ScalarField::from_u32(554913781), ScalarField::from_u32(272386993), ScalarField::from_u32(1985923679), ScalarField::from_u32(875943852),
            ScalarField::from_u32(1424435281), ScalarField::from_u32(887238006), ScalarField::from_u32(1870024845), ScalarField::from_u32(1496347261),
            ScalarField::from_u32(307147512), ScalarField::from_u32(362458382), ScalarField::from_u32(1436017488), ScalarField::from_u32(268708203),
            ScalarField::from_u32(1495708123), ScalarField::from_u32(1122668534), ScalarField::from_u32(149766476), ScalarField::from_u32(236957968),
            ScalarField::from_u32(139805602), ScalarField::from_u32(1254996531), ScalarField::from_u32(1424943280), ScalarField::from_u32(1069407759),
            ScalarField::from_u32(1264862582), ScalarField::from_u32(1262913776), ScalarField::from_u32(138139662), ScalarField::from_u32(643656826),
            ScalarField::from_u32(1303204475), ScalarField::from_u32(1186043635), ScalarField::from_u32(1899594205), ScalarField::from_u32(1288950539),
            ScalarField::from_u32(877594892), ScalarField::from_u32(1638281480), ScalarField::from_u32(2029396066), ScalarField::from_u32(377743645),
            ScalarField::from_u32(1821092124), ScalarField::from_u32(326810630), ScalarField::from_u32(1415720172), ScalarField::from_u32(50453286),
            ScalarField::from_u32(1946062643), ScalarField::from_u32(548995668), ScalarField::from_u32(871336575), ScalarField::from_u32(1307728760),
            ScalarField::from_u32(835752578), ScalarField::from_u32(173336400), ScalarField::from_u32(727472941), ScalarField::from_u32(1150223664),
            ScalarField::from_u32(1900208420), ScalarField::from_u32(1819528405), ScalarField::from_u32(1074366416), ScalarField::from_u32(2038270456),
            ScalarField::from_u32(275152789), ScalarField::from_u32(835175741), ScalarField::from_u32(109790318), ScalarField::from_u32(2119512393),
            ScalarField::from_u32(920061589), ScalarField::from_u32(599928663), ScalarField::from_u32(1377012175), ScalarField::from_u32(1860905991),
            ScalarField::from_u32(1552493810), ScalarField::from_u32(992250247), ScalarField::from_u32(642176955), ScalarField::from_u32(864542730),
            ScalarField::from_u32(179972512), ScalarField::from_u32(441327643), ScalarField::from_u32(1609308498), ScalarField::from_u32(1779914860),
            ScalarField::from_u32(1538136745), ScalarField::from_u32(399621182), ScalarField::from_u32(1033559052), ScalarField::from_u32(1584442957),
            ScalarField::from_u32(765541557), ScalarField::from_u32(1801069699), ScalarField::from_u32(764060118), ScalarField::from_u32(125581587),
            ScalarField::from_u32(1733795677), ScalarField::from_u32(305502848), ScalarField::from_u32(686942290), ScalarField::from_u32(1551244029),
            ScalarField::from_u32(176815448), ScalarField::from_u32(1736471542), ScalarField::from_u32(1492336822), ScalarField::from_u32(324383831),
            ScalarField::from_u32(1963235463), ScalarField::from_u32(215994837), ScalarField::from_u32(523244192), ScalarField::from_u32(1880840234),
            ScalarField::from_u32(1765510882), ScalarField::from_u32(452783068), ScalarField::from_u32(1948807934), ScalarField::from_u32(403904878),
            ScalarField::from_u32(631045078), ScalarField::from_u32(1530893872), ScalarField::from_u32(1869554695), ScalarField::from_u32(843321112),
            ScalarField::from_u32(1853049444), ScalarField::from_u32(899980558), ScalarField::from_u32(746565841), ScalarField::from_u32(1123803537),
            ScalarField::from_u32(970917397), ScalarField::from_u32(562179796), ScalarField::from_u32(1763004912), ScalarField::from_u32(956816550),
            ScalarField::from_u32(508277609), ScalarField::from_u32(498852363), ScalarField::from_u32(1910433806), ScalarField::from_u32(1503192390),
            ScalarField::from_u32(1986292742), ScalarField::from_u32(2010191043), ScalarField::from_u32(1253346262), ScalarField::from_u32(513901733),
            ScalarField::from_u32(490650176), ScalarField::from_u32(1060833994), ScalarField::from_u32(1172500488), ScalarField::from_u32(1701871715),
            ScalarField::from_u32(49554586), ScalarField::from_u32(1511360450), ScalarField::from_u32(1419156693), ScalarField::from_u32(1257750595),
            ScalarField::from_u32(797245405), ScalarField::from_u32(1128572452), ScalarField::from_u32(1788208931), ScalarField::from_u32(972281966),
            ScalarField::from_u32(1168699463), ScalarField::from_u32(58680411), ScalarField::from_u32(1832036717), ScalarField::from_u32(1886611600),
            ScalarField::from_u32(693018562), ScalarField::from_u32(1236257595), ScalarField::from_u32(436708491), ScalarField::from_u32(1405275070),
            ScalarField::from_u32(298055391), ScalarField::from_u32(1186047691), ScalarField::from_u32(1662227636), ScalarField::from_u32(74150862),
            ScalarField::from_u32(1818316773), ScalarField::from_u32(1638470581), ScalarField::from_u32(873716020), ScalarField::from_u32(2094394026),
            ScalarField::from_u32(1644066393), ScalarField::from_u32(1454612100), ScalarField::from_u32(819939998), ScalarField::from_u32(124089541),
            ScalarField::from_u32(1961645908), ScalarField::from_u32(950451518), ScalarField::from_u32(1607874430), ScalarField::from_u32(342777922),
            ScalarField::from_u32(1513498599), ScalarField::from_u32(1368920355), ScalarField::from_u32(744271559), ScalarField::from_u32(288201264),
            ScalarField::from_u32(1080496030), ScalarField::from_u32(151283737), ScalarField::from_u32(27585998), ScalarField::from_u32(1161080485),
            ScalarField::from_u32(1048466512), ScalarField::from_u32(1910918717), ScalarField::from_u32(1948576034), ScalarField::from_u32(1506966942),
            ScalarField::from_u32(2035478663), ScalarField::from_u32(154529039), ScalarField::from_u32(1810622493), ScalarField::from_u32(1956773457),
            ScalarField::from_u32(1360804866), ScalarField::from_u32(968452724), ScalarField::from_u32(637801132), ScalarField::from_u32(797436491),
            ScalarField::from_u32(651814562), ScalarField::from_u32(665940034), ScalarField::from_u32(942144263), ScalarField::from_u32(279532022),
            ScalarField::from_u32(1705839731), ScalarField::from_u32(1323313271), ScalarField::from_u32(1267360167), ScalarField::from_u32(52769292),
            ScalarField::from_u32(1966094340), ScalarField::from_u32(98697907), ScalarField::from_u32(1052090510), ScalarField::from_u32(329111666),
            ScalarField::from_u32(1616685967), ScalarField::from_u32(1401469342), ScalarField::from_u32(1485323049), ScalarField::from_u32(2130413924),
            ScalarField::from_u32(129000096), ScalarField::from_u32(139701208), ScalarField::from_u32(1089989756), ScalarField::from_u32(953306362),
            ScalarField::from_u32(1853303056), ScalarField::from_u32(797351937), ScalarField::from_u32(941912951), ScalarField::from_u32(784558348),
            ScalarField::from_u32(855202225), ScalarField::from_u32(1302064280), ScalarField::from_u32(1937145831), ScalarField::from_u32(974665314),
            ScalarField::from_u32(1608904915), ScalarField::from_u32(1821919482), ScalarField::from_u32(942100369), ScalarField::from_u32(1898005361),
            ScalarField::from_u32(1972333512), ScalarField::from_u32(122370364), ScalarField::from_u32(641210489), ScalarField::from_u32(1620052348),
            ScalarField::from_u32(715040110), ScalarField::from_u32(1012467976), ScalarField::from_u32(1368800962), ScalarField::from_u32(774136950),
            ScalarField::from_u32(1674521270), ScalarField::from_u32(1925175436), ScalarField::from_u32(1683806752), ScalarField::from_u32(1703329928),
            ScalarField::from_u32(405586049), ScalarField::from_u32(955216122), ScalarField::from_u32(395231874), ScalarField::from_u32(1283190097),
            ScalarField::from_u32(585879171), ScalarField::from_u32(979742359), ScalarField::from_u32(990296214), ScalarField::from_u32(659401201),
            ScalarField::from_u32(474329956), ScalarField::from_u32(522478150), ScalarField::from_u32(1442338092), ScalarField::from_u32(150770485),
            ScalarField::from_u32(286918005), ScalarField::from_u32(1449320497), ScalarField::from_u32(104911871), ScalarField::from_u32(888549696),
            ScalarField::from_u32(1082300105), ScalarField::from_u32(997479867), ScalarField::from_u32(1942589107), ScalarField::from_u32(2017873415),
            ScalarField::from_u32(1261755847), ScalarField::from_u32(1438434135), ScalarField::from_u32(1048322054), ScalarField::from_u32(1569971902),
            ScalarField::from_u32(1506677588), ScalarField::from_u32(2039732839), ScalarField::from_u32(1428051110), ScalarField::from_u32(904507734),
            ScalarField::from_u32(1223072319), ScalarField::from_u32(1144819072), ScalarField::from_u32(1759934323), ScalarField::from_u32(268760935),
            ScalarField::from_u32(1049482236), ScalarField::from_u32(1379941191), ScalarField::from_u32(712119671), ScalarField::from_u32(354887349),
            ScalarField::from_u32(1097722367), ScalarField::from_u32(460471787), ScalarField::from_u32(725132773), ScalarField::from_u32(1142004660),
            ScalarField::from_u32(666182744), ScalarField::from_u32(2024591106), ScalarField::from_u32(1269653270), ScalarField::from_u32(1489721639),
            ScalarField::from_u32(751022300), ScalarField::from_u32(1507060924), ScalarField::from_u32(1259220046), ScalarField::from_u32(186835864),
            ScalarField::from_u32(218972416), ScalarField::from_u32(729918549), ScalarField::from_u32(1150792576), ScalarField::from_u32(1271800225),
            ScalarField::from_u32(173756859), ScalarField::from_u32(1771395347), ScalarField::from_u32(1939801529), ScalarField::from_u32(1791538380),
            ScalarField::from_u32(420336754), ScalarField::from_u32(1062426754), ScalarField::from_u32(556495932), ScalarField::from_u32(460363610),
            ScalarField::from_u32(898476488), ScalarField::from_u32(281205668), ScalarField::from_u32(1138792738), ScalarField::from_u32(140830313),
            ScalarField::from_u32(390835881), ScalarField::from_u32(1550203560), ScalarField::from_u32(609621593), ScalarField::from_u32(207435049),
            ScalarField::from_u32(1682343017), ScalarField::from_u32(460640050), ScalarField::from_u32(1307173056), ScalarField::from_u32(918097326),
            ScalarField::from_u32(240701323), ScalarField::from_u32(1827211156), ScalarField::from_u32(10672104), ScalarField::from_u32(1729209499),
            ScalarField::from_u32(1116215327), ScalarField::from_u32(914176268), ScalarField::from_u32(641459635), ScalarField::from_u32(1855496762),
            ScalarField::from_u32(2052785472), ScalarField::from_u32(733255740), ScalarField::from_u32(318948503), ScalarField::from_u32(1144257786),
            ScalarField::from_u32(1026326720), ScalarField::from_u32(2026973129), ScalarField::from_u32(1156097768), ScalarField::from_u32(1193472736),
            ScalarField::from_u32(1333731425), ScalarField::from_u32(1754186243), ScalarField::from_u32(993351395), ScalarField::from_u32(1462484545),
            ScalarField::from_u32(1913684030), ScalarField::from_u32(1843585953), ScalarField::from_u32(789518453), ScalarField::from_u32(763782866),
            ScalarField::from_u32(1930506136), ScalarField::from_u32(379581355), ScalarField::from_u32(1516136727), ScalarField::from_u32(1803817838),
            ScalarField::from_u32(462022582), ScalarField::from_u32(1146946754), ScalarField::from_u32(887150801), ScalarField::from_u32(1211750447),
            ScalarField::from_u32(1319004921), ScalarField::from_u32(16830435), ScalarField::from_u32(2005693552), ScalarField::from_u32(1946012633),
            ScalarField::from_u32(873154094), ScalarField::from_u32(71888803), ScalarField::from_u32(478396410), ScalarField::from_u32(734945535),
            ScalarField::from_u32(482413625), ScalarField::from_u32(1538155645), ScalarField::from_u32(662153922), ScalarField::from_u32(1203659754),
            ScalarField::from_u32(2076183326), ScalarField::from_u32(1860894582), ScalarField::from_u32(942103424), ScalarField::from_u32(1058885109),
            ScalarField::from_u32(33741477), ScalarField::from_u32(410432622), ScalarField::from_u32(1952328560), ScalarField::from_u32(1257964203),
            ScalarField::from_u32(1765233261), ScalarField::from_u32(28509543), ScalarField::from_u32(1455147538), ScalarField::from_u32(1583241559),
            ScalarField::from_u32(1608015704), ScalarField::from_u32(874118111), ScalarField::from_u32(2000748243), ScalarField::from_u32(670329438),
            ScalarField::from_u32(2133473674), ScalarField::from_u32(366351394), ScalarField::from_u32(881775286), ScalarField::from_u32(2122409662),
            ScalarField::from_u32(574230943), ScalarField::from_u32(273768373), ScalarField::from_u32(2020015724), ScalarField::from_u32(1208153365),
            ScalarField::from_u32(436212495), ScalarField::from_u32(887323261), ScalarField::from_u32(1719082382), ScalarField::from_u32(182192),
            ScalarField::from_u32(1324223846), ScalarField::from_u32(1708556934), ScalarField::from_u32(424053431), ScalarField::from_u32(422312292),
            ScalarField::from_u32(1902244124), ScalarField::from_u32(1116097208), ScalarField::from_u32(2128440118), ScalarField::from_u32(803770959),
            ScalarField::from_u32(1631254947), ScalarField::from_u32(1086593403), ScalarField::from_u32(816574194), ScalarField::from_u32(419117370),
            ScalarField::from_u32(267923808), ScalarField::from_u32(371399314), ScalarField::from_u32(1956513668), ScalarField::from_u32(1428269486),
            ScalarField::from_u32(867621253), ScalarField::from_u32(734464114), ScalarField::from_u32(1510675753), ScalarField::from_u32(2124432451),
            ScalarField::from_u32(262730040), ScalarField::from_u32(1796758823), ScalarField::from_u32(1146123806), ScalarField::from_u32(1335253369),
            ScalarField::from_u32(2113562587), ScalarField::from_u32(1758160178), ScalarField::from_u32(1702549217), ScalarField::from_u32(1120932688),
            ScalarField::from_u32(926410249), ScalarField::from_u32(751933267), ScalarField::from_u32(947066646), ScalarField::from_u32(1010721228),
            ScalarField::from_u32(2925084), ScalarField::from_u32(762775638), ScalarField::from_u32(1932252091), ScalarField::from_u32(570599496),
            ScalarField::from_u32(1879742862), ScalarField::from_u32(1405100116), ScalarField::from_u32(627428297), ScalarField::from_u32(740950556),
            ScalarField::from_u32(497873628), ScalarField::from_u32(991411203), ScalarField::from_u32(2035042784), ScalarField::from_u32(1685953604),
            ScalarField::from_u32(1392594946), ScalarField::from_u32(84734891), ScalarField::from_u32(239966238), ScalarField::from_u32(1100134303),
            ScalarField::from_u32(860502012), ScalarField::from_u32(1273822642), ScalarField::from_u32(265038420), ScalarField::from_u32(541502111),
            ScalarField::from_u32(520355901), ScalarField::from_u32(1719230052), ScalarField::from_u32(1202420013), ScalarField::from_u32(1913444366),
            ScalarField::from_u32(1980750654), ScalarField::from_u32(67834589), ScalarField::from_u32(870920287), ScalarField::from_u32(66474991),
            ScalarField::from_u32(1153745488), ScalarField::from_u32(1841553357), ScalarField::from_u32(1678783671), ScalarField::from_u32(75631742),
            ScalarField::from_u32(1435288216), ScalarField::from_u32(354253984), ScalarField::from_u32(387954225), ScalarField::from_u32(106228371),
            ScalarField::from_u32(674441371), ScalarField::from_u32(1114809215), ScalarField::from_u32(1324106436), ScalarField::from_u32(311683215),
            ScalarField::from_u32(1602502619), ScalarField::from_u32(980466350), ScalarField::from_u32(986124760), ScalarField::from_u32(380262332),
            ScalarField::from_u32(635513485), ScalarField::from_u32(12390338), ScalarField::from_u32(13066332), ScalarField::from_u32(1390722195),
            ScalarField::from_u32(1826540105), ScalarField::from_u32(2118304867), ScalarField::from_u32(244935298), ScalarField::from_u32(974036541),
            ScalarField::from_u32(454232627), ScalarField::from_u32(966726748), ScalarField::from_u32(710707797), ScalarField::from_u32(1464927550),
            ScalarField::from_u32(1841024618), ScalarField::from_u32(712137923), ScalarField::from_u32(1081959297), ScalarField::from_u32(1759032127),
            ScalarField::from_u32(679792130), ScalarField::from_u32(1023671493), ScalarField::from_u32(904990910), ScalarField::from_u32(1851944173),
            ScalarField::from_u32(474499131), ScalarField::from_u32(887948466), ScalarField::from_u32(489312096), ScalarField::from_u32(3956892),
            ScalarField::from_u32(986883326), ScalarField::from_u32(137969330), ScalarField::from_u32(1761079190), ScalarField::from_u32(1106059137),
            ScalarField::from_u32(216572935), ScalarField::from_u32(1813289598), ScalarField::from_u32(1464350725), ScalarField::from_u32(368547196),
            ScalarField::from_u32(226529364), ScalarField::from_u32(1118651629), ScalarField::from_u32(120798314), ScalarField::from_u32(197088573),
            ScalarField::from_u32(452178301), ScalarField::from_u32(1148542694), ScalarField::from_u32(1159014645), ScalarField::from_u32(1724104093),
            ScalarField::from_u32(1792431461), ScalarField::from_u32(1581540092), ScalarField::from_u32(329365655), ScalarField::from_u32(1048879093),
            ScalarField::from_u32(1387128414), ScalarField::from_u32(1478598050), ScalarField::from_u32(236160429), ScalarField::from_u32(1659700207),
            ScalarField::from_u32(1211862055), ScalarField::from_u32(40320833), ScalarField::from_u32(1349010637), ScalarField::from_u32(1340770304),
            ScalarField::from_u32(1030900139), ScalarField::from_u32(1840125030), ScalarField::from_u32(66586326), ScalarField::from_u32(7504566),
            ScalarField::from_u32(1445010113), ScalarField::from_u32(604360748), ScalarField::from_u32(2099791157), ScalarField::from_u32(625417202),
            ScalarField::from_u32(2060595769), ScalarField::from_u32(1186048786), ScalarField::from_u32(1032629410), ScalarField::from_u32(1719277924),
            ScalarField::from_u32(1844918201), ScalarField::from_u32(275390495), ScalarField::from_u32(1750147095), ScalarField::from_u32(146657454),
            ScalarField::from_u32(373437184), ScalarField::from_u32(1662977908), ScalarField::from_u32(1223596031), ScalarField::from_u32(1593381642),
            ScalarField::from_u32(750157699), ScalarField::from_u32(375579794), ScalarField::from_u32(670924608), ScalarField::from_u32(1901927063),
            ScalarField::from_u32(431803923), ScalarField::from_u32(811993642), ScalarField::from_u32(922844133), ScalarField::from_u32(755127623),
            ScalarField::from_u32(100298373), ScalarField::from_u32(1940790941), ScalarField::from_u32(623431004), ScalarField::from_u32(859518984),
            ScalarField::from_u32(1465132695), ScalarField::from_u32(188583724), ScalarField::from_u32(396643333), ScalarField::from_u32(1045134065),
            ScalarField::from_u32(239286196), ScalarField::from_u32(948263895), ScalarField::from_u32(109085955), ScalarField::from_u32(1667415274),
            ScalarField::from_u32(1908257490), ScalarField::from_u32(1338248462), ScalarField::from_u32(1655620848), ScalarField::from_u32(338896255),
            ScalarField::from_u32(200498478), ScalarField::from_u32(98606084), ScalarField::from_u32(1381583592), ScalarField::from_u32(1583308823),
            ScalarField::from_u32(2045763890), ScalarField::from_u32(754871725), ScalarField::from_u32(1719995281), ScalarField::from_u32(704276593),
            ScalarField::from_u32(828781275), ScalarField::from_u32(455309901), ScalarField::from_u32(2117204438), ScalarField::from_u32(1622508949),
            ScalarField::from_u32(2090190726), ScalarField::from_u32(1125421132), ScalarField::from_u32(302519262), ScalarField::from_u32(1914456113),
        ];
        assert_eq!(golden_data, result);
    }
}
