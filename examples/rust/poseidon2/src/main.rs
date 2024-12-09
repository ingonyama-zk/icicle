use icicle_core::{
    hash::{HashConfig, Hasher}, 
    poseidon2::Poseidon2, traits::FieldImpl
};
use icicle_babybear::field::ScalarField as Frbb;
use icicle_m31::field::ScalarField as Frm31;

use icicle_runtime::memory::HostSlice;

pub fn hash_test<F:FieldImpl>(
    test_vec: Vec<F>,
    config: HashConfig,
    hash: Hasher,
) {
let input_slice = HostSlice::from_slice(&test_vec);
let out_init:F = F::zero();
let mut binding = [out_init];
let out_init_slice = HostSlice::from_mut_slice(&mut binding);
hash.hash(input_slice, &config, out_init_slice).unwrap();
println!("computed digest: {:?} ",out_init_slice.as_slice().to_vec()[0]);

}
pub fn main(){
// digest = output_state[1]
// Sage output Baby bear
// t = 2
// Input state (0, 1)
// Output state [869011615, 833751247]
// t= 3
// Input state (0, 1, 2)
// Output state [1704654064, 1850148672, 1532353406]
// t = 4
// Input state (0, 1, 2, 3)
// Output state [741579827, 472702774, 852055751, 1266116070]
// t= 8 
// Input state (0, 1, 2, 3, 4, 5, 6, 7)
// Output state [1231724177, 1077997898, 146576824, 919391229, 302461086, 1311223212, 679569792, 681685934]
// t = 12
// Input state (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
// Output state 1540343413, 1605336739, 1201446587, 1251783394, 440826505, 1691696232, 904498569, 1312737773, 1464207073, 133812423, 1144748001, 1160609856]
// t = 16
// Input state (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
// Output state [896560466, 771677727, 128113032, 1378976435, 160019712, 1452738514, 682850273, 223500421, 501450187, 1804685789, 1671399593, 1788755219, 1736880027, 1352180784, 1928489698, 1128802977]
// t = 20
// Input state (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
// Output state [1637625426, 1224149815, 185762176, 1743975927, 215506, 846181926, 1805239884, 1583247763, 40890463, 1769635047, 1593365708, 543030243, 190381160, 114174693, 528766946, 107317631, 199017750, 946546831, 188856465, 89693326]
// t = 24
// Input state (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
// Output state [785637949, 311566256, 241540729, 1641553353, 851108667, 1648913123, 510139232, 616108837, 707720633, 1357404478, 1539840236, 275323287, 899761440, 732341189, 664618988, 1426148993, 1498654335, 792736017, 1804085503, 402731039, 659103866, 1036635937, 1016617890, 1470732388]
let t_vec = [2,3,4,8,12,16,20,24];
let expected_digest_bb:Vec<Frbb> = vec![Frbb::from_u32(833751247),Frbb::from_u32(1850148672),Frbb::from_u32(472702774),Frbb::from_u32(1077997898),Frbb::from_u32(1605336739),Frbb::from_u32(771677727),Frbb::from_u32(1224149815),Frbb::from_u32(311566256)];
println!("Baby Bear");
let config = HashConfig::default();
for (t,digest) in t_vec.iter().zip(expected_digest_bb.iter()){
    let input_state_bb:Vec<Frbb> = (0..*t).map(Frbb::from_u32).collect();
    println!("test vector {:?}",input_state_bb);
    println!("expected digest {:?}",digest);
    hash_test::<Frbb>(input_state_bb, config.clone(), Poseidon2::new::<Frbb>(*t,None).unwrap());
    println!(" ");
}

// digest = output_state[1]
// Sage output m31
//t=2
// Input state (0, 1)
// Output state [1259525573, 1321841424]
//t=3
// Input state (0, 1, 2)
// Output state [1965998969, 1808522380, 1146513877]
// t=4
// Input state (0, 1, 2, 3)
// Output state [1062794947, 1937028579, 518022994, 1790851810]
// t= 8
// Input state (0, 1, 2, 3, 4, 5, 6, 7)
// Output state [1587676993, 1040745210, 1362579098, 1364533986, 505714447, 371333953, 24021099, 1307077870]
// t =12
// Input state (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
// Output state [1352296093, 495013829, 721412628, 551472485, 1402861161, 1099939525, 56806196, 322927204, 1743775127, 1737182096, 1637144312, 482990946]
// t= 16
// Input state (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
// Output state [1348310665, 996460804, 2044919169, 1269301599, 615961333, 595876573, 1377780500, 1776267289, 715842585, 1823756332, 1870636634, 1979645732, 311256455, 1364752356, 58674647, 323699327]
// t = 20
// Input state (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
// Output state [2145869251, 33722680, 323999981, 1338601227, 1335935383, 1569616976, 1025767832, 1219571145, 1312283131, 517961801, 1182517165, 1896142496, 1426432276, 386540698, 1519857378, 840037603, 431686357, 2045496595, 609478066, 1695781828]
// t = 24
// Input state (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
// Output state [813042329, 956159494, 2017691352, 906353481, 1909737181, 1568930368, 1051192156, 1915448194, 114779228, 1695016063, 56353577, 991257558, 1283398606, 1782986529, 89100699, 1011002020, 71058136, 1382771657, 1734747710, 184579357, 1201113333, 2002016011, 1347833245, 1026595486]

let expected_digest_m31:Vec<Frm31> = vec![Frm31::from_u32(1321841424),Frm31::from_u32(1808522380),Frm31::from_u32(1937028579),Frm31::from_u32(1040745210),
Frm31::from_u32(495013829),Frm31::from_u32(996460804),Frm31::from_u32(33722680),Frm31::from_u32(956159494)];
println!("M31");
let config = HashConfig::default();
for (t,digest) in t_vec.iter().zip(expected_digest_m31.iter()){
    let input_state_m31:Vec<Frm31> = (0..*t).map(Frm31::from_u32).collect();
    println!("test vector {:?}",input_state_m31);
    println!("expected digest {:?}",digest);
    hash_test::<Frm31>(input_state_m31, config.clone(), Poseidon2::new::<Frm31>(*t,None).unwrap());
    println!(" ");
}

}