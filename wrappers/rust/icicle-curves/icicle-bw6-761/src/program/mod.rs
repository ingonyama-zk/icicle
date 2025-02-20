pub mod bw6_761 {
  // Link program defined using bls12-377's base field to bw6-761
  use icicle_bls12_377::program::bw6_761::Program as bw6_761_program;
  use icicle_bls12_377::program::bw6_761::ReturningValueProgram as bw6_761_returning_program;
  pub type Program = bw6_761_program;
  pub type ReturningValueProgram = bw6_761_returning_program;
}