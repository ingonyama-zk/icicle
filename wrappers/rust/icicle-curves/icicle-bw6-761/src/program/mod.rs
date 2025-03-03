pub mod bw6_761 {
  // Link program defined using bls12-377's base field to bw6-761
  use icicle_bls12_377::program::bw6_761::FieldProgram as bw6_761_program;
  use icicle_bls12_377::program::bw6_761::FieldReturningValueProgram as bw6_761_returning_program;
  pub type FieldProgram = bw6_761_program;
  pub type FieldReturningValueProgram = bw6_761_returning_program;
}