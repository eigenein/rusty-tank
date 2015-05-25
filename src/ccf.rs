//! Clustering-based collaboration filtering for Tankopoisk.

extern crate rand;
extern crate time;

mod corr;
mod csr;
mod encyclopedia;
mod helpers;
mod kmeans;
mod protobuf;
mod stats;

/// Minimum battles count.
const MIN_BATTLES: u32 = 10;

#[allow(dead_code)]
fn main() {
    let mut input = helpers::get_input();
    let encyclopedia = encyclopedia::Encyclopedia::new();
    let (train_table, test_table) = helpers::read_stats(&mut input, MIN_BATTLES, &encyclopedia);
}
