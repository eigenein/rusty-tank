//! Slope One for Tankopoisk.

use std::collections::HashMap;

extern crate rand;
extern crate time;

mod csr;
mod encyclopedia;
mod helpers;
mod protobuf;
mod stats;

const MIN_BATTLES: u32 = 10;

struct Model {
    average_difference: HashMap<(u16, u16), f64>,
    rating_count: Vec<usize>,
}

impl Model {

}

impl helpers::AbstractModel for Model {
    fn predict(&self, row_index: usize, column_index: usize) -> f64 {
        0.0
    }
}

#[allow(dead_code)]
fn main() {
    //
}
