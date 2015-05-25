//! K-Means Clustering.

use csr::Csr;
use helpers::AbstractModel;

pub struct Model {
    // TODO.
}

impl Model {
    /// Creates a new model.
    pub fn new(row_count: usize, column_count: usize, cluster_count: usize) -> Self {
        Model { /* TODO */ }
    }

    pub fn make_step(&mut self, table: &Csr) {
        // TODO.
    }
}
