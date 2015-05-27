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
const MIN_BATTLES: u32 = 5;
/// Row count per a cluster.
const ROWS_PER_CLUSTER: usize = 10000;
/// K-Means run count.
const RUN_COUNT: usize = 2;

#[allow(dead_code)]
fn main() {
    use std::cmp::max;

    let (encyclopedia, train_matrix, test_matrix) = helpers::get_stats(MIN_BATTLES, helpers::identity);
    let cluster_count = max(2, train_matrix.row_count() / ROWS_PER_CLUSTER);
    println!("Starting clustering. Cluster count: {}.", cluster_count);
    let (_, error) = train(&train_matrix, encyclopedia.len(), cluster_count);
    println!("Finished clustering. Error: {0:.6}.", error);
}

/// Trains the model.
fn train(matrix: &csr::Csr, column_count: usize, cluster_count: usize) -> (kmeans::Model, f64) {
    use std::f64;

    let mut best_model = None;
    let mut error_min = f64::INFINITY;

    for run in 0..RUN_COUNT {
        let mut model = kmeans::Model::new(matrix.row_count(), column_count, cluster_count);
        let mut previous_error = f64::INFINITY;
        for step in 0.. {
            let error = model.make_step(matrix);
            println!(
                "#{0}/{1} of {5} | clustering | best E: {2:.6} | E: {3:.6} | dE: {4:.9}",
                step, run, error_min, error, previous_error - error, RUN_COUNT
            );
            if previous_error < error {
                break;
            }
            previous_error = error;
        }
        if previous_error < error_min {
            best_model = Some(model);
            error_min = previous_error;
        }
    }

    (best_model.unwrap(), error_min)
}
