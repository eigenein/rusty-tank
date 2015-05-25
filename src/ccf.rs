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
/// Row count per a cluster.
const ROWS_PER_CLUSTER: usize = 10;

#[allow(dead_code)]
fn main() {
    let (encyclopedia, train_table, test_table) = helpers::get_stats(MIN_BATTLES);
    println!("Initializing model.");
    let mut model = kmeans::Model::new(train_table.row_count(), encyclopedia.len(), train_table.row_count() / ROWS_PER_CLUSTER);
    println!("Cluster count: {}.", model.cluster_count());
    train(&mut model, &train_table);
}

/// Trains the model.
fn train(model: &mut kmeans::Model, train_table: &csr::Csr) {
    use time::now;

    let clustering_start_time = now();
    println!("Clustering started at {}.", clustering_start_time.ctime());
    for step in 0.. {
        let changed_count = model.make_step(train_table);
        println!("#{0} | clustering | changed: {1}", step, changed_count);
        if changed_count == 0 {
            break;
        }
    }
    println!("Clustering finished in {:.1}s.", helpers::get_seconds(clustering_start_time));
}
