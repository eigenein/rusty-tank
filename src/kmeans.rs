//! K-Means Clustering.

use rand::{Rng, thread_rng};

use corr;
use csr::{Csr, Row};
use helpers::AbstractModel;

pub struct Model {
    row_count: usize,
    column_count: usize,
    cluster_count: usize,
    centroids: Csr,
    row_clusters: Vec<usize>,
}

impl Model {
    /// Creates a new model.
    pub fn new(row_count: usize, column_count: usize, cluster_count: usize) -> Self {
        let mut centroids = Csr::new();
        let mut rng = thread_rng();

        for _ in 0..cluster_count {
            centroids.start();
            for column_index in 0..column_count {
                centroids.next(column_index, rng.gen_range(0.0, 100.0));
            }
        }
        centroids.start();

        Model {
            row_count: row_count,
            column_count: column_count,
            cluster_count: cluster_count,
            centroids: centroids,
            row_clusters: vec![0; row_count],
        }
    }

    /// Gets cluster count.
    pub fn cluster_count(&self) -> usize {
        self.cluster_count
    }

    /// Gets row cluster index.
    pub fn get_cluster(&self, row_index: usize) -> usize {
        self.row_clusters[row_index]
    }

    /// Gets cluster centroid.
    pub fn get_centroid(&self, index: usize) -> Row {
        self.centroids.get_row(index)
    }

    /// Makes clustering step.
    pub fn make_step(&mut self, table: &Csr) -> usize {
        let mut changed_count = 0;
        // Assign nearest centroids.
        for row_index in 0..self.row_count {
            let nearest_centroid_index = self.get_nearest_centroid(table.get_row(row_index));
            if nearest_centroid_index != self.row_clusters[row_index] {
                changed_count += 1;
            }
            self.row_clusters[row_index] = nearest_centroid_index;
        }
        if changed_count == 0 {
            return changed_count;
        }
        // Reset centroids.
        for centroid_index in 0..self.cluster_count {
            let row = self.centroids.get_mutable_row(centroid_index);
            for value in row.iter_mut() {
                value.value = 0.0;
            }
        }
        // Sum up values.
        let mut value_count = vec![0usize; self.cluster_count * self.column_count];
        for row_index in 0..self.row_count {
            for value in table.get_row(row_index) {
                let cluster_index = self.row_clusters[row_index];
                // Increase column value count.
                value_count[cluster_index * self.column_count + value.column] += 1;
                // Increase centroid value.
                self.centroids.get_mutable_row(cluster_index)[value.column].value += value.value;
            }
        }
        // Divide by value count.
        for cluster_index in 0..self.cluster_count {
            for value in self.centroids.get_mutable_row(cluster_index) {
                value.value /= value_count[cluster_index * self.column_count + value.column] as f64;
            }
        }

        changed_count
    }

    /// Gets the nearest centroid by the given row.
    fn get_nearest_centroid(&self, row: Row) -> usize {
        use std::f64;

        let mut max_correlation = f64::NEG_INFINITY;
        let mut centroid_index = 0;

        for i in 0..self.cluster_count {
            let correlation = corr::pearson(row, self.centroids.get_row(i));
            if correlation > max_correlation {
                max_correlation = correlation;
                centroid_index = i;
            }
        }

        centroid_index
    }
}
