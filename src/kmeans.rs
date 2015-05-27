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
    row_clusters: Vec<Option<usize>>,
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
            row_clusters: vec![None; row_count],
        }
    }

    /// Makes clustering step.
    pub fn make_step(&mut self, matrix: &Csr) -> f64 {
        let mut total_count = 0;
        let mut error_sum = 0.0;
        // Assign nearest centroids.
        for row_index in 0..self.row_count {
            let row = matrix.get_row(row_index);
            if row.len() < 3 {
                // FIX: this row correlates to any other one so it should be skipped.
                continue;
            }
            let (cluster_index, distance) = self.get_nearest_centroid(row);
            self.row_clusters[row_index] = Some(cluster_index);
            total_count += 1;
            error_sum += distance * distance;
        }
        // Reset centroids.
        for cluster_index in 0..self.cluster_count {
            let row = self.centroids.get_mutable_row(cluster_index);
            for value in row.iter_mut() {
                value.value = 0.0;
            }
        }
        // Sum up values.
        let mut value_count = vec![0usize; self.cluster_count * self.column_count];
        for row_index in 0..self.row_count {
            if let Some(cluster_index) = self.row_clusters[row_index] {
                for value in matrix.get_row(row_index) {
                    // Increase column value count.
                    value_count[cluster_index * self.column_count + value.column] += 1;
                    // Increase centroid value.
                    self.centroids.get_mutable_row(cluster_index)[value.column].value += value.value;
                }
            }
        }
        // Divide by value count.
        for cluster_index in 0..self.cluster_count {
            for value in self.centroids.get_mutable_row(cluster_index) {
                value.value /= value_count[cluster_index * self.column_count + value.column] as f64;
            }
        }

        error_sum / total_count as f64
    }

    /// Gets the nearest centroid by the given row.
    fn get_nearest_centroid(&self, row: Row) -> (usize, f64) {
        use std::f64;

        let mut min_distance = f64::INFINITY;
        let mut cluster_index = 0;

        for i in 0..self.cluster_count {
            let distance = 1.0 - corr::pearson(row, self.centroids.get_row(i));
            if distance < min_distance {
                min_distance = distance;
                cluster_index = i;
            }
        }

        (cluster_index, min_distance)
    }
}

impl AbstractModel for Model {
    /// Quick and dirty implementation.
    fn predict(&self, train_matrix: &Csr, row_index: usize, column_index: usize) -> f64 {
        use std::f64;

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        if let Some(cluster_index) = self.row_clusters[row_index] {
            // The row is clustered.
            let row = train_matrix.get_row(row_index);
            for other_index in 0..self.row_count {
                if other_index != row_index && self.row_clusters[other_index] == Some(cluster_index) {
                    // Found row in the same cluster.
                    let other_row = train_matrix.get_row(other_index);
                    for value in other_row {
                        if value.column == column_index {
                            // Found value at the column.
                            let weight = corr::pearson(row, other_row);
                            if weight > 0.0 {
                                weighted_sum += value.value * weight;
                                weight_sum += weight;
                            }
                            break;
                        }
                    }
                }
            }
            weighted_sum / weight_sum
        } else {
            f64::NAN
        }
    }
}
