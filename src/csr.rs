//! Compressed Sparse Row (and Compressed Sparse Column) implementation.

/// Value with the corresponding column.
#[derive(Debug)]
pub struct ColumnValue {
    pub column: usize,
    pub value: f64,
}

pub type Row<'a> = &'a[ColumnValue];

/// Compressed Sparse Row matrix.
#[derive(Debug)]
pub struct Csr {
    values: Vec<ColumnValue>,
    pointers: Vec<usize>,
}

/// Value with the corresponding row and column.
#[derive(Debug)]
struct RowColumnValue {
    pub row: usize,
    pub column_value: ColumnValue,
}

impl Csr {
    pub fn new() -> Self {
        Csr { values: Vec::new(), pointers: Vec::new() }
    }

    /// Starts a new row.
    ///
    /// Ensure that you have started a new row after the very last value.
    pub fn start(&mut self) {
        self.pointers.push(self.values.len());
    }

    /// Adds a new value to the current row.
    pub fn next(&mut self, column: usize, value: f64) {
        self.values.push(ColumnValue { value: value, column: column });
    }

    /// Gets value count.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Gets row count.
    pub fn row_count(&self) -> usize {
        self.pointers.len() - 1
    }

    /// Gets a slice to the row.
    pub fn get_row(&self, index: usize) -> Row {
        assert!(index < self.pointers.len());
        &self.values[self.pointers[index]..self.pointers[index + 1]]
    }

    /// Transposes matrix.
    pub fn transpose(&mut self) {
        // Make temporary COO matrix.
        let mut row_column_values: Vec<RowColumnValue> = Vec::new();
        for row_index in 0..self.row_count() {
            for column_value in self.get_row(row_index) {
                row_column_values.push(RowColumnValue {
                    row: row_index,
                    column_value: ColumnValue { value: column_value.value, column: column_value.column },
                });
            }
        }
        // Drop this matrix.
        self.values.clear();
        self.pointers = Vec::new();
        // Sort the COO matrix by column then by row.
        row_column_values.sort_by(|a, b| (a.column_value.column, a.row).cmp(&(b.column_value.column, b.row)));
        // Reconstruct this matrix as a transposed one. Group by column index.
        let mut current_column = None;
        for row_column_value in row_column_values {
            if current_column != Some(row_column_value.column_value.column) {
                current_column = Some(row_column_value.column_value.column);
                self.start();
            }
            self.next(row_column_value.row, row_column_value.column_value.value);
        }
        self.start();
    }
}

#[test]
fn test_start() {
    let mut matrix = Csr::new();
    matrix.start();
    matrix.next(0, 1.0);
    matrix.next(2, 2.0);
    matrix.start();

    assert_eq!(matrix.pointers.len(), 2);
    assert_eq!(matrix.pointers[0], 0);
    assert_eq!(matrix.pointers[1], 2);
}

#[test]
fn test_next() {
    let mut matrix = Csr::new();
    matrix.start();
    matrix.next(0, 1.0);
    matrix.next(2, 2.0);
    matrix.next(5, 3.0);
    matrix.start();

    assert_eq!(matrix.values.len(), 3);
}

#[test]
fn test_len() {
    let mut matrix = Csr::new();
    matrix.start();
    matrix.next(0, 1.0);
    matrix.next(2, 2.0);
    matrix.start();

    assert_eq!(matrix.len(), 2);
}

#[test]
fn test_row_count() {
    let mut matrix = Csr::new();
    matrix.start();
    matrix.next(0, 1.0);
    matrix.next(2, 2.0);
    matrix.start();

    assert_eq!(matrix.row_count(), 1);
}

#[test]
fn test_get_row() {
    let mut matrix = Csr::new();
    matrix.start();
    matrix.next(0, 1.0);
    matrix.start();
    matrix.next(2, 2.0);
    matrix.next(5, 3.0);
    matrix.start();
    matrix.next(1, 7.0);
    matrix.start();

    assert_eq!(matrix.get_row(1).len(), 2);
    assert_eq!(matrix.get_row(1)[0].column, 2);
}

#[test]
fn test_transpose() {
    let mut matrix = Csr::new();
    matrix.start();
    matrix.next(0, 1.0);
    matrix.start();
    matrix.next(2, 2.0);
    matrix.next(5, 3.0);
    matrix.start();
    matrix.next(1, 7.0);
    matrix.start();

    matrix.transpose();

    assert_eq!(matrix.pointers, vec![0, 1, 2, 3, 4]);
    assert_eq!(matrix.values.iter().map(|value| value.column).collect::<Vec<usize>>(), vec![0, 2, 1, 1]);
    assert_eq!(matrix.values.iter().map(|value| value.value).collect::<Vec<f64>>(), vec![1.0, 7.0, 2.0, 3.0]);
}
