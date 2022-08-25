
#![allow(dead_code)]
use pyo3::prelude::*;

mod ops;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}


// NOTE: This docstring is unshown in Python level
/// A Python module implemented in Rust.
#[pymodule]
fn furiosa_models_vision_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(ops::nms::nms_index, m)?)?;
    Ok(())
}
