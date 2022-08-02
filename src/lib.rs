use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

// FIXME: This docstring is unshown in Python level
// FIXME: Since identifiers cannot contain [DOT] characters,
//        pyo3 module name is set to furiosa, not furiosa.models.
//        This will drop furiosa.cpython-*.so in /python/furiosa/
/// A Python module implemented in Rust.
#[pymodule]
fn furiosa(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
