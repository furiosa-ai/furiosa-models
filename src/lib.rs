use pyo3::prelude::*;

mod ssd_mobilenet;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

// NOTE: This docstring is unshown in Python level
/// A Python module implemented in Rust.
#[pymodule]
fn furiosa_models_native(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    let ssd_mobilenet_mod = PyModule::new(py, "ssd_mobilenet")?;
    ssd_mobilenet::register_module(py, &ssd_mobilenet_mod)?;

    m.add_submodule(ssd_mobilenet_mod)?;

    Ok(())
}