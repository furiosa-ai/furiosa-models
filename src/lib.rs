use pyo3::prelude::*;
use pyo3::wrap_pymodule;

mod ssd_mobilenet;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

// NOTE: This docstring is unshown in Python level
/// A Python module implemented in Rust.
#[pymodule]
fn furiosa_models_vision_native(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    let ssd_mobilenet_module = pyo3::wrap_pymodule!(ssd_mobilenet::ssd_mobilenet);
    m.add_wrapped(ssd_mobilenet_module)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("furiosa_models_native.ssd_mobilenet", ssd_mobilenet_module(py))?;

    Ok(())
}