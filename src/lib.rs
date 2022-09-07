use pyo3::prelude::*;

mod ssd_mobilenet;

#[pymodule]
fn furiosa_models_vision_native(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let ssd_mobilenet_module = pyo3::wrap_pymodule!(ssd_mobilenet::ssd_mobilenet);
    m.add_wrapped(ssd_mobilenet_module)?;

    py.import("sys")?.getattr("modules")?.set_item(
        "furiosa_models_native.ssd_mobilenet",
        ssd_mobilenet_module(py),
    )?;

    Ok(())
}
