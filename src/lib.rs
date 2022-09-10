use pyo3::prelude::*;

mod common;
mod ssd_mobilenet;
mod ssd_resnet34;

#[pymodule]
fn furiosa_models_vision_native(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let ssd_mobilenet_module = pyo3::wrap_pymodule!(ssd_mobilenet::ssd_mobilenet);
    let ssd_resnet34_module = pyo3::wrap_pymodule!(ssd_resnet34::ssd_resnet34);

    m.add_wrapped(ssd_mobilenet_module)?;
    m.add_wrapped(ssd_resnet34_module)?;

    py.import("sys")?.getattr("modules")?.set_item(
        "furiosa_models_native.ssd_mobilenet",
        ssd_mobilenet_module(py),
    )?;
    py.import("sys")?.getattr("modules")?.set_item(
        "furiosa_models_native.ssd_resnet34",
        ssd_resnet34_module(py),
    )?;

    Ok(())
}
