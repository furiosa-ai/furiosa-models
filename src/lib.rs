use pyo3::prelude::*;

mod common;
mod resnet50;
mod ssd_mobilenet;

#[pymodule]
fn furiosa_models_vision_native(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let ssd_mobilenet_module = pyo3::wrap_pymodule!(ssd_mobilenet::ssd_mobilenet);
    let resnet50_module = pyo3::wrap_pymodule!(resnet50::resnet50);

    m.add_wrapped(ssd_mobilenet_module)?;
    m.add_wrapped(resnet50_module)?;

    py.import("sys")?.getattr("modules")?.set_item(
        "furiosa_models_native.ssd_mobilenet",
        ssd_mobilenet_module(py),
    )?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("furiosa_models_native.resnet50", resnet50_module(py))?;

    Ok(())
}
