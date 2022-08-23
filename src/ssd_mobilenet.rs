use mlperf_postprocess::common::graph::{GraphInfo, create_graph_from_binary};
use mlperf_postprocess::ssd_small as native;
use pyo3::{PyResult, Python};
use pyo3::types::PyModule;

use crate::{pyclass, pymethods};

#[pyclass]
pub struct RustPostprocessor(native::RustPostprocessor);

#[pyclass]
pub struct CppPostprocessor(native::CppPostprocessor);

pub(crate) fn register_module(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustPostprocessor>()?;
    m.add_class::<CppPostprocessor>()?;

    Ok(())
}

#[pymethods]
impl RustPostprocessor {
    #[new]
    fn new(dfg: &[u8]) -> Self {
        let graph = create_graph_from_binary(dfg).unwrap();
        RustPostprocessor(native::RustPostprocessor::new(&graph))
    }
}

#[pymethods]
impl CppPostprocessor {
    #[new]
    fn new(dfg: &[u8]) -> Self {
        let graph = create_graph_from_binary(dfg).unwrap();
        CppPostprocessor(native::CppPostprocessor::new(&graph))
    }
}

