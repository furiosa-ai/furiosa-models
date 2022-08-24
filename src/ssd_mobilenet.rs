use mlperf_postprocess::common::graph::{create_graph_from_binary_with_header, GraphInfo};
use mlperf_postprocess::common::ssd_postprocess::Postprocess;
use mlperf_postprocess::ssd_small as native;
use pyo3::{PyObject, PyRef, PyResult, Python};
use pyo3::types::PyModule;
use numpy::{
    PyReadonlyArrayDyn,
    ndarray::ArrayViewD,
};


use crate::{pyclass, pymethods, pymodule};

#[pyclass]
pub struct RustPostprocessor(native::RustPostprocessor);

#[pyclass]
pub struct CppPostprocessor(native::CppPostprocessor);

#[pymodule]
pub(crate) fn ssd_mobilenet(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustPostprocessor>()?;
    m.add_class::<CppPostprocessor>()?;

    Ok(())
}

#[pymethods]
impl RustPostprocessor {
    #[new]
    fn new(dfg: &[u8]) -> Self {
        let graph = create_graph_from_binary_with_header(dfg).unwrap();
        RustPostprocessor(native::RustPostprocessor::new(&graph))
    }

    fn process(self_: PyRef<'_, Self>, inputs: PyReadonlyArrayDyn<'_, i8>) -> PyResult<()> {
        eprintln!("inputs len: {}", inputs.len());
        //self.0.postprocess(0, )
        Ok(())
    }
}

#[pymethods]
impl CppPostprocessor {
    #[new]
    fn new(dfg: &[u8]) -> Self {
        let graph = create_graph_from_binary_with_header(dfg).unwrap();
        CppPostprocessor(native::CppPostprocessor::new(&graph))
    }
}

