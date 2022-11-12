use pyo3::{
    self,
    exceptions::PyValueError,
    types::{PyList, PyModule},
    PyErr, PyResult, Python,
};

use crate::common::{convert_to_slices, PyDetectionResult};
use crate::{pyclass, pymethods, pymodule};
use mlperf_postprocess::common::graph::create_graph_from_binary_with_header;
use mlperf_postprocess::common::ssd_postprocess::Postprocess;
use mlperf_postprocess::ssd_large as native;

const OUTPUT_NUM: usize = 12;

/// RustPostProcessor
///
/// It takes a DFG whose unlower part is removed.
/// The DFG binary must have magic number in its head.
///
/// Args:
///     dfg (bytes): a binary of DFG IR
#[pyclass]
#[pyo3(text_signature = "(dfg: bytes)")]
pub struct RustPostProcessor(native::RustPostprocessor);

#[pymodule]
pub(crate) fn ssd_resnet34(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustPostProcessor>()?;

    Ok(())
}

#[pymethods]
impl RustPostProcessor {
    #[new]
    fn new(dfg: &[u8]) -> PyResult<Self> {
        let graph = create_graph_from_binary_with_header(dfg)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("invalid DFG format: {}", e)))?;

        Ok(Self(native::RustPostprocessor::new(&graph)))
    }

    /// Evaluate the postprocess
    ///
    /// Args:
    ///     inputs (Sequence[numpy.ndarray]): Input tensors
    ///
    /// Returns:
    ///     List[PyDetectionResult]: Output tensors
    #[pyo3(text_signature = "(self, inputs: Sequence[numpy.ndarray])")]
    fn eval(&self, inputs: &PyList) -> PyResult<Vec<PyDetectionResult>> {
        if inputs.len() != OUTPUT_NUM {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "expected {} input tensors but got {}",
                OUTPUT_NUM,
                inputs.len()
            )));
        }

        let slices = convert_to_slices(inputs)?;
        Ok(self
            .0
            .postprocess(0f32, &slices)
            .0
            .into_iter()
            .map(PyDetectionResult::new)
            .collect())
    }
}
