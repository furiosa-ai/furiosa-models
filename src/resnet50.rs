use mlperf_postprocess::common::graph::create_graph_from_binary_with_header;
use mlperf_postprocess::resnet50 as native;
use pyo3::{
    self,
    exceptions::PyValueError,
    types::{PyList, PyModule},
    PyErr, PyResult, Python,
};

use crate::common::convert_to_slices;
use crate::{pyclass, pymethods, pymodule};

const OUTPUT_NUM: usize = 1;

/// PostProcessor
///
/// It takes a DFG whose unlower part is removed.
/// The DFG binary must have magic number in its head.
///
/// Args:
///     dfg (bytes): a binary of DFG IR
#[pyclass]
#[pyo3(text_signature = "(dfg: bytes)")]
pub struct PostProcessor(native::Resnet50PostProcessor);

#[pymodule]
pub(crate) fn resnet50(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PostProcessor>()?;

    Ok(())
}

#[pymethods]
impl PostProcessor {
    #[new]
    fn new(dfg: &[u8]) -> PyResult<Self> {
        let graph = create_graph_from_binary_with_header(dfg)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("invalid DFG format: {}", e)))?;

        Ok(Self(native::Resnet50PostProcessor::new(&graph)))
    }

    /// Evaluate the postprocess
    ///
    /// Args:
    ///     inputs (Sequence[numpy.ndarray]): Input tensors
    ///
    /// Returns:
    ///     List[PyDetectionResult]: Output tensors
    #[pyo3(text_signature = "(self, inputs: Sequence[numpy.ndarray])")]
    fn eval(&self, inputs: &PyList) -> PyResult<usize> {
        if inputs.len() != OUTPUT_NUM {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "expected {} input tensors but got {}",
                OUTPUT_NUM,
                inputs.len()
            )));
        }

        let slices = convert_to_slices(inputs)?;
        Ok(self.0.postprocess(slices[0]))
    }
}
