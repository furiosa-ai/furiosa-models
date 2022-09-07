use mlperf_postprocess::common::graph::create_graph_from_binary_with_header;
use mlperf_postprocess::common::ssd_postprocess::DetectionResult;
use mlperf_postprocess::common::uninitialized_vec;
use mlperf_postprocess::ssd_small as native;
use numpy::PyArrayDyn;
use pyo3::{
    self,
    exceptions::PyValueError,
    types::{PyList, PyModule},
    PyErr, PyResult, Python,
};

use crate::{pyclass, pymethods, pymodule};

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

/// CppPostProcessor
///
/// It takes a DFG whose unlower part is removed.
/// The DFG binary must have magic number in its head.
///
/// Args:
///     dfg (bytes): a binary of DFG IR
#[pyclass]
#[pyo3(text_signature = "(dfg: bytes)")]
pub struct CppPostProcessor(native::CppPostprocessor);

#[pymodule]
pub(crate) fn ssd_mobilenet(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustPostProcessor>()?;
    m.add_class::<CppPostProcessor>()?;

    Ok(())
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct BoundingBox {
    #[pyo3(get)]
    left: f32,
    #[pyo3(get)]
    top: f32,
    #[pyo3(get)]
    right: f32,
    #[pyo3(get)]
    bottom: f32,
}

#[pymethods]
impl BoundingBox {
    fn __repr__(&self) -> String {
        format!(
            "BoundingBox(left: {}, top: {}, right: {}, bottom: {})",
            self.left, self.top, self.right, self.bottom
        )
    }

    fn __str__(&self) -> String {
        format!(
            "(left: {}, top: {}, right: {}, bottom: {})",
            self.left, self.top, self.right, self.bottom
        )
    }
}

#[pyclass]
#[derive(Debug)]
pub struct PyDetectionResult {
    #[pyo3(get)]
    pub left: f32,
    #[pyo3(get)]
    pub right: f32,
    #[pyo3(get)]
    pub top: f32,
    #[pyo3(get)]
    pub bottom: f32,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub class_id: i32,
}

#[pymethods]
impl PyDetectionResult {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        format!("{:?}", self)
    }
}

impl PyDetectionResult {
    pub fn new(r: DetectionResult) -> Self {
        PyDetectionResult {
            left: r.bbox.px1,
            right: r.bbox.px2,
            top: r.bbox.py1,
            bottom: r.bbox.py2,
            score: r.score,
            class_id: r.class as i32,
        }
    }
}

fn convert_to_slices(inputs: &PyList) -> PyResult<Vec<&[u8]>> {
    if inputs.len() != OUTPUT_NUM {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "expected 12 input tensors but got {}",
            inputs.len()
        )));
    }

    let mut memories: Vec<&[u8]> = unsafe { uninitialized_vec(OUTPUT_NUM) };
    for (index, tensor) in inputs.into_iter().enumerate() {
        let tensor = tensor.downcast::<PyArrayDyn<i8>>()?;
        if !tensor.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "{}th tensor is not C-contiguous".to_string(),
            ));
        }
        let slice: &[u8] = unsafe {
            let raw_slice = tensor.as_slice()?;
            std::slice::from_raw_parts(raw_slice.as_ptr() as *const u8, raw_slice.len())
        };
        memories[index] = slice;
    }

    Ok(memories)
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
        let slices = convert_to_slices(inputs)?;
        Ok(self
            .0
            .postprocess_impl(0f32, &slices)
            .0
            .into_iter()
            .map(PyDetectionResult::new)
            .collect())
    }
}

#[pymethods]
impl CppPostProcessor {
    #[new]
    fn new(dfg: &[u8]) -> PyResult<Self> {
        let graph = create_graph_from_binary_with_header(dfg)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("invalid DFG format: {}", e)))?;

        Ok(Self(native::CppPostprocessor::new(&graph)))
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
        let slices = convert_to_slices(inputs)?;
        Ok(self
            .0
            .postprocess_impl2(0f32, &slices)
            .0
            .into_iter()
            .map(PyDetectionResult::new)
            .collect())
    }
}
