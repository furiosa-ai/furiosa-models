use itertools::Itertools;
use mlperf_postprocess::common::graph::{create_graph_from_binary_with_header, GraphInfo};
use mlperf_postprocess::common::{uninitialized_vec, ssd_postprocess::Postprocess};
use mlperf_postprocess::ssd_small as native;
use numpy::{
    ndarray::ArrayViewD,
    PyArrayDyn,
    PyArray,
    PyReadonlyArrayDyn,
};
use pyo3::{AsPyPointer, exceptions::PyValueError, PyErr, PyObject, PyRef, PyResult, Python, types::{PyList, PyModule}};

use crate::{pyclass, pymethods, pymodule};

const OUTPUT_NUM: usize = 12;

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

    fn process(&self, inputs: &PyList) -> PyResult<()> {
        // eprintln!("inputs len: [{}]", inputs.shape().iter().map(|s| s.to_string()).join(","));
        // self.0.postprocess(0,);

        if inputs.len() != OUTPUT_NUM {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "expected 12 input tensors but got {}",
                inputs.len()
            )))
        }

        let mut memories: Vec<&[u8]> = unsafe { uninitialized_vec(OUTPUT_NUM) };
        for tensor in inputs {
            let tensor = tensor.downcast::<PyArrayDyn<i8>>()?;
            if !tensor.is_c_contiguous() {
                return Err(PyErr::new::<PyValueError, _>(
                    "{}th tensor is not C-contiguous".to_string()))
            }
            let slice: &[u8] = unsafe {
                let raw_slice = tensor.as_slice()?;
                std::slice::from_raw_parts(raw_slice.as_ptr() as *const u8, raw_slice.len())
            };
            memories.push(slice);
        }

        let results = self.0.postprocess(0f32, &memories);
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

    fn process(&self, inputs: &PyList) -> PyResult<()> {
        // eprintln!("inputs len: [{}]", inputs.shape().iter().map(|s| s.to_string()).join(","));
        // self.0.postprocess(0,);

        if inputs.len() != OUTPUT_NUM {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "expected 12 input tensors but got {}",
                inputs.len()
            )))
        }

        let mut memories: Vec<&[u8]> = unsafe { uninitialized_vec(OUTPUT_NUM) };
        for tensor in inputs {
            let tensor = tensor.downcast::<PyArrayDyn<i8>>()?;
            if !tensor.is_c_contiguous() {
                return Err(PyErr::new::<PyValueError, _>(
                    "{}th tensor is not C-contiguous".to_string()))
            }
            let slice: &[u8] = unsafe {
                let raw_slice = tensor.as_slice()?;
                std::slice::from_raw_parts(raw_slice.as_ptr() as *const u8, raw_slice.len())
            };
            memories.push(slice);
        }

        let results = self.0.postprocess(0f32, &memories);
        Ok(())
    }
}

