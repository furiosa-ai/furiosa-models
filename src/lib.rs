
#![allow(dead_code)]
use pyo3::prelude::*;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};


create_exception!(furiosa_models_native, NativeException, PyException, "Some description.");
/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[derive(Debug)]
struct CanBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    area: f32,
}

impl CanBox {
    fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self { 
        Self { x1, y1, x2, y2, 
            area: (x2-x1)*(y2-y1) } 
    }
}

fn fmax(a: f32, b: f32) -> f32 {
    if b.is_nan() || b <= a { a } else { b }
}
fn fmin(a: f32, b: f32) -> f32 {
    if b.is_nan() || b >= a { a } else { b }
}

#[pyfunction]
#[pyo3(name = "nms_internal_ops_fast_rust")]
fn nms_index(boxes: PyReadonlyArray2<f32>, 
    scores: PyReadonlyArray1<f32>,
    iou_threshold: f32,
    eps: f32 ) -> PyResult<Vec<usize>>{
        let x = boxes.as_array();
        let scores = scores.as_array();
        let num_boxes = x.shape()[0];
        if num_boxes == 0 {
            return Ok(Vec::new());
        }
        let x_stride: usize = x.shape()[1];
        let x = x.as_slice().ok_or(
            NativeException::new_err("slice error".to_string())
        )?;

        let scores: Vec<f32> = scores.iter().map( |v| *v).collect();
        let mut box_index: Vec<usize> = (0..num_boxes).into_iter().collect();
        let can_box: Vec<CanBox> = (0..num_boxes).into_iter().map( |i| {
            let index = i * x_stride;
            return CanBox::new( x[index+0],x[index+1],x[index+2],x[index+3] );
        }).collect();

        box_index.sort_unstable_by(|&i, &j| 
            scores[i].partial_cmp(&scores[j]).unwrap()
        );
        let mut pick_indices: Vec<usize> = Vec::new();

        while box_index.len() > 0 {
            let last = *box_index.last().unwrap();
            pick_indices.push(last);

            let c_last = &can_box[last];
            box_index = box_index.into_iter().map( |i| {
                    let ci = &can_box[i];
                    let xx1: f32 = fmax(ci.x1, c_last.x1);
                    let yy1: f32 = fmax(ci.y1, c_last.y1);
                    let xx2: f32 = fmin(ci.x2, c_last.x2);
                    let yy2: f32 = fmin(ci.y2, c_last.y2);

                    let w = fmax(0., xx2 - xx1);
                    let h = fmax(0., yy2 - yy1);

                    let inter = w*h;
                    let iou = inter / (ci.area + c_last.area - inter + eps);
                    return (i, iou);
                }).filter(|v| v.1 <= iou_threshold)
                .map( |v| v.0)
                .collect();
        }
        return Ok(pick_indices);
}

// NOTE: This docstring is unshown in Python level
/// A Python module implemented in Rust.
#[pymodule]
fn furiosa_models_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(nms_index, m)?)?;
    Ok(())
}
