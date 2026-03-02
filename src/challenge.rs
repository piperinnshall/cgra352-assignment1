use anyhow::Result;
use opencv::core::{CV_32F, Mat, MatExprTraitConst, Vector};

// Sum pixel values
// Divide values by sum (probability)
// 2 (value) / 10 (sum) = 8 (20% chance of being intensity 8)
fn probability_density() { }

/**
Converts an image Matrix to a histogram
*/
pub fn convert_histogram(m: &Mat) -> Result<Mat> {
    let images = Vector::<Mat>::from(vec![m.clone()]);
    let channels = Vector::from_slice(&[0]);
    let mut hist = Mat::zeros(256, 1, CV_32F)?.to_mat()?;
    let hist_size = Vector::from_slice(&[256]);
    let ranges = Vector::from_slice(&[0f32, 256f32]);

    opencv::imgproc::calc_hist(
        &images,
        &channels,
        &Mat::default(), // no mask
        &mut hist,
        &hist_size,
        &ranges,
        false,
    )?;

    Ok(hist)
}
