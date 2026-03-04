use anyhow::Result;
use opencv::core::{Mat, MatTraitConst, Vector};
use std::{fs::File, io::Write, path::Path};

// Sum pixel values
// Divide values by sum (probability)
// 2 (value) / 10 (sum) = 8 (20% chance of being intensity 8)
fn _probability_density() {}

/**
Converts an image Matrix to a histogram
*/
pub fn convert_histogram(m: &Mat) -> Result<Mat> {
    let images = Vector::<Mat>::from(vec![m.clone()]);
    let channels = Vector::from_slice(&[0]);
    let mut hist = Mat::default();
    let hist_size = Vector::from_slice(&[256]);
    let ranges = Vector::from_slice(&[0f32, 256f32]);

    opencv::imgproc::calc_hist(
        &images,
        &channels,
        &Mat::default(), // No mask.
        &mut hist,
        &hist_size,
        &ranges,
        false,
    )?;

    Ok(hist)
}

/**
Saves a single-channel histogram `Mat` to a CSV file
*/
pub fn histogram_csv(filename: impl AsRef<Path>, hist: &Mat) -> Result<()> {
    let mut file = File::create(filename)?;
    for i in 0..hist.rows() {
        let val = *hist.at_2d::<f32>(i, 0)?; // Single-channel.
        writeln!(file, "{}", val)?;
    }
    Ok(())
}
