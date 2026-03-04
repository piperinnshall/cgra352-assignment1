use anyhow::Result;
use opencv::core::{Mat, MatTraitConst, Vector};
use std::{fs::File, io::Write, path::Path};

/**
Saves a single-channel histogram `Mat` to a CSV file
*/
pub fn histogram_csv(filename: impl AsRef<Path>, hist: &Mat) -> Result<()> {
    let mut file = File::create(filename)?;
    for i in 0..hist.rows() {
        let val = *hist.at_2d::<f32>(i, 0)?;
        writeln!(file, "{}", val)?;
    }
    Ok(())
}

/**
Computes the probability density for each intensity level.
PDF[i] = pixel_count[i] / total_pixels
*/
fn probability_density(hist: &Mat) -> Result<Vec<f32>> {
    let total: f32 = (0..hist.rows())
        .map(|i| hist.at_2d::<f32>(i, 0).copied().unwrap_or(0.0))
        .sum();

    let pdf = (0..hist.rows())
        .map(|i| {
            let count = *hist.at_2d::<f32>(i, 0).unwrap_or(&0.0);
            if total > 0.0 {
                count / total
            } else {
                0.0
            }
        })
        .collect();

    Ok(pdf)
}

/**
Builds a lookup table (LUT) via the CDF of the PDF. equalized[i] = round(CDF[i] * 255)
*/
fn equalization_lut(pdf: &[f32]) -> Vec<u8> {
    let mut cdf = 0.0f32;
    pdf.iter()
        .map(|&p| {
            cdf += p;
            (cdf * 255.0).round() as u8
        })
        .collect()
}

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
        &Mat::default(),
        &mut hist,
        &hist_size,
        &ranges,
        false,
    )?;
    Ok(hist)
}

/**
Equalizes a grayscale image using its histogram's PDF.
Computes the histogram, derives the PDF and CDF, then applies
the resulting LUT to remap every pixel intensity.
*/
pub fn equalize_histogram(hist: &Mat, m: &Mat) -> Result<Mat> {
    let pdf = probability_density(&hist)?;
    let lut_data = equalization_lut(&pdf);
    let lut_mat = Mat::from_slice(&lut_data)?;
    let mut dst = Mat::default();
    opencv::core::lut(m, &lut_mat, &mut dst)?;
    Ok(dst)
}
