mod core;
mod completion;
mod challenge;

use std::error::Error;

use anyhow::Result;
use opencv::{core::Vector, imgcodecs, imgproc};

fn main() -> Result<(), Box<dyn Error>> {
    let params = Vector::default();
    core(&params)?;
    completion(&params)?;
    challenge(&params)?;
    Ok(())
}

fn core(params: &Vector<i32>) -> Result<()> {
    let flower_bgr = imgcodecs::imread("assets/Flower.jpg", imgcodecs::IMREAD_UNCHANGED)?;
    let flower_hsv = core::convert_matrix_color_space(&flower_bgr, imgproc::COLOR_BGR2HSV)?;

    // Core 1
    let mut core_1 = Vec::new();
    core_1.append(&mut core::split_channels(&flower_bgr)?);
    core_1.append(&mut core::split_channels(&flower_hsv)?);

    let core_1_large = core::create_large_image(&core_1, 2, 3, opencv::core::CV_8UC1)?;
    imgcodecs::imwrite("assets/Core1.jpg", &core_1_large, &params)?;

    // Core 2
    let mut core_2 = Vec::new();
    let mut flower_hsv_split = core::multiply_channel(&flower_hsv)?;
    core_2.append(&mut flower_hsv_split.0);
    core_2.append(&mut flower_hsv_split.1);
    core_2.append(&mut flower_hsv_split.2);

    let core_2_large = core::create_large_image(&core_2, 3, 5, opencv::core::CV_8UC3)?;
    let core_2_large = core::convert_matrix_color_space(&core_2_large, imgproc::COLOR_HSV2BGR)?;
    imgcodecs::imwrite("assets/Core2.jpg", &core_2_large, &params)?;

    // Core 3
    let core_3 = core::euclidean_mask(&flower_bgr)?;
    imgcodecs::imwrite("assets/Core3.jpg", &core_3, &params)?;

    Ok(())
}

fn completion(params: &Vector<i32>) -> Result<()> {
    let flower_grey = imgcodecs::imread("assets/Flower.jpg", imgcodecs::IMREAD_GRAYSCALE)?;

    // Completion
    let (completion_1, completion_2, completion_3) = completion::edge_detection(&flower_grey)?;

    imgcodecs::imwrite("assets/Completion1.jpg", &completion_1, &params)?;
    imgcodecs::imwrite("assets/Completion2.jpg", &completion_2, &params)?;
    imgcodecs::imwrite("assets/Completion3.jpg", &completion_3, &params)?;

    Ok(())
}

fn challenge(params: &Vector<i32>) -> Result<(), Box<dyn Error>> {
    let building_grey = imgcodecs::imread("assets/Building.jpg", imgcodecs::IMREAD_GRAYSCALE)?;

    let histogram = challenge::convert_histogram(&building_grey)?;
    Ok(())
}

