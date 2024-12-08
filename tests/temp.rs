#[cfg(test)]
mod tests {
    use std::char::decode_utf16;
    use moseiik::main::compute_mosaic;
    use moseiik::main::Options;
    use image::open;
    use image::ImageReader;

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_x86() {

        // test avx2 or sse2 if available


        let image_path = "assets/kit.jpeg";     // path of the reference image 
        let output_path = "assets/out_x86.png"; // path of the output image
        let tiles_path = "assets/moseiik_test_images/images".to_owned();

        let opt = Options{
            image: image_path.to_owned(),
            output: output_path.to_owned(),
            tiles: tiles_path,
            scaling: 1,
            tile_size: 25,
            remove_used: false,
            verbose: false,
            simd: true,         // set to true because it's necessary to use the l1_x86_sse2 function 
            num_thread: 8,
        };

        // Call the function under test
        compute_mosaic(opt);

        // Load the generated image corresponding to the output
        let generated_image = ImageReader::open(output_path).unwrap().decode().unwrap().into_rgb8();

        // Load the ground truth image corresponding to the reference
        let truth_image_path = "assets/ground-truth-kit.png";
        let truth_image = ImageReader::open(truth_image_path).unwrap().decode().unwrap().into_rgb8();

        // Compare images dimensions between the output and the reference
        assert_eq!(generated_image.dimensions(),truth_image.dimensions());

    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64() {


        let image_path = "assets/kit.jpeg"; // path of the reference image
        let output_path = "assets/out_aarch64.png"; // path of the output image
        let tiles_path = "assets/moseiik_test_images/images".to_owned();

        let opt = Options{
            image: image_path.to_owned(),
            output: output_path.to_owned(),
            tiles: tiles_path,
            scaling: 1,
            tile_size: 25,
            remove_used: false,
            verbose: false,
            simd: true,         // set to true because it's necessary to use the l1_neon function 
            num_thread: 8,
        };

        // Call the function under test
        compute_mosaic(opt);

        // Load the generated image corresponding to the output
        let generated_image = ImageReader::open(output_path).unwrap().decode().unwrap().into_rgb8();

        // Load the ground truth image corresponding to the reference
        let truth_image_path = "assets/ground-truth-kit.png";
        let truth_image = ImageReader::open(truth_image_path).unwrap().decode().unwrap().into_rgb8();

        // Compare images dimensions between the output and the reference
        assert_eq!(generated_image.dimensions(),truth_image.dimensions());

    }

    #[test]
    fn test_generic() {



        let image_path = "assets/kit.jpeg"; // path of the reference image
        let output_path = "assets/out_generic.png";  // path of the output image
        let tiles_path = "assets/moseiik_test_images/images".to_owned();

        let opt = Options{
            image: image_path.to_owned(),   // to_owned() function used to change the type to String
            output: output_path.to_owned(),
            tiles: tiles_path,
            scaling: 1,
            tile_size: 25,
            remove_used: false,
            verbose: false,
            simd: false,        // set to false in the generic case
            num_thread: 8,
        };

        // Call the function under test
        compute_mosaic(opt);

        // Load the generated image corresponding to the output
        let generated_image = ImageReader::open(output_path).unwrap().decode().unwrap().into_rgb8();

        // Load the ground truth image corresponding to the reference
        let truth_image_path = "assets/ground-truth-kit.png";
        let truth_image = ImageReader::open(truth_image_path).unwrap().decode().unwrap().into_rgb8();

        // Compare images dimensions between the output and the reference
        assert_eq!(generated_image.dimensions(),truth_image.dimensions());
        
        
    }
}
