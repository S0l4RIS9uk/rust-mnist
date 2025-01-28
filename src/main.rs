use mnist::*;
use ndarray::prelude::*;
use serde::{Serialize, Deserialize};


impl Network {
    fn new(shape: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        let biases = shape[1..]
            .iter()
            .map(|&y| Array2::random_using((y, 1), rand::distributions::StandardNormal, &mut rng))
            .map(|&y| Array2::random((y, 1), rand::distributions::StandardNormal))
        let weights = shape.windows(2)
            .map(|w| Array2::random_using((w[1], w[0]), rand::distributions::StandardNormal, &mut rng))
            .map(|w| Array2::random((w[1], w[0]), rand::distributions::StandardNormal))
        Network { shape, biases, weights }
    }
}

#[derive(Serialize, Deserialize)]
struct Network {
    shape: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let nework = Network::new(args.split(',').map(|x| x.parse().unwrap()).collect());

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);

    let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let _test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);
}


fn save_network(network: Network) {
    let shape = network.shape.iter().map(usize::to_string).collect::<Vec<String>>().join(",");
    let json = serde_json::to_string(&network).unwrap();
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    std::fs::write(format!("{}-{}.json", shape, date), json).unwrap();
}