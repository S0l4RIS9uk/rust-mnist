use mnist::*;
use ndarray::prelude::*;
use ndarray_rand::RandomExt; // Make sure RandomExt is imported
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Network {
    shape: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    fn new(shape: Vec<usize>) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let uniform = Uniform::new(0.0, 1.0);

        for i in 1..shape.len() {
            let weight_matrix = Array2::<f64>::random((shape[i], shape[i - 1]), uniform.clone());
            let bias_matrix = Array2::<f64>::random((shape[i], 1), uniform.clone());
            weights.push(weight_matrix);
            biases.push(bias_matrix);
        }

        Network {
            shape,
            biases,
            weights,
        }
    }
}


fn main() {
    let args: Vec<String> = std::env::args().collect();
    let network = Network::new(args[1].split(',').map(|x| x.parse().unwrap()).collect());
    save_network(network);
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
    let shape = network
        .shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<String>>()
        .join(",");
    let json = serde_json::to_string(&network).unwrap();
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    std::fs::write(format!("./models/{}-{}.json", shape, date), json).unwrap();
}
