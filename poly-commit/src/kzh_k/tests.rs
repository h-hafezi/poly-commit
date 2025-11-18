use ark_bn254::Bn254;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::test_rng;
use rand_chacha::ChaCha20Rng;
use crate::kzh_k::KZH;
use crate::LabeledPolynomial;
use crate::utils::test_sponge;
use ark_bn254::Fr as Bn254Fr;
use ark_poly::MultilinearExtension;
use rand::SeedableRng;
use crate::PolynomialCommitment;

type KzhBn254 = KZH<Bn254, DenseMultilinearExtension<Bn254Fr>>;

// ******** auxiliary test functions ********

fn rand_poly<F: PrimeField>(num_vars: Option<usize>, rng: &mut ChaCha20Rng) -> DenseMultilinearExtension<F> {
    match num_vars {
        Some(n) => DenseMultilinearExtension::rand(n, rng),
        None => panic!("Must specify the number of variables"),
    }
}

fn constant_poly<F: PrimeField>(num_vars: Option<usize>, rng: &mut ChaCha20Rng) -> DenseMultilinearExtension<F> {
    match num_vars {
        Some(0) => DenseMultilinearExtension::rand(0, rng),
        _ => panic!("Must specify the number of variables: 0"),
    }
}

fn rand_point<F: PrimeField>(num_vars: Option<usize>, rng: &mut ChaCha20Rng) -> Vec<F> {
    match num_vars {
        Some(n) => (0..n).map(|_| F::rand(rng)).collect(),
        None => panic!("Must specify the number of variables"),
    }
}

// ****************** tests ******************

#[test]
fn test_kzh_construction() {
    // Desired number of variables
    let n = 21;

    let chacha = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();

    let pp = KzhBn254::setup(1, Some(n), chacha).unwrap();

    let (ck, vk) = KzhBn254::trim(&pp, 1, 1, None).unwrap();

    let l_poly = LabeledPolynomial::new(
        "test_poly".to_string(),
        rand_poly::<Bn254Fr>(Some(n), chacha).clone(),
        None,
        None,
    );

    let point: Vec<Bn254Fr> = rand_point(Some(n), chacha);
    let value = l_poly.evaluate(&point);

    let (c, rands) = KzhBn254::commit(&ck, &[l_poly.clone()], Some(chacha)).unwrap();

    // Dummy argument
    let mut test_sponge = test_sponge::<Bn254Fr>();

    let proof = KzhBn254::open(
        &ck,
        &[l_poly],
        &c,
        &point,
        &mut (test_sponge.clone()),
        &rands,
        Some(chacha),
    ).unwrap();


    assert!(KzhBn254::check(
        &vk,
        &c,
        &point,
        [value],
        &proof,
        &mut test_sponge,
        Some(chacha),
    ).unwrap());
}