use burn::tensor::{backend::Backend, Tensor};

pub mod visual_navigation;
pub mod pendulum;
pub mod maze;
pub mod bouncing_ball;

pub trait Environment {
    fn reset<B: Backend>(&mut self, device: &B::Device) -> Tensor<B, 3>;
    fn step<B: Backend>(
        &mut self,
        action: &[f32],
        device: &B::Device,
    ) -> (Tensor<B, 3>, f32, bool);
    fn obs_shape(&self) -> [usize; 3];
    fn action_dim(&self) -> usize;
    fn max_steps(&self) -> usize;
}
