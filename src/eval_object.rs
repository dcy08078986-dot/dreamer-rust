//! Object-Centric Evaluation Metrics
//!
//! Computes position MSE, velocity MSE, ADE, FDE, collision accuracy
//! from ObjectWorld ground-truth state vs model predictions.
//!
//! Usage: after training, run
//!   DREAMER_EVAL=1 DREAMER_ENV=object_world ./target/release/dreamer-rust

use crate::envs::object_world::{ObjectWorldState, Ball};

/// Position metrics for a single ball prediction vs ground truth.
#[derive(Clone, Debug)]
pub struct PositionMetrics {
    pub mse: f32,     // mean squared error over trajectory
    pub ade: f32,     // average displacement error
    pub fde: f32,     // final displacement error
    pub vel_mse: f32, // velocity MSE
}

/// Compute position/velocity metrics over a predicted trajectory.
///
/// `pred_balls[t][i]` = predicted ball i at timestep t.
/// `gt_states[t]` = ground truth state at timestep t.
pub fn compute_tracking_metrics(
    pred_balls: &[Vec<Ball>],
    gt_states: &[ObjectWorldState],
) -> Vec<PositionMetrics> {
    let num_balls = gt_states[0].balls.len();
    let t_steps = pred_balls.len().min(gt_states.len());
    if t_steps < 2 {
        return vec![PositionMetrics { mse: 0.0, ade: 0.0, fde: 0.0, vel_mse: 0.0 }; num_balls];
    }

    let mut metrics = Vec::new();
    for i in 0..num_balls {
        let mut pos_err_sum = 0.0f32;
        let mut vel_err_sum = 0.0f32;
        let mut displacement_sum = 0.0f32;
        let mut final_err = 0.0f32;

        for t in 0..t_steps {
            let gt = &gt_states[t].balls[i];
            let pred = &pred_balls[t][i];
            let dx = pred.x - gt.x;
            let dy = pred.y - gt.y;
            pos_err_sum += dx * dx + dy * dy;
            displacement_sum += (dx * dx + dy * dy).sqrt();

            if t > 0 {
                let gt_vx = gt.x - gt_states[t - 1].balls[i].x;
                let gt_vy = gt.y - gt_states[t - 1].balls[i].y;
                let pred_vx = pred.x - pred_balls[t - 1][i].x;
                let pred_vy = pred.y - pred_balls[t - 1][i].y;
                vel_err_sum += (pred_vx - gt_vx).powi(2) + (pred_vy - gt_vy).powi(2);
            }
        }

        let last_gt = &gt_states[t_steps - 1].balls[i];
        let last_pred = &pred_balls[t_steps - 1][i];
        let dx = last_pred.x - last_gt.x;
        let dy = last_pred.y - last_gt.y;
        final_err = (dx * dx + dy * dy).sqrt();

        metrics.push(PositionMetrics {
            mse: pos_err_sum / t_steps as f32,
            ade: displacement_sum / t_steps as f32,
            fde: final_err,
            vel_mse: if t_steps > 1 { vel_err_sum / (t_steps - 1) as f32 } else { 0.0 },
        });
    }
    metrics
}

/// Collision prediction accuracy.
#[derive(Clone, Debug)]
pub struct CollisionMetrics {
    pub accuracy: f32,         // whether predicted collision in correct time window
    pub time_error: f32,       // mean absolute error in collision time
    pub recall: f32,           // how many real collisions were predicted
    pub precision: f32,        // how many predicted collisions were real
}

/// Compare predicted collision events with ground truth.
/// `pred_events` = Vec<(timestep, ball_a, ball_b)> from model.
/// `gt_events` = Vec<(timestep, ball_a, ball_b)> from env.
pub fn compute_collision_metrics(
    pred_events: &[(usize, usize, usize)],
    gt_events: &[(usize, usize, usize)],
    time_window: usize,
) -> CollisionMetrics {
    let mut matched = 0;
    let mut time_errors = Vec::new();
    let mut pred_matched = vec![false; pred_events.len()];
    let mut gt_matched = vec![false; gt_events.len()];

    for (pi, &(pt, pa, pb)) in pred_events.iter().enumerate() {
        for (gi, &(gt, ga, gb)) in gt_events.iter().enumerate() {
            if (pa == ga && pb == gb) || (pa == gb && pb == ga) {
                let dt = (pt as isize - gt as isize).unsigned_abs();
                if dt <= time_window {
                    matched += 1;
                    time_errors.push(dt as f32);
                    pred_matched[pi] = true;
                    gt_matched[gi] = true;
                    break;
                }
            }
        }
    }

    let accuracy = if gt_events.is_empty() {
        if pred_events.is_empty() { 1.0 } else { 0.0 }
    } else {
        matched as f32 / gt_events.len().max(1) as f32
    };

    let time_error = if time_errors.is_empty() {
        0.0
    } else {
        time_errors.iter().sum::<f32>() / time_errors.len() as f32
    };

    let recall = if gt_events.is_empty() { 1.0 } else {
        gt_matched.iter().filter(|&&x| x).count() as f32 / gt_events.len() as f32
    };

    let precision = if pred_events.is_empty() { 0.0 } else {
        pred_matched.iter().filter(|&&x| x).count() as f32 / pred_events.len() as f32
    };

    CollisionMetrics { accuracy, time_error, recall, precision }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::object_world::Ball;

    #[test]
    fn test_tracking_metrics_perfect() {
        let balls = vec![Ball { x: 0.5, y: 0.5, vx: 0.0, vy: 0.0, radius: 0.04, mass: 0.0016, color: [255; 3] }];
        let pred = vec![balls.clone(), balls.clone()];
        let gt = vec![
            ObjectWorldState { balls: balls.clone(), walls: vec![], target: crate::envs::object_world::Target { x: 0.5, y: 0.9, radius: 0.04 }, step: 0, collision_events: vec![] },
            ObjectWorldState { balls: balls.clone(), walls: vec![], target: crate::envs::object_world::Target { x: 0.5, y: 0.9, radius: 0.04 }, step: 1, collision_events: vec![] },
        ];
        let m = compute_tracking_metrics(&pred, &gt);
        assert!(m[0].mse < 1e-6);
        assert!(m[0].fde < 1e-6);
    }

    #[test]
    fn test_collision_metrics() {
        let pred = vec![(5, 0, 1)];
        let gt = vec![(4, 0, 1)];
        let m = compute_collision_metrics(&pred, &gt, 2);
        assert!(m.accuracy > 0.9);
        assert!((m.time_error - 1.0).abs() < 1e-6);
    }
}
