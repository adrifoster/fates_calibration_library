"""Functions for tensorflow calibraition"""

def squared_z_loss(y_pred, target, stdev, y_var=None):
    z = tf.abs((y_pred - target)/(stdev + 1e-8))
    return tf.reduce_sum(z**2, axis=1)

def implausibility_loss(y_pred, target, stdev, y_var):
    total_variance = y_var + stdev**2 + 1e-8 
    z = tf.abs(y_pred - target) / tf.sqrt(total_variance)
    return tf.reduce_sum(z**2, axis=1)

def default_penalty_l1(X, X_default):
    return tf.reduce_sum(tf.abs(X - X_default), axis=1)

def barrier_penalty(X):
    return tf.reduce_mean(1.0 / (X + 1e-6) + 1.0 / (1.0 - X + 1e-6))

@tf.function
def optimization_step_batch(X, X_fixed, indices,
                            emulator_array, targets, stdevs, X_default, optimizer,
                            config):
    
    # grab and check config values
    lambda_penalty = config.get('lambda_penalty', None)
    if lambda_penalty is not None:
        if lambda_penalty <= 0.0:
            raise ValueError("lambda_penalty must be > 0.")
    
    barrier_strength = config.get('barrier_strenth', 0.0)
    if barrier_strength < 0.0:
        raise ValueError("barrier_strength must be >= 0.")
    
    earlystop_pct = config.get('earlystop_pct', 75.0)
    if earlystop_pct <= 0.0:
        raise ValueError("earlystop_pct must be > 0.")
    
    # batch size
    batch_size = tf.shape(X)[0]  
    
    # tile default params to match batch size
    X_default_tiled = tf.tile(X_default, [batch_size, 1])

    with tf.GradientTape() as tape:
        
        per_model_losses = []
        
        # loop over each emulator/target/stddev
        for emulator, target, stddev in zip(emulator_array, targets, stdevs):
            
            # expand target and stdev to batch shape
            target_tiled = tf.tile(tf.reshape(target, (1, -1)), [batch_size, 1])
            stdev_tiled = tf.tile(tf.reshape(stddev, (1, -1)), [batch_size, 1])
            
            X_full = rebuild_full_X(X, X_fixed, indices)
            y_pred, y_var = emulator(X_full)
            
            # calculate loss
            z = config['loss_fn'](y_pred, target_tiled, stdev_tiled, y_var)
            loss_i = tf.reshape(z, [-1])
            per_model_losses.append(loss_i)
            
        combined_loss = tf.add_n(per_model_losses)
        
        if lambda_penalty is not None:
            penalty_per_sample = config['default_penalty_fn'](X, X_default_tiled)  # shape: [batch]
            default_penalty = tf.maximum(penalty_per_sample / lambda_penalty, 1.0)
        else:
            default_penalty = tf.ones_like(combined_loss)
            
        if barrier_strength > 0.0:
            # penalty for moving too close to bounds [0, 1]
            barrier = config['barrier_penalty_fn'](X)
            barrier_penalty = (1.0 + barrier_strength * barrier)
        else:
            barrier_penalty = 1.0

        penalized_loss = combined_loss * default_penalty * barrier_penalty
        
        total_loss = tf.reduce_mean(penalized_loss)
        max_z = tfp.stats.percentile(combined_loss, earlystop_pct, 
                                     interpolation='midpoint')
         
    grads = tape.gradient(total_loss, [X])
    optimizer.apply_gradients(zip(grads, [X]))
    X.assign(tf.clip_by_value(X, 0.0, 1.0))

    return total_loss, max_z, penalized_loss

    
def run_optimization_tf(X: tf.Variable, X_fixed, indices, emulator_array: List[Callable], targets: List[tf.Tensor], 
                     stdevs: List[tf.Tensor], x_default: tf.Tensor, config: Dict) -> Tuple[tf.Tensor, Dict[str, List[float]]]:
    """Run optmization loop with configurable parameters

    Args:
        X (tf.Variable): Optimizable parameter tensor
        emulator_array (List[Callable]): List of emulators returning (mean, variance)
        targets (List[tf.Tensor]): List of observation targets (1D tensors)
        stdevs (List[tf.Tensor]): List of observational standard deviations (1D tensors)
        x_default (tf.Tensor): Default parameter tensor
        config (Dict): Dictionary containing all config options.

    Returns:
        Tuple[tf.Tensor, Dict[str, List[float]]]:
            - x_opt: final optimized parameters (numpy array)
            - logs: Dictionary of loss histories
    """
    
    if 'checkpoint_dir' not in config:
        raise ValueError("Missing required config key: 'checkpoint_dir'")
    
    checkpoint_dir = config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # learning rate schedule
    learning_rate = config.get('learning_rate', 1e-3)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=config.get('lr_decay_steps', 300),
        decay_rate=0.5,
        staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # ensure optimizer tracks 'x'
    _ = optimizer.iterations # touch optimizer to ensure it is initialized
    optimizer.apply_gradients([(tf.zeros_like(X), X)]) 
        
    # history trackers
    logs = {
        'total_loss': [],
        'max_z': [],
        'losses': [],
    }
    
    for step in range(config.get('maxiter', 3000)):
        
        total_loss, max_z, losses = optimization_step_batch(X, X_fixed, indices, emulator_array, targets, stdevs, x_default, optimizer, config)
        
        # log history
        logs['total_loss'].append(total_loss.numpy())
        logs['max_z'].append(max_z.numpy())
        logs['losses'].append(losses.numpy())
        
        # periodic printout
        if step % 10 == 0:
            tf.print(f"Step {step:03d}: total={total_loss:.6f} max_z={max_z:.6f}")
            
        # save checkpoints
        if step % config.get('checkpoint_n', 10) == 0:
            checkpoint = {
                'step': step,
                'params': X.numpy(),
                'loss': total_loss.numpy()
            }
            path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pkl')
            try:
                with open(path, 'wb') as f:
                    pickle.dump(checkpoint, f)
            except Exception as e:
                print(f"WARNING: Failed to save checkpoint at step {step}: {e}")
                
        # early stopping based on max implausibility
        if tf.reduce_max(max_z) <= config.get('epsilon', 0.5):
            print(f"Converged at step {step}")
            tf.print(f"Step {step:03d}: total={total_loss:.6f} max_z={max_z:.6f}")
            break

    return X.numpy(), logs