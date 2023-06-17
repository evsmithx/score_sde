# Current problems
1. Line 750: pmap not working when `params` are defined to be a static argument:
    ```
    score = pmap_score_fn(score_model, params, sample, time_steps)
    ValueError: Non-hashable static arguments are not supported. An error occurred during a call to 'score_fn' while trying to hash an object of type <class 'flax.core.frozen_dict.FrozenDict'>, FrozenDict({ 
    [...]
    ```

# Nice to have
1. Smaller network (EmmasTinyScoreNet). 
  1. Concatenation doesn't work.
2. Reduce the number of training images in the dataset.

