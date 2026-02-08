# Spatial Transformation Framework for Deep Learning (GeoDL)

The tree-based version is at: [Tree-based-models](/Tree-based-models)

## Major updates
- Added a new class `GeoDL` in `GeoDL.py` with standard functions: `fit`, `predict`, and `evaluate`.
- Added a simplified demo script `STAR_main.py` to show end-to-end usage.
- Standardized configuration in `config.py` (with a compatibility shim `paras.py`).

## Key differences from standard deep learning
- GeoDL requires an additional input `X_group` that assigns each sample to a spatial group.
- Groups are the minimum spatial units for partitioning. You can define them using raw coordinates or grid-based locations.

## Sample demo data
- Features X ([X_example.npy](https://drive.google.com/file/d/1-DbkQusMbpcS72NYuKe3kWN_tPNonQD3/view?usp=sharing))
- Labels y ([y_example.npy](https://drive.google.com/file/d/1-H7ZE8OoqJfhXSCCccFtpZp7vSZXypLC/view?usp=share_link))

## Example usage
For details, see `STAR_main.py`.

Create a new model:
```
from GeoDL import GeoDL
geodl = GeoDL(model_choice="DNN")  # or "UNet"
```

Load demo data:
```
from data import load_demo_data
X, y = load_demo_data()
```

Define groups and split:
```
from initialization import init_X_info
X_group, X_set, _, _ = init_X_info(X, y)
```

If you have raw locations, you can also generate groups explicitly:
```
from customize import GroupGenerator
group_gen = GroupGenerator(xmin, xmax, ymin, ymax, step_size)
X_group = group_gen.get_groups(X_loc)
```

Train and evaluate:
```
geodl.fit(X, y, X_group, X_set=X_set)
test_list = (X_set == 2)
pre, rec, f1 = geodl.evaluate(X[test_list], y[test_list], X_group[test_list])
```

## Notes for segmentation (UNet)
- Use `--use_segmentation` in `STAR_main.py` to load image patches.
- Adjust `IMG_SIZE`, `PATCH_SIZE`, and `PATCH_STEP_SIZE` in `config.py` if needed.
