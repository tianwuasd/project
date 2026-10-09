# U-Net author-source alignment

User request: replace the custom U-Net backbone with one aligned to the original authors' published implementation. User explicitly allows retaining gate as an optional extension. Keep separate data/train/predict/evaluate tasks and server integration; publish to the existing project repository.

Sources: original Freiburg 2015 Caffe release phseg_v5-train.prototxt and official 2016 3D no-BN network. Preserve exact source files, archive hashes and licenses. This is a project PyTorch port, never an official PyTorch release or a claim of original CHD-paper equivalence.

Core: valid 3x3 convolutions, ReLU, max pooling, learned upconvolution plus ReLU, centrally cropped skip features, original channel schedules, 2D dropout and no 3D normalization. Raw core returns smaller valid output. Optional probability gate stays outside this core. Image-grid adapter uses symmetric mirror extension at input and centered output crop so existing labels remain aligned. No resizing of valid outputs to disguise lost field of view.

Default CHD profiles keep practical widths but official depth (3D four / 2D five); smoke alone uses reduced depth/width. Add author-width profile (3D base32/four, 2D base64/five), gate on for the four anatomical stages. User explicitly requires gate ON by default, with a per-stage disable option; blood2d retains its previous ungated behavior. Old checkpoint format rejected with retrain guidance. Metadata records source/architecture, adapter and gate.

Validation: compare actual forward layer signatures with bundled source definitions; verify exact official shapes using meta tensors; independent functional execution of the source graph at reduced width; real forward/backward; odd/small grid geometry; gate on/off; old checkpoint rejection; whole six-stage smoke and package install. Original Caffe numeric parity and full GPU performance remain unverified.
