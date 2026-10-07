# CHD CT implementation plan

Goal: runnable independent reproduction with explicit evidence boundaries.

Architecture: data contracts → stage-specific networks/training → aligned fusion → physical features → transparent rule evaluation. Python 3.11+, PyTorch 2.6+, NIfTI, YAML. Spec: [design.md](design.md).

## Global constraints

No patient data, source PDF, pretrained claims, or credentials in Git. Patient-separated manifests; original-space outputs; unknown evidence stays unknown. CPU smoke configuration must be visibly distinct from paper configuration.

## Tasks

- [x] Data: strict manifest/NIfTI validation, normalization, ROI and target construction; tests for leakage, orientation, empty masks, grid mismatch.
- [x] Networks: dimension-generic U-Net with spatial gate; bidirectional ConvLSTM; weighted Dice+CE; tests for gradients, shape, sequence dependence.
- [x] Training/inference: all eight train stages, validation-based checkpoints, center-slice recurrent supervision, fold ensemble, six-way class-compatible fusion, region growth; synthetic end-to-end test.
- [x] Features/diagnosis: anisotropic graph radius and widest path, contacts/components/position, user-supplied exact anatomical features, tri-state JSON rules; test disconnected and missing structures and threshold boundaries.
- [x] Evaluation/docs: Dice and multilabel metrics with explicit denominators, data guide, paper discrepancy map, CPU demo, independent review and verified publication to existing repository.

## Review focus

1. Reoriented or oblique input must not silently invert right/left.
2. No ground-truth segmentation may enter inference; absent classes must not become negative evidence.
3. Initial seven-class labels must not be interpreted as full eleven-class labels.
4. Cross-validation and undefined metric denominators must not inflate results.
5. Incompatible/random checkpoints must be identifiable; demo outputs must remain marked synthetic.
