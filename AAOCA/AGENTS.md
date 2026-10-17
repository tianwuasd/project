# AAOCA code maintenance

Read README.md, PROJECT_HISTORY.md and 项目升级使用说明.md first. This directory is the public code component of a local congenital-heart research project.

- Keep this directory independently runnable. Do not replace entry points with redirects into a private absolute path.
- Use `run_project.py` and the GUI for incremental projects. The legacy `run_pipeline.py` remains for old standalone workflows; do not run it over an upgraded project database.
- Preserve original data and successful outputs. Keep independent units isolated, repairs optional, evidence durable and historical records intact.
- PDF text extraction insufficiency is an informational note. Availability requires successful CT/US output or confirmed PDF identity. Count source series separately from phase volumes.
- Name matching requires a unique, non-contradictory identity. Never infer missing pixels, ambiguous phases or ultrasound timing. User-confirmed identity policies are local private data, not public code constants.
- Reports distinguish identity, source-data defects and missing inputs. Display specific reasons instead of vague failure wording. Use the result project's report_title; do not rename stored paths to change display labels.
- Tests use synthetic data and short temporary paths. Desktop pipeline requires Python 3.12+ with Tcl/Tk. Research resampling has a separate optional environment.
- Publish only code, synthetic tests and documentation. No scans, patient identifiers/names, policies, databases, snapshots, reports, local memory or model outputs.
- Preserve unrelated changes in the parent monorepo. Review the staged file list and diff before pushing; no force push.
- A research resampling pilot is not a full preprocessing run, segmentation, registration or intensity normalization. Keep preprocessing configuration and validation records separate from the source images.
