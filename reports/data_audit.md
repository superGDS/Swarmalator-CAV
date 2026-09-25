# Data audit

This Stage 1 study uses generated, deterministic simulation data. No observed trajectory or external traffic dataset was supplied. The raw input is the versioned scenario/configuration definition; units are SI, time is seconds, and vehicle IDs are M/R/F/B. Initial-state variants change positions and speeds explicitly. The F preparation disturbance is a bounded scripted acceleration applied only to the environment. Every output row carries method, scenario, disturbance, initial variant, vehicle, and time.

The study therefore supports implementation and mechanism checks, not empirical calibration or external validity.
